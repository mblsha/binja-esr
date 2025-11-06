use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct MetadataEnvelope {
    version: u32,
    instructions: Vec<InstructionRecord>,
}

#[derive(Deserialize)]
struct InstructionRecord {
    opcode: u32,
    mnemonic: String,
    length: u32,
    asm: String,
    il: Vec<String>,
    llil: LlilProgramRecord,
}

#[derive(Deserialize)]
struct LlilProgramRecord {
    expressions: Vec<LlilExprRecord>,
    nodes: Vec<LlilNodeRecord>,
    #[serde(default)]
    label_count: Option<u32>,
}

#[derive(Deserialize)]
struct LlilExprRecord {
    op: String,
    #[serde(rename = "full_op")]
    full_op: String,
    suffix: Option<String>,
    width: Option<u32>,
    flags: Option<String>,
    operands: Vec<LlilOperandRecord>,
    intrinsic: Option<LlilIntrinsicRecord>,
}

#[derive(Deserialize)]
struct LlilOperandRecord {
    kind: String,
    expr: Option<u32>,
    value: Option<i64>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct LlilIntrinsicRecord {
    name: String,
}

#[derive(Deserialize)]
struct LlilNodeRecord {
    kind: String,
    expr: Option<u32>,
    cond: Option<u32>,
    #[serde(rename = "true")]
    true_label: Option<u32>,
    #[serde(rename = "false")]
    false_label: Option<u32>,
    label: Option<u32>,
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let script_path = manifest_dir.join("../tools/generate_opcode_metadata.py");
    println!("cargo:rerun-if-changed={}", script_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("../pysc62015/instr/opcode_table.py")
            .display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let table_path = out_dir.join("opcode_table.rs");
    let handlers_path = out_dir.join("opcode_handlers.rs");

    match run_generator(&script_path) {
        Ok(envelope) => {
            if envelope.version != 1 {
                eprintln!(
                    "Unexpected opcode metadata version {} (expected 1); using fallback table",
                    envelope.version
                );
                write_fallback(&table_path).expect("failed to write fallback table");
                write_handler_fallback(&handlers_path).expect("failed to write fallback handlers");
            } else {
                write_table(&table_path, &envelope.instructions)
                    .expect("failed to write opcode table");
                write_handlers(&handlers_path, &envelope.instructions)
                    .expect("failed to write opcode handlers");
            }
        }
        Err(err) => {
            eprintln!("warning: opcode metadata generation failed: {err}");
            write_fallback(&table_path).expect("failed to write fallback table");
            write_handler_fallback(&handlers_path).expect("failed to write fallback handlers");
        }
    }
}

fn run_generator(script_path: &Path) -> Result<MetadataEnvelope, String> {
    if !script_path.exists() {
        return Err(format!(
            "metadata generator not found at {}",
            script_path.display()
        ));
    }

    let python = find_python().ok_or_else(|| "python3/python not found in PATH".to_string())?;
    let output = Command::new(python)
        .arg(script_path)
        .env("FORCE_BINJA_MOCK", "1")
        .output()
        .map_err(|err| format!("failed to spawn generator: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "generator exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("failed to parse generator output: {err}"))
}

fn find_python() -> Option<&'static str> {
    for candidate in ["python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Some(candidate);
        }
    }
    None
}

fn write_table(path: &Path, instructions: &[InstructionRecord]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file)?;

    let mut sorted: Vec<&InstructionRecord> = instructions.iter().collect();
    sorted.sort_by_key(|record| record.opcode);

    let mut entries: Vec<String> = Vec::with_capacity(sorted.len());

    for record in sorted {
        let opcode = (record.opcode & 0xFF) as u8;
        let mnemonic = serde_json::to_string(&record.mnemonic).unwrap();
        let asm = serde_json::to_string(&record.asm).unwrap();
        let length = (record.length.min(0xFF)) as u8;
        let il_literal = format_string_array(&record.il);

        let expr_array_name = format!("LLIL_EXPR_{opcode:02X}");
        let node_array_name = format!("LLIL_NODE_{opcode:02X}");

        write_llil_expressions(&mut file, &expr_array_name, &record.llil.expressions)?;
        write_llil_nodes(&mut file, &node_array_name, &record.llil.nodes)?;

        let label_count = record
            .llil
            .label_count
            .unwrap_or(0)
            .min(u32::from(u16::MAX)) as u16;

        let entry = format!(
            "OpcodeMetadata {{ opcode: 0x{opcode:02X}, mnemonic: {mnemonic}, length: {length}, asm: {asm}, il: {il_literal}, llil: LlilProgram {{ expressions: {expr_array_name}, nodes: {node_array_name}, label_count: {label_count} }} }}"
        );
        entries.push(entry);
    }

    writeln!(file, "pub static OPCODES: &[OpcodeMetadata] = &[")?;
    for entry in entries {
        writeln!(file, "    {entry},")?;
    }
    writeln!(file, "];")?;
    Ok(())
}

fn write_handlers(path: &Path, instructions: &[InstructionRecord]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "// @generated by build.rs — do not edit by hand\nuse crate::executor::{{ExecutionResult, LlilRuntime}};\npub type OpcodeHandler = fn(&mut LlilRuntime) -> ExecutionResult;\n"
    )?;

    let mut sorted: Vec<&InstructionRecord> = instructions.iter().collect();
    sorted.sort_by_key(|record| record.opcode);

    for (index, record) in sorted.iter().enumerate() {
        let opcode = (record.opcode & 0xFF) as u8;
        writeln!(
            file,
            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.execute_program(&crate::OPCODES[{index}].llil)\n}}\n"
        )?;
    }

    writeln!(
        file,
        "#[allow(dead_code)]\nfn handler_unimplemented(_: &mut LlilRuntime) -> ExecutionResult {{\n    Err(crate::executor::ExecutionError::UnsupportedOpcode)\n}}\n"
    )?;

    let mut handler_names = vec!["handler_unimplemented".to_string(); 256];
    for record in &sorted {
        let opcode = (record.opcode & 0xFF) as usize;
        handler_names[opcode] = format!("handler_{:02X}", opcode);
    }

    writeln!(file, "pub static OPCODE_HANDLERS: [OpcodeHandler; 256] = [")?;
    for name in handler_names {
        writeln!(file, "    {name},")?;
    }
    writeln!(file, "];")?;
    Ok(())
}

fn write_llil_expressions(
    file: &mut File,
    name: &str,
    expressions: &[LlilExprRecord],
) -> std::io::Result<()> {
    writeln!(file, "pub static {name}: &[LlilExpr] = &[")?;
    for expr in expressions {
        let operands = format_operands(&expr.operands);
        let flags = format_option_str(&expr.flags);
        let suffix = format_option_str(&expr.suffix);
        let width = format_option_u8(expr.width);
        let intrinsic = expr
            .intrinsic
            .as_ref()
            .map(|i| format!("Some({})", serde_json::to_string(&i.name).unwrap()))
            .unwrap_or_else(|| "None".to_string());

        writeln!(
            file,
            "    LlilExpr {{ op: {}, full_op: {}, suffix: {suffix}, width: {width}, flags: {flags}, operands: {operands}, intrinsic: {intrinsic} }},",
            serde_json::to_string(&expr.op).unwrap(),
            serde_json::to_string(&expr.full_op).unwrap()
        )?;
    }
    writeln!(file, "];")?;
    Ok(())
}

fn write_llil_nodes(file: &mut File, name: &str, nodes: &[LlilNodeRecord]) -> std::io::Result<()> {
    writeln!(file, "pub static {name}: &[LlilNode] = &[")?;
    for node in nodes {
        let rendered = match node.kind.as_str() {
            "expr" => {
                let expr_id = node
                    .expr
                    .expect("expr node missing expr index")
                    .min(u32::from(u16::MAX)) as u16;
                format!("LlilNode::Expr {{ expr: {expr_id} }}")
            }
            "if" => {
                let cond = node
                    .cond
                    .expect("if node missing cond index")
                    .min(u32::from(u16::MAX)) as u16;
                let true_label = node
                    .true_label
                    .expect("if node missing true label")
                    .min(u32::from(u16::MAX)) as u16;
                let false_label = node
                    .false_label
                    .expect("if node missing false label")
                    .min(u32::from(u16::MAX)) as u16;
                format!(
                    "LlilNode::If {{ cond: {cond}, true_label: {true_label}, false_label: {false_label} }}"
                )
            }
            "goto" => {
                let label = node
                    .label
                    .expect("goto node missing label")
                    .min(u32::from(u16::MAX)) as u16;
                format!("LlilNode::Goto {{ label: {label} }}")
            }
            "label" => {
                let label = node
                    .label
                    .expect("label node missing label index")
                    .min(u32::from(u16::MAX)) as u16;
                format!("LlilNode::Label {{ label: {label} }}")
            }
            other => panic!("Unsupported LLIL node kind {other}"),
        };
        writeln!(file, "    {rendered},")?;
    }
    writeln!(file, "];")?;
    Ok(())
}

fn write_header(file: &mut File) -> std::io::Result<()> {
    writeln!(
        file,
        "// @generated by build.rs — do not edit by hand\n\
         #[derive(Debug)]\n\
         pub enum LlilOperand {{\n    Expr(u16),\n    Imm(i64),\n    Reg(&'static str),\n    Flag(&'static str),\n    None,\n}}\n"
    )?;
    writeln!(
        file,
        "#[derive(Debug)]\n\
         pub struct LlilExpr {{\n    pub op: &'static str,\n    pub full_op: &'static str,\n    pub suffix: Option<&'static str>,\n    pub width: Option<u8>,\n    pub flags: Option<&'static str>,\n    pub operands: &'static [LlilOperand],\n    pub intrinsic: Option<&'static str>,\n}}\n"
    )?;
    writeln!(
        file,
        "#[derive(Debug)]\n\
         pub enum LlilNode {{\n    Expr {{ expr: u16 }},\n    If {{ cond: u16, true_label: u16, false_label: u16 }},\n    Goto {{ label: u16 }},\n    Label {{ label: u16 }},\n}}\n"
    )?;
    writeln!(
        file,
        "#[derive(Debug)]\n\
         pub struct LlilProgram {{\n    pub expressions: &'static [LlilExpr],\n    pub nodes: &'static [LlilNode],\n    pub label_count: u16,\n}}\n"
    )?;
    writeln!(
        file,
        "#[derive(Debug)]\n\
         pub struct OpcodeMetadata {{\n    pub opcode: u8,\n    pub mnemonic: &'static str,\n    pub length: u8,\n    pub asm: &'static str,\n    pub il: &'static [&'static str],\n    pub llil: LlilProgram,\n}}\n"
    )?;
    Ok(())
}

fn write_fallback(path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file)?;
    writeln!(file, "pub static OPCODES: &[OpcodeMetadata] = &[];")?;
    Ok(())
}

fn write_handler_fallback(path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "// @generated fallback\nuse crate::executor::{{ExecutionResult, LlilRuntime}};\npub type OpcodeHandler = fn(&mut LlilRuntime) -> ExecutionResult;\n#[allow(dead_code)]\nfn handler_unimplemented(_: &mut LlilRuntime) -> ExecutionResult {{ Err(crate::executor::ExecutionError::UnsupportedOpcode) }}\npub static OPCODE_HANDLERS: [OpcodeHandler; 256] = [handler_unimplemented; 256];"
    )?;
    Ok(())
}

fn format_string_array(values: &[String]) -> String {
    if values.is_empty() {
        "&[]".to_string()
    } else {
        let entries: Vec<String> = values
            .iter()
            .map(|line| serde_json::to_string(line).unwrap())
            .collect();
        format!("&[{}]", entries.join(", "))
    }
}

fn format_operands(operands: &[LlilOperandRecord]) -> String {
    if operands.is_empty() {
        "&[]".to_string()
    } else {
        let parts: Vec<String> = operands
            .iter()
            .map(|operand| match operand.kind.as_str() {
                "expr" => {
                    let expr = operand
                        .expr
                        .expect("expr operand missing index")
                        .min(u32::from(u16::MAX)) as u16;
                    format!("LlilOperand::Expr({expr})")
                }
                "imm" => {
                    let value = operand.value.expect("imm operand missing value");
                    format!("LlilOperand::Imm({value})")
                }
                "reg" => {
                    let name = operand.name.as_ref().expect("reg operand missing name");
                    format!("LlilOperand::Reg({})", serde_json::to_string(name).unwrap())
                }
                "flag" => {
                    let name = operand.name.as_ref().expect("flag operand missing name");
                    format!(
                        "LlilOperand::Flag({})",
                        serde_json::to_string(name).unwrap()
                    )
                }
                "none" => "LlilOperand::None".to_string(),
                other => panic!("Unsupported operand kind {other}"),
            })
            .collect();
        format!("&[{}]", parts.join(", "))
    }
}

fn format_option_str(opt: &Option<String>) -> String {
    opt.as_ref()
        .map(|value| format!("Some({})", serde_json::to_string(value).unwrap()))
        .unwrap_or_else(|| "None".to_string())
}

fn format_option_u8(opt: Option<u32>) -> String {
    opt.map(|value| {
        let clamped = value.min(u32::from(u8::MAX)) as u8;
        format!("Some({clamped})")
    })
    .unwrap_or_else(|| "None".to_string())
}
