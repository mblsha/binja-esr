use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct MetadataEnvelope {
    version: u32,
    instructions: Vec<InstructionRecord>,
}

#[derive(Clone, Deserialize)]
#[allow(dead_code)]
struct LoweringPlanEnvelope {
    instructions: Vec<LoweringInstructionRecord>,
}

#[derive(Clone, Deserialize)]
#[allow(dead_code)]
struct LoweringInstructionRecord {
    opcode: u32,
    mnemonic: String,
    length: u32,
    expressions: Vec<LoweringExprRecord>,
    nodes: Vec<LlilNodeRecord>,
}

impl LoweringInstructionRecord {
    fn is_side_effect_free(&self) -> bool {
        self.expressions.iter().all(|expr| !expr.side_effect)
    }
}

#[derive(Clone, Deserialize)]
#[allow(dead_code)]
struct LoweringExprRecord {
    index: u32,
    op: String,
    full_op: String,
    width: Option<u32>,
    flags: Option<String>,
    suffix: Option<String>,
    deps: Vec<u32>,
    side_effect: bool,
    intrinsic: Option<String>,
    operands: Vec<LlilOperandRecord>,
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

#[derive(Clone, Deserialize)]
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

#[derive(Clone, Deserialize)]
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
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let project_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&manifest_dir)
        .to_path_buf();
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
    let metadata_json_path = out_dir.join("opcode_metadata.json");
    let lowering_plan_path = out_dir.join("lowering_plan.json");

    match run_generator(
        &script_path,
        &metadata_json_path,
        &lowering_plan_path,
        &project_root,
    ) {
        Ok((envelope, lowering_plan)) => {
            if envelope.version != 1 {
                eprintln!(
                    "Unexpected opcode metadata version {} (expected 1); using fallback table",
                    envelope.version
                );
                write_fallback(&table_path).expect("failed to write fallback table");
                write_handler_fallback(&handlers_path).expect("failed to write fallback handlers");
            } else {
                let lowering_map: HashMap<u8, LoweringInstructionRecord> = lowering_plan
                    .instructions
                    .into_iter()
                    .map(|entry| (((entry.opcode & 0xFF) as u8), entry))
                    .collect();
                #[cfg(feature = "print_lowering_stats")]
                {
                    let count = lowering_map
                        .values()
                        .filter(|plan| plan.is_side_effect_free())
                        .count();
                    println!("cargo:warning={} side-effect-free opcodes", count);
                }
                println!(
                    "cargo:warning=lowering plan recorded at {}",
                    lowering_plan_path.display()
                );
                write_table(&table_path, &envelope.instructions)
                    .expect("failed to write opcode table");
                let coverage =
                    write_handlers(&handlers_path, &envelope.instructions, Some(&lowering_map))
                        .expect("failed to write opcode handlers");
                println!(
                    "cargo:warning=opcode lowering coverage: {} specialized / {} total ({} LLIL fallbacks)",
                    coverage.specialized_count(),
                    envelope.instructions.len(),
                    coverage.fallback_count()
                );
            }
        }
        Err(err) => {
            eprintln!("warning: opcode metadata generation failed: {err}");
            write_fallback(&table_path).expect("failed to write fallback table");
            write_handler_fallback(&handlers_path).expect("failed to write fallback handlers");
        }
    }
}

fn run_generator(
    script_path: &Path,
    metadata_json_path: &Path,
    lowering_plan_path: &Path,
    project_root: &Path,
) -> Result<(MetadataEnvelope, LoweringPlanEnvelope), String> {
    if !script_path.exists() {
        return Err(format!(
            "metadata generator not found at {}",
            script_path.display()
        ));
    }

    let python = find_python().ok_or_else(|| "python3/python not found in PATH".to_string())?;
    let mut pythonpath = project_root.as_os_str().to_os_string();
    for version in ["python3.12", "python3.11", "python3.10"] {
        let venv_site = project_root
            .join(".venv")
            .join("lib")
            .join(version)
            .join("site-packages");
        if venv_site.exists() {
            pythonpath.push(OsString::from(":"));
            pythonpath.push(venv_site.as_os_str());
        }
    }
    let output = Command::new(&python)
        .arg(script_path)
        .arg("--output")
        .arg(metadata_json_path)
        .arg("--lowering-plan")
        .arg(lowering_plan_path)
        .env("FORCE_BINJA_MOCK", "1")
        .env("PYTHONPATH", pythonpath)
        .output()
        .map_err(|err| format!("failed to spawn generator: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "generator exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let metadata_bytes = fs::read(metadata_json_path).map_err(|err| {
        format!(
            "failed to read metadata json {}: {err}",
            metadata_json_path.display()
        )
    })?;
    let metadata: MetadataEnvelope = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| format!("failed to parse generator output: {err}"))?;
    let lowering_plan_bytes = fs::read(lowering_plan_path).map_err(|err| {
        format!(
            "failed to read lowering plan {}: {err}",
            lowering_plan_path.display()
        )
    })?;
    let lowering_plan: LoweringPlanEnvelope = serde_json::from_slice(&lowering_plan_bytes)
        .map_err(|err| format!("failed to parse lowering plan: {err}"))?;
    Ok((metadata, lowering_plan))
}

fn find_python() -> Option<String> {
    if let Ok(explicit) = env::var("PYTHON_GENERATOR") {
        return Some(explicit);
    }
    for candidate in ["python3.12", "python3.11", "python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Some(candidate.to_string());
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
    let mut call_effects: Vec<i8> = Vec::with_capacity(sorted.len());

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
        call_effects.push(call_stack_effect(record));
    }

    writeln!(file, "pub static OPCODES: &[OpcodeMetadata] = &[")?;
    for entry in entries {
        writeln!(file, "    {entry},")?;
    }
    writeln!(file, "];")?;
    writeln!(
        file,
        "pub static CALL_STACK_EFFECTS: [i8; {}] = [",
        call_effects.len()
    )?;
    for effect in call_effects {
        writeln!(file, "    {effect},")?;
    }
    writeln!(file, "];")?;
    Ok(())
}

struct HandlerStats {
    specialized: Vec<u8>,
    fallback: Vec<u8>,
}

impl HandlerStats {
    fn new() -> Self {
        Self {
            specialized: Vec::new(),
            fallback: Vec::new(),
        }
    }

    fn mark_specialized(&mut self, opcode: u8) {
        self.specialized.push(opcode);
    }

    fn mark_fallback(&mut self, opcode: u8) {
        self.fallback.push(opcode);
    }

    fn finalize(&mut self) {
        self.specialized.sort_unstable();
        self.fallback.sort_unstable();
    }

    fn specialized_count(&self) -> usize {
        self.specialized.len()
    }

    fn fallback_count(&self) -> usize {
        self.fallback.len()
    }
}

fn emit_specialized(
    file: &mut File,
    stats: &mut HandlerStats,
    opcode: u8,
    handler: String,
) -> std::io::Result<()> {
    stats.mark_specialized(opcode);
    writeln!(file, "{handler}")
}

fn write_handlers(
    path: &Path,
    instructions: &[InstructionRecord],
    lowering_map: Option<&HashMap<u8, LoweringInstructionRecord>>,
) -> std::io::Result<HandlerStats> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "// @generated by build.rs — do not edit by hand\nuse crate::executor::{{ExecutionResult, LlilRuntime}};\nuse crate::lowering;\npub type OpcodeHandler = fn(&mut LlilRuntime) -> ExecutionResult;\n"
    )?;

    let mut sorted: Vec<&InstructionRecord> = instructions.iter().collect();
    sorted.sort_by_key(|record| record.opcode);
    let mut stats = HandlerStats::new();

    for (index, record) in sorted.iter().enumerate() {
        let opcode = (record.opcode & 0xFF) as u8;
        let length = record.length.min(0xFF) as u8;

        if opcode == 0x08 {
            emit_specialized(
                &mut file,
                &mut stats,
                opcode,
                format!(
                    "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let next_pc = ctx.read_named_register(\"PC\")?;\n    let base = next_pc - ({length} as i64);\n    let value = ctx.read_memory_value(base + 1, 1)?;\n    ctx.write_named_register(\"A\", value, Some(1))?;\n    Ok(())\n}}\n"
                ),
            )?;
            continue;
        }

        if opcode == 0x40 {
            emit_specialized(
                &mut file,
                &mut stats,
                opcode,
                format!(
                    "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let next_pc = ctx.read_named_register(\"PC\")?;\n    let base = next_pc - ({length} as i64);\n    let lhs = ctx.read_named_register(\"A\")?;\n    let rhs = ctx.read_memory_value(base + 1, 1)?;\n    let (value, flags) = lowering::add(1, lhs, rhs);\n    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register(\"A\", value, Some(1))?;\n    Ok(())\n}}\n"
                ),
            )?;
            continue;
        }

        if opcode == 0xEF {
            emit_specialized(
                &mut file,
                &mut stats,
                opcode,
                format!(
                    "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    ctx.write_named_register(\"I\", 0, Some(2))?;\n    Ok(())\n}}\n"
                ),
            )?;
            continue;
        }
        if let Some(plan_map) = lowering_map {
            if let Some(plan) = plan_map.get(&opcode) {
                if let Some(handler) = try_emit_const_set_reg(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_reg_move(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_simple_add(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_simple_sub(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_simple_logical(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_load_from_reg(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_call(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_jumps(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_ret(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_store(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_rotate(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_rotate_through_carry(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_push_only(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_pop_flags(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_flag_only_binary(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_flag_assignments(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_intrinsic(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                if let Some(handler) = try_emit_unimpl(opcode, length, plan) {
                    emit_specialized(&mut file, &mut stats, opcode, handler)?;
                    continue;
                }
                writeln!(
                    file,
                    "// lowering plan: {} expressions, {} CFG nodes",
                    plan.expressions.len(),
                    plan.nodes.len()
                )?;
                if plan.is_side_effect_free() {
                    emit_specialized(
                        &mut file,
                        &mut stats,
                        opcode,
                        format!(
                            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    Ok(())\n}}\n"
                        ),
                    )?;
                    continue;
                }
            } else {
                writeln!(
                    file,
                    "// lowering plan unavailable for opcode 0x{opcode:02X}"
                )?;
            }
        }
        stats.mark_fallback(opcode);
        writeln!(
            file,
            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    ctx.execute_program(&crate::OPCODES[{index}].llil)\n}}\n"
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

    stats.finalize();
    writeln!(
        file,
        "pub const OPCODE_LOWERING_SPECIALIZED: usize = {};",
        stats.specialized.len()
    )?;
    writeln!(
        file,
        "pub const OPCODE_LOWERING_FALLBACK: usize = {};",
        stats.fallback.len()
    )?;
    writeln!(
        file,
        "pub const OPCODE_LLIL_FALLBACKS: [u8; {}] = {:?};",
        stats.fallback.len(),
        stats.fallback
    )?;
    Ok(stats)
}

fn try_emit_const_set_reg(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let source = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    if setter.op != "SET_REG" {
        return None;
    }
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    if source.op != "CONST" && source.op != "CONST_PTR" {
        return None;
    }
    let value = source.operands.get(0).and_then(|operand| operand.value)?;
    let width_expr = format_option_u8_literal(setter.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = {value}_i64;\n    ctx.write_named_register({:?}, value, {width_expr})?;\n    Ok(())\n}}\n",
        target_reg
    ))
}

fn try_emit_reg_move(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let source = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    if source.op != "REG" {
        return None;
    }
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let src_reg = source
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width_expr = format_option_u8_literal(setter.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = ctx.read_named_register({:?})?;\n    ctx.write_named_register({:?}, value, {width_expr})?;\n    Ok(())\n}}\n",
        src_reg,
        target_reg
    ))
}

fn call_stack_effect(record: &InstructionRecord) -> i8 {
    let mut saw_push = false;
    let mut saw_jump = false;
    for expr in &record.llil.expressions {
        match expr.op.as_str() {
            "RET" => return -1,
            "CALL" => return 1,
            "PUSH" => saw_push = true,
            "JUMP" => saw_jump = true,
            _ => {}
        }
    }
    if saw_push && saw_jump {
        1
    } else {
        0
    }
}

fn try_emit_simple_add(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let add_expr = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    if add_expr.op != "ADD" {
        return None;
    }
    if add_expr.deps.len() != 2 {
        return None;
    }
    let lhs = value_snippet(plan, add_expr.deps[0])?;
    let rhs = value_snippet(plan, add_expr.deps[1])?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = add_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let lhs = {lhs};\n    let rhs = {rhs};\n    let (value, flags) = lowering::add({width}, lhs, rhs);\n    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register({:?}, value, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg,
        width_literal = width_literal,
    ))
}

fn try_emit_simple_sub(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let sub_expr = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    if sub_expr.op != "SUB" {
        return None;
    }
    if sub_expr.deps.len() != 2 {
        return None;
    }
    let lhs = value_snippet(plan, sub_expr.deps[0])?;
    let rhs = value_snippet(plan, sub_expr.deps[1])?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = sub_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let lhs = {lhs};\n    let rhs = {rhs};\n    let (value, flags) = lowering::sub({width}, lhs, rhs);\n    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register({:?}, value, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg,
    ))
}

fn try_emit_simple_logical(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let logic_expr = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    let helper = match logic_expr.op.as_str() {
        "AND" => "and",
        "OR" => "or",
        "XOR" => "xor",
        _ => return None,
    };
    if logic_expr.deps.len() != 2 {
        return None;
    }
    let lhs = value_snippet(plan, logic_expr.deps[0])?;
    let rhs = value_snippet(plan, logic_expr.deps[1])?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = logic_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    let flags_spec = logic_expr.flags.as_deref().unwrap_or_default();
    let writes_carry = flags_spec.contains('C');
    let writes_zero = flags_spec.contains('Z');
    let mut flag_mask = String::new();
    if !writes_carry {
        flag_mask.push_str("    flags.carry = None;\n");
    }
    if !writes_zero {
        flag_mask.push_str("    flags.zero = None;\n");
    }
    let mut_kw = if flag_mask.is_empty() { "" } else { "mut " };
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let lhs = {lhs};\n    let rhs = {rhs};\n    let (value, {mut_kw}flags) = lowering::{helper}({width}, lhs, rhs);\n{flag_mask}    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register({:?}, value, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg,
    ))
}

fn try_emit_load_from_reg(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let load_expr = plan
        .expressions
        .iter()
        .find(|expr| expr.index == setter.deps[0])?;
    if load_expr.op != "LOAD" {
        return None;
    }
    let address_expr_index = load_expr.deps.get(0)?;
    let address_snippet = value_snippet(plan, *address_expr_index)?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = load_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let addr = {address_snippet};\n    let value = ctx.read_memory_value(addr, {width})?;\n    ctx.write_named_register({:?}, value, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg
    ))
}

fn value_snippet(plan: &LoweringInstructionRecord, index: u32) -> Option<String> {
    let expr = find_expr(plan, index)?;
    match expr.op.as_str() {
        "REG" => {
            let name = expr
                .operands
                .get(0)
                .and_then(|operand| operand.name.as_deref())?;
            Some(format!("ctx.read_named_register({:?})?", name))
        }
        "FLAG" => {
            let name = expr
                .operands
                .get(0)
                .and_then(|operand| operand.name.as_deref())?;
            Some(format!("ctx.read_flag({:?})?", name))
        }
        "CONST" | "CONST_PTR" => {
            let value = expr.operands.get(0).and_then(|operand| operand.value)?;
            Some(format!("{value}_i64"))
        }
        "ADD" => value_binary_lowering(plan, expr, "add"),
        "SUB" => value_binary_lowering(plan, expr, "sub"),
        "AND" => value_binary_lowering(plan, expr, "and"),
        "OR" => value_binary_lowering(plan, expr, "or"),
        "XOR" => value_binary_lowering(plan, expr, "xor"),
        "CMP_E" => {
            if expr.deps.len() != 2 {
                return None;
            }
            let lhs = value_snippet(plan, expr.deps[0])?;
            let rhs = value_snippet(plan, expr.deps[1])?;
            Some(format!("lowering::cmp_eq({lhs}, {rhs})"))
        }
        "CMP_SLT" => {
            if expr.deps.len() != 2 {
                return None;
            }
            let lhs = value_snippet(plan, expr.deps[0])?;
            let rhs = value_snippet(plan, expr.deps[1])?;
            let width = clamp_width(expr.width);
            Some(format!("lowering::cmp_slt({width}, {lhs}, {rhs})"))
        }
        "CMP_UGT" => {
            if expr.deps.len() != 2 {
                return None;
            }
            let lhs = value_snippet(plan, expr.deps[0])?;
            let rhs = value_snippet(plan, expr.deps[1])?;
            let width = clamp_width(expr.width);
            Some(format!("lowering::cmp_ugt({width}, {lhs}, {rhs})"))
        }
        "LOAD" => {
            let address_expr = *expr.deps.get(0)?;
            let address = value_snippet(plan, address_expr)?;
            let width = clamp_width(expr.width);
            Some(format!(
                "{{ let addr = {address}; ctx.read_memory_value(addr, {width})? }}"
            ))
        }
        "POP" => {
            let width = clamp_width(expr.width);
            Some(format!("ctx.pop_value({width})?"))
        }
        "ROL" | "ROR" => {
            if expr.deps.len() != 2 {
                return None;
            }
            let value = value_snippet(plan, expr.deps[0])?;
            let count = value_snippet(plan, expr.deps[1])?;
            let width = clamp_width(expr.width);
            let left = if expr.op == "ROL" { "true" } else { "false" };
            Some(format!(
                "lowering::rotate({width}, {value}, {count}, {left}).0"
            ))
        }
        "LSL" => shift_snippet(plan, expr, true),
        "LSR" => shift_snippet(plan, expr, false),
        "RLC" | "RRC" => {
            if expr.deps.len() != 3 {
                return None;
            }
            let value = value_snippet(plan, expr.deps[0])?;
            let count_expr = find_expr(plan, expr.deps[1])?;
            let count = count_expr
                .operands
                .get(0)
                .and_then(|operand| operand.value)?;
            if count != 1 {
                return None;
            }
            let carry_in = value_snippet(plan, expr.deps[2])?;
            let width = clamp_width(expr.width);
            let left = if expr.op == "RLC" { "true" } else { "false" };
            Some(format!(
                "lowering::rotate_through_carry({width}, {value}, {carry_in}, {left}).0"
            ))
        }
        _ => None,
    }
}

fn try_emit_store(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let store = plan.expressions.iter().find(|expr| expr.op == "STORE")?;
    if store.deps.len() != 2 {
        return None;
    }
    let address_snippet = value_snippet(plan, store.deps[0])?;
    let value_snippet_str = value_snippet(plan, store.deps[1])?;
    let width = clamp_width(store.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let addr = {address_snippet};\n    let value = {value_snippet_str};\n    ctx.write_memory_value(addr, {width}, value)?;\n    Ok(())\n}}\n"
    ))
}

fn try_emit_call(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    if let Some(push) = plan.expressions.iter().find(|expr| expr.op == "PUSH") {
        let jump = plan.expressions.iter().find(|expr| expr.op == "JUMP")?;
        let width = clamp_width(push.width);
        let target = value_snippet(plan, *jump.deps.get(0)?)?;
        return Some(format!(
            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let target = {target};\n    ctx.call_absolute({width}, target)?;\n    Ok(())\n}}\n"
        ));
    }
    let call = plan.expressions.iter().find(|expr| expr.op == "CALL")?;
    if call.deps.len() != 1 {
        return None;
    }
    let target = value_snippet(plan, call.deps[0])?;
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let target = {target};\n    ctx.call_absolute(3, target)?;\n    Ok(())\n}}\n"
    ))
}

fn try_emit_ret(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let ret = plan.expressions.iter().find(|expr| expr.op == "RET")?;
    if ret.deps.len() != 1 {
        return None;
    }
    let target = value_snippet(plan, ret.deps[0])?;
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let target = {target};\n    ctx.write_named_register(\"PC\", target, Some(3))?;\n    Ok(())\n}}\n"
    ))
}

fn try_emit_jumps(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    if !plan.nodes.iter().any(|node| node.kind == "expr") {
        return None;
    }
    let jump_expr = plan.expressions.iter().find(|expr| expr.op == "JUMP")?;
    let target = value_snippet(plan, *jump_expr.deps.get(0)?)?;
    let condition = plan
        .nodes
        .iter()
        .find(|node| node.kind == "if")
        .and_then(|node| value_snippet(plan, node.cond?));
    if let Some(cond) = condition {
        Some(format!(
            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let cond = {cond};\n    if cond != 0 {{\n        let target = {target};\n        ctx.write_named_register(\"PC\", target, Some(3))?;\n    }}\n    Ok(())\n}}\n"
        ))
    } else {
        Some(format!(
            "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let target = {target};\n    ctx.write_named_register(\"PC\", target, Some(3))?;\n    Ok(())\n}}\n"
        ))
    }
}

fn try_emit_rotate(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let rotate_expr = find_expr(plan, setter.deps[0])?;
    let (helper, left) = match rotate_expr.op.as_str() {
        "ROR" => ("rotate", "false"),
        "ROL" => ("rotate", "true"),
        _ => return None,
    };
    if rotate_expr.deps.len() != 2 {
        return None;
    }
    let value = value_snippet(plan, rotate_expr.deps[0])?;
    let count = value_snippet(plan, rotate_expr.deps[1])?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = rotate_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = {value};\n    let count = {count};\n    let (result, flags) = lowering::{helper}({width}, value, count, {left});\n    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register({:?}, result, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg
    ))
}

fn try_emit_rotate_through_carry(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    let setter = plan.expressions.iter().find(|expr| expr.op == "SET_REG")?;
    if setter.deps.len() != 1 {
        return None;
    }
    let rotate_expr = find_expr(plan, setter.deps[0])?;
    let left = match rotate_expr.op.as_str() {
        "RRC" => "false",
        "RLC" => "true",
        _ => return None,
    };
    if rotate_expr.deps.len() != 3 {
        return None;
    }
    let count_expr = find_expr(plan, rotate_expr.deps[1])?;
    if extract_const_value(count_expr) != Some(1) {
        return None;
    }
    let value = value_snippet(plan, rotate_expr.deps[0])?;
    let carry = value_snippet(plan, rotate_expr.deps[2])?;
    let target_reg = setter
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?
        .to_string();
    let width = rotate_expr.width.or(setter.width).unwrap_or(1);
    let width_literal = format_option_u8_literal(Some(width));
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = {value};\n    let carry = {carry};\n    let (result, flags) = lowering::rotate_through_carry({width}, value, carry, {left});\n    ctx.apply_op_flags(flags)?;\n    ctx.write_named_register({:?}, result, {width_literal})?;\n    Ok(())\n}}\n",
        target_reg
    ))
}

fn try_emit_push_only(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let push = plan.expressions.iter().find(|expr| expr.op == "PUSH")?;
    if plan
        .expressions
        .iter()
        .any(|expr| expr.side_effect && expr.index != push.index)
    {
        return None;
    }
    let value_index = *push.deps.get(0)?;
    let value = value_snippet(plan, value_index)?;
    let width = clamp_width(push.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = {value};\n    ctx.push_value({width}, value)?;\n    Ok(())\n}}\n"
    ))
}

fn try_emit_pop_flags(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    let pop = plan.expressions.iter().find(|expr| expr.op == "POP")?;
    if plan
        .expressions
        .iter()
        .any(|expr| expr.side_effect && !matches!(expr.op.as_str(), "POP" | "SET_REG" | "SET_FLAG"))
    {
        return None;
    }
    let temp_assign = plan.expressions.iter().find(|expr| {
        expr.op == "SET_REG"
            && expr
                .deps
                .get(0)
                .map(|dep| *dep == pop.index)
                .unwrap_or(false)
    })?;
    let temp_name = temp_assign
        .operands
        .get(0)
        .and_then(|operand| operand.name.as_deref())?;
    if !temp_name.starts_with("TEMP") {
        return None;
    }
    let mut flag_lines = String::new();
    let mut found = 0;
    for setter in plan.expressions.iter().filter(|expr| expr.op == "SET_FLAG") {
        let flag_name = setter
            .operands
            .get(0)
            .and_then(|operand| operand.name.as_deref())?;
        let source_index = *setter.deps.get(0)?;
        let source = find_expr(plan, source_index)?;
        let mask = match source.op.as_str() {
            "AND" => parse_temp_mask(plan, source, temp_name)?,
            _ => return None,
        };
        let mask_literal = format!("{mask}_i64");
        flag_lines.push_str(&format!(
            "    ctx.write_flag({:?}, ((value & {mask_literal}) != 0) as i64)?;\n",
            flag_name,
        ));
        found += 1;
    }
    if found == 0 {
        return None;
    }
    let width = clamp_width(pop.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let value = ctx.pop_value({width})?;\n{flag_lines}    Ok(())\n}}\n"
    ))
}

fn try_emit_flag_only_binary(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    const BLOCKING_OPS: &[&str] = &[
        "SET_REG",
        "SET_FLAG",
        "STORE",
        "PUSH",
        "POP",
        "CALL",
        "RET",
        "JUMP",
        "INTRINSIC",
        "UNIMPL",
    ];
    if plan
        .expressions
        .iter()
        .any(|expr| BLOCKING_OPS.contains(&expr.op.as_str()))
    {
        return None;
    }
    let flag_expr = plan.expressions.iter().find(|expr| {
        matches!(expr.op.as_str(), "ADD" | "SUB" | "AND" | "OR" | "XOR")
            && expr
                .flags
                .as_deref()
                .map(|f| !f.is_empty())
                .unwrap_or(false)
    })?;
    if flag_expr.deps.len() != 2 {
        return None;
    }
    let helper = match flag_expr.op.as_str() {
        "ADD" => "add",
        "SUB" => "sub",
        "AND" => "and",
        "OR" => "or",
        "XOR" => "xor",
        _ => return None,
    };
    let lhs = value_snippet(plan, flag_expr.deps[0])?;
    let rhs = value_snippet(plan, flag_expr.deps[1])?;
    let width = clamp_width(flag_expr.width);
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    let lhs = {lhs};\n    let rhs = {rhs};\n    let (_, flags) = lowering::{helper}({width}, lhs, rhs);\n    ctx.apply_op_flags(flags)?;\n    Ok(())\n}}\n"
    ))
}

fn try_emit_flag_assignments(
    opcode: u8,
    length: u8,
    plan: &LoweringInstructionRecord,
) -> Option<String> {
    const BLOCKING_OPS: &[&str] = &[
        "SET_REG",
        "STORE",
        "PUSH",
        "POP",
        "CALL",
        "RET",
        "JUMP",
        "INTRINSIC",
        "UNIMPL",
    ];
    if plan
        .expressions
        .iter()
        .any(|expr| BLOCKING_OPS.contains(&expr.op.as_str()))
    {
        return None;
    }
    let setters: Vec<&LoweringExprRecord> = plan
        .expressions
        .iter()
        .filter(|expr| expr.op == "SET_FLAG")
        .collect();
    if setters.is_empty() {
        return None;
    }
    let mut body = String::new();
    for setter in setters {
        let source = *setter.deps.get(0)?;
        let value = value_snippet(plan, source)?;
        let flag_name = setter
            .operands
            .get(0)
            .and_then(|operand| operand.name.as_deref())?;
        body.push_str(&format!(
            "    let value = {value};\n    ctx.write_flag({:?}, value)?;\n",
            flag_name
        ));
    }
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n{body}    Ok(())\n}}\n"
    ))
}

fn try_emit_intrinsic(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    if plan
        .expressions
        .iter()
        .any(|expr| expr.side_effect && expr.op != "INTRINSIC")
    {
        return None;
    }
    let intrinsic = plan
        .expressions
        .iter()
        .find(|expr| expr.op == "INTRINSIC")?;
    let name = intrinsic.intrinsic.as_deref()?;
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    ctx.invoke_intrinsic({:?})?;\n    Ok(())\n}}\n",
        name
    ))
}

fn value_binary_lowering(
    plan: &LoweringInstructionRecord,
    expr: &LoweringExprRecord,
    helper: &str,
) -> Option<String> {
    if expr.deps.len() != 2 {
        return None;
    }
    let lhs = value_snippet(plan, expr.deps[0])?;
    let rhs = value_snippet(plan, expr.deps[1])?;
    let width = clamp_width(expr.width);
    Some(format!("lowering::{helper}({width}, {lhs}, {rhs}).0"))
}

fn shift_snippet(
    plan: &LoweringInstructionRecord,
    expr: &LoweringExprRecord,
    left: bool,
) -> Option<String> {
    if expr.deps.len() != 2 {
        return None;
    }
    let value = value_snippet(plan, expr.deps[0])?;
    let count = value_snippet(plan, expr.deps[1])?;
    let width = clamp_width(expr.width);
    let helper = if left { "shift_left" } else { "shift_right" };
    Some(format!("lowering::{helper}({width}, {value}, {count}).0"))
}

fn parse_temp_mask(
    plan: &LoweringInstructionRecord,
    expr: &LoweringExprRecord,
    temp_name: &str,
) -> Option<i64> {
    if expr.deps.len() != 2 {
        return None;
    }
    let first = find_expr(plan, expr.deps[0])?;
    let second = find_expr(plan, expr.deps[1])?;
    if is_temp_reg(first, temp_name) {
        return extract_const_value(second);
    }
    if is_temp_reg(second, temp_name) {
        return extract_const_value(first);
    }
    None
}

fn is_temp_reg(expr: &LoweringExprRecord, temp_name: &str) -> bool {
    expr.op == "REG"
        && expr
            .operands
            .get(0)
            .and_then(|operand| operand.name.as_deref())
            .map(|name| name == temp_name)
            .unwrap_or(false)
}

fn extract_const_value(expr: &LoweringExprRecord) -> Option<i64> {
    if expr.op != "CONST" && expr.op != "CONST_PTR" {
        return None;
    }
    expr.operands.get(0).and_then(|operand| operand.value)
}

fn try_emit_unimpl(opcode: u8, length: u8, plan: &LoweringInstructionRecord) -> Option<String> {
    if plan.expressions.iter().any(|expr| expr.op != "UNIMPL") {
        return None;
    }
    Some(format!(
        "#[allow(clippy::needless_pass_by_value, non_snake_case)]\npub fn handler_{opcode:02X}(ctx: &mut LlilRuntime) -> ExecutionResult {{\n    ctx.prepare_for_opcode(0x{opcode:02X}, {length});\n    Err(crate::executor::ExecutionError::Unimplemented(\"opcode 0x{opcode:02X}\"))\n}}\n"
    ))
}

fn find_expr<'a>(
    plan: &'a LoweringInstructionRecord,
    index: u32,
) -> Option<&'a LoweringExprRecord> {
    plan.expressions.iter().find(|expr| expr.index == index)
}

fn clamp_width(width: Option<u32>) -> u8 {
    width
        .map(|value| value.min(u32::from(u8::MAX)) as u8)
        .unwrap_or(1)
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
    writeln!(file, "pub static CALL_STACK_EFFECTS: [i8; 256] = [0; 256];")?;
    Ok(())
}

fn write_handler_fallback(path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let fallback_opcodes: Vec<u8> = (0..=255).collect();
    writeln!(
        file,
        "// @generated fallback\nuse crate::executor::{{ExecutionResult, LlilRuntime}};\npub type OpcodeHandler = fn(&mut LlilRuntime) -> ExecutionResult;\n#[allow(dead_code)]\nfn handler_unimplemented(_: &mut LlilRuntime) -> ExecutionResult {{ Err(crate::executor::ExecutionError::UnsupportedOpcode) }}\npub static OPCODE_HANDLERS: [OpcodeHandler; 256] = [handler_unimplemented; 256];\npub const OPCODE_LOWERING_SPECIALIZED: usize = 0;\npub const OPCODE_LOWERING_FALLBACK: usize = 256;\npub const OPCODE_LLIL_FALLBACKS: [u8; 256] = {:?};",
        fallback_opcodes
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

fn format_option_u8_literal(opt: Option<u32>) -> String {
    opt.map(|value| format!("Some({})", value.min(u32::from(u8::MAX)) as u8))
        .unwrap_or_else(|| "None".to_string())
}
