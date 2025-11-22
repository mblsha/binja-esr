use rust_scil::{
    ast::{Binder, Expr, Instr, PreLatch},
    bus::Bus,
    eval,
    state::State as RsState,
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Minimal view over the generated layout entry used by the decoder.
pub trait LayoutEntryView {
    fn key(&self) -> &str;
    fn kind(&self) -> &str;
    fn meta(&self) -> &HashMap<String, Value>;
}

/// Minimal view over the manifest entry used by the eval loop.
pub trait ManifestEntryView {
    type Layout: LayoutEntryView;

    fn opcode(&self) -> u8;
    fn pre(&self) -> Option<(String, String)>;
    fn binder(&self) -> &Map<String, Value>;
    fn instr(&self) -> &Value;
    fn layout(&self) -> &[Self::Layout];
    fn parsed_instr(&self) -> Option<&Instr> {
        None
    }
    fn parsed_binder(&self) -> Option<&Binder> {
        None
    }
}

/// Minimal view over a bound instruction (decoded operands).
pub trait BoundInstrView {
    fn opcode(&self) -> u32;
    fn operands(&self) -> &HashMap<String, Value>;
    fn pre(&self) -> Option<(String, String)>;
}

/// Execute a manifest entry against the provided bus/state using the generic
/// manifest/bound views.
pub fn eval_manifest_entry<M, B, BUS>(
    manifest: &[M],
    state: &mut RsState,
    bus: &mut BUS,
    bound: &B,
) -> Result<(), String>
where
    M: ManifestEntryView,
    B: BoundInstrView,
    BUS: Bus,
{
    let entry = find_entry(manifest, bound).ok_or_else(|| {
        format!(
            "no manifest entry for opcode {} pre {:?}",
            bound.opcode(),
            bound.pre()
        )
    })?;
    let binder_json = patch_binder(entry.binder(), bound.operands());
    let binder: Binder = if let Some(parsed) = entry.parsed_binder() {
        // Fast path: start from cached binder and override with patched values.
        merge_binder(parsed, &binder_json)
    } else {
        serde_json::from_value(Value::Object(binder_json))
            .map_err(|e| format!("binder json: {e}"))?
    };
    let instr = if let Some(parsed) = entry.parsed_instr() {
        parsed.clone()
    } else {
        serde_json::from_value(entry.instr().clone())
            .map_err(|e| format!("instr json: {e}"))?
    };
    let pre = bound.pre().map(|(first, second)| PreLatch { first, second });
    eval::step(state, bus, &instr, &binder, pre).map_err(|e| format!("{e}"))
}

fn find_entry<'a, M: ManifestEntryView, B: BoundInstrView>(
    manifest: &'a [M],
    bound: &'a B,
) -> Option<&'a M> {
    manifest.iter().find(|entry| {
        entry.opcode() as u32 == bound.opcode()
            && entry.pre() == bound.pre()
    })
}

fn patch_binder(
    template: &Map<String, Value>,
    operands: &HashMap<String, Value>,
) -> Map<String, Value> {
    let mut merged = Map::new();
    for (key, value) in operands {
        merged.insert(key.clone(), value.clone());
    }
    for (key, value) in template {
        merged.entry(key.clone()).or_insert_with(|| value.clone());
    }
    apply_operand_aliases(&mut merged, template, operands);
    merged
        .into_iter()
        .map(|(key, value)| (key, normalize_binder_value(&value)))
        .collect()
}

fn merge_binder(base: &Binder, overrides: &Map<String, Value>) -> Binder {
    let mut out = base.clone();
    for (k, v) in overrides {
        if let Some(expr) = decode_expr(v) {
            out.insert(k.clone(), expr);
        }
    }
    out
}

fn decode_expr(value: &Value) -> Option<Expr> {
    // Try constant forms first
    if let Some(val) = value.get("value").and_then(|n| n.as_u64()) {
        let size = value
            .get("size")
            .and_then(|n| n.as_u64())
            .map(|n| n as u8)
            .unwrap_or(8);
        return Some(Expr::Const {
            value: val as u32,
            size,
        });
    }
    // Bare number -> const
    if let Some(num) = value.as_u64() {
        return Some(Expr::Const {
            value: num as u32,
            size: 32,
        });
    }
    // Reg
    if value.get("type").and_then(|v| v.as_str()) == Some("reg") {
        if let (Some(name), Some(size)) = (
            value.get("name").and_then(|v| v.as_str()),
            value.get("size").and_then(|v| v.as_u64()),
        ) {
            return Some(Expr::Reg {
                name: name.to_string(),
                size: size as u8,
                bank: value.get("bank").and_then(|v| v.as_str()).map(|s| s.to_string()),
            });
        }
    }
    // Temp
    if value.get("type").and_then(|v| v.as_str()) == Some("tmp") {
        if let (Some(name), Some(size)) = (
            value.get("name").and_then(|v| v.as_str()),
            value.get("size").and_then(|v| v.as_u64()),
        ) {
            return Some(Expr::Tmp {
                name: name.to_string(),
                size: size as u8,
            });
        }
    }
    // Mem
    if value.get("type").and_then(|v| v.as_str()) == Some("mem") {
        let space = match value.get("space").and_then(|v| v.as_str()) {
            Some("int") | Some("Int") => rust_scil::ast::Space::Int,
            Some("ext") | Some("Ext") => rust_scil::ast::Space::Ext,
            Some("code") | Some("Code") => rust_scil::ast::Space::Code,
            _ => return None,
        };
        let size = value.get("size").and_then(|v| v.as_u64())? as u8;
        let addr_val = value.get("addr")?;
        let addr = decode_expr(addr_val)?;
        return Some(Expr::Mem {
            space,
            size,
            addr: Box::new(addr),
        });
    }
    // Flags
    if value.get("type").and_then(|v| v.as_str()) == Some("flag") {
        if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
            return Some(Expr::Flag {
                name: name.to_string(),
            });
        }
    }
    None
}

fn apply_operand_aliases(
    merged: &mut Map<String, Value>,
    template: &Map<String, Value>,
    operands: &HashMap<String, Value>,
) {
    for key in template.keys() {
        if operands.contains_key(key) {
            continue;
        }
        if let Some(value) = addr16_page_component(key, operands) {
            merged.insert(key.clone(), value);
            continue;
        }
        if let Some(value) = ptr_operand_component(key, operands) {
            merged.insert(key.clone(), value);
            continue;
        }
        if let Some(candidates) = binder_alias_sources(key) {
            for alias in candidates {
                if let Some(value) = merged.get(*alias) {
                    merged.insert(key.clone(), value.clone());
                    break;
                }
            }
        }
    }
}

fn binder_alias_sources(key: &str) -> Option<&'static [&'static str]> {
    match key {
        "imm8" => Some(&["n", "imm"]),
        "imm8_lo" => Some(&["lo", "dst", "hi"]),
        "imm8_mid" => Some(&["mid"]),
        "imm8_hi" => Some(&["hi", "dst", "lo"]),
        "dst_off" => Some(&["dst", "addr", "addr24", "src", "imm", "lo", "hi", "imem"]),
        "src_off" => Some(&["src", "addr", "addr24", "dst", "imem"]),
        "addr_ptr" => Some(&["addr24", "addr"]),
        "addr16" | "call_addr16" | "call_page_hi" | "page_hi" => Some(&["addr16_page"]),
        "call_addr24" => Some(&["addr24"]),
        "disp8" => Some(&["disp"]),
        _ => None,
    }
}

fn ptr_operand_component(key: &str, operands: &HashMap<String, Value>) -> Option<Value> {
    if key != "ptr_base" && key != "ptr_disp" {
        return None;
    }
    let ptr = operands.get("ptr")?;
    let obj = ptr.as_object()?;
    match key {
        "ptr_base" => obj.get("base").cloned(),
        "ptr_disp" => obj
            .get("disp")
            .cloned()
            .or_else(|| Some(const_from_u32(0, 8))),
        _ => None,
    }
}

fn addr16_page_component(key: &str, operands: &HashMap<String, Value>) -> Option<Value> {
    if key != "addr16"
        && key != "call_addr16"
        && key != "page_hi"
        && key != "call_page_hi"
    {
        return None;
    }
    let operand = operands.get("addr16_page")?;
    let obj = operand.as_object()?;
    let offset = obj.get("offs16").and_then(extract_const_u32)?;
    let page = obj.get("page20").and_then(|num| num.as_u64()).unwrap_or(0) as u32;
    match key {
        "addr16" | "call_addr16" => Some(const_from_u32(offset, 16)),
        "page_hi" | "call_page_hi" => Some(const_from_u32(page, 20)),
        _ => None,
    }
}

fn extract_const_u32(value: &Value) -> Option<u32> {
    value
        .get("value")
        .and_then(|num| num.as_u64())
        .map(|val| val as u32)
}

fn normalize_binder_value(value: &Value) -> Value {
    if let Some(kind) = value.get("type").and_then(|v| v.as_str()) {
        match kind {
            "imm8" => const_from_value(value, 8),
            "imm16" => const_from_value(value, 16),
            "imm20" => const_from_value(value, 20),
            "imm24" => const_from_value(value, 24),
            "disp8" => const_from_value(value, 8),
            "addr16_page" => {
                if let Some(offs) = value.get("offs16") {
                    let offs_val = normalize_binder_value(offs);
                    if let (Some(Value::Number(lo)), Some(Value::Number(page))) =
                        (offs_val.get("value"), value.get("page20"))
                    {
                        let lo_val = lo.as_u64().unwrap_or(0);
                        let page_val = page.as_u64().unwrap_or(0);
                        let combined = ((page_val as u32) << 16) | (lo_val as u32);
                        return const_from_u32(combined, 20);
                    }
                }
                const_from_u32(0, 20)
            }
            "addr24" => {
                if let Some(imm) = value.get("imm24") {
                    let imm_val = normalize_binder_value(imm);
                    if let Some(Value::Number(num)) = imm_val.get("value") {
                        let val = num.as_u64().unwrap_or(0);
                        return const_from_u32(val as u32, 24);
                    }
                }
                const_from_u32(0, 24)
            }
            "regsel" => normalize_regsel_value(value),
            "ext_reg_ptr" => normalize_ext_reg_ptr(value),
            "imem_ptr" => normalize_imem_ptr(value),
            _ => value.clone(),
        }
    } else {
        value.clone()
    }
}

fn const_from_value(value: &Value, size: u8) -> Value {
    value
        .get("value")
        .and_then(|num| num.as_u64())
        .map(|val| const_from_u32(val as u32, size))
        .unwrap_or_else(|| const_from_u32(0, size))
}

fn const_from_u32(value: u32, size: u8) -> Value {
    let mask = if size >= 32 {
        u32::MAX
    } else {
        (1 << size) - 1
    };
    json!({"type": "const", "value": value & mask, "size": size})
}

fn normalize_regsel_value(value: &Value) -> Value {
    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("A");
    let group = value
        .get("size_group")
        .and_then(|v| v.as_str())
        .unwrap_or("r1");
    let size = reg_size_from_group(group);
    json!({
        "type": "reg",
        "name": name,
        "size": size,
        "bank": "gpr",
    })
}

fn normalize_ext_reg_ptr(value: &Value) -> Value {
    let mut obj = Map::new();
    if let Some(ptr) = value.get("ptr") {
        obj.insert("ptr".to_string(), normalize_regsel_value(ptr));
    }
    if let Some(mode) = value.get("mode") {
        obj.insert("mode".to_string(), mode.clone());
    }
    if let Some(disp) = value.get("disp") {
        obj.insert("disp".to_string(), normalize_binder_value(disp));
    }
    obj.insert("type".to_string(), Value::String("ext_reg_ptr".to_string()));
    Value::Object(obj)
}

fn normalize_imem_ptr(value: &Value) -> Value {
    let base = value
        .get("base")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as i32;
    let disp = value
        .get("disp")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let combined = base + disp;
    const_from_u32(combined as u32, 8)
}

const REG_TABLE: [(&str, &str, u8); 8] = [
    ("A", "r1", 8),
    ("IL", "r1", 8),
    ("BA", "r2", 16),
    ("I", "r2", 16),
    ("X", "r3", 24),
    ("Y", "r3", 24),
    ("U", "r3", 24),
    ("S", "r3", 24),
];

fn reg_size_from_group(group: &str) -> u8 {
    let mut fallback = 8;
    for (_, g, size) in REG_TABLE.iter() {
        if g.eq_ignore_ascii_case(group) {
            return *size;
        }
        if group.is_empty() {
            fallback = *size;
        }
    }
    fallback
}
pub fn env_opcode_target(var: &str) -> Option<u8> {
    std::env::var(var)
        .ok()
        .and_then(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Some(hex) = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")) {
                u8::from_str_radix(hex, 16).ok()
            } else {
                trimmed.parse::<u8>().ok()
            }
        })
}
