use crate::eval::{eval_manifest_entry, BoundInstrView, LayoutEntryView, ManifestEntryView};
use crate::{
    collect_registers, keyboard::KeyboardMatrix, lcd::LcdController, memory::*,
    perfetto::PerfettoTracer, register_width, CoreError, MemoryImage, Result,
};
use rust_scil::{
    bus::{Bus, Space},
};
use once_cell::sync::Lazy;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::env;
use rust_scil::state::State as CpuState;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct OpcodeLookupKey {
    pub opcode: u8,
    pub pre: Option<(String, String)>,
}

pub trait OpcodeIndexView {
    fn opcode(&self) -> u8;
    fn pre(&self) -> Option<(String, String)>;
    fn manifest_index(&self) -> usize;
}

/// Lightweight map keyed by opcode + PRE tuple for quick manifest lookup.
pub struct OpcodeLookup<'a, M> {
    map: HashMap<OpcodeLookupKey, &'a M>,
}

impl<'a, M> OpcodeLookup<'a, M>
where
    M: ExecManifestEntry,
{
    pub fn new<I>(manifest: &'a [M], index: &'a [I]) -> Self
    where
        I: OpcodeIndexView,
    {
        let mut map = HashMap::new();
        for entry in index {
            let key = OpcodeLookupKey {
                opcode: entry.opcode(),
                pre: entry.pre(),
            };
            if let Some(manifest_entry) = manifest.get(entry.manifest_index()) {
                map.insert(key, manifest_entry);
            }
        }
        Self { map }
    }

    pub fn lookup(&self, opcode: u8, pre: Option<(String, String)>) -> Option<&'a M> {
        let key = OpcodeLookupKey { opcode, pre };
        self.map.get(&key).copied()
    }

    pub fn lookup_with_fallback(
        &self,
        opcode: u8,
        pre: Option<(String, String)>,
    ) -> Option<&'a M> {
        self.lookup(opcode, pre.clone())
            .or_else(|| self.lookup(opcode, None))
    }
}

pub trait ExecManifestEntry: ManifestEntryView {
    fn mnemonic(&self) -> &str;
    fn family(&self) -> Option<&str>;
}

pub trait BoundInstrBuilder: BoundInstrView + Sized {
    fn from_parts(
        opcode: u32,
        mnemonic: &str,
        family: Option<&str>,
        length: u8,
        pre: Option<(String, String)>,
        operands: HashMap<String, Value>,
    ) -> Self;
}

pub trait HostMemory {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32;
    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32);
    fn read_byte(&mut self, address: u32) -> u8;
    fn notify_lcd_write(&mut self, address: u32, value: u32);
}

pub trait BusProfiler {
    fn record_bus_load(&mut self) {}
    fn record_bus_store(&mut self) {}
    fn record_python_load(&mut self, _address: u32) {}
    fn record_python_store(&mut self, _address: u32) {}
}

#[derive(Default)]
pub struct NullProfiler;

impl BusProfiler for NullProfiler {}

pub struct HybridBus<'a, H, P>
where
    H: HostMemory,
    P: BusProfiler,
{
    memory: &'a mut MemoryImage,
    host: &'a mut H,
    profile: Option<&'a mut P>,
    lcd: Option<&'a mut LcdController>,
    keyboard: Option<&'a mut KeyboardMatrix>,
    perfetto: Option<&'a mut PerfettoTracer>,
    trace_pc: Option<u32>,
    trace_index: u64,
}

impl<'a, H, P> HybridBus<'a, H, P>
where
    H: HostMemory,
    P: BusProfiler,
{
    pub fn new(
        memory: &'a mut MemoryImage,
        host: &'a mut H,
        profile: Option<&'a mut P>,
        lcd: Option<&'a mut LcdController>,
        keyboard: Option<&'a mut KeyboardMatrix>,
        perfetto: Option<&'a mut PerfettoTracer>,
        trace_pc: Option<u32>,
        trace_index: u64,
    ) -> Self {
        Self {
            memory,
            host,
            profile,
            lcd,
            keyboard,
            perfetto,
            trace_pc,
            trace_index,
        }
    }
}

impl<'a, H, P> Bus for HybridBus<'a, H, P>
where
    H: HostMemory,
    P: BusProfiler,
{
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        if let Some(profile) = self.profile.as_mut() {
            profile.record_bus_load();
        }
        let absolute = resolve_bus_address(space, addr);
        let debug_watch = env::var("RUST_ROM_DEBUG").is_ok()
            && (0x0F2BD0..=0x0F2BD8).contains(&absolute);
        if debug_watch {
            eprintln!(
                "[rom-debug-bus] load addr=0x{addr:06X} space={space:?}",
                addr = absolute,
                space = space
            );
        }
        if let Some(lcd) = self.lcd.as_mut() {
            if lcd.handles(absolute) {
                if debug_watch {
                    eprintln!(
                        "[rom-debug-bus] lcd handled addr=0x{addr:06X}",
                        addr = absolute
                    );
                }
                return lcd.read(absolute);
            }
        }
        if let Some(offset) = MemoryImage::internal_offset(absolute) {
            if let Some(keyboard) = self.keyboard.as_mut() {
                if MemoryImage::is_keyboard_offset(offset) {
                    if let Some(value) = keyboard.handle_read(offset, self.memory) {
                        return value as u32;
                    }
                }
            }
        }
        if debug_watch {
            eprintln!(
                "[rom-debug-bus] checking python ranges addr=0x{addr:06X}",
                addr = absolute
            );
        }
        let needs_host = self.memory.requires_python(absolute);
        if debug_watch {
            eprintln!(
                "[rom-debug-bus] requires_python={needs_host} addr=0x{addr:06X}",
                addr = absolute
            );
        }
        if needs_host {
            if let Some(profile) = self.profile.as_mut() {
                profile.record_python_load(absolute);
            }
            if debug_watch {
                eprintln!("[rom-debug-bus] delegating to host addr=0x{addr:06X}", addr = absolute);
            }
            return self.host.load(space, addr, bits);
        }
        if let Some(value) = self.memory.load(absolute, bits) {
            if debug_watch {
                eprintln!(
                    "[rom-debug-bus] cached value addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                    addr = absolute,
                    bits = bits,
                    val = value & mask_bits(bits),
                );
            }
            return value;
        }
        if let Some(profile) = self.profile.as_mut() {
            profile.record_python_load(absolute);
        }
        if debug_watch {
            eprintln!(
                "[rom-debug-bus] cache miss -> host addr=0x{addr:06X}",
                addr = absolute
            );
        }
        let value = self.host.load(space, addr, bits);
        if debug_watch {
            eprintln!(
                "[rom-debug-bus] host returned addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                addr = absolute,
                bits = bits,
                val = value & mask_bits(bits),
            );
        }
        value
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        if let Some(profile) = self.profile.as_mut() {
            profile.record_bus_store();
        }
        let absolute = resolve_bus_address(space, addr);
        if matches!(space, Space::Int) && env::var("RUST_INT_STORE_TRACE").is_ok() {
            println!(
                "[rust-int-store] addr=0x{addr:02X} bits={bits} value=0x{value:02X}"
            );
        }
        if env::var("RUST_LCD_DEBUG").is_ok()
            && matches!(space, Space::Ext | Space::Code)
            && ((absolute & 0xF000) == 0x2000 || (absolute & 0xF000) == 0xA000)
        {
            println!(
                "[hybrid-store] space={:?} orig=0x{addr:05X} abs=0x{absolute:05X} val=0x{value:02X}",
                space
            );
        }
        let mut lcd_handled = false;
        if let Some(lcd) = self.lcd.as_mut() {
            if lcd.handles(absolute) {
                lcd.write(absolute, value as u8);
                lcd_handled = true;
            }
        }
        if env::var("RUST_LCD_TRACE").is_ok()
            && ((0x2000..=0x200F).contains(&absolute) || (0xA000..=0xAFFF).contains(&absolute))
        {
            println!(
                "[rust-lcd] space={:?} addr=0x{addr:05X} abs=0x{absolute:05X} val=0x{value:02X}",
                space
            );
        }
        if lcd_handled {
            self.host.notify_lcd_write(absolute, value);
            return;
        }
        if let Some(offset) = MemoryImage::internal_offset(absolute) {
            if let Some(keyboard) = self.keyboard.as_mut() {
                if MemoryImage::is_keyboard_offset(offset)
                    && keyboard.handle_write(offset, value as u8, self.memory)
                {
                    return;
                }
            }
        }
        if self.memory.requires_python(absolute) {
            if let Some(profile) = self.profile.as_mut() {
                profile.record_python_store(absolute);
            }
            self.host.store(space, addr, bits, value);
            return;
        }
        let stored_locally = self.memory.store(absolute, bits, value).is_some();
        if let (Some(tracer), Some(pc)) = (self.perfetto.as_deref_mut(), self.trace_pc) {
            let space_str = match space {
                Space::Int => "internal",
                Space::Ext => "external",
                Space::Code => "code",
            };
            tracer.record_mem_write(self.trace_index, pc, absolute, value, space_str, bits.max(8) / 8);
        }
        if matches!(space, Space::Int) {
            self.host.store(space, addr, bits, value);
        } else if !stored_locally {
            if let Some(profile) = self.profile.as_mut() {
                profile.record_python_store(absolute);
            }
            self.host.store(space, addr, bits, value);
        }
    }
}

pub fn resolve_bus_address(space: Space, addr: u32) -> u32 {
    match space {
        Space::Int => INTERNAL_MEMORY_START + (addr & INTERNAL_ADDR_MASK),
        Space::Ext | Space::Code => addr & ADDRESS_MASK,
    }
}

pub struct InstructionStream<'a, H>
where
    H: HostMemory,
{
    memory: &'a mut MemoryImage,
    host: &'a mut H,
    cursor: u32,
    consumed: usize,
    start_pc: u32,
}

impl<'a, H> InstructionStream<'a, H>
where
    H: HostMemory,
{
    pub fn new(memory: &'a mut MemoryImage, host: &'a mut H, start_pc: u32) -> Self {
        Self {
            memory,
            host,
            cursor: start_pc & ADDRESS_MASK,
            consumed: 0,
            start_pc: start_pc & ADDRESS_MASK,
        }
    }

    pub fn read_u8(&mut self) -> u8 {
        let absolute = self.cursor & ADDRESS_MASK;
        self.cursor = (self.cursor + 1) & ADDRESS_MASK;
        self.consumed += 1;
        if !self.memory.requires_python(absolute) {
            if let Some(byte) = self.memory.read_byte(absolute) {
                return byte;
            }
        }
        self.host.read_byte(absolute)
    }

    pub fn consumed(&self) -> usize {
        self.consumed
    }

    pub fn page20(&self) -> u32 {
        self.start_pc & 0xF0000
    }
}

pub struct DecodedInstr<B> {
    pub bound: B,
    pub opcode: u8,
    pub length: usize,
}

pub fn decode_bound_from_memory<'a, B, M, H>(
    memory: &'a mut MemoryImage,
    host: &'a mut H,
    lookup: &'a OpcodeLookup<M>,
    pc: u32,
) -> Result<DecodedInstr<B>>
where
    B: BoundInstrBuilder,
    M: ExecManifestEntry,
    H: HostMemory,
{
    let mut stream = InstructionStream::new(memory, host, pc);
    let mut pending_pre: Option<(String, String)> = None;
    let mut prefix_guard = 0usize;

    loop {
        let byte = stream.read_u8();
        if let Some(pair) = PRE_BY_OPCODE.get(&byte).cloned() {
            pending_pre = Some(pair);
            prefix_guard += 1;
            if prefix_guard > MAX_PREFIX_CHAIN {
                return Err(CoreError::Other(
                    "excessive PRE prefixes before instruction".to_string(),
                ));
            }
            continue;
        }

        let entry = lookup.lookup_with_fallback(byte, pending_pre.clone()).ok_or_else(|| {
            CoreError::Other(format!(
                "no manifest entry for opcode 0x{byte:02X} (pre={pending_pre:?})"
            ))
        })?;
        let page20 = stream.page20();
        let operands = decode_operands(&mut stream, entry.layout(), page20)?;
        let length = stream.consumed();
        let bound = B::from_parts(
            byte as u32,
            entry.mnemonic(),
            entry.family(),
            length as u8,
            entry.pre(),
            operands,
        );
        return Ok(DecodedInstr {
            bound,
            opcode: byte,
            length,
        });
    }
}

fn decode_operands<H, L>(
    stream: &mut InstructionStream<'_, H>,
    layout: &[L],
    page20: u32,
) -> Result<HashMap<String, Value>>
where
    H: HostMemory,
    L: LayoutEntryView,
{
    let mut operands = HashMap::new();
    for entry in layout {
        let value = decode_operand_entry(stream, entry, page20)?;
        if entry.kind() == "reg_pair" {
            if let Some(map) = value.as_object() {
                for (sub, sub_value) in map {
                    operands.insert(sub.clone(), sub_value.clone());
                }
                continue;
            }
        }
        operands.insert(entry.key().to_string(), value);
    }
    Ok(operands)
}

fn decode_operand_entry<H, L>(
    stream: &mut InstructionStream<'_, H>,
    entry: &L,
    page20: u32,
) -> Result<Value>
where
    H: HostMemory,
    L: LayoutEntryView,
{
    match entry.kind() {
        "imm8" => Ok(encode_imm8(stream.read_u8())),
        "imm16" => {
            let lo = stream.read_u8();
            let hi = stream.read_u8();
            Ok(encode_imm16(lo, hi))
        }
        "imm20" => {
            let lo = stream.read_u8();
            let mid = stream.read_u8();
            let hi = stream.read_u8() & 0x0F;
            Ok(encode_imm20(lo, mid, hi))
        }
        "addr16_page" => {
            let lo = stream.read_u8();
            let hi = stream.read_u8();
            Ok(encode_addr16(page20, lo, hi))
        }
        "addr24" => {
            let lo = stream.read_u8();
            let mid = stream.read_u8();
            let hi = stream.read_u8();
            Ok(encode_addr24(lo, mid, hi))
        }
        "imm24" => {
            let lo = stream.read_u8();
            let mid = stream.read_u8();
            let hi = stream.read_u8();
            Ok(encode_imm24(lo, mid, hi))
        }
        "ext_reg_ptr" => decode_ext_reg_ptr(stream),
        "imem_ptr" => decode_imem_ptr(stream),
        "regsel" => decode_regsel(stream, entry),
        "reg_pair" => decode_reg_pair(stream, entry),
        other => Err(CoreError::Other(format!(
            "unsupported layout kind {other}"
        ))),
    }
}

/// Decode and execute a single instruction using the core manifest and bus.
pub fn execute_step<B, M, H, P>(
    state: &mut CpuState,
    memory: &mut MemoryImage,
    host: &mut H,
    manifest: &[M],
    lookup: &OpcodeLookup<M>,
    keyboard: Option<&mut KeyboardMatrix>,
    lcd: Option<&mut LcdController>,
    perfetto: Option<&mut PerfettoTracer>,
    profiler: Option<&mut P>,
    instruction_index: u64,
) -> Result<(u8, u8)>
where
    B: BoundInstrBuilder,
    M: ExecManifestEntry,
    H: HostMemory,
    P: BusProfiler,
{
    let pc = state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;

    let decoded = decode_bound_from_memory::<B, M, H>(memory, host, lookup, pc)?;
    let mut perfetto = perfetto;
    let trace_pc = perfetto.as_ref().map(|_| pc);
    let mut bus = HybridBus::new(
        memory,
        host,
        profiler,
        lcd,
        keyboard,
        perfetto.as_deref_mut(),
        trace_pc,
        instruction_index,
    );
    eval_manifest_entry(manifest, state, &mut bus, &decoded.bound)
        .map_err(|e| CoreError::Other(e.to_string()))?;

    // Maintain PC advance if the instruction did not change it.
    let current_pc = state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;
    if current_pc == pc {
        let next_pc = (pc + decoded.length as u32) & ADDRESS_MASK;
        state.set_reg("PC", next_pc, register_width("PC"));
    }

    if let Some(tracer) = perfetto.as_deref_mut() {
        let regs = collect_registers(state);
        // Report reg_PC as the post-step PC for compatibility with Python traces.
        let reg_pc = state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;
        tracer.record_regs(instruction_index, pc, reg_pc, decoded.opcode, &regs);
    }

    Ok((decoded.opcode, decoded.length as u8))
}

fn decode_reg_pair<H, L>(
    stream: &mut InstructionStream<'_, H>,
    entry: &L,
) -> Result<Value>
where
    H: HostMemory,
    L: LayoutEntryView,
{
    let byte = stream.read_u8();
    let dst_idx = ((byte >> 4) & 0x07) as usize;
    let src_idx = (byte & 0x07) as usize;
    let (dst_name, dst_group, _) = REG_TABLE
        .get(dst_idx)
        .ok_or_else(|| CoreError::Other("invalid dst register index".to_string()))?;
    let (src_name, src_group, _) = REG_TABLE
        .get(src_idx)
        .ok_or_else(|| CoreError::Other("invalid src register index".to_string()))?;
    let dst_key = entry
        .meta()
        .get("dst_key")
        .and_then(|val| val.as_str())
        .unwrap_or("dst");
    let src_key = entry
        .meta()
        .get("src_key")
        .and_then(|val| val.as_str())
        .unwrap_or("src");
    let mut payload = Map::new();
    let dst_value = normalize_regsel_value(&encode_regsel(dst_name, dst_group));
    let src_value = normalize_regsel_value(&encode_regsel(src_name, src_group));
    payload.insert(dst_key.to_string(), dst_value);
    payload.insert(src_key.to_string(), src_value);
    Ok(Value::Object(payload))
}

fn encode_imm8(value: u8) -> Value {
    json!({"type": "imm8", "value": value})
}

fn encode_imm16(lo: u8, hi: u8) -> Value {
    let value = ((hi as u32) << 8) | (lo as u32);
    json!({"type": "imm16", "lo": lo, "hi": hi, "value": value})
}

fn encode_imm24(lo: u8, mid: u8, hi: u8) -> Value {
    let value = ((hi as u32) << 16) | ((mid as u32) << 8) | (lo as u32);
    json!({"type": "imm24", "lo": lo, "mid": mid, "hi": hi, "value": value})
}

fn encode_imm20(lo: u8, mid: u8, hi: u8) -> Value {
    let value = ((hi as u32) << 16) | ((mid as u32) << 8) | (lo as u32);
    json!({"type": "imm20", "lo": lo, "mid": mid, "hi": hi, "value": value})
}

fn encode_addr16(page20: u32, lo: u8, hi: u8) -> Value {
    json!({
        "type": "addr16_page",
        "offs16": encode_imm16(lo, hi),
        "page20": page20,
    })
}

fn encode_addr24(lo: u8, mid: u8, hi: u8) -> Value {
    json!({
        "type": "addr24",
        "imm24": encode_imm24(lo, mid, hi),
    })
}

fn encode_regsel(name: &str, group: &str) -> Value {
    json!({
        "type": "regsel",
        "size_group": group,
        "name": name,
    })
}

fn encode_reg(name: &str, group: &str) -> Value {
    json!({
        "type": "reg",
        "name": name,
        "size": reg_size_from_group(group),
        "bank": "gpr",
    })
}

fn decode_operands_disp(value: u8, sign: i8) -> i32 {
    let signed = if value & 0x80 != 0 {
        (value as i32) - 0x100
    } else {
        value as i32
    };
    if sign >= 0 { signed } else { -signed }
}

fn decode_ext_reg_ptr<H>(stream: &mut InstructionStream<'_, H>) -> Result<Value>
where
    H: HostMemory,
{
    let reg_byte = stream.read_u8();
    let raw_mode = (reg_byte >> 4) & 0x0F;
    let mode_code = normalize_ext_reg_mode(raw_mode);
    let mode = EXT_REG_MODES
        .iter()
        .find(|mode| mode.code == mode_code)
        .ok_or_else(|| CoreError::Other("unsupported ext-reg pointer mode".to_string()))?;
    let idx = (reg_byte & 0x07) as usize;
    let (name, group, _) = REG_TABLE
        .get(idx)
        .ok_or_else(|| CoreError::Other("invalid pointer register".to_string()))?;
    if env::var("EXT_PTR_TRACE_RAW").is_ok() {
        println!(
            "[ext-reg-decode] reg_byte=0x{reg_byte:02X} idx={idx} name={name} raw_mode=0x{raw_mode:X} mode_code=0x{mode_code:X}"
        );
    }
    let mut payload = Map::new();
    payload.insert("type".into(), Value::String("ext_reg_ptr".into()));
    payload.insert("ptr".into(), encode_reg(name, group));
    payload.insert("mode".into(), Value::String(mode.name.into()));
    if mode.needs_disp {
        let magnitude = stream.read_u8() as i32;
        let signed = if mode.disp_sign >= 0 {
            magnitude
        } else {
            -magnitude
        };
        payload.insert("disp".into(), const_from_u32(signed as u32, 8));
    }
    Ok(Value::Object(payload))
}

fn decode_imem_ptr<H>(stream: &mut InstructionStream<'_, H>) -> Result<Value>
where
    H: HostMemory,
{
    let mode_byte = stream.read_u8();
    let raw_code = mode_byte & 0xC0;
    let mode_code = match raw_code {
        0x40 => 0x00,
        code => code,
    };
    let mode = IMEM_MODES
        .iter()
        .find(|entry| entry.code == mode_code)
        .ok_or_else(|| {
            CoreError::Other(format!(
                "unsupported IMEM pointer mode (raw=0x{raw:02X} code=0x{code:02X})",
                raw = mode_byte,
                code = mode_code
            ))
        })?;
    let base = stream.read_u8();
    let mut payload = Map::new();
    payload.insert("type".into(), Value::String("imem_ptr".into()));
    payload.insert("base".into(), const_from_u32(base as u32, 8));
    payload.insert("mode".into(), Value::String(mode.name.into()));
    if mode.needs_disp {
        let raw = stream.read_u8();
        let value = decode_operands_disp(raw, mode.disp_sign);
        payload.insert("disp".into(), const_from_u32(value as u32, 8));
    }
    Ok(Value::Object(payload))
}

fn decode_regsel<H, L>(
    stream: &mut InstructionStream<'_, H>,
    entry: &L,
) -> Result<Value>
where
    H: HostMemory,
    L: LayoutEntryView,
{
    let allowed = entry.meta().get("allowed_groups").and_then(|value| {
        value.as_array().map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
    });
    let byte = stream.read_u8();
    let idx = (byte & 0x07) as usize;
    let (name, group, _) = REG_TABLE
        .get(idx)
        .ok_or_else(|| CoreError::Other("unsupported register index".to_string()))?;
    if let Some(groups) = allowed.as_ref() {
        if !groups.iter().any(|g| g.eq_ignore_ascii_case(group)) {
            return Err(CoreError::Other(format!(
                "register {name} not allowed for operand"
            )));
        }
    }
    Ok(normalize_regsel_value(&encode_regsel(name, group)))
}

fn mask_bits(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else if bits == 0 {
        0
    } else {
        (1u32 << bits) - 1
    }
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

fn normalize_ext_reg_mode(raw: u8) -> u8 {
    if raw & 0x8 != 0 {
        if raw & 0x4 != 0 {
            0xC
        } else {
            0x8
        }
    } else {
        match raw & 0x3 {
            0 | 1 => 0x0,
            2 => 0x2,
            _ => 0x3,
        }
    }
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

struct ExtRegMode {
    code: u8,
    name: &'static str,
    needs_disp: bool,
    disp_sign: i8,
}

const EXT_REG_MODES: [ExtRegMode; 5] = [
    ExtRegMode {
        code: 0x0,
        name: "simple",
        needs_disp: false,
        disp_sign: 0,
    },
    ExtRegMode {
        code: 0x2,
        name: "post_inc",
        needs_disp: false,
        disp_sign: 0,
    },
    ExtRegMode {
        code: 0x3,
        name: "pre_dec",
        needs_disp: false,
        disp_sign: 0,
    },
    ExtRegMode {
        code: 0x8,
        name: "offset",
        needs_disp: true,
        disp_sign: 1,
    },
    ExtRegMode {
        code: 0xC,
        name: "offset",
        needs_disp: true,
        disp_sign: -1,
    },
];

struct ImemMode {
    code: u8,
    name: &'static str,
    needs_disp: bool,
    disp_sign: i8,
}

const IMEM_MODES: [ImemMode; 3] = [
    ImemMode {
        code: 0x00,
        name: "simple",
        needs_disp: false,
        disp_sign: 0,
    },
    ImemMode {
        code: 0x80,
        name: "pos",
        needs_disp: true,
        disp_sign: 1,
    },
    ImemMode {
        code: 0xC0,
        name: "neg",
        needs_disp: true,
        disp_sign: -1,
    },
];

pub const PRE_PREFIXES: &[(u8, (&str, &str))] = &[
    (0x32, ("(n)", "(n)")),
    (0x30, ("(n)", "(BP+n)")),
    (0x33, ("(n)", "(PY+n)")),
    (0x31, ("(n)", "(BP+PY)")),
    (0x22, ("(BP+n)", "(n)")),
    (0x23, ("(BP+n)", "(PY+n)")),
    (0x21, ("(BP+n)", "(BP+PY)")),
    (0x36, ("(PX+n)", "(n)")),
    (0x34, ("(PX+n)", "(BP+n)")),
    (0x37, ("(PX+n)", "(PY+n)")),
    (0x35, ("(PX+n)", "(BP+PY)")),
    (0x26, ("(BP+PX)", "(n)")),
    (0x24, ("(BP+PX)", "(BP+n)")),
    (0x27, ("(BP+PX)", "(PY+n)")),
    (0x25, ("(BP+PX)", "(BP+PY)")),
];

const MAX_PREFIX_CHAIN: usize = 4;

static PRE_BY_OPCODE: Lazy<HashMap<u8, (String, String)>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (opcode, pair) in PRE_PREFIXES.iter() {
        map.insert(*opcode, (pair.0.to_string(), pair.1.to_string()));
    }
    map
});
