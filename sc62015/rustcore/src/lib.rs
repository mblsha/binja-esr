mod generated {
    pub mod types {
        include!("../generated/types.rs");
    }
    pub mod payload {
        include!("../generated/handlers.rs");
    }
    pub mod opcode_index {
        include!("../generated/opcode_index.rs");
    }
}

use generated::opcode_index::OPCODE_INDEX;
use generated::types::{BoundInstrRepr, LayoutEntry, ManifestEntry, PreInfo};
use once_cell::sync::Lazy;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use pyo3::wrap_pyfunction;
use rust_scil::{
    ast::{Binder, Instr, PreLatch},
    bus::Space,
    eval,
    state::State as RsState,
};
use serde_json::{self, json, Map, Value};
use std::collections::HashMap;

static MANIFEST: Lazy<Vec<ManifestEntry>> = Lazy::new(|| {
    serde_json::from_str(generated::payload::PAYLOAD).expect("manifest json")
});

const INTERNAL_MEMORY_START: u32 = 0x100000;
const ADDRESS_MASK: u32 = 0x00FF_FFFF;
const INTERNAL_ADDR_MASK: u32 = 0xFF;
const EXTERNAL_SPACE: usize = 0x100000;
const DEFAULT_REG_WIDTH: u8 = 24;

#[derive(Default)]
struct MemoryImage {
    external: Vec<u8>,
    dirty: Vec<(u32, u8)>,
    python_ranges: Vec<(u32, u32)>,
}

impl MemoryImage {
    fn new() -> Self {
        Self {
            external: vec![0; EXTERNAL_SPACE],
            dirty: Vec::new(),
            python_ranges: Vec::new(),
        }
    }

    fn load_external(&mut self, blob: &[u8]) {
        let limit = self.external.len().min(blob.len());
        self.external[..limit].copy_from_slice(&blob[..limit]);
        self.dirty.clear();
    }

    fn set_python_ranges(&mut self, ranges: Vec<(u32, u32)>) {
        self.python_ranges = ranges;
    }

    fn requires_python(&self, address: u32) -> bool {
        if address >= EXTERNAL_SPACE as u32 {
            return true;
        }
        for (start, end) in &self.python_ranges {
            if address >= *start && address <= *end {
                return true;
            }
        }
        false
    }

    fn read_byte(&self, address: u32) -> Option<u8> {
        self.external.get(address as usize).copied()
    }

    fn load(&self, address: u32, bits: u8) -> Option<u32> {
        let bytes = (bits / 8).max(1) as usize;
        let end = address as usize + bytes;
        if end > self.external.len() {
            return None;
        }
        let mut value = 0u32;
        for offset in 0..bytes {
            value |= (self.external[address as usize + offset] as u32) << (offset * 8);
        }
        Some(value)
    }

    fn store(&mut self, address: u32, bits: u8, value: u32) -> Option<()> {
        let bytes = (bits / 8).max(1) as usize;
        let end = address as usize + bytes;
        if end > self.external.len() {
            return None;
        }
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let slot = &mut self.external[address as usize + offset];
            if *slot != byte {
                *slot = byte;
                self.dirty.push((address + offset as u32, byte));
            }
        }
        Some(())
    }

    fn drain_dirty(&mut self) -> Vec<(u32, u8)> {
        std::mem::take(&mut self.dirty)
    }
}

#[pyclass(name = "CPU")]
struct Cpu {
    inner: Py<PyAny>,
}

struct PyBus {
    inner: Py<PyAny>,
}

impl PyBus {
    fn new(obj: PyObject) -> Self {
        Self { inner: obj.into() }
    }
}

impl rust_scil::bus::Bus for PyBus {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        Python::with_gil(|py| {
            self.inner
                .call_method(
                    py,
                    "load",
                    (space_to_str(&space), addr, bits),
                    None,
                )
                .and_then(|obj| obj.extract::<u32>(py))
                .expect("MemoryAdapter.load returned non-u32")
        })
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        Python::with_gil(|py| {
            let _ = self.inner.call_method(
                py,
                "store",
                (space_to_str(&space), addr, bits, value),
                None,
            );
        });
    }
}

struct MemoryProxyBus {
    memory: Py<PyAny>,
}

impl MemoryProxyBus {
    fn new(memory: Py<PyAny>) -> Self {
        Self { memory }
    }

    fn resolve(space: Space, addr: u32) -> u32 {
        match space {
            Space::Int => INTERNAL_MEMORY_START + (addr & INTERNAL_ADDR_MASK),
            Space::Ext | Space::Code => addr & ADDRESS_MASK,
        }
    }

    fn read_byte(&self, address: u32) -> u8 {
        Python::with_gil(|py| {
            self.memory
                .call_method(py, "read_byte", (address,), None)
                .and_then(|obj| obj.extract::<u8>(py))
                .expect("memory.read_byte must return int")
        })
    }

    fn write_byte(&self, address: u32, value: u8) {
        Python::with_gil(|py| {
            let _ = self
                .memory
                .call_method(py, "write_byte", (address, value), None);
        });
    }
}

impl rust_scil::bus::Bus for MemoryProxyBus {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        let bytes = (bits / 8).max(1);
        let mut value = 0u32;
        let base = Self::resolve(space, addr);
        for offset in 0..bytes {
            let byte = self.read_byte(base + offset as u32);
            value |= (byte as u32) << (offset * 8);
        }
        value
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        let bytes = (bits / 8).max(1);
        let base = Self::resolve(space, addr);
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            self.write_byte(base + offset as u32, byte);
        }
    }
}

struct HybridBus<'a> {
    memory: &'a mut MemoryImage,
    fallback: MemoryProxyBus,
}

impl<'a> HybridBus<'a> {
    fn new(memory: &'a mut MemoryImage, fallback: MemoryProxyBus) -> Self {
        Self { memory, fallback }
    }
}

impl<'a> rust_scil::bus::Bus for HybridBus<'a> {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        let absolute = MemoryProxyBus::resolve(space, addr);
        if self.memory.requires_python(absolute) {
            return self.fallback.load(space, addr, bits);
        }
        if let Some(value) = self.memory.load(absolute, bits) {
            return value;
        }
        self.fallback.load(space, addr, bits)
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        let absolute = MemoryProxyBus::resolve(space, addr);
        if self.memory.requires_python(absolute) {
            self.fallback.store(space, addr, bits, value);
            return;
        }
        if self.memory.store(absolute, bits, value).is_none() {
            self.fallback.store(space, addr, bits, value);
        }
    }
}

struct InstructionStream<'a> {
    memory: &'a mut MemoryImage,
    fallback: MemoryProxyBus,
    cursor: u32,
    consumed: usize,
    start_pc: u32,
}

impl<'a> InstructionStream<'a> {
    fn new(memory: &'a mut MemoryImage, fallback: MemoryProxyBus, start_pc: u32) -> Self {
        Self {
            memory,
            fallback,
            cursor: start_pc & ADDRESS_MASK,
            consumed: 0,
            start_pc: start_pc & ADDRESS_MASK,
        }
    }

    fn read_u8(&mut self) -> u8 {
        let absolute = self.cursor & ADDRESS_MASK;
        self.cursor = (self.cursor + 1) & ADDRESS_MASK;
        self.consumed += 1;
        if !self.memory.requires_python(absolute) {
            if let Some(byte) = self.memory.read_byte(absolute) {
                return byte;
            }
        }
        self.fallback.read_byte(absolute)
    }

    fn consumed(&self) -> usize {
        self.consumed
    }

    fn page20(&self) -> u32 {
        self.start_pc & 0xF0000
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

fn encode_imm8(value: u8) -> Value {
    json!({"kind": "imm8", "value": value})
}

fn encode_disp8(value: i32) -> Value {
    json!({"kind": "disp8", "value": value})
}

fn encode_imm16(lo: u8, hi: u8) -> Value {
    json!({"kind": "imm16", "lo": lo, "hi": hi})
}

fn encode_imm24(lo: u8, mid: u8, hi: u8) -> Value {
    json!({"kind": "imm24", "lo": lo, "mid": mid, "hi": hi})
}

fn encode_addr16(page20: u32, lo: u8, hi: u8) -> Value {
    json!({
        "kind": "addr16_page",
        "offs16": encode_imm16(lo, hi),
        "page20": page20,
    })
}

fn encode_addr24(lo: u8, mid: u8, hi: u8) -> Value {
    json!({
        "kind": "addr24",
        "imm24": encode_imm24(lo, mid, hi),
    })
}

fn encode_regsel(name: &str, group: &str) -> Value {
    json!({
        "kind": "regsel",
        "size_group": group,
        "name": name,
    })
}

fn decode_operands(
    stream: &mut InstructionStream<'_>,
    layout: &[LayoutEntry],
    page20: u32,
) -> PyResult<HashMap<String, Value>> {
    let mut operands = HashMap::new();
    for entry in layout {
        let value = decode_operand_entry(stream, entry, page20)?;
        operands.insert(entry.key.clone(), value);
    }
    Ok(operands)
}

fn decode_operand_entry(
    stream: &mut InstructionStream<'_>,
    entry: &LayoutEntry,
    page20: u32,
) -> PyResult<Value> {
    match entry.kind.as_str() {
        "imm8" => Ok(encode_imm8(stream.read_u8())),
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
        other => Err(PyValueError::new_err(format!(
            "unsupported layout kind {other}"
        ))),
    }
}

fn decode_regsel(
    stream: &mut InstructionStream<'_>,
    entry: &LayoutEntry,
) -> PyResult<Value> {
    let allowed = entry.meta.get("allowed_groups").and_then(|value| {
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
        .ok_or_else(|| PyValueError::new_err("unsupported register index"))?;
    if let Some(groups) = allowed.as_ref() {
        if !groups.iter().any(|g| g.eq_ignore_ascii_case(group)) {
            return Err(PyValueError::new_err(format!(
                "register {name} not allowed for operand"
            )));
        }
    }
    Ok(encode_regsel(name, group))
}

fn decode_ext_reg_ptr(stream: &mut InstructionStream<'_>) -> PyResult<Value> {
    let reg_byte = stream.read_u8();
    let mode_code = (reg_byte >> 4) & 0x0F;
    let mode = EXT_REG_MODES
        .iter()
        .find(|mode| mode.code == mode_code)
        .ok_or_else(|| PyValueError::new_err("unsupported ext-reg pointer mode"))?;
    let idx = (reg_byte & 0x07) as usize;
    let (name, group, _) = REG_TABLE
        .get(idx)
        .ok_or_else(|| PyValueError::new_err("invalid pointer register"))?;
    if !group.eq_ignore_ascii_case("r3") {
        return Err(PyValueError::new_err(format!(
            "pointer register must be r3, got {name}"
        )));
    }
    let mut payload = Map::new();
    payload.insert("kind".into(), Value::String("ext_reg_ptr".into()));
    payload.insert("ptr".into(), encode_regsel(name, group));
    payload.insert("mode".into(), Value::String(mode.name.into()));
    if mode.needs_disp {
        let magnitude = stream.read_u8() as i32;
        let signed = if mode.disp_sign >= 0 {
            magnitude
        } else {
            -magnitude
        };
        payload.insert("disp".into(), encode_disp8(signed));
    }
    Ok(Value::Object(payload))
}

fn decode_imem_ptr(stream: &mut InstructionStream<'_>) -> PyResult<Value> {
    let mode_byte = stream.read_u8();
    let mode = IMEM_MODES
        .iter()
        .find(|entry| entry.code == mode_byte)
        .ok_or_else(|| PyValueError::new_err("unsupported IMEM pointer mode"))?;
    let base = stream.read_u8();
    let mut payload = Map::new();
    payload.insert("kind".into(), Value::String("imem_ptr".into()));
    payload.insert("base".into(), encode_imm8(base));
    payload.insert("mode".into(), Value::String(mode.name.into()));
    if mode.needs_disp {
        let magnitude = stream.read_u8() as i32;
        let signed = if mode.disp_sign >= 0 {
            magnitude
        } else {
            -magnitude
        };
        payload.insert("disp".into(), encode_disp8(signed));
    }
    Ok(Value::Object(payload))
}

fn lookup_manifest_entry(
    opcode: u8,
    pre: Option<PreTuple>,
) -> PyResult<&'static ManifestEntry> {
    let key = OpcodeLookupKey { opcode, pre };
    OPCODE_LOOKUP
        .get(&key)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "no manifest entry for opcode 0x{opcode:02X} (pre={pre:?})"
            ))
        })
}

fn decode_bound_from_memory(
    memory: &mut MemoryImage,
    fallback: MemoryProxyBus,
    pc: u32,
) -> PyResult<(BoundInstrRepr, u8, usize)> {
    let mut stream = InstructionStream::new(memory, fallback, pc);
    let mut pending_pre: Option<PreTuple> = None;
    let mut prefix_guard = 0usize;

    loop {
        let byte = stream.read_u8();
        if let Some(pair) = PRE_BY_OPCODE.get(&byte).copied() {
            pending_pre = Some(pair);
            prefix_guard += 1;
            if prefix_guard > MAX_PREFIX_CHAIN {
                return Err(PyValueError::new_err(
                    "excessive PRE prefixes before instruction",
                ));
            }
            continue;
        }

        let entry = lookup_manifest_entry(byte, pending_pre)?;
        let page20 = stream.page20();
        let operands = decode_operands(&mut stream, &entry.layout, page20)?;
        let length = stream.consumed();
        let bound = BoundInstrRepr {
            opcode: byte as u32,
            mnemonic: entry.mnemonic.clone(),
            family: entry.family.clone(),
            length: length as u8,
            pre: entry.pre.clone(),
            operands,
        };
        return Ok((bound, byte, length));
    }
}

fn space_to_str(space: &Space) -> &'static str {
    match space {
        Space::Int => "int",
        Space::Ext => "ext",
        Space::Code => "code",
    }
}

fn find_entry(bound: &BoundInstrRepr) -> Option<&'static ManifestEntry> {
    (*MANIFEST).iter().find(|entry| {
        entry.opcode == bound.opcode && entry.pre == bound.pre
    })
}

fn register_width(name: &str) -> u8 {
    match name.to_ascii_uppercase().as_str() {
        "A" | "B" | "IL" | "IH" => 8,
        "BA" | "I" => 16,
        "X" | "Y" | "U" | "S" => 24,
        "F" => 8,
        "PC" => 20,
        _ => DEFAULT_REG_WIDTH,
    }
}

type PreTuple = (&'static str, &'static str);

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
struct OpcodeLookupKey {
    opcode: u8,
    pre: Option<PreTuple>,
}

static OPCODE_LOOKUP: Lazy<HashMap<OpcodeLookupKey, &'static ManifestEntry>> =
    Lazy::new(|| {
        let mut map = HashMap::new();
        for entry in OPCODE_INDEX.iter() {
            let key = OpcodeLookupKey {
                opcode: entry.opcode,
                pre: entry.pre.map(|pre| (pre.first, pre.second)),
            };
            let manifest_entry = &(*MANIFEST)[entry.manifest_index];
            map.insert(key, manifest_entry);
        }
        map
    });

const PRE_PREFIXES: &[(u8, PreTuple)] = &[
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

static PRE_BY_OPCODE: Lazy<HashMap<u8, PreTuple>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (opcode, pair) in PRE_PREFIXES.iter() {
        map.insert(*opcode, *pair);
    }
    map
});

fn patch_binder(
    template: &Map<String, Value>,
    operands: &HashMap<String, Value>,
) -> Map<String, Value> {
    let mut merged = template.clone();
    for (key, value) in operands {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

impl From<&PreInfo> for PreLatch {
    fn from(info: &PreInfo) -> Self {
        PreLatch {
            first: info.first.clone(),
            second: info.second.clone(),
        }
    }
}

fn eval_manifest_entry<B: rust_scil::bus::Bus>(
    state: &mut RsState,
    bus: &mut B,
    bound: &BoundInstrRepr,
) -> PyResult<()> {
    let entry = find_entry(bound).ok_or_else(|| {
        PyValueError::new_err(format!(
            "no manifest entry for opcode {} pre {:?}",
            bound.opcode, bound.pre
        ))
    })?;
    let binder_json = patch_binder(&entry.binder, &bound.operands);
    let binder: Binder = serde_json::from_value(Value::Object(binder_json))
        .map_err(|e| PyValueError::new_err(format!("binder json: {e}")))?;
    let instr: Instr = serde_json::from_value(entry.instr.clone())
        .map_err(|e| PyValueError::new_err(format!("instr json: {e}")))?;
    let pre = bound.pre.as_ref().map(|info| info.into());
    eval::step(state, bus, &instr, &binder, pre)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
}

#[pyclass]
struct Runtime {
    state: RsState,
    memory: Py<PyAny>,
    memory_image: MemoryImage,
}

#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let mut runtime = Self {
            state: RsState::default(),
            memory: memory.into(),
            memory_image: MemoryImage::new(),
        };
        if reset_on_init {
            runtime.power_on_reset();
        }
        Ok(runtime)
    }

    fn power_on_reset(&mut self) {
        self.state = RsState::default();
    }

    fn load_external_memory(&mut self, snapshot: &PyBytes) -> PyResult<()> {
        self.memory_image.load_external(snapshot.as_bytes());
        Ok(())
    }

    fn set_python_ranges(&mut self, ranges: Vec<(u32, u32)>) -> PyResult<()> {
        self.memory_image.set_python_ranges(ranges);
        Ok(())
    }

    fn drain_external_writes(&mut self) -> PyResult<Vec<(u32, u8)>> {
        Ok(self.memory_image.drain_dirty())
    }

    fn execute_instruction(&mut self) -> PyResult<(u8, u8)> {
        let pc = self.state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;
        let decode_proxy = MemoryProxyBus::new(self.memory.clone());
        let (bound, opcode, length) =
            decode_bound_from_memory(&mut self.memory_image, decode_proxy, pc)?;
        let fallback_bus = MemoryProxyBus::new(self.memory.clone());
        let mut bus = HybridBus::new(&mut self.memory_image, fallback_bus);
        eval_manifest_entry(&mut self.state, &mut bus, &bound)?;
        let current_pc = self.state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;
        if current_pc == pc {
            let next_pc = (pc + length as u32) & ADDRESS_MASK;
            self.state
                .set_reg("PC", next_pc, register_width("PC"));
        }
        Ok((opcode, length as u8))
    }

    fn execute_bound_repr(&mut self, bound_json: &str) -> PyResult<()> {
        let bound: BoundInstrRepr = serde_json::from_str(bound_json)
            .map_err(|e| PyValueError::new_err(format!("bound json: {e}")))?;
        let mut bus = MemoryProxyBus::new(self.memory.clone());
        eval_manifest_entry(&mut self.state, &mut bus, &bound)
    }

    fn read_register(&self, name: &str) -> PyResult<u32> {
        Ok(self.state.get_reg(name, register_width(name)))
    }

    fn write_register(&mut self, name: &str, value: u32) -> PyResult<()> {
        let width = register_width(name);
        self.state.set_reg(name, value, width);
        Ok(())
    }

    fn read_flag(&self, name: &str) -> PyResult<u8> {
        Ok(self.state.get_flag(name) as u8)
    }

    fn write_flag(&mut self, name: &str, value: u8) -> PyResult<()> {
        self.state.set_flag(name, value as u32);
        Ok(())
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.state)
            .map_err(|e| PyRuntimeError::new_err(format!("state serialize: {e}")))
    }

    fn load_snapshot_json(&mut self, payload: &str) -> PyResult<()> {
        self.state = serde_json::from_str(payload)
            .map_err(|e| PyValueError::new_err(format!("state json: {e}")))?;
        Ok(())
    }

    #[getter]
    fn halted(&self) -> bool {
        self.state.halted
    }

    #[setter]
    fn set_halted(&mut self, value: bool) {
        self.state.halted = value;
    }
}

#[pymethods]
impl Cpu {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(py: Python<'_>, memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let bridge_mod = PyModule::import(py, "sc62015.pysc62015._rust_bridge")?;
        let bridge_cls = bridge_mod.getattr("BridgeCPU")?;
        let bridge = bridge_cls.call1((memory, reset_on_init))?;
        Ok(Self {
            inner: bridge.into_py(py),
        })
    }

    fn execute_instruction(&self, py: Python<'_>, address: u32) -> PyResult<(u8, u8)> {
        let helper = self.inner.bind(py);
        helper
            .call_method1("execute_instruction", (address,))?
            .extract()
    }

    fn power_on_reset(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method0("power_on_reset")
            .map(|_| ())
    }

    fn read_register(&self, py: Python<'_>, name: &str) -> PyResult<u32> {
        self.inner
            .bind(py)
            .call_method1("read_register", (name,))?
            .extract()
    }

    fn write_register(&self, py: Python<'_>, name: &str, value: u32) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method1("write_register", (name, value))?
            .extract::<PyObject>()?;
        Ok(())
    }

    fn read_flag(&self, py: Python<'_>, name: &str) -> PyResult<u8> {
        self.inner
            .bind(py)
            .call_method1("read_flag", (name,))?
            .extract()
    }

    fn write_flag(&self, py: Python<'_>, name: &str, value: u8) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method1("write_flag", (name, value))?
            .extract::<PyObject>()?;
        Ok(())
    }

    fn snapshot_cpu_registers(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .bind(py)
            .call_method0("snapshot_cpu_registers")
            .map(|obj| obj.into())
    }

    fn load_cpu_snapshot(&self, py: Python<'_>, snapshot: PyObject) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method1("load_cpu_snapshot", (snapshot,))?
            .extract::<PyObject>()?;
        Ok(())
    }

    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .bind(py)
            .call_method0("get_stats")
            .map(|obj| obj.into())
    }

    #[getter]
    fn call_sub_level(&self, py: Python<'_>) -> PyResult<u32> {
        self.inner
            .bind(py)
            .getattr("call_sub_level")?
            .extract()
    }

    #[setter]
    fn set_call_sub_level(&self, py: Python<'_>, value: u32) -> PyResult<()> {
        self.inner
            .bind(py)
            .setattr("call_sub_level", value)?;
        Ok(())
    }

    #[getter]
    fn halted(&self, py: Python<'_>) -> PyResult<bool> {
        self.inner.bind(py).getattr("halted")?.extract()
    }

    #[setter]
    fn set_halted(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        self.inner.bind(py).setattr("halted", value)?;
        Ok(())
    }
}

#[pyfunction]
fn backend_name() -> &'static str {
    "rust"
}

#[pyfunction]
fn is_ready() -> bool {
    true
}

#[pyfunction(signature = (state_json, instr_json, binder_json, py_bus, pre_json=None))]
fn scil_step_json(
    state_json: &str,
    instr_json: &str,
    binder_json: &str,
    py_bus: PyObject,
    pre_json: Option<&str>,
) -> PyResult<String> {
    let mut state: RsState = serde_json::from_str(state_json)
        .map_err(|e| PyValueError::new_err(format!("state json: {e}")))?;
    let instr: Instr = serde_json::from_str(instr_json)
        .map_err(|e| PyValueError::new_err(format!("instr json: {e}")))?;
    let binder: Binder = serde_json::from_str(binder_json)
        .map_err(|e| PyValueError::new_err(format!("binder json: {e}")))?;
    let pre: Option<PreLatch> = match pre_json {
        Some(raw) => Some(
            serde_json::from_str(raw)
                .map_err(|e| PyValueError::new_err(format!("pre json: {e}")))?,
        ),
        None => None,
    };

    let mut bus = PyBus::new(py_bus);
    eval::step(&mut state, &mut bus, &instr, &binder, pre)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&state)
        .map_err(|e| PyRuntimeError::new_err(format!("state serialize: {e}")))
}

#[pyfunction]
fn execute_bound_repr(state_json: &str, bound_json: &str, py_bus: PyObject) -> PyResult<String> {
    let mut state: RsState = serde_json::from_str(state_json)
        .map_err(|e| PyValueError::new_err(format!("state json: {e}")))?;
    let bound: BoundInstrRepr = serde_json::from_str(bound_json)
        .map_err(|e| PyValueError::new_err(format!("bound json: {e}")))?;
    let mut bus = PyBus::new(py_bus);
    eval_manifest_entry(&mut state, &mut bus, &bound)?;
    serde_json::to_string(&state)
        .map_err(|e| PyRuntimeError::new_err(format!("state serialize: {e}")))
}

#[pymodule]
fn _sc62015_rustcore(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Cpu>()?;
    m.add_class::<Runtime>()?;
    m.add("__backend_name__", backend_name())?;
    m.add("HAS_CPU_IMPLEMENTATION", true)?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_ready, m)?)?;
    m.add_function(wrap_pyfunction!(scil_step_json, m)?)?;
    m.add_function(wrap_pyfunction!(execute_bound_repr, m)?)?;
    Ok(())
}
