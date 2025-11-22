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
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use rust_scil::{
    ast::{Binder, Expr, Instr, PreLatch},
    bus::Space,
    eval,
    state::State as RsState,
};
use serde_json::{self, Map, Value};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Instant;
use sc62015_core::{
    apply_registers, collect_registers, eval_manifest_entry as core_eval_manifest_entry,
    load_snapshot as core_load_snapshot,
    now_timestamp, register_width, save_snapshot as core_save_snapshot,
    BoundInstrBuilder, BoundInstrView, BusProfiler, CoreError, ExecManifestEntry, HostMemory,
    HybridBus, LayoutEntryView, ManifestEntryView, MemoryImage, OpcodeIndexView, OpcodeLookup,
    PerfettoTracer, SnapshotLoad, SnapshotMetadata, TimerContext, ADDRESS_MASK, execute_step,
    INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE, SNAPSHOT_MAGIC,
    SNAPSHOT_VERSION,
};
mod keyboard;
mod lcd;
use lcd::LcdController;
use keyboard::KeyboardMatrix;
#[derive(Clone)]
struct CachedManifestEntry {
    inner: ManifestEntry,
    instr_parsed: Instr,
    binder_template: Binder,
}

static MANIFEST: Lazy<Vec<CachedManifestEntry>> = Lazy::new(|| {
    let raw: Vec<ManifestEntry> =
        serde_json::from_str(generated::payload::PAYLOAD).expect("manifest json");
    raw.into_iter()
        .map(|entry| {
            let instr_parsed: Instr = serde_json::from_value(entry.instr.clone())
                .expect("manifest instr json");
            let binder_template: Binder =
                serde_json::from_value(Value::Object(entry.binder.clone())).expect("manifest binder json");
            CachedManifestEntry {
                inner: entry,
                instr_parsed,
                binder_template,
            }
        })
        .collect()
});

#[derive(Default)]
struct FallbackHistogram {
    shift: u32,
    load: HashMap<u32, u64>,
    store: HashMap<u32, u64>,
}

impl FallbackHistogram {
    fn new(shift: u32) -> Self {
        Self {
            shift,
            load: HashMap::new(),
            store: HashMap::new(),
        }
    }

    fn bucket(&self, address: u32) -> u32 {
        address >> self.shift
    }

    fn record_load(&mut self, address: u32) {
        let bucket = self.bucket(address);
        *self.load.entry(bucket).or_insert(0) += 1;
    }

    fn record_store(&mut self, address: u32) {
        let bucket = self.bucket(address);
        *self.store.entry(bucket).or_insert(0) += 1;
    }

    fn loads_dict(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        for (bucket, count) in &self.load {
            dict.set_item(bucket << self.shift, count).unwrap();
        }
        dict.into_py(py)
    }

    fn stores_dict(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        for (bucket, count) in &self.store {
            dict.set_item(bucket << self.shift, count).unwrap();
        }
        dict.into_py(py)
    }

    fn as_dict(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("bucket_shift", self.shift).unwrap();
        dict.set_item("loads", self.loads_dict(py)).unwrap();
        dict.set_item("stores", self.stores_dict(py)).unwrap();
        dict.into_py(py)
    }
}

#[derive(Default)]
struct RuntimeProfile {
    enabled: bool,
    total_calls: u64,
    total_ns: u128,
    decode_calls: u64,
    decode_ns: u128,
    eval_calls: u64,
    eval_ns: u128,
    bus_load_calls: u64,
    bus_load_python: u64,
    bus_store_calls: u64,
    bus_store_python: u64,
    fallback_hist: Option<FallbackHistogram>,
}

impl RuntimeProfile {
    fn from_env() -> Self {
        let enabled = env::var("SCIL_RUNTIME_PROFILE")
            .map(|value| value != "0")
            .unwrap_or(false);
        let fallback_hist = env::var("SCIL_RUNTIME_PROFILE_HIST")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(FallbackHistogram::new);
        Self {
            enabled,
            fallback_hist,
            ..Default::default()
        }
    }

    fn timer_start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn record_decode(&mut self, start: Option<Instant>) {
        if let (true, Some(instant)) = (self.enabled, start) {
            self.decode_calls += 1;
            self.decode_ns += instant.elapsed().as_nanos();
        }
    }

    fn record_eval(&mut self, start: Option<Instant>) {
        if let (true, Some(instant)) = (self.enabled, start) {
            self.eval_calls += 1;
            self.eval_ns += instant.elapsed().as_nanos();
        }
    }

    fn record_total(&mut self, start: Option<Instant>) {
        if let (true, Some(instant)) = (self.enabled, start) {
            self.total_calls += 1;
            self.total_ns += instant.elapsed().as_nanos();
        }
    }

    fn reset(&mut self) {
        let enabled = self.enabled;
        let hist_shift = self.fallback_hist.as_ref().map(|hist| hist.shift);
        *self = RuntimeProfile {
            enabled,
            fallback_hist: hist_shift.map(FallbackHistogram::new),
            ..Default::default()
        };
        self.enabled = enabled;
    }

    fn as_dict(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("enabled", self.enabled).unwrap();
        dict.set_item("total_calls", self.total_calls).unwrap();
        dict.set_item("total_time_ns", self.total_ns).unwrap();
        dict.set_item("decode_calls", self.decode_calls).unwrap();
        dict.set_item("decode_time_ns", self.decode_ns).unwrap();
        dict.set_item("eval_calls", self.eval_calls).unwrap();
        dict.set_item("eval_time_ns", self.eval_ns).unwrap();
        dict.set_item("bus_load_calls", self.bus_load_calls).unwrap();
        dict.set_item("bus_load_python", self.bus_load_python).unwrap();
        dict.set_item("bus_store_calls", self.bus_store_calls).unwrap();
        dict.set_item("bus_store_python", self.bus_store_python).unwrap();
        if let Some(hist) = &self.fallback_hist {
            dict.set_item("python_fallback_hist", hist.as_dict(py))
                .unwrap();
        }
        dict.into_py(py)
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
    fn new(obj: PyObject, py: Python<'_>) -> Self {
        Self { inner: obj.into_py(py) }
    }
}

impl rust_scil::bus::Bus for PyBus {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        Python::with_gil(|py| {
            self.inner
                .bind(py)
                .call_method1("load", (space_to_str(&space), addr, bits))
                .and_then(|obj| obj.extract::<u32>())
                .expect("MemoryAdapter.load returned non-u32")
        })
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        Python::with_gil(|py| {
            let _ = self
                .inner
                .bind(py)
                .call_method1("store", (space_to_str(&space), addr, bits, value));
        });
    }
}

struct PythonHost {
    memory: Py<PyAny>,
}

impl PythonHost {
    fn new(py: Python<'_>, memory: &Py<PyAny>) -> Self {
        Self {
            memory: memory.clone_ref(py),
        }
    }
}

impl HostMemory for PythonHost {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        Python::with_gil(|py| {
            self.memory
                .bind(py)
                .call_method1("load", (space_to_str(&space), addr, bits))
                .and_then(|obj| obj.extract::<u32>())
                .expect("MemoryAdapter.load returned non-u32")
        })
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        Python::with_gil(|py| {
            let _ = self
                .memory
                .bind(py)
                .call_method1("store", (space_to_str(&space), addr, bits, value));
        });
    }

    fn read_byte(&mut self, address: u32) -> u8 {
        Python::with_gil(|py| {
            self.memory
                .bind(py)
                .call_method1("read_byte", (address,))
                .and_then(|obj| obj.extract::<u8>())
                .expect("memory.read_byte must return int")
        })
    }

    fn notify_lcd_write(&mut self, address: u32, value: u32) {
        Python::with_gil(|py| {
            let bound = self.memory.bind(py);
            if let Ok(callback) = bound.getattr("_rust_lcd_write") {
                let _ = callback.call1((address, value, py.None()));
            }
        });
    }
}

impl BusProfiler for RuntimeProfile {
    fn record_bus_load(&mut self) {
        self.bus_load_calls += 1;
    }

    fn record_bus_store(&mut self) {
        self.bus_store_calls += 1;
    }

    fn record_python_load(&mut self, address: u32) {
        self.bus_load_python += 1;
        if let Some(hist) = self.fallback_hist.as_mut() {
            hist.record_load(address);
        }
    }

    fn record_python_store(&mut self, address: u32) {
        self.bus_store_python += 1;
        if let Some(hist) = self.fallback_hist.as_mut() {
            hist.record_store(address);
        }
    }
}

impl LayoutEntryView for LayoutEntry {
    fn key(&self) -> &str {
        &self.key
    }

    fn kind(&self) -> &str {
        &self.kind
    }

    fn meta(&self) -> &HashMap<String, Value> {
        &self.meta
    }
}

impl ManifestEntryView for CachedManifestEntry {
    type Layout = LayoutEntry;

    fn opcode(&self) -> u8 {
        self.inner.opcode as u8
    }

    fn pre(&self) -> Option<(String, String)> {
        self.inner
            .pre
            .as_ref()
            .map(|info| (info.first.clone(), info.second.clone()))
    }

    fn binder(&self) -> &Map<String, Value> {
        &self.inner.binder
    }

    fn instr(&self) -> &Value {
        &self.inner.instr
    }

    fn layout(&self) -> &[Self::Layout] {
        &self.inner.layout
    }

    fn parsed_instr(&self) -> Option<&Instr> {
        Some(&self.instr_parsed)
    }

    fn parsed_binder(&self) -> Option<&Binder> {
        Some(&self.binder_template)
    }
}

impl ExecManifestEntry for CachedManifestEntry {
    fn mnemonic(&self) -> &str {
        &self.inner.mnemonic
    }

    fn family(&self) -> Option<&str> {
        self.inner.family.as_deref()
    }
}

impl BoundInstrView for BoundInstrRepr {
    fn opcode(&self) -> u32 {
        self.opcode
    }

    fn operands(&self) -> &HashMap<String, Value> {
        &self.operands
    }

    fn pre(&self) -> Option<(String, String)> {
        self.pre
            .as_ref()
            .map(|pre| (pre.first.clone(), pre.second.clone()))
    }
}

impl BoundInstrBuilder for BoundInstrRepr {
    fn from_parts(
        opcode: u32,
        mnemonic: &str,
        family: Option<&str>,
        length: u8,
        pre: Option<(String, String)>,
        operands: HashMap<String, Value>,
    ) -> Self {
        Self {
            opcode,
            mnemonic: mnemonic.to_string(),
            family: family.map(|f| f.to_string()),
            length,
            pre: pre.map(|(first, second)| PreInfo { first, second }),
            operands,
        }
    }
}

impl OpcodeIndexView for generated::opcode_index::OpcodeIndexEntry {
    fn opcode(&self) -> u8 {
        self.opcode
    }

    fn pre(&self) -> Option<(String, String)> {
        self.pre
            .map(|pre| (pre.first.to_string(), pre.second.to_string()))
    }

    fn manifest_index(&self) -> usize {
        self.manifest_index
    }
}

static OPCODE_LOOKUP: Lazy<OpcodeLookup<'static, CachedManifestEntry>> =
    Lazy::new(|| OpcodeLookup::new(&*MANIFEST, OPCODE_INDEX));

fn eval_manifest_entry<B: rust_scil::bus::Bus>(
    state: &mut RsState,
    bus: &mut B,
    bound: &BoundInstrRepr,
) -> PyResult<()> {
    core_eval_manifest_entry(&*MANIFEST, state, bus, bound)
        .map_err(|e| PyRuntimeError::new_err(e))
}

fn space_to_str(space: &Space) -> &'static str {
    match space {
        Space::Int => "int",
        Space::Ext => "ext",
        Space::Code => "code",
    }
}

#[pyclass]
struct Runtime {
    state: RsState,
    memory: Py<PyAny>,
    memory_image: MemoryImage,
    profile: RuntimeProfile,
    lcd: Option<LcdController>,
    keyboard: Option<KeyboardMatrix>,
    cycle_count: u64,
    instruction_count: u64,
    timer: TimerContext,
    perfetto_path: Option<String>,
    perfetto: Option<PerfettoTracer>,
}

fn perform_reset_side_effects(memory: &mut MemoryImage, state: &mut RsState) {
    // IMEM offsets
    const UCR: u32 = 0xF7;
    const USR: u32 = 0xF8;
    const ISR: u32 = 0xFC;
    const SCR: u32 = 0xFD;
    const LCC: u32 = 0xFE;
    const SSR: u32 = 0xFF;
    const RESET_VEC: u32 = 0xFFFFA;

    // Reset UCR, ISR, SCR to 0
    memory.write_internal_byte(UCR, 0x00);
    memory.write_internal_byte(ISR, 0x00);
    memory.write_internal_byte(SCR, 0x00);

    // Reset LCC bit 7 to 0 (if present)
    let lcc = memory.read_internal_byte(LCC).unwrap_or(0);
    memory.write_internal_byte(LCC, lcc & !0x80);

    // USR: clear bits 0-5, set bits 3 and 4
    let usr = memory.read_internal_byte(USR).unwrap_or(0);
    let usr_val = (usr & !0x3F) | 0x18;
    memory.write_internal_byte(USR, usr_val);

    // SSR: clear bit 2
    let ssr = memory.read_internal_byte(SSR).unwrap_or(0);
    memory.write_internal_byte(SSR, ssr & !0x04);

    // Read reset vector (24-bit little-endian) from external memory
    let lo = memory.read_byte(RESET_VEC).unwrap_or(0) as u32;
    let mid = memory.read_byte(RESET_VEC + 1).unwrap_or(0) as u32;
    let hi = memory.read_byte(RESET_VEC + 2).unwrap_or(0) as u32;
    let pc = ((hi << 16) | (mid << 8) | lo) & ADDRESS_MASK;
    state.set_reg("PC", pc, register_width("PC"));
}

#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let use_rust_lcd = env::var("RUST_PURE_LCD").map(|v| v == "1").unwrap_or(false);
        let use_rust_keyboard =
            env::var("RUST_PURE_KEYBOARD").map(|v| v == "1").unwrap_or(false);
        let timer_in_rust = env::var("RUST_TIMER_IN_RUST")
            .map(|v| v == "1")
            .unwrap_or(false);
        let perfetto_path = env::var("RUST_PERFETTO_PATH").ok();
        let mut runtime = Self {
            state: RsState::default(),
            memory: memory.into(),
            memory_image: MemoryImage::new(),
            profile: RuntimeProfile::from_env(),
            lcd: use_rust_lcd.then(LcdController::new),
            keyboard: use_rust_keyboard.then(KeyboardMatrix::new),
            cycle_count: 0,
            instruction_count: 0,
            timer: TimerContext::new(timer_in_rust, 500, 5000),
            perfetto: None,
            perfetto_path,
        };
        runtime
            .memory_image
            .set_keyboard_bridge(runtime.keyboard.is_some());
        if reset_on_init {
            runtime.power_on_reset();
        }
        Ok(runtime)
    }

    fn power_on_reset(&mut self) {
        self.state = RsState::default();
        if let Some(lcd) = self.lcd.as_mut() {
            lcd.reset();
        }
        if let Some(kbd) = self.keyboard.as_mut() {
            kbd.reset(&mut self.memory_image);
        }
        self.timer.reset();
        perform_reset_side_effects(&mut self.memory_image, &mut self.state);
        self.cycle_count = 0;
        self.instruction_count = 0;
    }

    #[pyo3(signature = (path=None))]
    fn set_perfetto_trace(&mut self, path: Option<&str>) -> PyResult<()> {
        let resolved = path
            .map(|p| p.to_string())
            .or_else(|| self.perfetto_path.clone());
        self.perfetto = resolved.map(|p| PerfettoTracer::new(PathBuf::from(p)));
        Ok(())
    }

    fn flush_perfetto(&mut self) -> PyResult<()> {
        if let Some(tracer) = self.perfetto.take() {
            tracer
                .finish()
                .map_err(|e| PyRuntimeError::new_err(format!("perfetto finish: {e}")))?;
        }
        Ok(())
    }

    fn load_external_memory(&mut self, snapshot: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.memory_image.load_external(snapshot.as_bytes());
        Ok(())
    }

    fn load_internal_memory(&mut self, payload: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.memory_image.load_internal(payload.as_bytes());
        Ok(())
    }

    fn export_lcd_snapshot(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        if let Some(lcd) = &self.lcd {
            let (meta, payload) = lcd.export_snapshot();
            let meta_str = serde_json::to_string(&meta)
                .map_err(|e| PyRuntimeError::new_err(format!("lcd meta serialize: {e}")))?;
            let json_mod = PyModule::import_bound(py, "json")
                .map_err(|e| PyRuntimeError::new_err(format!("import json: {e}")))?;
            let meta_py = json_mod
                .call_method1("loads", (meta_str,))?
                .into_py(py);
            let payload_py = PyBytes::new_bound(py, &payload).into_py(py);
            Ok((meta_py, payload_py))
        } else {
            Ok((py.None(), py.None()))
        }
    }

    fn set_python_ranges(&mut self, ranges: Vec<(u32, u32)>) -> PyResult<()> {
        self.memory_image.set_python_ranges(ranges);
        Ok(())
    }

    fn set_readonly_ranges(&mut self, ranges: Vec<(u32, u32)>) -> PyResult<()> {
        self.memory_image.set_readonly_ranges(ranges);
        Ok(())
    }

    fn set_keyboard_bridge(&mut self, enabled: bool) -> PyResult<()> {
        if enabled {
            if self.keyboard.is_none() {
                self.keyboard = Some(KeyboardMatrix::new());
            }
            if let Some(kbd) = self.keyboard.as_mut() {
                kbd.reset(&mut self.memory_image);
            }
        } else {
            self.keyboard = None;
        }
        self.memory_image.set_keyboard_bridge(enabled);
        Ok(())
    }

    fn drain_external_writes(&mut self) -> PyResult<Vec<(u32, u8)>> {
        Ok(self.memory_image.drain_dirty())
    }

    fn drain_internal_writes(&mut self) -> PyResult<Vec<(u32, u8)>> {
        Ok(self.memory_image.drain_dirty_internal())
    }

    fn read_external_segment(
        &self,
        py: Python<'_>,
        start: u32,
        length: usize,
    ) -> PyResult<Py<PyBytes>> {
        let start = start as usize;
        if let Some(slice) = self.memory_image.external_segment(start, length) {
            Ok(PyBytes::new_bound(py, slice).unbind())
        } else {
            Err(PyValueError::new_err("start offset outside external memory"))
        }
    }

    fn apply_host_write(&mut self, address: u32, value: u8) -> PyResult<()> {
        self.memory_image.apply_host_write(address, value);
        Ok(())
    }

    #[getter]
    fn profile_enabled(&self) -> bool {
        self.profile.enabled
    }

    #[setter]
    fn set_profile_enabled(&mut self, value: bool) {
        self.profile.enabled = value;
    }

    fn reset_profile_stats(&mut self) {
        self.profile.reset();
    }

    fn get_profile_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.profile.as_dict(py))
    }

    fn keyboard_press(&mut self, matrix_code: u8) -> PyResult<()> {
        if let Some(kbd) = self.keyboard.as_mut() {
            kbd.press_matrix_code(matrix_code & 0x7F, &mut self.memory_image);
        }
        Ok(())
    }

    fn keyboard_release(&mut self, matrix_code: u8) -> PyResult<()> {
        if let Some(kbd) = self.keyboard.as_mut() {
            kbd.release_matrix_code(matrix_code & 0x7F, &mut self.memory_image);
        }
        Ok(())
    }

    fn keyboard_scan_tick(&mut self) -> PyResult<u32> {
        if let Some(kbd) = self.keyboard.as_mut() {
            let count = kbd.scan_tick();
            if count > 0 {
                kbd.write_fifo_to_memory(&mut self.memory_image);
            }
            return Ok(count as u32);
        }
        Ok(0)
    }

    fn keyboard_fifo_snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(kbd) = &self.keyboard {
            let snap = kbd.fifo_snapshot();
            return Ok(PyBytes::new_bound(py, &snap).into_py(py));
        }
        Ok(PyBytes::new_bound(py, &[]).into_py(py))
    }

    fn keyboard_irq_count(&self) -> PyResult<u32> {
        Ok(self.keyboard.as_ref().map(|k| k.irq_count()).unwrap_or(0))
    }

    #[pyo3(signature = (
        pending,
        imr,
        isr,
        next_mti,
        next_sti,
        source=None,
        in_interrupt=false,
        interrupt_stack=None,
        next_interrupt_id=0
    ))]
    fn set_interrupt_state(
        &mut self,
        pending: bool,
        imr: u8,
        isr: u8,
        next_mti: i32,
        next_sti: i32,
        source: Option<String>,
        in_interrupt: bool,
        interrupt_stack: Option<Vec<u32>>,
        next_interrupt_id: u32,
    ) {
        self.timer.set_interrupt_state(
            pending,
            imr,
            isr,
            next_mti,
            next_sti,
            source,
            in_interrupt,
            interrupt_stack,
            next_interrupt_id,
        );
    }

    fn tick_timers(&mut self) {
        self.timer
            .tick_timers(&mut self.memory_image, &mut self.cycle_count);
    }

    fn execute_instruction(&mut self) -> PyResult<(u8, u8)> {
        let total_timer = self.profile.timer_start();
        let mut host = Python::with_gil(|py| PythonHost::new(py, &self.memory));
        let mut profile_ref = if self.profile.enabled {
            Some(&mut self.profile)
        } else {
            None
        };
        let (opcode, length) = execute_step::<BoundInstrRepr, CachedManifestEntry, PythonHost, RuntimeProfile>(
            &mut self.state,
            &mut self.memory_image,
            &mut host,
            MANIFEST.as_slice(),
            &OPCODE_LOOKUP,
            self.keyboard.as_mut().map(|k| k as &mut KeyboardMatrix),
            self.lcd.as_mut().map(|l| l as &mut LcdController),
            self.perfetto.as_mut().map(|t| t as &mut PerfettoTracer),
            profile_ref.as_deref_mut(),
            self.instruction_count,
        )
        .map_err(|e| PyValueError::new_err(format!("decode: {e}")))?;
        // Run keyboard scan after each instruction to simulate MTI-driven polling
        if let Some(kbd) = self.keyboard.as_mut() {
            if kbd.scan_tick() > 0 {
                kbd.write_fifo_to_memory(&mut self.memory_image);
            }
        }
        self.tick_timers();
        self.instruction_count = self.instruction_count.wrapping_add(1);
        self.profile.record_total(total_timer);
        Ok((opcode, length))
    }

    fn drain_pending_irq(&mut self) -> PyResult<Option<String>> {
        Ok(self.timer.drain_pending_irq())
    }

    fn export_external_memory_full(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new_bound(py, self.memory_image.external_slice()).unbind())
    }

    fn export_internal_memory_full(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new_bound(py, self.memory_image.internal_slice()).unbind())
    }

    fn save_snapshot(&mut self, path: &str) -> PyResult<()> {
        let (lcd_meta, lcd_payload) = if let Some(lcd) = &self.lcd {
            let (meta, payload) = lcd.export_snapshot();
            (Some(meta), payload)
        } else {
            (None, Vec::new())
        };

        let (timer, interrupts) = self.timer.snapshot_info();
        let metadata = SnapshotMetadata {
            magic: SNAPSHOT_MAGIC.to_string(),
            version: SNAPSHOT_VERSION,
            backend: "rust".to_string(),
            created: now_timestamp(),
            instruction_count: self.instruction_count,
            cycle_count: self.cycle_count,
            pc: self.state.get_reg("PC", register_width("PC")) & ADDRESS_MASK,
            timer,
            interrupts,
            fallback_ranges: self.memory_image.python_ranges().to_vec(),
            readonly_ranges: self.memory_image.readonly_ranges().to_vec(),
            internal_ram: (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32),
            imem: (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32),
            memory_dump_pc: 0,
            fast_mode: false,
            memory_image_size: self.memory_image.external_len(),
            lcd_payload_size: lcd_payload.len(),
            lcd: lcd_meta,
        };
        let registers = collect_registers(&self.state);
        let lcd_payload_ref = if lcd_payload.is_empty() {
            None
        } else {
            Some(lcd_payload.as_slice())
        };

        core_save_snapshot(
            std::path::Path::new(path),
            &metadata,
            &registers,
            &self.memory_image,
            lcd_payload_ref,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("snapshot save: {e}")))
    }

    fn load_snapshot(&mut self, path: &str) -> PyResult<()> {
        let loaded = core_load_snapshot(std::path::Path::new(path), &mut self.memory_image)
            .map_err(|e| match e {
                CoreError::InvalidSnapshot(msg) => PyValueError::new_err(msg),
                other => PyRuntimeError::new_err(format!("snapshot load: {other}")),
            })?;

        let SnapshotLoad {
            metadata,
            registers,
            lcd_payload,
        } = loaded;
        apply_registers(&mut self.state, &registers);

        if let (Some(meta), Some(payload)) = (metadata.lcd.as_ref(), lcd_payload.as_ref()) {
            if let Some(lcd) = self.lcd.as_mut() {
                lcd.load_snapshot(meta, payload)
                    .map_err(|err| PyRuntimeError::new_err(err))?;
            }
        }

        // Hardware reset invariants that are implicit in the Python backend snapshots:
        // ensure USR reports TXR/TXE ready (bits 3/4 set) even if the snapshot omits them.
        let usr = self.memory_image.read_internal_byte(0xF8).unwrap_or(0);
        if usr & 0x18 != 0x18 {
            self.memory_image.write_internal_byte(0xF8, usr | 0x18);
        }
        // Seed IMR with a sane default when snapshots omit it (match Python backend behaviour).
        let imr = self.memory_image.read_internal_byte(0xFB).unwrap_or(0);
        if imr == 0 {
            self.memory_image.write_internal_byte(0xFB, 0xC3);
        }

        // Timer/IRQ bookkeeping
        self.instruction_count = metadata.instruction_count;
        self.cycle_count = metadata.cycle_count;
        self.timer
            .apply_snapshot_info(&metadata.timer, &metadata.interrupts);
        Ok(())
    }

    fn execute_bound_repr(&mut self, bound_json: &str) -> PyResult<()> {
        let bound: BoundInstrRepr = serde_json::from_str(bound_json)
            .map_err(|e| PyValueError::new_err(format!("bound json: {e}")))?;
        let mut exec_host = Python::with_gil(|py| PythonHost::new(py, &self.memory));
        let eval_timer = self.profile.timer_start();
        let mut profile_ref = if self.profile.enabled {
            Some(&mut self.profile)
        } else {
            None
        };
        let mut bus = HybridBus::new(
            &mut self.memory_image,
            &mut exec_host,
            profile_ref.as_deref_mut(),
            self.lcd.as_mut(),
            self.keyboard.as_mut(),
            None,
            None,
            self.instruction_count,
        );
        let result = eval_manifest_entry(&mut self.state, &mut bus, &bound);
        if result.is_ok() {
            self.profile.record_eval(eval_timer);
        }
        result
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
        let bridge_mod = PyModule::import_bound(py, "sc62015.pysc62015._rust_bridge")?;
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

    fn save_snapshot(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let helper = self.inner.bind(py);
        if helper.hasattr("save_snapshot")? {
            helper.call_method1("save_snapshot", (path,))?;
        }
        Ok(())
    }

    fn load_snapshot(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let helper = self.inner.bind(py);
        if helper.hasattr("load_snapshot")? {
            helper.call_method1("load_snapshot", (path,))?;
        }
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

    fn export_lcd_snapshot(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        self.inner
            .bind(py)
            .call_method0("export_lcd_snapshot")?
            .extract()
    }

    fn mark_memory_dirty(&self, py: Python<'_>) -> PyResult<()> {
        let helper = self.inner.bind(py);
        if helper.hasattr("_memory_synced")? {
            helper.setattr("_memory_synced", false)?;
        }
        Ok(())
    }

    fn is_memory_synced(&self, py: Python<'_>) -> PyResult<bool> {
        let helper = self.inner.bind(py);
        if helper.hasattr("_memory_synced")? {
            return helper.getattr("_memory_synced")?.extract();
        }
        Ok(false)
    }

    fn notify_host_write(
        &self,
        py: Python<'_>,
        address: u32,
        value: u8,
    ) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method1("notify_host_write", (address, value))?;
        Ok(())
    }

    #[pyo3(signature = (
        pending,
        imr,
        isr,
        next_mti,
        next_sti,
        source=None,
        in_interrupt=false,
        interrupt_stack=None,
        next_interrupt_id=0
    ))]
    fn set_interrupt_state(
        &self,
        py: Python<'_>,
        pending: bool,
        imr: u8,
        isr: u8,
        next_mti: i32,
        next_sti: i32,
        source: Option<&str>,
        in_interrupt: bool,
        interrupt_stack: Option<Vec<u32>>,
        next_interrupt_id: u32,
    ) -> PyResult<()> {
        let stack = interrupt_stack.unwrap_or_default();
        self.inner.bind(py).call_method1(
            "set_interrupt_state",
            (
                pending,
                imr,
                isr,
                next_mti,
                next_sti,
                source,
                in_interrupt,
                stack,
                next_interrupt_id,
            ),
        )?;
        Ok(())
    }

    fn set_runtime_profile_enabled(&self, py: Python<'_>, enabled: bool) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method1("set_runtime_profile_enabled", (enabled,))?;
        Ok(())
    }

    fn reset_runtime_profile_stats(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .bind(py)
            .call_method0("reset_runtime_profile_stats")?;
        Ok(())
    }

    fn get_runtime_profile_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .bind(py)
            .call_method0("get_runtime_profile_stats")
            .map(|obj| obj.into())
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
    py: Python<'_>,
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

    let mut bus = PyBus::new(py_bus, py);
    eval::step(&mut state, &mut bus, &instr, &binder, pre)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&state)
        .map_err(|e| PyRuntimeError::new_err(format!("state serialize: {e}")))
}

#[pyfunction]
fn execute_bound_repr(
    py: Python<'_>,
    state_json: &str,
    bound_json: &str,
    py_bus: PyObject,
) -> PyResult<String> {
    let mut state: RsState = serde_json::from_str(state_json)
        .map_err(|e| PyValueError::new_err(format!("state json: {e}")))?;
    let bound: BoundInstrRepr = serde_json::from_str(bound_json)
        .map_err(|e| PyValueError::new_err(format!("bound json: {e}")))?;
    let mut bus = PyBus::new(py_bus, py);
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
