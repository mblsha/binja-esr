// PY_SOURCE: sc62015/pysc62015/cpu.py:CPU
// PY_SOURCE: sc62015/pysc62015/emulator.py:Emulator
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyBytes, PyDict, PyModule, PyTuple};
use pyo3::Bound;
use retrobus_perfetto::AnnotationValue;
use sc62015_core::{
    keyboard::KeyboardMatrix,
    llama::{
        eval::{perfetto_last_pc, power_on_reset, reset_perf_counters, LlamaBus, LlamaExecutor},
        opcodes::RegName as LlamaRegName,
        state::LlamaState,
    },
    memory::MemoryImage,
    snapshot::save_snapshot,
    timer::TimerContext,
    PerfettoTracer, SnapshotMetadata, ADDRESS_MASK, EXTERNAL_SPACE, INTERNAL_MEMORY_START,
    PERFETTO_TRACER,
};
use serde_json::{json, to_value, Value as JsonValue};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

const IMEM_KOL_OFFSET: u32 = 0xF0;
const IMEM_KOH_OFFSET: u32 = 0xF1;
const IMEM_KIL_OFFSET: u32 = 0xF2;
const IMEM_IMR_OFFSET: u32 = 0xFB;
const IMEM_ISR_OFFSET: u32 = 0xFC;
const IMEM_SCR_OFFSET: u32 = 0xFD;
const IMEM_SSR_OFFSET: u32 = 0xFF;

fn llama_reg_from_name(name: &str) -> Option<LlamaRegName> {
    match name.to_ascii_uppercase().as_str() {
        "A" => Some(LlamaRegName::A),
        "B" => Some(LlamaRegName::B),
        "BA" => Some(LlamaRegName::BA),
        "IL" => Some(LlamaRegName::IL),
        "IH" => Some(LlamaRegName::IH),
        "I" => Some(LlamaRegName::I),
        "X" => Some(LlamaRegName::X),
        "Y" => Some(LlamaRegName::Y),
        "U" => Some(LlamaRegName::U),
        "S" => Some(LlamaRegName::S),
        "PC" => Some(LlamaRegName::PC),
        "F" => Some(LlamaRegName::F),
        "FC" => Some(LlamaRegName::FC),
        "FZ" => Some(LlamaRegName::FZ),
        "IMR" => Some(LlamaRegName::IMR),
        _ => None,
    }
}

fn llama_flag_from_name(name: &str) -> Option<LlamaRegName> {
    match name.to_ascii_uppercase().as_str() {
        "C" | "FC" => Some(LlamaRegName::FC),
        "Z" | "FZ" => Some(LlamaRegName::FZ),
        _ => None,
    }
}

#[derive(Clone)]
struct ContractEvent {
    kind: &'static str,
    address: u32,
    value: u8,
    pc: Option<u32>,
    detail: Option<u8>,
}

#[derive(Clone)]
struct LcdShadow {
    page: u8,
    y: u8,
    on: bool,
    start_line: u8,
    vram: [[u8; 64]; 8],
}

impl LcdShadow {
    fn new() -> Self {
        Self {
            page: 0,
            y: 0,
            on: true,
            start_line: 0,
            vram: [[0; 64]; 8],
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.page = 0;
        self.y = 0;
        self.on = true;
        self.start_line = 0;
        self.vram = [[0; 64]; 8];
    }

    fn apply_instruction(&mut self, instr: u8, data: u8) {
        match instr {
            0b00 => {
                // On/Off
                self.on = (data & 1) != 0;
            }
            0b01 => {
                // Set Y address
                self.y = data & 0b0011_1111;
            }
            0b10 => {
                // Set Page
                self.page = data & 0b0000_0111;
            }
            0b11 => {
                // Start line (ignore)
                self.start_line = data & 0b0011_1111;
            }
            _ => {}
        }
    }

    fn apply_data(&mut self, value: u8) {
        let page = (self.page as usize) % 8;
        let y = (self.y as usize) % 64;
        self.vram[page][y] = value;
        self.y = ((self.y as usize + 1) % 64) as u8;
    }

    fn flatten(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 64);
        for page in 0..8 {
            out.extend_from_slice(&self.vram[page]);
        }
        out
    }
}

fn capture_lcd_snapshot(
    py: Python<'_>,
    memory: &Bound<PyAny>,
) -> PyResult<Option<(JsonValue, Vec<u8>)>> {
    let controller = match memory.getattr("_lcd_controller") {
        Ok(ctrl) => ctrl,
        Err(_) => return Ok(None),
    };
    if controller.is_none() {
        return Ok(None);
    }
    let snapshot = match controller.call_method0("get_snapshot") {
        Ok(snap) => snap,
        Err(_) => return Ok(None),
    };
    let chips: Vec<PyObject> = snapshot
        .getattr("chips")?
        .extract()
        .map_err(|e| PyRuntimeError::new_err(format!("lcd snapshot chips: {e}")))?;
    if chips.is_empty() {
        return Ok(None);
    }

    let mut meta_chips = Vec::with_capacity(chips.len());
    let mut payload: Vec<u8> = Vec::new();
    let mut pages = 0usize;
    let mut width = 0usize;
    for chip_obj in &chips {
        let chip = chip_obj.bind(py);
        let on: bool = chip.getattr("on")?.extract()?;
        let start_line: u8 = chip.getattr("start_line")?.extract()?;
        let page: u8 = chip.getattr("page")?.extract()?;
        let y_address: u8 = chip.getattr("y_address")?.extract()?;
        let vram: Vec<Vec<u8>> = chip.getattr("vram")?.extract()?;
        pages = vram.len();
        width = vram.first().map(|row| row.len()).unwrap_or(0);
        for page_rows in &vram {
            for byte in page_rows {
                payload.push(*byte);
            }
        }
        let instr_count: u32 = chip.getattr("instruction_count")?.extract()?;
        let data_write_count: u32 = chip.getattr("data_write_count")?.extract()?;
        meta_chips.push(json!({
            "on": on,
            "start_line": start_line,
            "page": page,
            "y_address": y_address,
            "instruction_count": instr_count,
            "data_write_count": data_write_count,
            "data_read_count": 0,
        }));
    }
    let cs_both = controller
        .getattr("cs_both_count")
        .and_then(|v| v.extract::<u32>())
        .unwrap_or(0);
    let cs_left = controller
        .getattr("cs_left_count")
        .and_then(|v| v.extract::<u32>())
        .unwrap_or(0);
    let cs_right = controller
        .getattr("cs_right_count")
        .and_then(|v| v.extract::<u32>())
        .unwrap_or(0);
    let meta = json!({
        "chip_count": chips.len(),
        "pages": pages,
        "width": width,
        "chips": meta_chips,
        "cs_both_count": cs_both,
        "cs_left_count": cs_left,
        "cs_right_count": cs_right,
    });
    Ok(Some((meta, payload)))
}

fn event_to_dict(py: Python<'_>, evt: &ContractEvent) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("kind", evt.kind)?;
    dict.set_item("address", evt.address)?;
    dict.set_item("value", evt.value)?;
    if let Some(detail) = evt.detail {
        dict.set_item("detail", detail)?;
    }
    if let Some(pc) = evt.pc {
        dict.set_item("pc", pc)?;
    } else {
        dict.set_item("pc", py.None())?;
    }
    Ok(dict.into_py(py))
}

fn is_lcd_addr(addr: u32) -> bool {
    (0x2000..=0x200F).contains(&(addr & ADDRESS_MASK))
        || (0xA000..=0xAFFF).contains(&(addr & ADDRESS_MASK))
}

#[pyclass(unsendable, name = "LlamaContractBus")]
struct LlamaContractBus {
    memory: MemoryImage,
    events: Vec<ContractEvent>,
    timer: TimerContext,
    cycles: u64,
    host_memory: Option<Py<PyAny>>,
    last_lcd_status: Option<u8>,
    lcd_log: Vec<(u32, u8)>,
    lcd_shadow: [LcdShadow; 2],
    keyboard: KeyboardMatrix,
}

#[pymethods]
impl LlamaContractBus {
    #[new]
    fn new() -> Self {
        let mut bus = Self {
            memory: MemoryImage::new(),
            events: Vec::new(),
            timer: TimerContext::new(true, 0, 0),
            cycles: 0,
            host_memory: None,
            last_lcd_status: None,
            lcd_log: Vec::new(),
            lcd_shadow: [LcdShadow::new(), LcdShadow::new()],
            keyboard: KeyboardMatrix::new(),
        };
        bus.keyboard.reset(&mut bus.memory);
        bus
    }

    fn load_external(&mut self, blob: &[u8]) {
        self.memory.load_external(blob);
        self.events.clear();
    }

    fn load_internal(&mut self, blob: &[u8]) {
        self.memory.load_internal(blob);
        self.events.clear();
    }

    fn set_python_ranges(&mut self, ranges: Vec<(u32, u32)>) {
        self.memory.set_python_ranges(ranges);
    }

    fn set_readonly_ranges(&mut self, ranges: Vec<(u32, u32)>) {
        self.memory.set_readonly_ranges(ranges);
    }

    fn set_keyboard_bridge(&mut self, enabled: bool) {
        self.memory.set_keyboard_bridge(enabled);
    }

    /// Optional host memory hook for addresses that require Python overlays (e.g., ON/ONK).
    fn set_host_memory(&mut self, memory: Py<PyAny>) {
        self.host_memory = Some(memory);
    }

    fn requires_python(&self, address: u32) -> bool {
        self.memory.requires_python(address)
    }

    #[pyo3(signature = (mti_period, sti_period, *, enabled = true))]
    fn configure_timer(&mut self, mti_period: i32, sti_period: i32, enabled: bool) {
        self.timer.enabled = enabled;
        self.timer.mti_period = mti_period.max(0) as u64;
        self.timer.sti_period = sti_period.max(0) as u64;
        self.timer.reset(self.cycles);
    }

    #[pyo3(signature = (steps = 1))]
    fn tick_timers(&mut self, steps: u32) {
        for _ in 0..steps {
            self.cycles = self.cycles.wrapping_add(1);
            // Keep timer mirrors in sync with current memory before ticking.
            if let Some(imr) = self.memory.read_internal_byte(0xFB) {
                self.timer.irq_imr = imr;
            }
            if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                self.timer.irq_isr = isr;
            }

            let kb_irq_enabled = self.timer.kb_irq_enabled;
            let (mti, sti, key_events, _kb_stats) = self.timer.tick_timers_with_keyboard(
                &mut self.memory,
                self.cycles,
                |mem| {
                    let events = self.keyboard.scan_tick(kb_irq_enabled);
                    if events > 0 {
                        self.keyboard.write_fifo_to_memory(mem, kb_irq_enabled);
                    }
                    (
                        events,
                        self.keyboard.fifo_len() > 0,
                        Some(self.keyboard.telemetry()),
                    )
                },
                None,
                None,
            );
            if kb_irq_enabled {
                if mti && key_events > 0 && self.keyboard.fifo_len() > 0 {
                    // Ensure KEYI is asserted; tick_timers_with_keyboard already wrote ISR, but keep parity.
                    if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                        if (isr & 0x04) == 0 {
                            self.memory.write_internal_byte(0xFC, isr | 0x04);
                        }
                    }
                }
                if sti && self.keyboard.fifo_len() > 0 {
                    if let Some(cur) = self.memory.read_internal_byte(0xFC) {
                        if (cur & 0x04) == 0 {
                            self.memory.write_internal_byte(0xFC, cur | 0x04);
                        }
                    }
                }
            }
            if mti || sti {
                let mut value = 0u8;
                if mti {
                    value |= 0x01;
                }
                if sti {
                    value |= 0x02;
                }
                self.events.push(ContractEvent {
                    kind: "timer",
                    address: INTERNAL_MEMORY_START + 0xFC,
                    value,
                    pc: None,
                    detail: None,
                });
            }
            if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                self.timer.irq_isr = isr;
            }
        }
    }

    #[pyo3(signature = (address, pc=None))]
    fn read_byte(&mut self, address: u32, pc: Option<u32>) -> PyResult<u8> {
        let addr = address & ADDRESS_MASK;
        // Defer to host memory when Python overlays are required.
        if self.memory.requires_python(addr) {
            if let Some(host) = &self.host_memory {
                return Python::with_gil(|py| {
                    let bound = host.bind(py);
                    let value = bound
                        .call_method1("read_byte", (addr, pc))
                        .or_else(|err| {
                            if err.is_instance_of::<PyTypeError>(py) {
                                bound.call_method1("read_byte", (addr,))
                            } else {
                                Err(err)
                            }
                        })
                        .and_then(|val| val.extract::<u8>())?;
                    // Parity: still bump counters and record a read event even when the host services it.
                    self.memory.bump_read_count();
                    self.events.push(ContractEvent {
                        kind: "read",
                        address: addr,
                        value,
                        pc: pc.map(|v| v & ADDRESS_MASK),
                        detail: None,
                    });
                    return Ok(value);
                });
            }
        }
        if addr >= INTERNAL_MEMORY_START {
            if let Some(offset) = addr.checked_sub(INTERNAL_MEMORY_START) {
                if let Some(value) = self.keyboard.handle_read(offset, &mut self.memory) {
                    self.events.push(ContractEvent {
                        kind: "read",
                        address: addr,
                        value,
                        pc: pc.map(|v| v & ADDRESS_MASK),
                        detail: None,
                    });
                    return Ok(value);
                }
            }
        }
        let value = self.memory.read_byte(addr).unwrap_or(0);
        if is_lcd_addr(address) && (address & 0x3) == 0x1 {
            self.last_lcd_status = Some(value);
        }
        self.events.push(ContractEvent {
            kind: "read",
            address: addr,
            value,
            pc: pc.map(|v| v & ADDRESS_MASK),
            detail: None,
        });
        Ok(value)
    }

    #[pyo3(signature = (address, value, pc=None))]
    fn write_byte(&mut self, address: u32, value: u8, pc: Option<u32>) -> PyResult<()> {
        let addr = address & ADDRESS_MASK;
        if self.memory.requires_python(addr) {
                if let Some(host) = &self.host_memory {
                    Python::with_gil(|py| {
                        let bound = host.bind(py);
                        bound
                            .call_method1("write_byte", (addr, value, pc))
                            .or_else(|err| {
                            if err.is_instance_of::<PyTypeError>(py) {
                                bound.call_method1("write_byte", (addr, value))
                            } else {
                                Err(err)
                            }
                        })
                        .map(|_| ())
                    })?;
                    // Parity: for host-handled IMEM writes, avoid mirroring keyboard/E-port overlays
                    // into internal memory; Python overlays do not mutate IMEM for KIO/ONK.
                    let should_mirror = if MemoryImage::is_internal(addr) {
                        let offset = addr - INTERNAL_MEMORY_START;
                        !MemoryImage::is_keyboard_offset(offset) && offset != 0xF5 && offset != 0xF6
                    } else {
                        true
                    };
                    if should_mirror {
                        self.memory
                            .apply_host_write_with_cycle(addr, value, None, pc.map(|v| v & ADDRESS_MASK));
                    }
                    self.events.push(ContractEvent {
                        kind: "write",
                        address: addr,
                        value,
                    pc: pc.map(|v| v & ADDRESS_MASK),
                    detail: None,
                });
                return Ok(());
            }
        }
        if addr >= INTERNAL_MEMORY_START {
            if let Some(offset) = addr.checked_sub(INTERNAL_MEMORY_START) {
                if self.keyboard.handle_write(offset, value, &mut self.memory) {
                    self.events.push(ContractEvent {
                        kind: "write",
                        address: addr,
                        value,
                        pc: pc.map(|v| v & ADDRESS_MASK),
                        detail: None,
                    });
                    return Ok(());
                }
            }
        }
        let _ = self.memory.store(addr, 8, value as u32);
        if is_lcd_addr(addr) {
            self.lcd_log.push((address & ADDRESS_MASK, value));
            let addr_lo = address & 0x0FFF;
            let rw = addr_lo & 1;
            if rw == 0 {
                let di = (addr_lo >> 1) & 1;
                let cs_bits = (addr_lo >> 2) & 0b11;
                let targets: &[usize] = match cs_bits {
                    0b00 => &[0, 1],
                    0b01 => &[1],
                    0b10 => &[0],
                    _ => &[],
                };
                if di == 0 {
                    let instr = value >> 6;
                    let data = value & 0b0011_1111;
                    for idx in targets {
                        if let Some(shadow) = self.lcd_shadow.get_mut(*idx) {
                            shadow.apply_instruction(instr, data);
                        }
                    }
                } else {
                    for idx in targets {
                        if let Some(shadow) = self.lcd_shadow.get_mut(*idx) {
                            shadow.apply_data(value);
                        }
                    }
                }
            }
        }
        self.events.push(ContractEvent {
            kind: "write",
            address: address & ADDRESS_MASK,
            value,
            pc: pc.map(|v| v & ADDRESS_MASK),
            detail: None,
        });
        Ok(())
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "internal",
            PyBytes::new_bound(py, self.memory.internal_slice()),
        )?;
        dict.set_item(
            "external",
            PyBytes::new_bound(py, self.memory.external_slice()),
        )?;
        dict.set_item("external_len", self.memory.external_len())?;
        // Surface IMR/ISR for contract assertions.
        let internal = self.memory.internal_slice();
        let imr = *internal.get(0xFB).unwrap_or(&0);
        let isr = *internal.get(0xFC).unwrap_or(&0);
        dict.set_item("imr", imr)?;
        dict.set_item("isr", isr)?;
        dict.set_item("irq_pending", self.timer.irq_pending)?;
        if let Some(src) = self.timer.irq_source.as_deref() {
            dict.set_item("irq_source", src)?;
        } else {
            dict.set_item("irq_source", py.None())?;
        }
        // Capture LCD-facing events without draining the event log.
        let mut seq: Vec<PyObject> = Vec::new();
        for evt in self.events.iter().filter(|e| is_lcd_addr(e.address)) {
            let e = event_to_dict(py, evt)?;
            seq.push(e);
        }
        dict.set_item("lcd_events", seq)?;
        if let Some(status) = self.last_lcd_status {
            dict.set_item("lcd_status", status)?;
        }
        let lcd_log: Vec<PyObject> = self
            .lcd_log
            .iter()
            .map(|(addr, val)| {
                let entry = PyDict::new_bound(py);
                let _ = entry.set_item("address", *addr);
                let _ = entry.set_item("value", *val);
                entry.into_py(py)
            })
            .collect();
        dict.set_item("lcd_log", lcd_log)?;
        // Expose a simple VRAM snapshot derived from shadowed writes.
        let mut merged = Vec::new();
        merged.extend_from_slice(&self.lcd_shadow[0].flatten());
        merged.extend_from_slice(&self.lcd_shadow[1].flatten());
        dict.set_item("lcd_vram", PyBytes::new_bound(py, &merged))?;
        dict.set_item("lcd_meta", "chips=2,pages=8,width=64")?;
        Ok(dict.into_py(py))
    }

    fn press_on_key(&mut self) {
        let ssr_offset = IMEM_SSR_OFFSET;
        let isr_offset = IMEM_ISR_OFFSET;
        let ssr_addr = INTERNAL_MEMORY_START + ssr_offset;
        let ssr = self.memory.read_internal_byte(ssr_offset).unwrap_or(0);
        self.events.push(ContractEvent {
            kind: "read",
            address: ssr_addr,
            value: ssr,
            pc: None,
            detail: None,
        });
        self.memory.write_internal_byte(ssr_offset, ssr | 0x08);
        self.events.push(ContractEvent {
            kind: "write",
            address: ssr_addr,
            value: ssr | 0x08,
            pc: None,
            detail: None,
        });
        let isr_addr = INTERNAL_MEMORY_START + isr_offset;
        let isr = self.memory.read_internal_byte(isr_offset).unwrap_or(0);
        let new_isr = isr | 0x08;
        self.events.push(ContractEvent {
            kind: "read",
            address: isr_addr,
            value: isr,
            pc: None,
            detail: None,
        });
        self.memory.write_internal_byte(isr_offset, new_isr);
        self.events.push(ContractEvent {
            kind: "write",
            address: isr_addr,
            value: new_isr,
            pc: None,
            detail: None,
        });
        self.timer
            .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
        self.timer.irq_pending = true;
        self.timer.irq_source = Some("ONK".to_string());
        self.timer.last_fired = self.timer.irq_source.clone();
        self.timer.irq_isr = new_isr;
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            let mut payload = HashMap::new();
            payload.insert(
                "pc".to_string(),
                AnnotationValue::Pointer(perfetto_last_pc() as u64),
            );
            payload.insert(
                "imr".to_string(),
                AnnotationValue::UInt(self.timer.irq_imr as u64),
            );
            payload.insert(
                "isr".to_string(),
                AnnotationValue::UInt(self.timer.irq_isr as u64),
            );
            payload.insert(
                "src".to_string(),
                AnnotationValue::Str("ONK".to_string()),
            );
            tracer.record_irq_event("KeyIRQ", payload);
        }
    }

    fn release_on_key(&mut self) {
        let ssr_offset = IMEM_SSR_OFFSET;
        let isr_offset = IMEM_ISR_OFFSET;
        let ssr_addr = INTERNAL_MEMORY_START + ssr_offset;
        let ssr = self.memory.read_internal_byte(ssr_offset).unwrap_or(0);
        self.events.push(ContractEvent {
            kind: "read",
            address: ssr_addr,
            value: ssr,
            pc: None,
            detail: None,
        });
        self.memory.write_internal_byte(ssr_offset, ssr & !0x08);
        self.events.push(ContractEvent {
            kind: "write",
            address: ssr_addr,
            value: ssr & !0x08,
            pc: None,
            detail: None,
        });
        let isr_addr = INTERNAL_MEMORY_START + isr_offset;
        let isr = self.memory.read_internal_byte(isr_offset).unwrap_or(0);
        let new_isr = isr & !0x08;
        self.events.push(ContractEvent {
            kind: "read",
            address: isr_addr,
            value: isr,
            pc: None,
            detail: None,
        });
        self.memory.write_internal_byte(isr_offset, new_isr);
        self.events.push(ContractEvent {
            kind: "write",
            address: isr_addr,
            value: new_isr,
            pc: None,
            detail: None,
        });
        self.timer
            .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
        self.timer.irq_isr = new_isr;
    }

    fn drain_events<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let mut drained = Vec::with_capacity(self.events.len());
        for evt in self.events.drain(..) {
            drained.push(event_to_dict(py, &evt)?);
        }
        Ok(drained)
    }
}

struct LlamaPyBus {
    memory: Py<PyAny>,
    pc: u32,
    lcd_hook: Option<Py<PyAny>>,
    memory_reads: u64,
    memory_writes: u64,
    has_wait_cycles: bool,
    timer: *mut TimerContext,
    keyboard: *mut KeyboardMatrix,
    mirror: *mut MemoryImage,
    cycles_ptr: *mut u64,
}

impl LlamaPyBus {
    fn new(
        py: Python<'_>,
        memory: &Py<PyAny>,
        pc: u32,
        has_wait_cycles: bool,
        timer: *mut TimerContext,
        keyboard: *mut KeyboardMatrix,
        mirror: *mut MemoryImage,
        cycles_ptr: *mut u64,
    ) -> Self {
        // Optional LCD hook used when Python overlays are disabled (pure LLAMA LCD path).
        let lcd_hook = memory
            .getattr(py, "_llama_lcd_write")
            .ok()
            .and_then(|obj| obj.extract::<Py<PyAny>>(py).ok());
        Self {
            memory: memory.clone_ref(py),
            pc,
            lcd_hook,
            memory_reads: 0,
            memory_writes: 0,
            has_wait_cycles,
            timer,
            keyboard,
            mirror,
            cycles_ptr,
        }
    }

    fn is_lcd_addr(addr: u32) -> bool {
        (0x2000..=0x200F).contains(&(addr & ADDRESS_MASK))
            || (0xA000..=0xAFFF).contains(&(addr & ADDRESS_MASK))
    }

    fn read_byte(&mut self, addr: u32) -> u8 {
        Python::with_gil(|py| self.read_byte_with_gil(py, addr))
    }

    fn read_byte_with_gil(&mut self, py: Python<'_>, addr: u32) -> u8 {
        let bound = self.memory.bind(py);
        let addr = addr & ADDRESS_MASK;
        let value = match bound.call_method1("read_byte", (addr, self.pc)) {
            Ok(obj) => obj.extract::<u8>().unwrap_or(0),
            Err(err) => {
                if err.is_instance_of::<PyTypeError>(py) {
                    bound
                        .call_method1("read_byte", (addr,))
                        .and_then(|obj| obj.extract::<u8>())
                        .unwrap_or(0)
                } else {
                    0
                }
            }
        };
        // Count one logical read per byte.
        self.memory_reads += 1;
        value
    }

    fn write_byte(&mut self, addr: u32, value: u8) {
        Python::with_gil(|py| self.write_byte_with_gil(py, addr, value));
    }

    fn write_byte_with_gil(&mut self, py: Python<'_>, addr: u32, value: u8) {
        let bound = self.memory.bind(py);
        let addr = addr & ADDRESS_MASK;
        let _ = match bound.call_method1("write_byte", (addr, value, self.pc)) {
            Ok(_) => Ok(()),
            Err(err) => {
                if err.is_instance_of::<PyTypeError>(py) {
                    bound.call_method1("write_byte", (addr, value)).map(|_| ())
                } else {
                    Ok(())
                }
            }
        };
        // Count one logical write per byte.
        self.memory_writes += 1;
        if Self::is_lcd_addr(addr) {
            if let Some(hook) = &self.lcd_hook {
                let _ = hook.call1(py, (addr, value, self.pc));
            }
        }
    }
}

impl LlamaBus for LlamaPyBus {
    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        // Respect the requested width so multi-byte loads match the Python emulator.
        let bytes = bits.div_ceil(8).max(1);
        let addr = addr & ADDRESS_MASK;
        let mut value = 0u32;
        for i in 0..bytes {
            let absolute = addr.wrapping_add(i as u32) & ADDRESS_MASK;
            let byte = self.read_byte(absolute) as u32;
            value |= byte << (8 * i);
            if absolute >= INTERNAL_MEMORY_START {
                let offset = absolute - INTERNAL_MEMORY_START;
                let mut tracer_ok = false;
                if matches!(offset, IMEM_KIL_OFFSET | IMEM_KOL_OFFSET | IMEM_KOH_OFFSET) {
                    let mut guard = PERFETTO_TRACER.enter();
                    if let Some(tracer) = guard.as_mut() {
                        tracer.record_kio_read(Some(self.pc), offset as u8, byte as u8, None);
                        tracer_ok = true;
                    }
                    // Mirror into Python's dispatcher so the main Perfetto trace sees KIO reads.
                    Python::with_gil(|py| {
                        let _ = self
                            .memory
                            .bind(py)
                            .call_method1("trace_kio_from_rust", (offset, byte, self.pc));
                    });
                    eprintln!(
                        "[kio-read-pybus] pc=0x{pc:06X} offset=0x{offset:02X} value=0x{val:02X} tracer={tracer}",
                        pc = self.pc,
                        offset = offset,
                        val = byte,
                        tracer = if tracer_ok { "Y" } else { "N" }
                    );
                } else if should_trace_addr(absolute) && trace_loads() {
                    eprintln!(
                        "[pybus-load] pc=0x{pc:06X} addr=0x{addr:06X} bits={bits} byte=0x{val:02X}",
                        pc = self.pc,
                        addr = absolute,
                        bits = bits,
                        val = byte
                    );
                }
            }
        }
        if bits == 0 || bits >= 32 {
            value
        } else {
            value & ((1u32 << bits) - 1)
        }
    }

    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        // Try to read IMEM without emitting Python-side tracer callbacks.
        Python::with_gil(|py| {
            let bound = self.memory.bind(py);
            // Prefer an internal silent helper if available.
            if let Ok(obj) = bound.getattr("internal_memory") {
                if let Ok(method) = obj.getattr("read_byte") {
                    if let Ok(val) = method.call1((INTERNAL_MEMORY_START + offset,)) {
                        if let Ok(byte) = val.extract::<u8>() {
                            return byte;
                        }
                    }
                }
            }
            // Fallback to the standard read_byte path.
            bound
                .call_method1("read_byte", (INTERNAL_MEMORY_START + offset,))
                .ok()
                .and_then(|v| v.extract::<u8>().ok())
                .unwrap_or(0)
        })
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        // Mirror KIO writes into the Python tracer so overlays see LLAMA traffic.
        let absolute = addr & ADDRESS_MASK;
        if absolute >= INTERNAL_MEMORY_START {
            let offset = absolute - INTERNAL_MEMORY_START;
            if matches!(offset, IMEM_KIL_OFFSET | IMEM_KOL_OFFSET | IMEM_KOH_OFFSET) {
                Python::with_gil(|py| {
                    let _ = self
                        .memory
                        .bind(py)
                        .call_method1("trace_kio_from_rust", (offset, value & 0xFF, self.pc));
                });
                let mut guard = PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_kio_read(Some(self.pc), offset as u8, value as u8, None);
                }
            } else if matches!(offset, IMEM_IMR_OFFSET | IMEM_ISR_OFFSET | IMEM_SCR_OFFSET) {
                // Mirror IRQ register writes into Python tracer for parity runs.
                Python::with_gil(|py| {
                    let payload = PyDict::new_bound(py);
                    let _ = payload.set_item("offset", offset & 0xFF);
                    let _ = payload.set_item("value", value & 0xFF);
                    let _ = payload.set_item("pc", self.pc & ADDRESS_MASK);
                    let _ = self.memory.bind(py).call_method1(
                        "trace_irq_from_rust",
                        (
                            match offset {
                                IMEM_IMR_OFFSET => "IMR_Write",
                                IMEM_ISR_OFFSET => "ISR_Write",
                                IMEM_SCR_OFFSET => "SCR_Write",
                                _ => "IRQ_Write",
                            },
                            payload,
                        ),
                    );
                });
            }
        }
        match bits {
            0 | 8 => self.write_byte(addr, value as u8),
            16 => {
                let low = addr & ADDRESS_MASK;
                let high = addr.wrapping_add(1) & ADDRESS_MASK;
                if should_trace_addr(low) {
                    eprintln!(
                        "[pybus-store] pc=0x{pc:06X} addr=0x{addr:06X} bits={bits} byte=0x{val:02X}",
                        pc = self.pc,
                        addr = low,
                        bits = bits,
                        val = value & 0xFF
                    );
                }
                if should_trace_addr(high) {
                    eprintln!(
                        "[pybus-store] pc=0x{pc:06X} addr=0x{addr:06X} bits={bits} byte=0x{val:02X}",
                        pc = self.pc,
                        addr = high,
                        bits = bits,
                        val = (value >> 8) & 0xFF
                    );
                }
                self.write_byte(low, (value & 0xFF) as u8);
                self.write_byte(high, ((value >> 8) & 0xFF) as u8);
            }
            24 => {
                let b0 = addr & ADDRESS_MASK;
                let b1 = addr.wrapping_add(1) & ADDRESS_MASK;
                let b2 = addr.wrapping_add(2) & ADDRESS_MASK;
                for (byte_addr, shift) in [(b0, 0), (b1, 8), (b2, 16)] {
                    if should_trace_addr(byte_addr) {
                        eprintln!(
                            "[pybus-store] pc=0x{pc:06X} addr=0x{addr:06X} bits={bits} byte=0x{val:02X}",
                            pc = self.pc,
                            addr = byte_addr,
                            bits = bits,
                            val = (value >> shift) & 0xFF
                        );
                    }
                    self.write_byte(byte_addr, ((value >> shift) & 0xFF) as u8);
                }
            }
            _ => {
                let bytes = bits.div_ceil(8);
                for i in 0..bytes {
                    let byte = ((value >> (8 * i)) & 0xFF) as u8;
                    let absolute = addr.wrapping_add(i as u32) & ADDRESS_MASK;
                    if should_trace_addr(absolute) {
                        eprintln!(
                            "[pybus-store] pc=0x{pc:06X} addr=0x{addr:06X} bits={bits} byte=0x{val:02X}",
                            pc = self.pc,
                            addr = absolute,
                            bits = bits,
                            val = byte
                        );
                    }
                    self.write_byte(absolute, byte);
                }
            }
        }
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }

    fn wait_cycles(&mut self, cycles: u32) {
        // Prefer the Python host hook; otherwise, tick the Rust timer/keyboard locally for parity.
        if self.has_wait_cycles {
            Python::with_gil(|py| {
                let bound = self.memory.bind(py);
                let _ = bound.call_method1("wait_cycles", (cycles.max(1),));
            });
            return;
        }

        let ticks = cycles.max(1);
        unsafe {
            if self.timer.is_null()
                || self.keyboard.is_null()
                || self.mirror.is_null()
                || self.cycles_ptr.is_null()
            {
                return;
            }
            let timer = &mut *self.timer;
            let keyboard = &mut *self.keyboard;
            let mirror = &mut *self.mirror;
            let cycles_counter = &mut *self.cycles_ptr;

            // Keep IMR/ISR mirrors up to date before ticking.
            Python::with_gil(|py| {
                let bound = self.memory.bind(py);
                for offset in [IMEM_IMR_OFFSET, IMEM_ISR_OFFSET] {
                    if let Ok(val) = bound
                        .call_method1("read_byte", (INTERNAL_MEMORY_START + offset,))
                        .and_then(|obj| obj.extract::<u8>())
                    {
                        let _ = mirror.store(INTERNAL_MEMORY_START + offset, 8, val as u32);
                    }
                }
            });

            for _ in 0..ticks {
                *cycles_counter = cycles_counter.wrapping_add(1);
                if let Some(imr) = mirror.read_internal_byte(IMEM_IMR_OFFSET) {
                    timer.irq_imr = imr;
                }
                if let Some(isr) = mirror.read_internal_byte(IMEM_ISR_OFFSET) {
                    timer.irq_isr = isr;
                }

                let kb_irq_enabled = timer.kb_irq_enabled;
                let (mti, sti, key_events, _kb_stats) = timer.tick_timers_with_keyboard(
                    mirror,
                    *cycles_counter,
                    |mem| {
                        let events = keyboard.scan_tick(kb_irq_enabled);
                        if events > 0 {
                            keyboard.write_fifo_to_memory(mem, kb_irq_enabled);
                        }
                        (
                            events,
                            keyboard.fifo_len() > 0,
                            Some(keyboard.telemetry()),
                        )
                    },
                    None,
                    None,
                );
                if kb_irq_enabled && (key_events > 0 || keyboard.fifo_len() > 0) {
                    if let Some(cur) = mirror.read_internal_byte(IMEM_ISR_OFFSET) {
                        if (cur & 0x04) == 0 {
                            mirror.write_internal_byte(IMEM_ISR_OFFSET, cur | 0x04);
                        }
                    }
                }
                if mti || sti {
                    let mut value = mirror.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                    if mti {
                        value |= 0x01;
                    }
                    if sti {
                        value |= 0x02;
                    }
                    mirror.write_internal_byte(IMEM_ISR_OFFSET, value);
                }
            }

            // Flush mirror deltas back to the Python memory so traces/state stay aligned.
            Python::with_gil(|py| {
                let bound = self.memory.bind(py);
                for (addr, value) in mirror.drain_dirty_internal() {
                    let _ = bound.call_method1("write_byte", (addr, value));
                }
                for (addr, value) in mirror.drain_dirty() {
                    let _ = bound.call_method1("write_byte", (addr, value));
                }
            });
        }
    }
}

/// Lazy parse a comma-separated list of absolute addresses in `TRACE_ADDRS`.
fn should_trace_addr(addr: u32) -> bool {
    static WATCH: OnceLock<Vec<u32>> = OnceLock::new();
    let list = WATCH.get_or_init(|| {
        std::env::var("TRACE_ADDRS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|tok| {
                        u32::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    list.contains(&addr)
}

fn trace_loads() -> bool {
    static LOADS: OnceLock<bool> = OnceLock::new();
    *LOADS.get_or_init(|| {
        std::env::var("TRACE_ADDRS_LOAD")
            .ok()
            .is_some_and(|v| v != "0")
    })
}

#[pyclass(unsendable, name = "LlamaCPU")]
struct LlamaCpu {
    state: LlamaState,
    executor: LlamaExecutor,
    memory: Py<PyAny>,
    call_sub_level: u32,
    temps: HashMap<u32, u32>,
    mirror: MemoryImage,
    keyboard: KeyboardMatrix,
    timer: TimerContext,
    memory_synced: bool,
    memory_reads: u64,
    memory_writes: u64,
    cycles: u64,
}

#[pymethods]
impl LlamaCpu {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let mut cpu = Self {
            state: LlamaState::new(),
            executor: LlamaExecutor::new(),
            memory,
            call_sub_level: 0,
            temps: HashMap::new(),
            mirror: MemoryImage::new(),
            keyboard: KeyboardMatrix::new(),
            timer: TimerContext::new(false, 0, 0),
            memory_synced: false,
            memory_reads: 0,
            memory_writes: 0,
            cycles: 0,
        };
        if reset_on_init {
            cpu.power_on_reset()?;
        }
        Ok(cpu)
    }

    fn sync_temps_from_state(&mut self) {
        for idx in 0..14u32 {
            let reg = LlamaRegName::Temp(idx as u8);
            let val = self.state.get_reg(reg) & 0xFF_FFFF;
            self.temps.insert(idx, val);
        }
    }

    fn apply_temps_to_state(&mut self) {
        for (idx, val) in self.temps.clone() {
            let reg = LlamaRegName::Temp(idx as u8);
            self.state.set_reg(reg, val);
        }
    }

    fn power_on_reset(&mut self) -> PyResult<()> {
        // Apply RESET intrinsic semantics using Python memory for reads/writes so the
        // reset vector and IMEM updates match the Python emulator, while keeping the
        // mirror in sync.
        Python::with_gil(|py| {
            struct ResetBus<'py, 'a> {
                py: Python<'py>,
                mem: Py<PyAny>,
                mirror: &'a mut MemoryImage,
            }

            impl<'py, 'a> ResetBus<'py, 'a> {
                fn read_byte(&self, addr: u32) -> u8 {
                    let addr = addr & ADDRESS_MASK;
                    let bound = self.mem.bind(self.py);
                    bound
                        .call_method1("read_byte", (addr,))
                        .and_then(|obj| obj.extract::<u8>())
                        .unwrap_or(0)
                }

                fn write_byte(&mut self, addr: u32, value: u8) {
                    let addr = addr & ADDRESS_MASK;
                    let bound = self.mem.bind(self.py);
                    let _ = bound.call_method1("write_byte", (addr, value));
                    let _ = self.mirror.store(addr, 8, value as u32);
                }
            }

            impl<'py, 'a> LlamaBus for ResetBus<'py, 'a> {
                fn load(&mut self, addr: u32, bits: u8) -> u32 {
                    let bytes = bits.div_ceil(8).max(1);
                    let mut value = 0u32;
                    for i in 0..bytes {
                        let byte = self.read_byte(addr.wrapping_add(i as u32));
                        value |= (byte as u32) << (8 * i);
                    }
                    if bits == 0 || bits >= 32 {
                        value
                    } else {
                        value & ((1u32 << bits) - 1)
                    }
                }

                fn store(&mut self, addr: u32, bits: u8, value: u32) {
                    let bytes = bits.div_ceil(8).max(1);
                    for i in 0..bytes {
                        let byte = ((value >> (8 * i)) & 0xFF) as u8;
                        self.write_byte(addr.wrapping_add(i as u32), byte);
                    }
                }

                fn resolve_emem(&mut self, base: u32) -> u32 {
                    base
                }

                fn peek_imem(&mut self, offset: u32) -> u8 {
                    self.read_byte(INTERNAL_MEMORY_START + offset)
                }

                fn peek_imem_silent(&mut self, offset: u32) -> u8 {
                    self.read_byte(INTERNAL_MEMORY_START + offset)
                }
            }

            let mem = self.memory.clone_ref(py);
            let mut bus = ResetBus { py, mem, mirror: &mut self.mirror };
            power_on_reset(&mut bus, &mut self.state);
            Ok::<(), pyo3::PyErr>(())
        })?;

        self.memory_synced = true;
        reset_perf_counters();
        Python::with_gil(|py| self.sync_mirror(py));
        self.cycles = 0;
        self.timer.reset_full(self.cycles);
        self.timer.irq_imr = self
            .mirror
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        self.timer.irq_isr = self
            .mirror
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        Ok(())
    }

    fn execute_instruction(&mut self, py: Python<'_>, address: u32) -> PyResult<(u8, u8)> {
        self.state.set_pc(address & ADDRESS_MASK);
        let entry_pc = self.state.get_reg(LlamaRegName::PC);
        let has_wait_cycles = {
            let bound = self.memory.bind(py);
            bound.hasattr("wait_cycles")?
        };
        let mut bus = LlamaPyBus::new(
            py,
            &self.memory,
            entry_pc,
            has_wait_cycles,
            &mut self.timer,
            &mut self.keyboard,
            &mut self.mirror,
            &mut self.cycles,
        );
        let opcode = bus.read_byte(entry_pc & ADDRESS_MASK);
        let len = self
            .executor
            .execute(opcode, &mut self.state, &mut bus)
            .map_err(|e| PyRuntimeError::new_err(format!("llama execute: {e}")))?;
        self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
        self.memory_writes = self.memory_writes.saturating_add(bus.memory_writes);
        self.call_sub_level = self.state.call_sub_level();
        self.sync_temps_from_state();
        if opcode == 0xFE {
            // IR: interrupt entry
            let (imr, isr) = self.read_irq_registers(py, &mut bus);
            self.emit_irq_trace(
                py,
                "IRQ_Enter",
                HashMap::from([
                    ("pc", entry_pc & ADDRESS_MASK),
                    ("vector", self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK),
                    ("imr", imr as u32),
                    ("isr", isr as u32),
                ]),
            );
        } else if opcode == 0x01 {
            // RETI: interrupt exit
            let (imr, isr) = self.read_irq_registers(py, &mut bus);
            self.emit_irq_trace(
                py,
                "IRQ_Return",
                HashMap::from([
                    ("pc", entry_pc & ADDRESS_MASK),
                    ("ret", self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK),
                    ("imr", imr as u32),
                    ("isr", isr as u32),
                ]),
            );
        }
        Ok((opcode, len))
    }

    fn read_register(&self, name: &str) -> PyResult<u32> {
        let upper = name.to_ascii_uppercase();
        if let Some(reg) = llama_reg_from_name(&upper) {
            return Ok(self.state.get_reg(reg));
        }
        if let Some(rest) = upper.strip_prefix("TEMP") {
            if let Ok(idx) = rest.parse::<u32>() {
                return Ok(*self.temps.get(&idx).unwrap_or(&0));
            }
        }
        Err(PyValueError::new_err(format!("unknown register {name}")))
    }

    fn write_register(&mut self, name: &str, value: u32) -> PyResult<()> {
        let upper = name.to_ascii_uppercase();
        if let Some(reg) = llama_reg_from_name(&upper) {
            self.state.set_reg(reg, value);
            return Ok(());
        }
        if let Some(rest) = upper.strip_prefix("TEMP") {
            if let Ok(idx) = rest.parse::<u32>() {
                if value != 0 {
                    self.temps.insert(idx, value & ADDRESS_MASK);
                } else {
                    self.temps.remove(&idx);
                }
                return Ok(());
            }
        }
        Err(PyValueError::new_err(format!("unknown register {name}")))
    }

    fn read_flag(&self, name: &str) -> PyResult<u8> {
        let reg = llama_flag_from_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown flag {name}")))?;
        Ok(self.state.get_reg(reg) as u8)
    }

    fn write_flag(&mut self, name: &str, value: u8) -> PyResult<()> {
        let reg = llama_flag_from_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown flag {name}")))?;
        self.state.set_reg(reg, value as u32);
        Ok(())
    }

    fn snapshot_cpu_registers(&self, py: Python<'_>) -> PyResult<PyObject> {
        let module = PyModule::import_bound(py, "sc62015.pysc62015.stepper")
            .map_err(|e| PyRuntimeError::new_err(format!("import stepper: {e}")))?;
        let cls = module.getattr("CPURegistersSnapshot")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("pc", self.state.get_reg(LlamaRegName::PC))?;
        kwargs.set_item("ba", self.state.get_reg(LlamaRegName::BA))?;
        kwargs.set_item("i", self.state.get_reg(LlamaRegName::I))?;
        kwargs.set_item("x", self.state.get_reg(LlamaRegName::X))?;
        kwargs.set_item("y", self.state.get_reg(LlamaRegName::Y))?;
        kwargs.set_item("u", self.state.get_reg(LlamaRegName::U))?;
        kwargs.set_item("s", self.state.get_reg(LlamaRegName::S))?;
        kwargs.set_item("f", self.state.get_reg(LlamaRegName::F))?;
        let temps = PyDict::new_bound(py);
        for idx in 0..14u32 {
            let reg = LlamaRegName::Temp(idx as u8);
            let val = self.state.get_reg(reg) & 0xFF_FFFF;
            temps.set_item(idx, val)?;
        }
        kwargs.set_item("temps", temps)?;
        kwargs.set_item("call_sub_level", self.call_sub_level)?;
        cls.call((), Some(&kwargs)).map(|obj| obj.into())
    }

    fn load_cpu_snapshot(&mut self, py: Python<'_>, snapshot: PyObject) -> PyResult<()> {
        let snap = snapshot.bind(py);
        let fields = [
            ("pc", LlamaRegName::PC),
            ("ba", LlamaRegName::BA),
            ("i", LlamaRegName::I),
            ("x", LlamaRegName::X),
            ("y", LlamaRegName::Y),
            ("u", LlamaRegName::U),
            ("s", LlamaRegName::S),
            ("f", LlamaRegName::F),
        ];
        for (attr, reg) in fields {
            if let Ok(value) = snap.getattr(attr).and_then(|obj| obj.extract::<u32>()) {
                self.state.set_reg(reg, value);
            }
        }
        if let Ok(temps_obj) = snap.getattr("temps") {
            if let Ok(mapping) = temps_obj.extract::<HashMap<u32, u32>>() {
                self.temps = mapping;
                self.apply_temps_to_state();
            }
        }
        if let Ok(call_depth) = snap
            .getattr("call_sub_level")
            .and_then(|obj| obj.extract::<u32>())
        {
            self.call_sub_level = call_depth;
            self.state.set_call_sub_level(call_depth);
        }
        // Restore timer/interrupt mirrors if present to match Python snapshot semantics.
        if let Ok(interrupts) = snap.getattr("interrupts") {
            if let Ok(imr) = interrupts.getattr("imr").and_then(|o| o.extract::<u8>()) {
                self.timer.irq_imr = imr;
                self.mirror.write_internal_byte(IMEM_IMR_OFFSET, imr);
            }
            if let Ok(isr) = interrupts.getattr("isr").and_then(|o| o.extract::<u8>()) {
                self.timer.irq_isr = isr;
                self.mirror.write_internal_byte(IMEM_ISR_OFFSET, isr);
            }
            if let Ok(pending) = interrupts
                .getattr("pending")
                .and_then(|o| o.extract::<bool>())
            {
                self.timer.irq_pending = pending;
            }
            if let Ok(in_irq) = interrupts
                .getattr("in_interrupt")
                .and_then(|o| o.extract::<bool>())
            {
                self.timer.in_interrupt = in_irq;
            }
            if let Ok(src) = interrupts
                .getattr("source")
                .and_then(|o| o.extract::<Option<String>>())
            {
                self.timer.irq_source = src;
            }
            if let Ok(stack) = interrupts
                .getattr("stack")
                .and_then(|o| o.extract::<Vec<u32>>())
            {
                self.timer.interrupt_stack = stack;
            }
            if let Ok(next_id) = interrupts
                .getattr("next_id")
                .and_then(|o| o.extract::<u32>())
            {
                self.timer.next_interrupt_id = next_id;
            }
            if let Ok(counts) = interrupts
                .getattr("irq_counts")
                .and_then(|o| o.extract::<HashMap<String, u32>>())
            {
                self.timer.irq_total = *counts.get("total").unwrap_or(&0);
                self.timer.irq_key = *counts.get("KEY").unwrap_or(&0);
                self.timer.irq_mti = *counts.get("MTI").unwrap_or(&0);
                self.timer.irq_sti = *counts.get("STI").unwrap_or(&0);
            }
            if let Ok(last_irq_obj) = interrupts.getattr("last_irq") {
                if let Ok(dict) = last_irq_obj.downcast::<PyDict>() {
                    self.timer.last_irq_src = dict
                        .get_item("src")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok());
                    self.timer.last_irq_pc = dict
                        .get_item("pc")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<u64>().ok())
                        .map(|v| v as u32);
                    self.timer.last_irq_vector = dict
                        .get_item("vector")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<u64>().ok())
                        .map(|v| v as u32);
                }
            }
        }
        Ok(())
    }

    fn keyboard_press_matrix_code(&mut self, py: Python<'_>, code: u8) -> PyResult<bool> {
        let events = self
            .keyboard
            .inject_matrix_event(
                code & 0x7F,
                false,
                &mut self.mirror,
                self.timer.kb_irq_enabled,
            );
        if self.timer.kb_irq_enabled && (events > 0 || self.keyboard.fifo_len() > 0) {
            // Mirror Python scheduler: latch KEYI and pending IRQ immediately.
            self.timer.irq_pending = true;
            self.timer.irq_source = Some("KEY".to_string());
            self.timer.last_fired = self.timer.irq_source.clone();
            self.timer.irq_isr = self
                .mirror
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(self.timer.irq_isr);
            self.timer.irq_imr = self
                .mirror
                .read_internal_byte(IMEM_IMR_OFFSET)
                .unwrap_or(self.timer.irq_imr);
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let mut payload = HashMap::new();
                payload.insert("imr".to_string(), AnnotationValue::UInt(self.timer.irq_imr as u64));
                payload.insert("isr".to_string(), AnnotationValue::UInt(self.timer.irq_isr as u64));
                payload.insert("src".to_string(), AnnotationValue::Str("KEY".to_string()));
                tracer.record_irq_event("KeyIRQ", payload);
            }
        }
        self.sync_mirror(py);
        Ok(events > 0)
    }

    fn keyboard_press_on_key(&mut self, py: Python<'_>) -> PyResult<bool> {
        // Emulate ON key: set SSR.ONK and ISR.ONKI, mirror to internal memory.
        let ssr_offset = 0xFF;
        let isr_offset = IMEM_ISR_OFFSET;
        let ssr = self.mirror.read_internal_byte(ssr_offset).unwrap_or(0);
        self.mirror.write_internal_byte(ssr_offset, ssr | 0x08);
        let isr = self.mirror.read_internal_byte(isr_offset).unwrap_or(0);
        let new_isr = isr | 0x08;
        self.mirror.write_internal_byte(isr_offset, new_isr);
        // Parity: mirror CoreRuntime press_on_key side-effects so IRQ delivery and tracing match Python.
        self.timer
            .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
        self.timer.irq_pending = true;
        self.timer.irq_source = Some("ONK".to_string());
        self.timer.last_fired = self.timer.irq_source.clone();
        self.timer.irq_isr = self
            .mirror
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .mirror
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            let mut payload = HashMap::new();
            payload.insert("pc".to_string(), AnnotationValue::Pointer(perfetto_last_pc() as u64));
            payload.insert("imr".to_string(), AnnotationValue::UInt(self.timer.irq_imr as u64));
            payload.insert("isr".to_string(), AnnotationValue::UInt(self.timer.irq_isr as u64));
            payload.insert("src".to_string(), AnnotationValue::Str("ONK".to_string()));
            tracer.record_irq_event("KeyIRQ", payload);
        }
        self.sync_mirror(py);
        Ok(true)
    }

    fn keyboard_release_on_key(&mut self, py: Python<'_>) -> PyResult<()> {
        let ssr_offset = 0xFF;
        let isr_offset = IMEM_ISR_OFFSET;
        let ssr = self.mirror.read_internal_byte(ssr_offset).unwrap_or(0);
        self.mirror.write_internal_byte(ssr_offset, ssr & !0x08);
        let isr = self.mirror.read_internal_byte(isr_offset).unwrap_or(0);
        let new_isr = isr & !0x08;
        self.mirror.write_internal_byte(isr_offset, new_isr);
        self.timer
            .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
        self.timer.irq_isr = new_isr;
        self.sync_mirror(py);
        Ok(())
    }

    fn keyboard_release_matrix_code(&mut self, py: Python<'_>, code: u8) -> PyResult<bool> {
        let events = self
            .keyboard
            .inject_matrix_event(
                code & 0x7F,
                true,
                &mut self.mirror,
                self.timer.kb_irq_enabled,
            );
        if self.timer.kb_irq_enabled && self.keyboard.fifo_len() > 0 {
            self.timer.irq_pending = true;
            self.timer.irq_source = Some("KEY".to_string());
            self.timer.last_fired = self.timer.irq_source.clone();
            self.timer.irq_isr = self
                .mirror
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(self.timer.irq_isr);
            self.timer.irq_imr = self
                .mirror
                .read_internal_byte(IMEM_IMR_OFFSET)
                .unwrap_or(self.timer.irq_imr);
        }
        self.sync_mirror(py);
        Ok(events > 0)
    }

    fn is_memory_synced(&self) -> bool {
        self.memory_synced
    }

    fn mark_memory_dirty(&mut self) {
        self.memory_synced = false;
    }

    fn _initialise_rust_memory(&mut self, py: Python<'_>) -> PyResult<()> {
        let bound = self.memory.bind(py);
        let exported = bound
            .call_method0("export_flat_memory")
            .map_err(|e| PyRuntimeError::new_err(format!("export_flat_memory: {e}")))?;
        let exported = exported.downcast::<PyTuple>()?;
        if exported.len() < 3 {
            return Err(PyRuntimeError::new_err(
                "export_flat_memory returned an unexpected shape",
            ));
        }
        let flat_item = exported.get_item(0)?;
        let flat_bytes = flat_item
            .downcast::<PyBytes>()
            .map_err(|e| PyRuntimeError::new_err(format!("flattened memory: {e}")))?
            .as_bytes()
            .to_vec();
        let fallback_ranges: Vec<(u32, u32)> = exported
            .get_item(1)?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(format!("fallback ranges: {e}")))?;
        let readonly_ranges: Vec<(u32, u32)> = exported
            .get_item(2)?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(format!("readonly ranges: {e}")))?;

        self.mirror = MemoryImage::new();
        if flat_bytes.len() == EXTERNAL_SPACE {
            let _ = self.mirror.copy_external_from(&flat_bytes);
        }
        if let Ok(imem_obj) = bound.call_method0("get_internal_memory_bytes") {
            if let Ok(imem_bytes) = imem_obj.downcast::<PyBytes>() {
                self.mirror.load_internal(imem_bytes.as_bytes());
            }
        }
        self.mirror.set_python_ranges(fallback_ranges);
        self.mirror.set_readonly_ranges(readonly_ranges);
        self.memory_synced = true;
        Ok(())
    }

    fn save_snapshot(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        let bound = self.memory.bind(py);
        let exported = bound
            .call_method0("export_flat_memory")
            .map_err(|e| PyRuntimeError::new_err(format!("export_flat_memory: {e}")))?;
        let exported = exported.downcast::<PyTuple>()?;
        if exported.len() < 3 {
            return Err(PyRuntimeError::new_err(
                "export_flat_memory returned an unexpected shape",
            ));
        }

        let flat_item = exported.get_item(0)?;
        let flat_bytes = flat_item
            .downcast::<PyBytes>()
            .map_err(|e| PyRuntimeError::new_err(format!("flattened memory: {e}")))?
            .as_bytes()
            .to_vec();
        let fallback_ranges: Vec<(u32, u32)> = exported
            .get_item(1)?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(format!("fallback ranges: {e}")))?;
        let readonly_ranges: Vec<(u32, u32)> = exported
            .get_item(2)?
            .extract()
            .map_err(|e| PyRuntimeError::new_err(format!("readonly ranges: {e}")))?;

        let imem_obj = bound
            .call_method0("get_internal_memory_bytes")
            .map_err(|e| PyRuntimeError::new_err(format!("get_internal_memory_bytes: {e}")))?;
        let imem_bytes = imem_obj
            .downcast::<PyBytes>()
            .map_err(|e| PyRuntimeError::new_err(format!("imem bytes: {e}")))?
            .as_bytes()
            .to_vec();

        let mut image = MemoryImage::new();
        if flat_bytes.len() != EXTERNAL_SPACE {
            return Err(PyRuntimeError::new_err(format!(
                "flattened memory length mismatch (got {}, expected {})",
                flat_bytes.len(),
                EXTERNAL_SPACE
            )));
        }
        image
            .copy_external_from(&flat_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("copy external: {e}")))?;
        image.load_internal(&imem_bytes);
        image.set_python_ranges(fallback_ranges.clone());
        image.set_readonly_ranges(readonly_ranges.clone());

        let temps: std::collections::HashMap<String, u32> = (0..14u32)
            .map(|idx| {
                let reg = LlamaRegName::Temp(idx as u8);
                (idx.to_string(), self.state.get_reg(reg) & ADDRESS_MASK)
            })
            .collect();
        let kb_state = self.keyboard.snapshot_state();
        let mut metadata = SnapshotMetadata {
            backend: "llama".to_string(),
            pc: self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK,
            memory_image_size: image.external_len(),
            fallback_ranges,
            readonly_ranges,
            memory_dump_pc: 0,
            memory_reads: self.memory_reads,
            memory_writes: self.memory_writes,
            call_depth: self.state.call_depth(),
            call_sub_level: self.state.call_sub_level(),
            temps,
            keyboard: to_value(&kb_state).ok(),
            kb_metrics: Some(json!({
                "irq_count": kb_state.irq_count,
                "strobe_count": kb_state.strobe_count,
                "column_hist": kb_state.column_histogram,
                "last_cols": kb_state.active_columns,
                "last_kol": kb_state.kol,
                "last_koh": kb_state.koh,
                "kil_reads": kb_state.fifo_len,
                "kb_irq_enabled": true,
            })),
            ..SnapshotMetadata::default()
        };
        // Capture timer/interrupt mirrors for parity with Python snapshots.
        let (timer_info, interrupt_info) = self.timer.snapshot_info();
        metadata.timer = timer_info;
        metadata.interrupts = interrupt_info;
        if let Some(imr) = image.read_internal_byte(IMEM_IMR_OFFSET) {
            metadata.interrupts.imr = imr;
        }
        if let Some(isr) = image.read_internal_byte(IMEM_ISR_OFFSET) {
            metadata.interrupts.isr = isr;
        }

        // Mirror interrupt scheduler state where available.
        metadata.interrupts.pending = false;
        metadata.interrupts.in_interrupt = false;
        metadata.interrupts.source = None;
        metadata.interrupts.stack = Vec::new();
        metadata.interrupts.next_id = 0;

        let (lcd_meta, lcd_payload) = match capture_lcd_snapshot(py, bound) {
            Ok(Some(pair)) => (Some(pair.0), Some(pair.1)),
            _ => (None, None),
        };
        metadata.lcd = lcd_meta;
        metadata.lcd_payload_size = lcd_payload.as_ref().map(|v| v.len()).unwrap_or(0);

        let regs = sc62015_core::collect_registers(&self.state);
        save_snapshot(
            std::path::Path::new(path),
            &metadata,
            &regs,
            &image,
            lcd_payload.as_deref(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("save_snapshot: {e}")))?;
        Ok(())
    }

    fn notify_host_write(&self, py: Python<'_>, address: u32, value: u8) -> PyResult<()> {
        let bound = self.memory.bind(py);
        let _ = bound.call_method1("write_byte", (address, value));
        Ok(())
    }

    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("backend", "llama")?;
        Ok(dict.into_py(py))
    }

    #[getter]
    fn call_sub_level(&self) -> u32 {
        self.call_sub_level
    }

    #[setter]
    fn set_call_sub_level(&mut self, value: u32) {
        self.call_sub_level = value;
        self.state.set_call_sub_level(value);
    }

    #[getter]
    fn halted(&self) -> bool {
        self.state.is_halted()
    }

    #[setter]
    fn set_halted(&mut self, value: bool) {
        self.state.set_halted(value);
    }

    #[pyo3(signature = (path=None))]
    fn set_perfetto_trace(&mut self, path: Option<&str>) -> PyResult<()> {
        if let Some(p) = path {
            reset_perf_counters();
            let tracer = PerfettoTracer::new(PathBuf::from(p));
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(tracer);
            println!("[perfetto-tracer] started at {}", p);
        } else {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = None;
            println!("[perfetto-tracer] cleared");
        }
        Ok(())
    }

    fn flush_perfetto(&mut self) -> PyResult<()> {
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.take() {
            let _ = tracer.finish();
        }
        Ok(())
    }
}

impl LlamaCpu {
    fn emit_irq_trace(&self, py: Python<'_>, name: &str, payload: HashMap<&'static str, u32>) {
        let dict = PyDict::new_bound(py);
        for (k, v) in payload {
            let _ = dict.set_item(k, v);
        }
        let _ = self
            .memory
            .bind(py)
            .call_method1("trace_irq_from_rust", (name, dict));
    }

    fn read_irq_registers(&self, py: Python<'_>, bus: &mut LlamaPyBus) -> (u8, u8) {
        let imr = bus.read_byte_with_gil(py, INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
        let isr = bus.read_byte_with_gil(py, INTERNAL_MEMORY_START + IMEM_ISR_OFFSET);
        (imr, isr)
    }

    fn sync_mirror(&mut self, py: Python<'_>) {
        let bound = self.memory.bind(py);
        for (addr, value) in self.mirror.drain_dirty_internal() {
            let _ = bound.call_method1("write_byte", (addr, value));
        }
        for (addr, value) in self.mirror.drain_dirty() {
            let _ = bound.call_method1("write_byte", (addr, value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_bus_tick_timers_sets_mti_and_isr() {
        let mut bus = LlamaContractBus::new();
        bus.configure_timer(1, 0, true);
        bus.tick_timers(1);
        let isr = bus
            .memory
            .read_internal_byte(0xFC)
            .unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01, "MTI bit should be set in ISR");
        assert!(bus.timer.irq_pending, "MTI should mark irq_pending");
    }

    #[test]
    fn wait_fallback_ticks_timers_when_wait_cycles_missing() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100000 + 0x100)
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.data[addr & 0xFFFFFF] = val & 0xFF
        return None
"#;
            let module = PyModule::from_code_bound(py, code, "mem_mod.py", "mem_mod")
                .expect("mem module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();

            let mut cpu = LlamaCpu::new(mem.to_object(py), false).expect("cpu init");
            // Seed WAIT at PC=0 and configure a fast timer tick.
            let mem_obj = cpu.memory.clone_ref(py);
            let bound_before = mem_obj.bind(py);
            let _ = bound_before.call_method1("write_byte", (0u32, 0xEFu8));
            cpu.state.set_reg(LlamaRegName::PC, 0);
            cpu.state.set_reg(LlamaRegName::I, 2);
            cpu.timer.enabled = true;
            cpu.timer.mti_period = 1;
            cpu.timer.reset(0);

            let (_opcode, _len) = cpu.execute_instruction(py, 0).expect("execute WAIT");

            // MTI should have fired during fallback wait_cycles and set IRQ pending.
            assert!(cpu.timer.irq_pending, "timer should pend after WAIT fallback");
            let bound_after = mem_obj.bind(py);
            let isr = bound_after
                .call_method1("read_byte", (INTERNAL_MEMORY_START + IMEM_ISR_OFFSET,))
                .and_then(|obj| obj.extract::<u8>())
                .unwrap_or(0);
            assert!(
                isr & 0x01 != 0,
                "ISR MTI bit should set when wait_cycles fallback ticks timers"
            );
            assert_eq!(
                cpu.state.get_reg(LlamaRegName::I),
                0,
                "WAIT should clear I even with fallback"
            );
        });
    }
}

#[pymodule]
fn _sc62015_rustcore(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("HAS_CPU_IMPLEMENTATION", false)?;
    m.add("HAS_LLAMA_IMPLEMENTATION", true)?;
    m.add_class::<LlamaCpu>()?;
    m.add_class::<LlamaContractBus>()?;
    m.add(
        "record_irq_event",
        pyo3::wrap_pyfunction!(record_irq_event_py, m)?,
    )?;
    Ok(())
}

/// Helper to emit an IRQ event from Python into the Rust tracer when available.
#[pyfunction]
fn record_irq_event_py(name: &str, payload: HashMap<String, u64>) -> PyResult<()> {
    let mut guard = PERFETTO_TRACER.enter();
    if let Some(tracer) = guard.as_mut() {
        let mut converted = HashMap::new();
        for (k, v) in payload {
            converted.insert(k, AnnotationValue::UInt(v));
        }
        tracer.record_irq_event(name, converted);
    }
    Ok(())
}
