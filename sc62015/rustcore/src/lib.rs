// PY_SOURCE: sc62015/pysc62015/cpu.py:CPU
// PY_SOURCE: sc62015/pysc62015/emulator.py:Emulator
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyBytes, PyDict, PyModule, PyTuple};
use pyo3::Bound;
use retrobus_perfetto::AnnotationValue;
use sc62015_core::{
    keyboard::{KeyboardMatrix, KeyboardSnapshot},
    llama::{
        eval::{
            fetch_validated_vector, perfetto_last_instr_index, perfetto_last_pc,
            power_on_reset_with_transfer, prepare_validated_vector, reset_perf_counters,
            set_perf_instr_counter, validate_vector_transfer_with_length, LlamaBus, LlamaExecutor,
            ValidatedVectorTransfer,
        },
        opcodes::RegName as LlamaRegName,
        state::{validate_f_image, CallMetricsSnapshot, LlamaState, PowerState},
    },
    memory::MemoryImage,
    pce500::ROM_RESET_VECTOR_ADDR,
    snapshot::save_snapshot,
    timer::TimerContext,
    InterruptInfo, PerfettoTracer, SnapshotMetadata, TimerInfo, ADDRESS_MASK, EXTERNAL_SPACE,
    INTERNAL_MEMORY_START, INTERNAL_SPACE, NUM_TEMP_REGISTERS, PERFETTO_TRACER,
};
use serde_json::{json, to_value, Value as JsonValue};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;

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
    busy: bool,
    start_line: u8,
    vram: [[u8; 64]; 8],
}

impl LcdShadow {
    fn new() -> Self {
        Self {
            page: 0,
            y: 0,
            on: false,
            busy: false,
            start_line: 0,
            vram: [[0; 64]; 8],
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.page = 0;
        self.y = 0;
        self.on = false;
        self.busy = false;
        self.start_line = 0;
        self.vram = [[0; 64]; 8];
    }

    fn apply_instruction(&mut self, instr: u8, data: u8) {
        self.busy = true;
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
        self.busy = true;
        let page = (self.page as usize) % 8;
        let y = (self.y as usize) % 64;
        self.vram[page][y] = value;
        self.y = ((self.y as usize + 1) % 64) as u8;
    }

    fn read_status(&mut self) -> u8 {
        let mut status = if self.busy { 0x80 } else { 0x00 };
        if !self.on {
            status |= 0x20;
        }
        self.busy = false;
        status
    }

    fn read_data(&mut self) -> u8 {
        let page = (self.page as usize) % 8;
        let y = (self.y as usize) % 64;
        let read_col = if y == 0 { 63 } else { y - 1 };
        let value = self.vram[page][read_col];
        self.y = ((y + 1) % 64) as u8;
        value
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
        Err(err) if err.is_instance_of::<PyAttributeError>(py) => return Ok(None),
        Err(err) => return Err(err),
    };
    if controller.is_none() {
        return Ok(None);
    }
    // Once a controller is present, snapshot capture is part of the explicit
    // LLAMA snapshot contract. Do not silently omit it when its getter or
    // snapshot method is broken.
    let snapshot = controller.call_method0("get_snapshot")?;
    let chips: Vec<PyObject> = snapshot
        .getattr("chips")?
        .extract()
        .map_err(|e| PyRuntimeError::new_err(format!("lcd snapshot chips: {e}")))?;
    if chips.is_empty() {
        return Err(PyValueError::new_err(
            "lcd snapshot must contain at least one chip",
        ));
    }

    let mut meta_chips = Vec::with_capacity(chips.len());
    let mut payload: Vec<u8> = Vec::new();
    let mut pages = 0usize;
    let mut width = 0usize;
    for (chip_index, chip_obj) in chips.iter().enumerate() {
        let chip = chip_obj.bind(py);
        let on: bool = chip.getattr("on")?.extract()?;
        let start_line: u8 = chip.getattr("start_line")?.extract()?;
        let page: u8 = chip.getattr("page")?.extract()?;
        let y_address: u8 = chip.getattr("y_address")?.extract()?;
        let vram: Vec<Vec<u8>> = chip.getattr("vram")?.extract()?;
        let chip_pages = vram.len();
        let chip_width = vram.first().map(|row| row.len()).unwrap_or(0);
        if chip_pages == 0 || chip_width == 0 {
            return Err(PyValueError::new_err(format!(
                "lcd snapshot chip {chip_index} has empty VRAM"
            )));
        }
        if let Some((row_index, row)) = vram
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != chip_width)
        {
            return Err(PyValueError::new_err(format!(
                "lcd snapshot chip {chip_index} VRAM row {row_index} has width {}, expected {chip_width}",
                row.len()
            )));
        }
        if chip_index == 0 {
            pages = chip_pages;
            width = chip_width;
        } else if chip_pages != pages || chip_width != width {
            return Err(PyValueError::new_err(format!(
                "lcd snapshot chip {chip_index} VRAM shape is {chip_pages}x{chip_width}, expected {pages}x{width}"
            )));
        }
        for page_rows in &vram {
            for byte in page_rows {
                payload.push(*byte);
            }
        }
        let instr_count: u32 = chip.getattr("instruction_count")?.extract()?;
        let data_write_count: u32 = chip.getattr("data_write_count")?.extract()?;
        let data_read_count: u32 = chip.getattr("data_read_count")?.extract()?;
        let on_off_count: u32 = chip.getattr("on_off_count")?.extract()?;
        meta_chips.push(json!({
            "on": on,
            "start_line": start_line,
            "page": page,
            "y_address": y_address,
            "instruction_count": instr_count,
            "data_write_count": data_write_count,
            "data_read_count": data_read_count,
            "on_off_count": on_off_count,
        }));
    }
    let optional_counter = |name: &str| -> PyResult<u32> {
        match controller.getattr(name) {
            Ok(value) => value.extract::<u32>(),
            Err(err) if err.is_instance_of::<PyAttributeError>(py) => Ok(0),
            Err(err) => Err(err),
        }
    };
    let cs_both = optional_counter("cs_both_count")?;
    let cs_left = optional_counter("cs_left_count")?;
    let cs_right = optional_counter("cs_right_count")?;
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

struct ContractHostMemory {
    read_callback: Py<PyAny>,
    write_callback: Py<PyAny>,
    read_with_pc: bool,
    write_with_pc: bool,
}

#[pyclass(unsendable, name = "LlamaContractBus")]
struct LlamaContractBus {
    memory: MemoryImage,
    events: Vec<ContractEvent>,
    timer: TimerContext,
    cycles: u64,
    host_memory: Option<ContractHostMemory>,
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

    fn add_ram_overlay(&mut self, start: u32, size: usize, name: &str) {
        self.memory.add_ram_overlay(start, size, name);
    }

    fn add_rom_overlay(&mut self, start: u32, data: &[u8], name: &str) {
        self.memory.add_rom_overlay(start, data, name);
    }

    fn load_memory_card(&mut self, data: &[u8]) -> PyResult<()> {
        self.memory
            .load_memory_card(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn remove_overlay(&mut self, name: &str) {
        self.memory.remove_overlay(name);
    }

    fn overlay_read_log(&self, py: Python<'_>) -> PyResult<PyObject> {
        let log = self
            .memory
            .overlay_read_log()
            .into_iter()
            .map(|entry| {
                let dict = PyDict::new_bound(py);
                let _ = dict.set_item("kind", "read");
                let _ = dict.set_item("address", entry.address);
                let _ = dict.set_item("value", entry.value);
                let _ = dict.set_item("overlay", entry.overlay);
                if let Some(pc) = entry.pc {
                    let _ = dict.set_item("pc", pc);
                }
                dict.into_py(py)
            })
            .collect::<Vec<_>>();
        Ok(log.into_py(py))
    }

    fn overlay_write_log(&self, py: Python<'_>) -> PyResult<PyObject> {
        let log = self
            .memory
            .overlay_write_log()
            .into_iter()
            .map(|entry| {
                let dict = PyDict::new_bound(py);
                let _ = dict.set_item("kind", "write");
                let _ = dict.set_item("address", entry.address);
                let _ = dict.set_item("value", entry.value);
                let _ = dict.set_item("overlay", entry.overlay);
                if let Some(pc) = entry.pc {
                    let _ = dict.set_item("pc", pc);
                }
                if let Some(prev) = entry.previous {
                    let _ = dict.set_item("previous", prev);
                }
                dict.into_py(py)
            })
            .collect::<Vec<_>>();
        Ok(log.into_py(py))
    }

    /// Optional host memory hook for addresses that require Python overlays (e.g., ON/ONK).
    fn set_host_memory(&mut self, py: Python<'_>, memory: Py<PyAny>) -> PyResult<()> {
        let read_callback = memory.getattr(py, "read_byte")?;
        let write_callback = memory.getattr(py, "write_byte")?;
        if !read_callback.bind(py).is_callable() || !write_callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "host memory read_byte and write_byte attributes must be callable",
            ));
        }
        let read_with_pc = callback_uses_pc(py, read_callback.bind(py), 2)?;
        let write_with_pc = callback_uses_pc(py, write_callback.bind(py), 3)?;
        self.host_memory = Some(ContractHostMemory {
            read_callback,
            write_callback,
            read_with_pc,
            write_with_pc,
        });
        Ok(())
    }

    fn requires_python(&self, address: u32) -> bool {
        self.memory.requires_python(address)
    }

    fn set_timer_scale(&mut self, scale: f64) {
        self.timer.set_timer_scale(scale);
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
            let (mti, sti, _key_events, _kb_stats) = self.timer.tick_timers_with_keyboard(
                &mut self.memory,
                self.cycles,
                |mem| {
                    let events = self.keyboard.scan_tick(mem, kb_irq_enabled);
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
        let raw_addr = address & ADDRESS_MASK;
        // Parity: Python treats any address with bit20 set as selecting the 256-byte IMEM window
        // via `(addr - 0x100000) & 0xFF`. The core `MemoryImage` is range-limited to 0x100000..0x1000FF,
        // so normalize here for contract harness runs.
        let addr = if raw_addr >= INTERNAL_MEMORY_START {
            INTERNAL_MEMORY_START + ((raw_addr - INTERNAL_MEMORY_START) & 0xFF)
        } else {
            raw_addr
        };
        // Defer to host memory when Python overlays are required.
        if self.memory.requires_python(addr) {
            if let Some(host) = &self.host_memory {
                return Python::with_gil(|py| {
                    let result = if host.read_with_pc {
                        host.read_callback.bind(py).call1((raw_addr, pc))
                    } else {
                        host.read_callback.bind(py).call1((raw_addr,))
                    };
                    let value = result.and_then(|val| val.extract::<u8>())?;
                    // Parity: still bump counters and record a read event even when the host services it.
                    self.memory.bump_read_count();
                    self.events.push(ContractEvent {
                        kind: "read",
                        address: raw_addr,
                        value,
                        pc: pc.map(|v| v & ADDRESS_MASK),
                        detail: None,
                    });
                    Ok(value)
                });
            }
        }
        if addr >= INTERNAL_MEMORY_START {
            let offset = (raw_addr - INTERNAL_MEMORY_START) & 0xFF;
            if let Some(value) = self.keyboard.handle_read(offset, &mut self.memory) {
                self.events.push(ContractEvent {
                    kind: "read",
                    address: raw_addr,
                    value,
                    pc: pc.map(|v| v & ADDRESS_MASK),
                    detail: None,
                });
                return Ok(value);
            }
        }
        if is_lcd_addr(raw_addr) {
            let addr_lo = raw_addr & 0x0FFF;
            let rw = addr_lo & 1;
            if rw == 1 {
                let di = (addr_lo >> 1) & 1;
                let cs_bits = (addr_lo >> 2) & 0b11;
                let idx = match cs_bits {
                    0b10 => Some(0),
                    0b01 => Some(1),
                    _ => None,
                };
                if let Some(idx) = idx {
                    let value = if di == 0 {
                        self.lcd_shadow[idx].read_status()
                    } else {
                        self.lcd_shadow[idx].read_data()
                    };
                    if (raw_addr & 0x3) == 0x1 {
                        self.last_lcd_status = Some(value);
                    }
                    self.events.push(ContractEvent {
                        kind: "read",
                        address: raw_addr,
                        value,
                        pc: pc.map(|v| v & ADDRESS_MASK),
                        detail: None,
                    });
                    return Ok(value);
                }
            }
        }
        let value = self
            .memory
            .load_with_pc(addr, 8, pc.map(|v| v & ADDRESS_MASK))
            .unwrap_or(0) as u8;
        if is_lcd_addr(raw_addr) && (raw_addr & 0x3) == 0x1 {
            self.last_lcd_status = Some(value);
        }
        self.events.push(ContractEvent {
            kind: "read",
            address: raw_addr,
            value,
            pc: pc.map(|v| v & ADDRESS_MASK),
            detail: None,
        });
        Ok(value)
    }

    #[pyo3(signature = (address, value, pc=None))]
    fn write_byte(&mut self, address: u32, value: u8, pc: Option<u32>) -> PyResult<()> {
        let raw_addr = address & ADDRESS_MASK;
        let addr = if raw_addr >= INTERNAL_MEMORY_START {
            INTERNAL_MEMORY_START + ((raw_addr - INTERNAL_MEMORY_START) & 0xFF)
        } else {
            raw_addr
        };
        if self.memory.requires_python(addr) {
            if let Some(host) = &self.host_memory {
                Python::with_gil(|py| {
                    if host.write_with_pc {
                        host.write_callback
                            .bind(py)
                            .call1((raw_addr, value, pc))
                            .map(|_| ())
                    } else {
                        host.write_callback
                            .bind(py)
                            .call1((raw_addr, value))
                            .map(|_| ())
                    }
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
                    self.memory.apply_host_write_with_cycle(
                        addr,
                        value,
                        None,
                        pc.map(|v| v & ADDRESS_MASK),
                    );
                }
                self.events.push(ContractEvent {
                    kind: "write",
                    address: raw_addr,
                    value,
                    pc: pc.map(|v| v & ADDRESS_MASK),
                    detail: None,
                });
                return Ok(());
            }
        }
        if addr >= INTERNAL_MEMORY_START {
            let offset = (raw_addr - INTERNAL_MEMORY_START) & 0xFF;
            if self.keyboard.handle_write(offset, value, &mut self.memory) {
                self.events.push(ContractEvent {
                    kind: "write",
                    address: raw_addr,
                    value,
                    pc: pc.map(|v| v & ADDRESS_MASK),
                    detail: None,
                });
                return Ok(());
            }
        }
        let _ = self
            .memory
            .store_with_pc(addr, 8, value as u32, pc.map(|v| v & ADDRESS_MASK));
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
            address: raw_addr,
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
        let mut external = self.memory.external_slice().to_vec();
        let internal = self.memory.internal_slice();
        if external.len() >= internal.len() {
            let start = external.len() - internal.len();
            external[start..].copy_from_slice(internal);
        }
        dict.set_item("external", PyBytes::new_bound(py, &external))?;
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
        guard.with_some(|tracer| {
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
            payload.insert("src".to_string(), AnnotationValue::Str("ONK".to_string()));
            tracer.record_irq_event("KeyIRQ", payload);
        });
    }

    fn keyboard_press_matrix_code(&mut self, code: u8) -> PyResult<bool> {
        let events = self.keyboard.inject_matrix_event(
            code & 0x7F,
            false,
            &mut self.memory,
            self.timer.kb_irq_enabled,
        );
        Ok(events > 0)
    }

    fn keyboard_release_matrix_code(&mut self, code: u8) -> PyResult<bool> {
        let events = self.keyboard.inject_matrix_event(
            code & 0x7F,
            true,
            &mut self.memory,
            self.timer.kb_irq_enabled,
        );
        Ok(events > 0)
    }

    fn release_on_key(&mut self) {
        let ssr_offset = IMEM_SSR_OFFSET;
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
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
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
    read_callback: Py<PyAny>,
    write_callback: Py<PyAny>,
    preflight_read_callback: Py<PyAny>,
    read_with_pc: bool,
    write_with_pc: bool,
    preflight_read_with_pc: bool,
    pc: u32,
    lcd_hook: Option<Py<PyAny>>,
    kio_trace_hook: Option<Py<PyAny>>,
    irq_trace_hook: Option<Py<PyAny>>,
    timer_phase_clear_hook: Option<Py<PyAny>>,
    memory_reads: u64,
    memory_writes: u64,
    has_wait_cycles: bool,
    timer: *mut TimerContext,
    keyboard: *mut KeyboardMatrix,
    mirror: *mut MemoryImage,
    cycles_ptr: *mut u64,
    callback_error: Option<PyErr>,
    provenance_error: RefCell<Option<PyErr>>,
    provenance_failure_nonce: Cell<u64>,
    write_attempts: Vec<u32>,
}

fn optional_callable_attr(
    py: Python<'_>,
    object: &Py<PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    match object.getattr(py, name) {
        Ok(value) => {
            if !value.bind(py).is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "optional callback {name} is present but is not callable"
                )));
            }
            Ok(Some(value))
        }
        Err(err) if err.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

fn python_vector_provenance(py: Python<'_>, memory: &Py<PyAny>) -> PyResult<(usize, u64)> {
    match optional_callable_attr(py, memory, "vector_transfer_provenance")? {
        Some(provider) => provider.bind(py).call0()?.extract::<(usize, u64)>(),
        None => Ok((memory.as_ptr() as usize, 0)),
    }
}

fn require_python_instruction_stability(
    py: Python<'_>,
    memory: &Py<PyAny>,
    method_name: &str,
    addresses: impl IntoIterator<Item = u32>,
) -> PyResult<()> {
    let checker = optional_callable_attr(py, memory, method_name)?.ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "prepared SC62015 vector transfer requires memory.{method_name}"
        ))
    })?;
    for address in addresses {
        let stable = checker
            .bind(py)
            .call1((address & ADDRESS_MASK,))?
            .extract::<bool>()?;
        if !stable {
            return Err(PyRuntimeError::new_err(format!(
                "prepared SC62015 vector transfer rejects dynamic byte 0x{:05X}",
                address & ADDRESS_MASK
            )));
        }
    }
    Ok(())
}

fn callback_accepts_positional(
    py: Python<'_>,
    callable: &Bound<'_, PyAny>,
    count: usize,
) -> PyResult<Option<bool>> {
    let inspect = py.import_bound("inspect")?;
    let signature = match inspect.call_method1("signature", (callable,)) {
        Ok(signature) => signature,
        Err(err)
            if err.is_instance_of::<PyTypeError>(py) || err.is_instance_of::<PyValueError>(py) =>
        {
            return Ok(None)
        }
        Err(err) => return Err(err),
    };
    let args = PyTuple::new_bound(py, (0..count).map(|_| py.None()));
    match signature.call_method("bind", args, None) {
        Ok(_) => Ok(Some(true)),
        Err(err) if err.is_instance_of::<PyTypeError>(py) => Ok(Some(false)),
        Err(err) => Err(err),
    }
}

fn callback_uses_pc(
    py: Python<'_>,
    callable: &Bound<'_, PyAny>,
    modern_count: usize,
) -> PyResult<bool> {
    match callback_accepts_positional(py, callable, modern_count)? {
        Some(true) => Ok(true),
        Some(false) => match callback_accepts_positional(py, callable, modern_count - 1)? {
            Some(true) => Ok(false),
            _ => Err(PyTypeError::new_err(format!(
                "memory callback accepts neither {modern_count} nor {} positional arguments",
                modern_count - 1
            ))),
        },
        // Some C-extension callables do not publish a signature. Pick the
        // modern contract once and propagate any body/signature error; never
        // retry a callback after it has run.
        None => Ok(true),
    }
}

fn flush_mirror_to_python(
    py: Python<'_>,
    write_callback: &Py<PyAny>,
    write_with_pc: bool,
    pc: u32,
    mirror: &mut MemoryImage,
    mut write_attempts: Option<&mut Vec<u32>>,
) -> PyResult<()> {
    let write_byte = |addr: u32, value: u8| {
        if write_with_pc {
            write_callback.bind(py).call1((addr, value, pc)).map(|_| ())
        } else {
            write_callback.bind(py).call1((addr, value)).map(|_| ())
        }
    };
    let internal = mirror.drain_dirty_internal();
    for (index, &(addr, value)) in internal.iter().enumerate() {
        if let Some(attempts) = write_attempts.as_deref_mut() {
            attempts.push(addr & ADDRESS_MASK);
        }
        if let Err(err) = write_byte(addr, value) {
            mirror.prepend_dirty_internal(internal[index..].to_vec());
            return Err(err);
        }
    }

    let external = mirror.drain_dirty();
    for (index, &(addr, value)) in external.iter().enumerate() {
        if let Some(attempts) = write_attempts.as_deref_mut() {
            attempts.push(addr & ADDRESS_MASK);
        }
        if let Err(err) = write_byte(addr, value) {
            mirror.prepend_dirty(external[index..].to_vec());
            return Err(err);
        }
    }
    Ok(())
}

impl LlamaPyBus {
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        memory: &Py<PyAny>,
        read_callback: Py<PyAny>,
        write_callback: Py<PyAny>,
        read_with_pc: bool,
        write_with_pc: bool,
        pc: u32,
        has_wait_cycles: bool,
        timer: *mut TimerContext,
        keyboard: *mut KeyboardMatrix,
        mirror: *mut MemoryImage,
        cycles_ptr: *mut u64,
    ) -> PyResult<Self> {
        if !read_callback.bind(py).is_callable() || !write_callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "memory read_byte and write_byte attributes must be callable",
            ));
        }
        // Optional hooks are absent only on AttributeError. A broken property
        // or a present non-callable hook is a configuration error.
        let preflight_read_callback =
            optional_callable_attr(py, memory, "peek_byte_for_preflight")?.ok_or_else(|| {
                PyRuntimeError::new_err(
                    "LLAMA execution requires memory.peek_byte_for_preflight for \
                     side-effect-free instruction validation",
                )
            })?;
        let preflight_read_with_pc = callback_uses_pc(py, preflight_read_callback.bind(py), 2)?;
        let lcd_hook = optional_callable_attr(py, memory, "_llama_lcd_write")?;
        let kio_trace_hook = optional_callable_attr(py, memory, "trace_kio_from_rust")?;
        let irq_trace_hook = optional_callable_attr(py, memory, "trace_irq_from_rust")?;
        let timer_phase_clear_hook = optional_callable_attr(py, memory, "clear_timer_phases")?;
        let _ = python_vector_provenance(py, memory)?;
        Ok(Self {
            memory: memory.clone_ref(py),
            read_callback,
            write_callback,
            preflight_read_callback,
            read_with_pc,
            write_with_pc,
            preflight_read_with_pc,
            pc,
            lcd_hook,
            kio_trace_hook,
            irq_trace_hook,
            timer_phase_clear_hook,
            memory_reads: 0,
            memory_writes: 0,
            has_wait_cycles,
            timer,
            keyboard,
            mirror,
            cycles_ptr,
            callback_error: None,
            provenance_error: RefCell::new(None),
            provenance_failure_nonce: Cell::new(0),
            write_attempts: Vec::new(),
        })
    }

    fn record_callback_error(&mut self, err: PyErr) {
        if self.callback_error.is_none() {
            self.callback_error = Some(err);
        }
    }

    fn take_callback_error(&mut self) -> Option<PyErr> {
        self.callback_error
            .take()
            .or_else(|| self.provenance_error.borrow_mut().take())
    }

    fn take_write_attempts(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.write_attempts)
    }

    fn is_lcd_addr(addr: u32) -> bool {
        (0x2000..=0x200F).contains(&(addr & ADDRESS_MASK))
            || (0xA000..=0xAFFF).contains(&(addr & ADDRESS_MASK))
    }

    fn read_byte(&mut self, addr: u32) -> u8 {
        Python::with_gil(|py| self.read_byte_with_gil(py, addr))
    }

    fn read_byte_with_gil(&mut self, py: Python<'_>, addr: u32) -> u8 {
        if self.callback_error.is_some() {
            return 0;
        }
        let addr = addr & ADDRESS_MASK;
        let result = if self.read_with_pc {
            self.read_callback.bind(py).call1((addr, self.pc))
        } else {
            self.read_callback.bind(py).call1((addr,))
        };
        let value = match result {
            Ok(obj) => match obj.extract::<u8>() {
                Ok(value) => value,
                Err(err) => {
                    self.record_callback_error(err);
                    0
                }
            },
            Err(err) => {
                self.record_callback_error(err);
                0
            }
        };
        // Count one logical read per byte.
        self.memory_reads += 1;
        value
    }

    fn peek_byte_for_preflight_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
        if self.callback_error.is_some() {
            return None;
        }
        let addr = addr & ADDRESS_MASK;
        let result = Python::with_gil(|py| {
            let callback = self.preflight_read_callback.bind(py);
            let value = if self.preflight_read_with_pc {
                callback.call1((addr, context_pc & ADDRESS_MASK))
            } else {
                callback.call1((addr,))
            }?;
            value.extract::<u8>()
        });
        match result {
            Ok(value) => Some(value),
            Err(err) => {
                self.record_callback_error(err);
                None
            }
        }
    }

    fn peek_byte_for_preflight(&mut self, addr: u32) -> Option<u8> {
        self.peek_byte_for_preflight_at(addr, self.pc)
    }

    fn write_byte(&mut self, addr: u32, value: u8) {
        Python::with_gil(|py| self.write_byte_with_gil(py, addr, value));
    }

    fn write_byte_with_gil(&mut self, py: Python<'_>, addr: u32, value: u8) {
        if self.callback_error.is_some() {
            return;
        }
        let addr = addr & ADDRESS_MASK;
        self.write_attempts.push(addr);
        let result = if self.write_with_pc {
            self.write_callback
                .bind(py)
                .call1((addr, value, self.pc))
                .map(|_| ())
        } else {
            self.write_callback
                .bind(py)
                .call1((addr, value))
                .map(|_| ())
        };
        if let Err(err) = result {
            self.record_callback_error(err);
            return;
        }
        // Count one logical write per byte.
        self.memory_writes += 1;
        if Self::is_lcd_addr(addr) {
            if let Some(hook) = &self.lcd_hook {
                if let Err(err) = hook.call1(py, (addr, value, self.pc)) {
                    self.record_callback_error(err);
                }
            }
        }
    }

    fn prepare_byte_write(&mut self, absolute: u32, value: u8) {
        if self.callback_error.is_some() || absolute < INTERNAL_MEMORY_START {
            return;
        }
        let offset = absolute - INTERNAL_MEMORY_START;
        if matches!(offset, IMEM_KIL_OFFSET | IMEM_KOL_OFFSET | IMEM_KOH_OFFSET) {
            unsafe {
                if !self.keyboard.is_null() && !self.mirror.is_null() {
                    let keyboard = &mut *self.keyboard;
                    let mirror = &mut *self.mirror;
                    // Keep the Rust keyboard/mirror in sync with Python
                    // overlays so timer-driven scans observe every byte of a
                    // wide KIO write, not only its starting address.
                    keyboard.handle_write(offset, value, mirror);
                }
            }
        }
        if matches!(offset, IMEM_IMR_OFFSET | IMEM_ISR_OFFSET | IMEM_SCR_OFFSET) {
            let result = Python::with_gil(|py| -> PyResult<()> {
                let payload = PyDict::new_bound(py);
                payload.set_item("offset", offset & 0xFF)?;
                payload.set_item("value", value)?;
                payload.set_item("pc", self.pc & ADDRESS_MASK)?;
                if let Some(hook) = &self.irq_trace_hook {
                    hook.bind(py).call1((
                        match offset {
                            IMEM_IMR_OFFSET => "IMR_Write",
                            IMEM_ISR_OFFSET => "ISR_Write",
                            IMEM_SCR_OFFSET => "SCR_Write",
                            _ => "IRQ_Write",
                        },
                        payload,
                    ))?;
                }
                Ok(())
            });
            if let Err(err) = result {
                self.record_callback_error(err);
            }
        } else if matches!(offset, IMEM_KIL_OFFSET | IMEM_KOL_OFFSET | IMEM_KOH_OFFSET) {
            let trace_result = match &self.kio_trace_hook {
                Some(hook) => Python::with_gil(|py| {
                    hook.bind(py)
                        .call1((offset, value, self.pc & ADDRESS_MASK))
                        .map(|_| ())
                }),
                None => Ok(()),
            };
            if let Err(err) = trace_result {
                self.record_callback_error(err);
            }
        }
    }

    fn commit_byte_write(&mut self, absolute: u32, value: u8) {
        unsafe {
            if !self.mirror.is_null() {
                // The host callback accepted this CPU-originated byte. Mark
                // it committed in the native mirror so a later flush cannot
                // replay KIO writes produced by `handle_write`.
                (*self.mirror).sync_committed_host_write(absolute, value);
            }
            if !self.timer.is_null() && absolute >= INTERNAL_MEMORY_START {
                match absolute - INTERNAL_MEMORY_START {
                    IMEM_IMR_OFFSET => (*self.timer).irq_imr = value,
                    IMEM_ISR_OFFSET => (*self.timer).irq_isr = value,
                    _ => {}
                }
            }
        }
    }
}

impl LlamaBus for LlamaPyBus {
    fn supports_wait_cycles(&self) -> bool {
        true
    }

    fn supports_timer_phase_clear(&self) -> bool {
        self.timer_phase_clear_hook.is_some()
    }

    fn clear_timer_phases(&mut self, clear_sti: bool, clear_mti: bool) {
        let Some(hook) = self.timer_phase_clear_hook.as_ref() else {
            return;
        };
        let result = Python::with_gil(|py| hook.bind(py).call1((clear_sti, clear_mti)).map(|_| ()));
        if let Err(err) = result {
            self.record_callback_error(err);
        }
    }

    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        // Respect the requested width so multi-byte loads match the Python emulator.
        let bytes = bits.div_ceil(8).max(1);
        let addr = addr & ADDRESS_MASK;
        let mut value = 0u32;
        for i in 0..bytes {
            let absolute = addr.wrapping_add(i as u32) & ADDRESS_MASK;
            let byte = self.read_byte(absolute) as u32;
            if self.callback_error.is_some() {
                break;
            }
            value |= byte << (8 * i);
            if absolute >= INTERNAL_MEMORY_START {
                let offset = absolute - INTERNAL_MEMORY_START;
                if matches!(offset, IMEM_KIL_OFFSET | IMEM_KOL_OFFSET | IMEM_KOH_OFFSET) {
                    let mut guard = PERFETTO_TRACER.enter();
                    guard.with_some(|tracer| {
                        tracer.record_kio_read(Some(self.pc), offset as u8, byte as u8, None);
                    });
                    // Mirror into Python's dispatcher so the main Perfetto trace sees KIO reads.
                    let trace_result = match &self.kio_trace_hook {
                        Some(hook) => Python::with_gil(|py| {
                            hook.bind(py).call1((offset, byte, self.pc)).map(|_| ())
                        }),
                        None => Ok(()),
                    };
                    if let Err(err) = trace_result {
                        self.record_callback_error(err);
                    }
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

    fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
        self.peek_byte_for_preflight(addr)
    }

    fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
        self.peek_byte_for_preflight_at(addr, context_pc)
    }

    fn vector_transfer_provenance(&self) -> (usize, u64) {
        Python::with_gil(|py| match python_vector_provenance(py, &self.memory) {
            Ok(provenance) => provenance,
            Err(error) => {
                if self.provenance_error.borrow().is_none() {
                    *self.provenance_error.borrow_mut() = Some(error);
                }
                let nonce = self.provenance_failure_nonce.get().wrapping_add(1);
                self.provenance_failure_nonce.set(nonce);
                (usize::MAX, nonce)
            }
        })
    }

    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        self.peek_byte_for_preflight(INTERNAL_MEMORY_START + offset)
            .unwrap_or(0)
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        if self.callback_error.is_some() {
            return;
        }
        let bytes = bits.div_ceil(8).max(1);
        for i in 0..bytes {
            let shift = 8 * i;
            let byte = if shift < 32 {
                ((value >> shift) & 0xFF) as u8
            } else {
                0
            };
            let absolute = addr.wrapping_add(i as u32) & ADDRESS_MASK;
            self.prepare_byte_write(absolute, byte);
            if self.callback_error.is_some() {
                break;
            }
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
            if self.callback_error.is_some() {
                break;
            }
            self.commit_byte_write(absolute, byte);
        }
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }

    fn wait_cycles(&mut self, cycles: u32) {
        if self.callback_error.is_some() {
            return;
        }
        // Prefer the Python host hook; otherwise, tick the Rust timer/keyboard locally for parity.
        if self.has_wait_cycles {
            let result = Python::with_gil(|py| {
                let bound = self.memory.bind(py);
                bound
                    .call_method1("wait_cycles", (cycles.max(1),))
                    .map(|_| ())
            });
            if let Err(err) = result {
                self.record_callback_error(err);
            }
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
            let sync_result = Python::with_gil(|py| -> PyResult<()> {
                for offset in [IMEM_IMR_OFFSET, IMEM_ISR_OFFSET] {
                    let addr = INTERNAL_MEMORY_START + offset;
                    let value = if self.read_with_pc {
                        self.read_callback.bind(py).call1((addr, self.pc))
                    } else {
                        self.read_callback.bind(py).call1((addr,))
                    }?;
                    let val = value.extract::<u8>()?;
                    let _ = mirror.store(INTERNAL_MEMORY_START + offset, 8, val as u32);
                }
                Ok(())
            });
            if let Err(err) = sync_result {
                self.record_callback_error(err);
                return;
            }

            for _ in 0..ticks {
                *cycles_counter = cycles_counter.wrapping_add(1);
                if let Some(imr) = mirror.read_internal_byte(IMEM_IMR_OFFSET) {
                    timer.irq_imr = imr;
                }
                if let Some(isr) = mirror.read_internal_byte(IMEM_ISR_OFFSET) {
                    timer.irq_isr = isr;
                }

                let kb_irq_enabled = timer.kb_irq_enabled;
                let (mti, sti, _key_events, _kb_stats) = timer.tick_timers_with_keyboard(
                    mirror,
                    *cycles_counter,
                    |mem| {
                        let events = keyboard.scan_tick(mem, kb_irq_enabled);
                        if events > 0 {
                            keyboard.write_fifo_to_memory(mem, kb_irq_enabled);
                        }
                        (events, keyboard.fifo_len() > 0, Some(keyboard.telemetry()))
                    },
                    None,
                    None,
                );
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
            let flush_result = Python::with_gil(|py| {
                flush_mirror_to_python(
                    py,
                    &self.write_callback,
                    self.write_with_pc,
                    self.pc,
                    mirror,
                    Some(&mut self.write_attempts),
                )
            });
            if let Err(err) = flush_result {
                self.record_callback_error(err);
            }
        }
    }
}

fn should_trace_addr(addr: u32) -> bool {
    let _ = addr;
    false
}

fn trace_loads() -> bool {
    false
}

#[pyclass(unsendable, name = "LlamaCPU")]
struct LlamaCpu {
    state: LlamaState,
    executor: LlamaExecutor,
    memory: Py<PyAny>,
    read_callback: Py<PyAny>,
    write_callback: Py<PyAny>,
    read_with_pc: bool,
    write_with_pc: bool,
    call_sub_level: i32,
    temps: HashMap<u32, u32>,
    mirror: MemoryImage,
    keyboard: KeyboardMatrix,
    timer: TimerContext,
    memory_synced: bool,
    memory_reads: u64,
    memory_writes: u64,
    cycles: u64,
    poisoned: Option<String>,
    uncertain_host_writes: Vec<u32>,
    pending_vector_transfer: Option<(ValidatedVectorTransfer, String)>,
    pending_scheduled_opcode: Option<(u32, u8)>,
}

struct CallbackRollbackState {
    state: LlamaState,
    timer: TimerContext,
    keyboard: KeyboardMatrix,
    cycles: u64,
    memory_reads: u64,
    memory_writes: u64,
    call_sub_level: i32,
    temps: HashMap<u32, u32>,
}

struct LoadedInterruptSnapshot {
    imr: u8,
    isr: u8,
    pending: bool,
    in_interrupt: bool,
    source: Option<String>,
    stack: Vec<u32>,
    next_id: u32,
    irq_total: u32,
    irq_key: u32,
    irq_mti: u32,
    irq_sti: u32,
    last_irq_src: Option<String>,
    last_irq_pc: Option<u32>,
    last_irq_vector: Option<u32>,
}

fn required_irq_count(counts: &HashMap<String, u32>, name: &str) -> PyResult<u32> {
    counts
        .get(name)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("snapshot irq_counts is missing {name:?}")))
}

fn extract_interrupt_snapshot(interrupts: &Bound<'_, PyAny>) -> PyResult<LoadedInterruptSnapshot> {
    let counts = interrupts
        .getattr("irq_counts")?
        .extract::<HashMap<String, u32>>()?;
    let last_irq_obj = interrupts.getattr("last_irq")?;
    let last_irq = last_irq_obj
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("snapshot last_irq must be a dict"))?;
    let last_irq_src = last_irq
        .get_item("src")?
        .ok_or_else(|| PyValueError::new_err("snapshot last_irq is missing \"src\""))?
        .extract::<Option<String>>()?;
    let last_irq_pc = last_irq
        .get_item("pc")?
        .ok_or_else(|| PyValueError::new_err("snapshot last_irq is missing \"pc\""))?
        .extract::<Option<u32>>()?;
    let last_irq_vector = last_irq
        .get_item("vector")?
        .ok_or_else(|| PyValueError::new_err("snapshot last_irq is missing \"vector\""))?
        .extract::<Option<u32>>()?;

    Ok(LoadedInterruptSnapshot {
        imr: interrupts.getattr("imr")?.extract::<u8>()?,
        isr: interrupts.getattr("isr")?.extract::<u8>()?,
        pending: interrupts.getattr("pending")?.extract::<bool>()?,
        in_interrupt: interrupts.getattr("in_interrupt")?.extract::<bool>()?,
        source: interrupts.getattr("source")?.extract::<Option<String>>()?,
        stack: interrupts.getattr("stack")?.extract::<Vec<u32>>()?,
        next_id: interrupts.getattr("next_id")?.extract::<u32>()?,
        irq_total: required_irq_count(&counts, "total")?,
        irq_key: required_irq_count(&counts, "KEY")?,
        irq_mti: required_irq_count(&counts, "MTI")?,
        irq_sti: required_irq_count(&counts, "STI")?,
        last_irq_src,
        last_irq_pc,
        last_irq_vector,
    })
}

fn validate_bridge_ranges(name: &str, ranges: &[(u32, u32)]) -> PyResult<()> {
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(format!(
                "{name} contains an inverted range 0x{start:06X}..0x{end:06X}"
            )));
        }
        if end > ADDRESS_MASK {
            return Err(PyValueError::new_err(format!(
                "{name} range 0x{start:06X}..0x{end:06X} exceeds the 24-bit bridge address space"
            )));
        }
    }
    Ok(())
}

fn extract_bridge_memory_image(py: Python<'_>, memory: &Py<PyAny>) -> PyResult<MemoryImage> {
    let bound = memory.bind(py);
    let exported = bound
        .call_method0("export_flat_memory")
        .map_err(|err| PyRuntimeError::new_err(format!("export_flat_memory: {err}")))?;
    let exported = exported
        .downcast::<PyTuple>()
        .map_err(|err| PyTypeError::new_err(format!("export_flat_memory result: {err}")))?;
    if exported.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "export_flat_memory returned {} items; expected exactly 3",
            exported.len()
        )));
    }

    let flat_item = exported.get_item(0)?;
    let flat_bytes = flat_item
        .downcast::<PyBytes>()
        .map_err(|err| PyTypeError::new_err(format!("flattened memory: {err}")))?
        .as_bytes();
    if flat_bytes.len() != EXTERNAL_SPACE {
        return Err(PyValueError::new_err(format!(
            "flattened memory length mismatch (got {}, expected {EXTERNAL_SPACE})",
            flat_bytes.len()
        )));
    }

    let fallback_ranges: Vec<(u32, u32)> = exported
        .get_item(1)?
        .extract()
        .map_err(|err| PyTypeError::new_err(format!("fallback ranges: {err}")))?;
    let readonly_ranges: Vec<(u32, u32)> = exported
        .get_item(2)?
        .extract()
        .map_err(|err| PyTypeError::new_err(format!("readonly ranges: {err}")))?;
    validate_bridge_ranges("fallback ranges", &fallback_ranges)?;
    validate_bridge_ranges("readonly ranges", &readonly_ranges)?;

    let imem_obj = bound
        .call_method0("get_internal_memory_bytes")
        .map_err(|err| PyRuntimeError::new_err(format!("get_internal_memory_bytes: {err}")))?;
    let imem_bytes = imem_obj
        .downcast::<PyBytes>()
        .map_err(|err| PyTypeError::new_err(format!("internal memory: {err}")))?
        .as_bytes();
    if imem_bytes.len() != INTERNAL_SPACE {
        return Err(PyValueError::new_err(format!(
            "internal memory length mismatch (got {}, expected {INTERNAL_SPACE})",
            imem_bytes.len()
        )));
    }

    // Construct the complete replacement off to the side. None of the Python
    // extraction or validation above can leave the active mirror half-replaced.
    let mut candidate = MemoryImage::new();
    candidate
        .copy_external_from(flat_bytes)
        .map_err(|err| PyRuntimeError::new_err(format!("copy external: {err}")))?;
    candidate.load_internal(imem_bytes);
    candidate.set_python_ranges(fallback_ranges);
    candidate.set_readonly_ranges(readonly_ranges);
    Ok(candidate)
}

fn populate_live_timer_snapshot_metadata(
    metadata: &mut SnapshotMetadata,
    timer: &TimerContext,
    image: &MemoryImage,
) {
    let (timer_info, mut interrupt_info) = timer.snapshot_info();
    if let Some(imr) = image.read_internal_byte(IMEM_IMR_OFFSET) {
        interrupt_info.imr = imr;
    }
    if let Some(isr) = image.read_internal_byte(IMEM_ISR_OFFSET) {
        interrupt_info.isr = isr;
    }
    metadata.timer = timer_info;
    metadata.interrupts = interrupt_info;
}

fn py_json_string(py: Python<'_>, value: PyObject, label: &str) -> PyResult<String> {
    py.import_bound("json")?
        .call_method1("dumps", (value,))?
        .extract::<String>()
        .map_err(|err| PyTypeError::new_err(format!("{label}: {err}")))
}

fn parse_power_state(value: &str) -> PyResult<PowerState> {
    match value.trim().to_ascii_lowercase().as_str() {
        "running" => Ok(PowerState::Running),
        "halted" => Ok(PowerState::Halted),
        "off" => Ok(PowerState::Off),
        _ => Err(PyValueError::new_err(format!(
            "unknown power state {value:?}; expected running, halted, or off"
        ))),
    }
}

fn validate_call_metrics(
    call_depth: u32,
    call_sub_level: u32,
    call_stack: Vec<u32>,
    call_page_stack: Vec<u32>,
    call_return_widths: Vec<u8>,
) -> PyResult<CallMetricsSnapshot> {
    if call_stack.len() != call_return_widths.len() {
        return Err(PyValueError::new_err(
            "snapshot call stack and return-width stack have different lengths",
        ));
    }
    if call_stack.iter().any(|address| *address > ADDRESS_MASK) {
        return Err(PyValueError::new_err(
            "snapshot call stack contains an address above 20 bits",
        ));
    }
    if call_page_stack
        .iter()
        .any(|page| *page > ADDRESS_MASK || page & 0xFFFF != 0)
    {
        return Err(PyValueError::new_err(
            "snapshot call page stack contains a noncanonical page",
        ));
    }
    if call_return_widths
        .iter()
        .any(|width| !matches!(*width, 0 | 16 | 24))
    {
        return Err(PyValueError::new_err(
            "snapshot call return-width stack contains an unsupported width",
        ));
    }
    if call_sub_level > i32::MAX as u32 {
        return Err(PyValueError::new_err(
            "snapshot call_sub_level exceeds the signed bridge range",
        ));
    }
    Ok(CallMetricsSnapshot {
        call_stack,
        call_depth,
        call_sub_level,
        call_page_stack,
        call_return_widths,
    })
}

fn validate_scheduler_metadata(timer: &TimerInfo, interrupts: &InterruptInfo) -> PyResult<()> {
    for (name, period) in [("MTI", timer.mti_period), ("STI", timer.sti_period)] {
        if timer.enabled && period == 0 {
            return Err(PyValueError::new_err(format!(
                "enabled snapshot timer has zero {name} period"
            )));
        }
        if period >= (1u64 << 63) {
            return Err(PyValueError::new_err(format!(
                "snapshot {name} period is ambiguous under wrapping deadline order"
            )));
        }
    }
    for (name, fired, cycle) in [
        (
            "MTI",
            timer.fired_mti_since_boundary,
            timer.last_mti_fire_cycle,
        ),
        (
            "STI",
            timer.fired_sti_since_boundary,
            timer.last_sti_fire_cycle,
        ),
    ] {
        if fired != cycle.is_some() {
            return Err(PyValueError::new_err(format!(
                "snapshot {name} phase flag/fire cycle disagree"
            )));
        }
    }
    if let Some(source) = interrupts.source.as_deref() {
        if !matches!(
            source,
            "RX" | "EX" | "TX" | "ONK" | "KEY" | "STI" | "MTI" | "IR" | "IRQ"
        ) {
            return Err(PyValueError::new_err(format!(
                "interrupt snapshot has unknown source {source:?}"
            )));
        }
    }
    if interrupts
        .stack
        .iter()
        .max()
        .is_some_and(|maximum| interrupts.next_id <= *maximum)
    {
        return Err(PyValueError::new_err(
            "interrupt snapshot next_id must exceed every active flow id",
        ));
    }
    if interrupts.pending && interrupts.isr == 0 {
        return Err(PyValueError::new_err(
            "interrupt snapshot cannot be pending with ISR == 0",
        ));
    }
    if interrupts.pending && interrupts.source.is_none() {
        return Err(PyValueError::new_err(
            "interrupt snapshot cannot be pending without a source",
        ));
    }
    if interrupts.pending && !interrupts.in_interrupt {
        if let Some(mask) = interrupts
            .source
            .as_deref()
            .and_then(|source| match source {
                "RX" => Some(0x20),
                "EX" => Some(0x40),
                "TX" => Some(0x10),
                "ONK" => Some(0x08),
                "KEY" => Some(0x04),
                "STI" => Some(0x02),
                "MTI" => Some(0x01),
                _ => None,
            })
        {
            if interrupts.isr & mask == 0 {
                return Err(PyValueError::new_err(
                    "interrupt snapshot pending source disagrees with ISR",
                ));
            }
        }
    }
    if let Some(counts) = interrupts.irq_counts.as_ref() {
        let counts = counts
            .as_object()
            .ok_or_else(|| PyTypeError::new_err("snapshot irq_counts must be an object"))?;
        if counts.len() != 4 {
            return Err(PyValueError::new_err(
                "snapshot irq_counts must contain exactly total/KEY/MTI/STI",
            ));
        }
        for name in ["total", "KEY", "MTI", "STI"] {
            let value = counts
                .get(name)
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "snapshot irq_counts is missing unsigned integer {name:?}"
                    ))
                })?;
            if value > u64::from(u32::MAX) {
                return Err(PyValueError::new_err(format!(
                    "snapshot irq_counts {name:?} exceeds u32"
                )));
            }
        }
    }
    if let Some(last_irq) = interrupts.last_irq.as_ref() {
        let last_irq = last_irq
            .as_object()
            .ok_or_else(|| PyTypeError::new_err("snapshot last_irq must be an object"))?;
        if last_irq.len() != 3
            || !["src", "pc", "vector"]
                .iter()
                .all(|name| last_irq.contains_key(*name))
        {
            return Err(PyValueError::new_err(
                "snapshot last_irq must contain exactly src/pc/vector",
            ));
        }
        for name in ["pc", "vector"] {
            if let Some(value) = last_irq.get(name).filter(|value| !value.is_null()) {
                let value = value.as_u64().ok_or_else(|| {
                    PyTypeError::new_err(format!(
                        "snapshot last_irq {name:?} must be an unsigned integer or null"
                    ))
                })?;
                if value > u64::from(ADDRESS_MASK) {
                    return Err(PyValueError::new_err(format!(
                        "snapshot last_irq {name:?} exceeds the 20-bit address space"
                    )));
                }
            }
        }
        if let Some(source) = last_irq.get("src").filter(|value| !value.is_null()) {
            if !source.is_string() {
                return Err(PyTypeError::new_err(
                    "snapshot last_irq \"src\" must be a string or null",
                ));
            }
        }
    }
    if let Some(watch) = interrupts.irq_bit_watch.as_ref() {
        let watch = watch
            .as_object()
            .ok_or_else(|| PyTypeError::new_err("snapshot irq_bit_watch must be an object"))?;
        if watch.len() != 2 || !["IMR", "ISR"].iter().all(|name| watch.contains_key(*name)) {
            return Err(PyValueError::new_err(
                "snapshot irq_bit_watch must contain exactly IMR/ISR",
            ));
        }
        for register in ["IMR", "ISR"] {
            let bits = watch[register].as_object().ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "snapshot irq_bit_watch {register} must be an object"
                ))
            })?;
            if bits.len() != 8 {
                return Err(PyValueError::new_err(format!(
                    "snapshot irq_bit_watch {register} must contain exactly eight bits"
                )));
            }
            for bit in 0..8 {
                let actions = bits
                    .get(&bit.to_string())
                    .and_then(JsonValue::as_object)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "snapshot irq_bit_watch {register} is missing bit {bit}"
                        ))
                    })?;
                if actions.len() != 2
                    || !["set", "clear"]
                        .iter()
                        .all(|name| actions.get(*name).is_some_and(JsonValue::is_array))
                {
                    return Err(PyValueError::new_err(format!(
                        "snapshot irq_bit_watch {register}.{bit} must contain set/clear arrays"
                    )));
                }
                for action in ["set", "clear"] {
                    for pc in actions[action].as_array().expect("validated array") {
                        if pc.as_u64().is_none_or(|pc| pc > u64::from(ADDRESS_MASK)) {
                            return Err(PyValueError::new_err(format!(
                                "snapshot irq_bit_watch {register}.{bit}.{action} contains an invalid PC"
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn keyboard_snapshot_candidate(
    py: Python<'_>,
    snapshot: PyObject,
    current: &KeyboardMatrix,
) -> PyResult<KeyboardMatrix> {
    // Python persists KeyboardHandler state with the matrix nested under
    // ``matrix``; native snapshots persist KeyboardSnapshot itself.
    // Normalize the two losslessly before serde validates the canonical
    // native shape.
    let encoded = py_json_string(py, snapshot, "keyboard snapshot")?;
    let raw: JsonValue = serde_json::from_str(&encoded)
        .map_err(|err| PyValueError::new_err(format!("keyboard snapshot: {err}")))?;
    let mut normalized = match raw {
        JsonValue::Object(mut wrapper) => wrapper
            .remove("matrix")
            .unwrap_or(JsonValue::Object(wrapper)),
        _ => return Err(PyTypeError::new_err("keyboard snapshot must be an object")),
    };
    let matrix = normalized
        .as_object_mut()
        .ok_or_else(|| PyTypeError::new_err("keyboard snapshot matrix must be an object"))?;
    if !matrix.contains_key("fifo_len") {
        let head = matrix
            .get("head")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| {
                PyValueError::new_err("keyboard snapshot is missing unsigned integer \"head\"")
            })?;
        let tail = matrix
            .get("tail")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| {
                PyValueError::new_err("keyboard snapshot is missing unsigned integer \"tail\"")
            })?;
        if head >= 8 || tail >= 8 {
            return Err(PyValueError::new_err(
                "keyboard snapshot FIFO head/tail must be below 8",
            ));
        }
        matrix.insert("fifo_len".to_string(), json!((tail + 8 - head) % 8));
    }
    if !matrix.contains_key("active_columns") {
        let kol = matrix
            .get("kol")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| {
                PyValueError::new_err("keyboard snapshot is missing unsigned integer \"kol\"")
            })? as u8;
        let koh = matrix
            .get("koh")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| {
                PyValueError::new_err("keyboard snapshot is missing unsigned integer \"koh\"")
            })? as u8;
        let active_high = matrix
            .get("columns_active_high")
            .and_then(JsonValue::as_bool)
            .ok_or_else(|| {
                PyValueError::new_err(
                    "keyboard snapshot is missing boolean \"columns_active_high\"",
                )
            })?;
        let active: Vec<u8> = (0..16u8)
            .filter(|column| {
                let bit = if *column < 8 {
                    (kol >> *column) & 1
                } else {
                    (koh >> (*column - 8)) & 1
                };
                (bit == 1) == active_high
            })
            .collect();
        matrix.insert("active_columns".to_string(), json!(active));
    }
    let candidate: KeyboardSnapshot = serde_json::from_value(normalized)
        .map_err(|err| PyValueError::new_err(format!("keyboard snapshot: {err}")))?;
    if candidate.fifo.len() != 8
        || candidate.fifo_len > 8
        || candidate.head >= 8
        || candidate.tail >= 8
        || candidate.column_histogram.len() != 16
    {
        return Err(PyValueError::new_err(
            "keyboard snapshot has invalid FIFO or column-histogram shape",
        ));
    }

    // Load into an off-side clone, then ensure no field was silently clamped,
    // recomputed inconsistently, or discarded. Both validation and restore
    // use this exact candidate builder.
    let expected = to_value(&candidate)
        .map_err(|err| PyRuntimeError::new_err(format!("keyboard snapshot: {err}")))?;
    let mut keyboard_candidate = current.clone();
    keyboard_candidate
        .load_snapshot_state(&candidate)
        .map_err(PyValueError::new_err)?;
    let restored = to_value(keyboard_candidate.snapshot_state())
        .map_err(|err| PyRuntimeError::new_err(format!("keyboard snapshot: {err}")))?;
    if restored != expected {
        return Err(PyValueError::new_err(
            "keyboard snapshot is not exactly representable by the native matrix",
        ));
    }
    Ok(keyboard_candidate)
}

fn scheduler_snapshot_candidate(
    py: Python<'_>,
    timer: PyObject,
    interrupts: PyObject,
    current_cycle: u64,
    current: &TimerContext,
) -> PyResult<(TimerContext, InterruptInfo)> {
    let timer_json = py_json_string(py, timer, "timer snapshot")?;
    let interrupt_json = py_json_string(py, interrupts, "interrupt snapshot")?;
    let timer_metadata: TimerInfo = serde_json::from_str(&timer_json)
        .map_err(|err| PyValueError::new_err(format!("timer snapshot: {err}")))?;
    let interrupt_metadata: InterruptInfo = serde_json::from_str(&interrupt_json)
        .map_err(|err| PyValueError::new_err(format!("interrupt snapshot: {err}")))?;
    validate_scheduler_metadata(&timer_metadata, &interrupt_metadata)?;

    let mut candidate = current.clone();
    candidate.apply_snapshot_info(&timer_metadata, &interrupt_metadata, current_cycle);
    Ok((candidate, interrupt_metadata))
}

#[pymethods]
impl LlamaCpu {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true, timer_scale = 1.0))]
    fn new(memory: PyObject, reset_on_init: bool, timer_scale: f64) -> PyResult<Self> {
        let (read_callback, write_callback, read_with_pc, write_with_pc) =
            Python::with_gil(|py| -> PyResult<_> {
                let read_callback = memory.getattr(py, "read_byte")?;
                let write_callback = memory.getattr(py, "write_byte")?;
                if !read_callback.bind(py).is_callable() || !write_callback.bind(py).is_callable() {
                    return Err(PyTypeError::new_err(
                        "memory read_byte and write_byte attributes must be callable",
                    ));
                }
                let read_with_pc = callback_uses_pc(py, read_callback.bind(py), 2)?;
                let write_with_pc = callback_uses_pc(py, write_callback.bind(py), 3)?;
                Ok((read_callback, write_callback, read_with_pc, write_with_pc))
            })?;
        let mut cpu = Self {
            state: LlamaState::new(),
            executor: LlamaExecutor::new(),
            memory,
            read_callback,
            write_callback,
            read_with_pc,
            write_with_pc,
            call_sub_level: 0,
            temps: HashMap::new(),
            mirror: MemoryImage::new(),
            keyboard: KeyboardMatrix::new(),
            timer: TimerContext::new(false, 0, 0),
            memory_synced: false,
            memory_reads: 0,
            memory_writes: 0,
            cycles: 0,
            poisoned: None,
            uncertain_host_writes: Vec::new(),
            pending_vector_transfer: None,
            pending_scheduled_opcode: None,
        };
        cpu.timer.set_timer_scale(timer_scale);
        if reset_on_init {
            Python::with_gil(|py| cpu.prepare_immediate_machine_reset_transfer(py))?;
            cpu.power_on_reset()?;
        }
        Ok(cpu)
    }

    /// Prepare RESET during construction with no scheduler gap. The generic
    /// public preparation API additionally requires callback-stability metadata
    /// because its proof may be retained across host work; construction fetches
    /// and consumes this proof immediately.
    fn prepare_immediate_machine_reset_transfer(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.pending_vector_transfer.is_some() {
            return Err(PyRuntimeError::new_err(
                "an SC62015 vector transfer is already prepared",
            ));
        }
        let has_wait_cycles = self.memory.bind(py).hasattr("wait_cycles")?;
        let source_pc = self.state.pc() & ADDRESS_MASK;
        let mut bus = LlamaPyBus::new(
            py,
            &self.memory,
            self.read_callback.clone_ref(py),
            self.write_callback.clone_ref(py),
            self.read_with_pc,
            self.write_with_pc,
            source_pc,
            has_wait_cycles,
            &mut self.timer,
            &mut self.keyboard,
            &mut self.mirror,
            &mut self.cycles,
        )?;
        let transfer_result = fetch_validated_vector(ROM_RESET_VECTOR_ADDR, &self.state, &mut bus);
        self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
        let callback_error = bus.take_callback_error();
        drop(bus);
        if let Some(error) = callback_error {
            return self.poison_callback_error(error);
        }
        let transfer = match transfer_result {
            Ok(transfer) => transfer,
            Err(error) => {
                let reason = format!("prepare SC62015 machine reset: {error}");
                if self.poisoned.is_none() {
                    self.poisoned = Some(reason.clone());
                }
                return Err(PyRuntimeError::new_err(reason));
            }
        };
        self.pending_vector_transfer = Some((transfer, "machine_reset".to_string()));
        Ok(())
    }

    fn sync_temps_from_state(&mut self) {
        for idx in 0..u32::from(NUM_TEMP_REGISTERS) {
            let reg = LlamaRegName::Temp(idx as u8);
            let val = self.state.get_reg(reg) & 0xFF_FFFF;
            self.temps.insert(idx, val);
        }
    }

    fn apply_temps_to_state(&mut self) {
        for idx in 0..u32::from(NUM_TEMP_REGISTERS) {
            let val = self.temps.get(&idx).copied().unwrap_or(0);
            let reg = LlamaRegName::Temp(idx as u8);
            self.state.set_reg(reg, val);
        }
    }

    #[pyo3(signature = (vector_address, source_pc, *, require_immutable = false, scope = "instruction"))]
    fn prepare_vector_transfer(
        &mut self,
        py: Python<'_>,
        vector_address: u32,
        source_pc: u32,
        require_immutable: bool,
        scope: &str,
    ) -> PyResult<u32> {
        if vector_address > ADDRESS_MASK {
            return Err(PyValueError::new_err(
                "SC62015 vector address must be canonical 20-bit",
            ));
        }
        if !matches!(scope, "instruction" | "machine_reset") {
            return Err(PyValueError::new_err(
                "SC62015 prepared vector scope must be instruction or machine_reset",
            ));
        }
        if self.pending_vector_transfer.is_some() {
            return Err(PyRuntimeError::new_err(
                "an SC62015 vector transfer is already prepared",
            ));
        }
        let stability_method = if require_immutable {
            "instruction_byte_is_immutable"
        } else {
            "instruction_byte_is_callback_free"
        };
        require_python_instruction_stability(
            py,
            &self.memory,
            stability_method,
            (0..3).map(|index| vector_address.wrapping_add(index) & ADDRESS_MASK),
        )?;

        let has_wait_cycles = self.memory.bind(py).hasattr("wait_cycles")?;
        let mut bus = LlamaPyBus::new(
            py,
            &self.memory,
            self.read_callback.clone_ref(py),
            self.write_callback.clone_ref(py),
            self.read_with_pc,
            self.write_with_pc,
            source_pc & ADDRESS_MASK,
            has_wait_cycles,
            &mut self.timer,
            &mut self.keyboard,
            &mut self.mirror,
            &mut self.cycles,
        )?;
        let mut source_state = self.state.clone();
        source_state.set_pc(source_pc & ADDRESS_MASK);
        let silent_result =
            validate_vector_transfer_with_length(vector_address, &source_state, &mut bus);
        let callback_error = bus.take_callback_error();
        drop(bus);
        if let Some(error) = callback_error {
            return Err(error);
        }
        let (target, target_len) = silent_result.map_err(|error| {
            PyRuntimeError::new_err(format!("prepare SC62015 vector transfer: {error}"))
        })?;
        require_python_instruction_stability(
            py,
            &self.memory,
            stability_method,
            (0..u32::from(target_len)).map(|index| target.wrapping_add(index) & ADDRESS_MASK),
        )?;

        // Re-certify both ranges immediately before binding the proof to the
        // live provenance epoch. IR retains a silent-only proof because its
        // architectural vector fetch occurs after the frame writes. RESET
        // keeps its fetched-before-mutation fail-closed contract.
        require_python_instruction_stability(
            py,
            &self.memory,
            stability_method,
            (0..3)
                .map(|index| vector_address.wrapping_add(index) & ADDRESS_MASK)
                .chain(
                    (0..u32::from(target_len))
                        .map(|index| target.wrapping_add(index) & ADDRESS_MASK),
                ),
        )?;
        let mut bus = LlamaPyBus::new(
            py,
            &self.memory,
            self.read_callback.clone_ref(py),
            self.write_callback.clone_ref(py),
            self.read_with_pc,
            self.write_with_pc,
            source_pc & ADDRESS_MASK,
            has_wait_cycles,
            &mut self.timer,
            &mut self.keyboard,
            &mut self.mirror,
            &mut self.cycles,
        )?;
        let transfer_result = if vector_address == ROM_RESET_VECTOR_ADDR {
            fetch_validated_vector(vector_address, &source_state, &mut bus)
        } else {
            prepare_validated_vector(vector_address, &source_state, &mut bus)
        };
        self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
        let callback_error = bus.take_callback_error();
        drop(bus);
        if let Some(error) = callback_error {
            return self.poison_callback_error(error);
        }
        let transfer = match transfer_result {
            Ok(transfer) => transfer,
            Err(error) => {
                let reason = format!("prepare SC62015 vector transfer: {error}");
                if self.poisoned.is_none() {
                    self.poisoned = Some(reason.clone());
                }
                return Err(PyRuntimeError::new_err(reason));
            }
        };
        if transfer.target() != target || transfer.target_len() != target_len {
            let reason = "SC62015 vector target changed after stability certification".to_string();
            if self.poisoned.is_none() {
                self.poisoned = Some(reason.clone());
            }
            return Err(PyRuntimeError::new_err(reason));
        }
        let prepared_target = transfer.target();
        self.pending_vector_transfer = Some((transfer, scope.to_string()));
        Ok(prepared_target)
    }

    fn cancel_prepared_vector_transfer(&mut self) {
        self.pending_vector_transfer = None;
        self.pending_scheduled_opcode = None;
    }

    fn prepare_scheduled_opcode(&mut self, address: u32, opcode: u8) -> PyResult<()> {
        if self.pending_scheduled_opcode.is_some() {
            return Err(PyRuntimeError::new_err(
                "an SC62015 scheduled opcode is already prepared",
            ));
        }
        self.pending_scheduled_opcode = Some((address & ADDRESS_MASK, opcode));
        Ok(())
    }

    fn power_on_reset(&mut self) -> PyResult<()> {
        let prepared_transfer = match self.pending_vector_transfer.take() {
            Some((transfer, scope)) if scope == "machine_reset" => transfer,
            Some((_transfer, _scope)) => {
                return Err(PyRuntimeError::new_err(
                    "prepared SC62015 vector transfer has the wrong scope for machine reset",
                ));
            }
            None => {
                return Err(PyRuntimeError::new_err(
                    "power-on reset requires a prepared SC62015 machine_reset vector transfer",
                ));
            }
        };
        self.pending_scheduled_opcode = None;
        let rollback_before = self.capture_callback_rollback_state();
        self.mirror.begin_rollback_capture();
        // Apply RESET intrinsic semantics using Python memory for reads/writes so the
        // reset vector and IMEM updates match the Python emulator, while keeping the
        // mirror in sync.
        let reset_result = Python::with_gil(|py| {
            struct ResetBus<'py, 'a> {
                py: Python<'py>,
                memory: Py<PyAny>,
                read_callback: Py<PyAny>,
                write_callback: Py<PyAny>,
                preflight_read_callback: Option<Py<PyAny>>,
                read_with_pc: bool,
                write_with_pc: bool,
                preflight_read_with_pc: bool,
                pc: u32,
                mirror: &'a mut MemoryImage,
                callback_error: Option<PyErr>,
                provenance_error: RefCell<Option<PyErr>>,
                provenance_failure_nonce: Cell<u64>,
                write_attempts: Vec<u32>,
            }

            impl<'py, 'a> ResetBus<'py, 'a> {
                fn record_callback_error(&mut self, err: PyErr) {
                    if self.callback_error.is_none() {
                        self.callback_error = Some(err);
                    }
                }

                fn read_byte(&mut self, addr: u32) -> u8 {
                    if self.callback_error.is_some() {
                        return 0;
                    }
                    let addr = addr & ADDRESS_MASK;
                    let result = if self.read_with_pc {
                        self.read_callback.bind(self.py).call1((addr, self.pc))
                    } else {
                        self.read_callback.bind(self.py).call1((addr,))
                    };
                    match result.and_then(|obj| obj.extract::<u8>()) {
                        Ok(value) => value,
                        Err(err) => {
                            self.record_callback_error(err);
                            0
                        }
                    }
                }

                fn write_byte(&mut self, addr: u32, value: u8) {
                    if self.callback_error.is_some() {
                        return;
                    }
                    let addr = addr & ADDRESS_MASK;
                    self.write_attempts.push(addr);
                    let result = if self.write_with_pc {
                        self.write_callback
                            .bind(self.py)
                            .call1((addr, value, self.pc))
                    } else {
                        self.write_callback.bind(self.py).call1((addr, value))
                    };
                    match result {
                        Ok(_) => {
                            self.mirror.sync_committed_host_write(addr, value);
                        }
                        Err(err) => self.record_callback_error(err),
                    }
                }

                fn peek_byte_for_preflight_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
                    if self.callback_error.is_some() {
                        return None;
                    }
                    let addr = addr & ADDRESS_MASK;
                    let callback = self.preflight_read_callback.as_ref()?;
                    let callback = callback.bind(self.py);
                    let result = if self.preflight_read_with_pc {
                        callback.call1((addr, context_pc & ADDRESS_MASK))
                    } else {
                        callback.call1((addr,))
                    }
                    .and_then(|value| value.extract::<u8>());
                    match result {
                        Ok(value) => Some(value),
                        Err(err) => {
                            self.record_callback_error(err);
                            None
                        }
                    }
                }

                fn peek_byte_for_preflight(&mut self, addr: u32) -> Option<u8> {
                    self.peek_byte_for_preflight_at(addr, self.pc)
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

                fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                    self.peek_byte_for_preflight(addr)
                }

                fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
                    self.peek_byte_for_preflight_at(addr, context_pc)
                }

                fn vector_transfer_provenance(&self) -> (usize, u64) {
                    match python_vector_provenance(self.py, &self.memory) {
                        Ok(provenance) => provenance,
                        Err(error) => {
                            if self.provenance_error.borrow().is_none() {
                                *self.provenance_error.borrow_mut() = Some(error);
                            }
                            let nonce = self.provenance_failure_nonce.get().wrapping_add(1);
                            self.provenance_failure_nonce.set(nonce);
                            (usize::MAX, nonce)
                        }
                    }
                }

                fn peek_imem(&mut self, offset: u32) -> u8 {
                    self.read_byte(INTERNAL_MEMORY_START + offset)
                }

                fn peek_imem_silent(&mut self, offset: u32) -> u8 {
                    self.peek_byte_for_preflight(INTERNAL_MEMORY_START + offset)
                        .unwrap_or(0)
                }
            }

            let mut bus = ResetBus {
                py,
                memory: self.memory.clone_ref(py),
                read_callback: self.read_callback.clone_ref(py),
                write_callback: self.write_callback.clone_ref(py),
                preflight_read_callback: None,
                read_with_pc: self.read_with_pc,
                write_with_pc: self.write_with_pc,
                preflight_read_with_pc: false,
                pc: self.state.pc() & ADDRESS_MASK,
                mirror: &mut self.mirror,
                callback_error: None,
                provenance_error: RefCell::new(None),
                provenance_failure_nonce: Cell::new(0),
                write_attempts: Vec::new(),
            };
            let semantic_result =
                power_on_reset_with_transfer(&mut bus, &mut self.state, prepared_transfer);
            let callback_error = bus
                .callback_error
                .take()
                .or_else(|| bus.provenance_error.borrow_mut().take());
            let (result, semantic_failure) = match callback_error {
                Some(err) => (Err(err), false),
                None => match semantic_result {
                    Ok(()) => (Ok(()), false),
                    Err(error) => (
                        Err(PyRuntimeError::new_err(format!(
                            "power-on reset vector transfer: {error}"
                        ))),
                        true,
                    ),
                },
            };
            (result, bus.write_attempts, semantic_failure)
        });
        let (reset_result, reset_write_attempts, semantic_failure) = reset_result;
        if let Err(err) = reset_result {
            self.restore_after_callback_failure(rollback_before);
            if semantic_failure {
                return Err(err);
            }
            self.record_uncertain_host_writes(reset_write_attempts.iter().copied());
            return self.poison_callback_error(err);
        }

        // A Python write callback may commit a byte and then raise. Native
        // rollback is still required for atomic CPU state, but the host is the
        // only authority for those uncertain addresses. A recovery RESET must
        // reconcile them before it is allowed to clear the poison.
        if let Err(err) = Python::with_gil(|py| self.resync_uncertain_host_writes(py)) {
            self.record_uncertain_host_writes(reset_write_attempts.iter().copied());
            self.restore_after_callback_failure(rollback_before);
            return self.poison_callback_error(err);
        }

        let mut flush_write_attempts = Vec::new();
        if let Err(err) =
            Python::with_gil(|py| self.sync_mirror_tracking(py, &mut flush_write_attempts))
        {
            self.record_uncertain_host_writes(reset_write_attempts.iter().copied());
            self.record_uncertain_host_writes(flush_write_attempts);
            self.restore_after_callback_failure(rollback_before);
            return self.poison_callback_error(err);
        }
        // Reset the process-global trace clock only after every fallible host
        // reconciliation and mirror flush has succeeded. A failed recovery
        // must leave both architectural and trace accounting untouched.
        reset_perf_counters();
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
        self.memory_synced = true;
        self.poisoned = None;
        self.uncertain_host_writes.clear();
        self.mirror.commit_rollback_capture();
        Ok(())
    }

    fn execute_instruction(&mut self, py: Python<'_>, address: u32) -> PyResult<(u8, u8)> {
        if let Some(reason) = &self.poisoned {
            return Err(PyRuntimeError::new_err(format!(
                "LLAMA CPU is poisoned after a Python callback failure; reset required: {reason}"
            )));
        }
        // Detach before any fallible callback/decode. Every return path then
        // burns the one-shot proof instead of leaving a stale capability in
        // the CPU for a later instruction or recovery RESET.
        let pending_transfer = self.pending_vector_transfer.take();
        let pending_opcode = self.pending_scheduled_opcode.take();
        if pending_transfer
            .as_ref()
            .is_some_and(|(_, scope)| scope != "instruction")
        {
            return Err(PyRuntimeError::new_err(
                "prepared SC62015 vector transfer has the wrong scope",
            ));
        }
        let rollback_before = self.capture_callback_rollback_state();
        self.mirror.begin_rollback_capture();
        let entry_pc = address & ADDRESS_MASK;
        let has_wait_cycles_result = {
            let bound = self.memory.bind(py);
            bound.hasattr("wait_cycles")
        };
        let has_wait_cycles = match has_wait_cycles_result {
            Ok(value) => value,
            Err(err) => {
                self.restore_after_callback_failure(rollback_before);
                return self.poison_callback_error(err);
            }
        };
        let bus_result = LlamaPyBus::new(
            py,
            &self.memory,
            self.read_callback.clone_ref(py),
            self.write_callback.clone_ref(py),
            self.read_with_pc,
            self.write_with_pc,
            entry_pc,
            has_wait_cycles,
            &mut self.timer,
            &mut self.keyboard,
            &mut self.mirror,
            &mut self.cycles,
        );
        let mut bus = match bus_result {
            Ok(bus) => bus,
            Err(err) => {
                self.restore_after_callback_failure(rollback_before);
                return self.poison_callback_error(err);
            }
        };
        self.state.set_pc(entry_pc);
        // A raw/direct caller has no one-shot scheduling proof. Classify the
        // opcode through the explicitly side-effect-free reader before the
        // sole architectural fetch, so FE/FF cannot consume a callback read
        // and then return a retryable "missing prepared transfer" error.
        let direct_preflight_opcode = if pending_opcode.is_none() && pending_transfer.is_none() {
            let opcode = bus.peek_byte_silent_at(entry_pc, entry_pc);
            if let Some(err) = bus.take_callback_error() {
                drop(bus);
                self.restore_after_callback_failure(rollback_before);
                return Err(err);
            }
            let opcode = match opcode {
                Some(opcode) => opcode,
                None => {
                    drop(bus);
                    self.restore_after_callback_failure(rollback_before);
                    return Err(PyRuntimeError::new_err(
                        "direct SC62015 execution requires side-effect-free opcode preflight",
                    ));
                }
            };
            if matches!(opcode, 0xFE | 0xFF) {
                drop(bus);
                self.restore_after_callback_failure(rollback_before);
                return Err(PyRuntimeError::new_err(
                    "SC62015 vector opcode requires a prepared instruction vector transfer",
                ));
            }
            Some(opcode)
        } else {
            None
        };
        let opcode = match pending_opcode {
            Some((prepared_pc, opcode)) if prepared_pc == entry_pc => opcode,
            Some((_prepared_pc, _opcode)) => {
                drop(bus);
                self.restore_after_callback_failure(rollback_before);
                return Err(PyRuntimeError::new_err(
                    "prepared SC62015 opcode does not match the current instruction",
                ));
            }
            None => bus.read_byte(entry_pc & ADDRESS_MASK),
        };
        if let Some(err) = bus.take_callback_error() {
            self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
            let write_attempts = bus.take_write_attempts();
            drop(bus);
            self.record_uncertain_host_writes(write_attempts);
            self.restore_after_callback_failure(rollback_before);
            return self.poison_callback_error(err);
        }
        if let Some(preflight_opcode) = direct_preflight_opcode {
            if opcode != preflight_opcode {
                self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
                drop(bus);
                self.restore_after_callback_failure(rollback_before);
                let error = PyRuntimeError::new_err(format!(
                    "architectural opcode changed after silent preflight at 0x{entry_pc:05X}: \
                     fetched 0x{opcode:02X}, preflight 0x{preflight_opcode:02X}"
                ));
                return self.poison_callback_error(error);
            }
        }
        if pending_transfer.is_some() && !matches!(opcode, 0xFE | 0xFF) {
            drop(bus);
            self.restore_after_callback_failure(rollback_before);
            return Err(PyRuntimeError::new_err(
                "prepared SC62015 vector transfer does not match the current instruction",
            ));
        }
        if pending_transfer.is_none() && matches!(opcode, 0xFE | 0xFF) {
            drop(bus);
            self.restore_after_callback_failure(rollback_before);
            return Err(PyRuntimeError::new_err(
                "SC62015 vector opcode requires a prepared instruction vector transfer",
            ));
        }
        let execute_result = self.executor.execute_with_vector_transfer(
            opcode,
            &mut self.state,
            &mut bus,
            pending_transfer.map(|(transfer, _scope)| transfer),
        );
        self.memory_reads = self.memory_reads.saturating_add(bus.memory_reads);
        self.memory_writes = self.memory_writes.saturating_add(bus.memory_writes);
        if let Some(err) = bus.take_callback_error() {
            let write_attempts = bus.take_write_attempts();
            drop(bus);
            self.record_uncertain_host_writes(write_attempts);
            self.restore_after_callback_failure(rollback_before);
            return self.poison_callback_error(err);
        }
        let len = match execute_result {
            Ok(len) => len,
            Err(err) => {
                let write_attempts = bus.take_write_attempts();
                drop(bus);
                let host_writes_may_have_committed = !write_attempts.is_empty();
                self.record_uncertain_host_writes(write_attempts);
                self.restore_after_callback_failure(rollback_before);
                let error = PyRuntimeError::new_err(format!("llama execute: {err}"));
                if host_writes_may_have_committed {
                    return self.poison_callback_error(error);
                }
                return Err(error);
            }
        };
        self.call_sub_level = self.state.call_sub_level() as i32;
        self.sync_temps_from_state();
        let trace_result = if opcode == 0xFE {
            // IR: interrupt entry
            self.read_irq_registers(py, &mut bus)
                .and_then(|(imr, isr)| {
                    self.emit_irq_trace(
                        py,
                        "IRQ_Enter",
                        HashMap::from([
                            ("pc", entry_pc & ADDRESS_MASK),
                            (
                                "vector",
                                self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK,
                            ),
                            ("imr", imr as u32),
                            ("isr", isr as u32),
                        ]),
                    )
                })
        } else if opcode == 0x01 {
            // RETI: interrupt exit
            self.read_irq_registers(py, &mut bus)
                .and_then(|(imr, isr)| {
                    self.emit_irq_trace(
                        py,
                        "IRQ_Return",
                        HashMap::from([
                            ("pc", entry_pc & ADDRESS_MASK),
                            ("ret", self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK),
                            ("imr", imr as u32),
                            ("isr", isr as u32),
                        ]),
                    )
                })
        } else {
            Ok(())
        };
        if let Err(err) = trace_result {
            let write_attempts = bus.take_write_attempts();
            drop(bus);
            self.record_uncertain_host_writes(write_attempts);
            self.restore_after_callback_failure(rollback_before);
            return self.poison_callback_error(err);
        }
        drop(bus);
        self.mirror.commit_rollback_capture();
        Ok((opcode, len))
    }

    fn read_register(&self, name: &str) -> PyResult<u32> {
        let upper = name.to_ascii_uppercase();
        if let Some(reg) = llama_reg_from_name(&upper) {
            return Ok(self.state.get_reg(reg));
        }
        if let Some(rest) = upper.strip_prefix("TEMP") {
            if let Ok(idx) = rest.parse::<u32>() {
                if idx < u32::from(NUM_TEMP_REGISTERS) {
                    return Ok(self.state.get_reg(LlamaRegName::Temp(idx as u8)));
                }
            }
        }
        Err(PyValueError::new_err(format!("unknown register {name}")))
    }

    fn write_register(&mut self, name: &str, value: u32) -> PyResult<()> {
        let upper = name.to_ascii_uppercase();
        if let Some(reg) = llama_reg_from_name(&upper) {
            if reg == LlamaRegName::F {
                validate_f_image(value).map_err(PyRuntimeError::new_err)?;
            }
            self.state.set_reg(reg, value);
            return Ok(());
        }
        if let Some(rest) = upper.strip_prefix("TEMP") {
            if let Ok(idx) = rest.parse::<u32>() {
                if idx < u32::from(NUM_TEMP_REGISTERS) {
                    let value = value & ADDRESS_MASK;
                    self.state.set_reg(LlamaRegName::Temp(idx as u8), value);
                    if value != 0 {
                        self.temps.insert(idx, value);
                    } else {
                        self.temps.remove(&idx);
                    }
                    return Ok(());
                }
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
        for idx in 0..u32::from(NUM_TEMP_REGISTERS) {
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
        // CPURegistersSnapshot's complete core shape is required. Extract and
        // validate every fallible field before the first mutation so missing
        // or malformed attributes cannot create a partial native state.
        let pc = snap.getattr("pc")?.extract::<u32>()?;
        let ba = snap.getattr("ba")?.extract::<u32>()?;
        let i = snap.getattr("i")?.extract::<u32>()?;
        let x = snap.getattr("x")?.extract::<u32>()?;
        let y = snap.getattr("y")?.extract::<u32>()?;
        let u = snap.getattr("u")?.extract::<u32>()?;
        let s = snap.getattr("s")?.extract::<u32>()?;
        let f = snap.getattr("f")?.extract::<u32>()?;
        validate_f_image(f).map_err(PyRuntimeError::new_err)?;
        let temps = snap.getattr("temps")?.extract::<HashMap<u32, u32>>()?;
        if let Some(invalid) = temps
            .keys()
            .copied()
            .find(|index| *index >= u32::from(NUM_TEMP_REGISTERS))
        {
            return Err(PyValueError::new_err(format!(
                "snapshot contains unknown temporary register TEMP{invalid}"
            )));
        }
        let raw_call_sub_level = snap.getattr("call_sub_level")?.extract::<i64>()?;
        if !(0..=i64::from(i32::MAX)).contains(&raw_call_sub_level) {
            return Err(PyValueError::new_err(
                "snapshot call_sub_level must be a non-negative signed 32-bit value",
            ));
        }
        let call_sub_level = raw_call_sub_level as i32;
        let interrupts = if snap.hasattr("interrupts")? {
            Some(extract_interrupt_snapshot(&snap.getattr("interrupts")?)?)
        } else {
            None
        };

        self.state.set_reg(LlamaRegName::PC, pc);
        self.state.set_reg(LlamaRegName::BA, ba);
        self.state.set_reg(LlamaRegName::I, i);
        self.state.set_reg(LlamaRegName::X, x);
        self.state.set_reg(LlamaRegName::Y, y);
        self.state.set_reg(LlamaRegName::U, u);
        self.state.set_reg(LlamaRegName::S, s);
        self.state.set_reg(LlamaRegName::F, f);
        self.temps = temps;
        self.apply_temps_to_state();
        self.call_sub_level = call_sub_level;
        self.state.set_call_sub_level(call_sub_level as u32);
        // Restore the optional legacy interrupt extension only after its
        // complete shape has been validated alongside the core snapshot.
        if let Some(interrupts) = interrupts {
            self.timer.irq_imr = interrupts.imr;
            self.timer.irq_isr = interrupts.isr;
            self.timer.irq_pending = interrupts.pending;
            self.timer.in_interrupt = interrupts.in_interrupt;
            self.timer.irq_source = interrupts.source;
            self.timer.interrupt_stack = interrupts.stack;
            self.timer.next_interrupt_id = interrupts.next_id;
            self.timer.irq_total = interrupts.irq_total;
            self.timer.irq_key = interrupts.irq_key;
            self.timer.irq_mti = interrupts.irq_mti;
            self.timer.irq_sti = interrupts.irq_sti;
            self.timer.last_irq_src = interrupts.last_irq_src;
            self.timer.last_irq_pc = interrupts.last_irq_pc;
            self.timer.last_irq_vector = interrupts.last_irq_vector;
            self.mirror
                .write_internal_byte(IMEM_IMR_OFFSET, interrupts.imr);
            self.mirror
                .write_internal_byte(IMEM_ISR_OFFSET, interrupts.isr);
        }
        Ok(())
    }

    fn keyboard_press_matrix_code(&mut self, py: Python<'_>, code: u8) -> PyResult<bool> {
        self.transactional_key_mutation(py, |cpu| {
            let events = cpu.keyboard.inject_matrix_event(
                code & 0x7F,
                false,
                &mut cpu.mirror,
                cpu.timer.kb_irq_enabled,
            );
            // Host injection updates physical matrix and FIFO bookkeeping.
            // The machine scheduler, not this helper, samples selected KIL
            // and asserts raw KEYI at an instruction boundary.
            events > 0
        })
    }

    fn keyboard_press_on_key(&mut self, py: Python<'_>) -> PyResult<bool> {
        self.transactional_key_mutation(py, |cpu| {
            // Emulate ON key: set SSR.ONK and ISR.ONKI, mirror to internal memory.
            let ssr_offset = 0xFF;
            let isr_offset = IMEM_ISR_OFFSET;
            let ssr = cpu.mirror.read_internal_byte(ssr_offset).unwrap_or(0);
            cpu.mirror.write_internal_byte(ssr_offset, ssr | 0x08);
            let isr = cpu.mirror.read_internal_byte(isr_offset).unwrap_or(0);
            let new_isr = isr | 0x08;
            cpu.mirror.write_internal_byte(isr_offset, new_isr);
            // Parity: mirror CoreRuntime press_on_key side-effects so IRQ delivery and tracing match Python.
            cpu.timer
                .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
            cpu.timer.irq_pending = true;
            cpu.timer.irq_source = Some("ONK".to_string());
            cpu.timer.last_fired = cpu.timer.irq_source.clone();
            cpu.timer.irq_isr = cpu
                .mirror
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(cpu.timer.irq_isr);
            cpu.timer.irq_imr = cpu
                .mirror
                .read_internal_byte(IMEM_IMR_OFFSET)
                .unwrap_or(cpu.timer.irq_imr);
            let mut guard = PERFETTO_TRACER.enter();
            guard.with_some(|tracer| {
                let mut payload = HashMap::new();
                payload.insert(
                    "pc".to_string(),
                    AnnotationValue::Pointer(perfetto_last_pc() as u64),
                );
                payload.insert(
                    "imr".to_string(),
                    AnnotationValue::UInt(cpu.timer.irq_imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    AnnotationValue::UInt(cpu.timer.irq_isr as u64),
                );
                payload.insert("src".to_string(), AnnotationValue::Str("ONK".to_string()));
                tracer.record_irq_event("KeyIRQ", payload);
            });
            true
        })
    }

    fn keyboard_release_on_key(&mut self, py: Python<'_>) -> PyResult<()> {
        self.transactional_key_mutation(py, |cpu| {
            let ssr_offset = 0xFF;
            let ssr = cpu.mirror.read_internal_byte(ssr_offset).unwrap_or(0);
            cpu.mirror.write_internal_byte(ssr_offset, ssr & !0x08);
            let isr = cpu
                .mirror
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(cpu.timer.irq_isr);
            cpu.timer.irq_isr = isr;
            if !cpu.timer.in_interrupt
                && cpu.timer.irq_source.as_deref() == Some("ONK")
                && isr & 0x08 == 0
            {
                cpu.timer.irq_source = if isr & 0x20 != 0 {
                    Some("RX".to_string())
                } else if isr & 0x40 != 0 {
                    Some("EX".to_string())
                } else if isr & 0x10 != 0 {
                    Some("TX".to_string())
                } else if isr & 0x04 != 0 {
                    Some("KEY".to_string())
                } else if isr & 0x02 != 0 {
                    Some("STI".to_string())
                } else if isr & 0x01 != 0 {
                    Some("MTI".to_string())
                } else {
                    None
                };
                cpu.timer.irq_pending = cpu.timer.irq_source.is_some();
            }
        })
    }

    fn keyboard_release_matrix_code(&mut self, py: Python<'_>, code: u8) -> PyResult<bool> {
        self.transactional_key_mutation(py, |cpu| {
            let events = cpu.keyboard.inject_matrix_event(
                code & 0x7F,
                true,
                &mut cpu.mirror,
                cpu.timer.kb_irq_enabled,
            );
            // Release changes physical/event state only. It neither
            // acknowledges an existing KEYI bit nor creates a new one.
            events > 0
        })
    }

    fn is_memory_synced(&self) -> bool {
        self.memory_synced
    }

    fn mark_memory_dirty(&mut self) {
        self.memory_synced = false;
    }

    fn _initialise_rust_memory(&mut self, py: Python<'_>) -> PyResult<()> {
        let candidate = extract_bridge_memory_image(py, &self.memory)?;
        self.mirror = candidate;
        self.memory_synced = true;
        Ok(())
    }

    fn save_snapshot(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        // Snapshot capture uses the same exact-shape, candidate-first bridge
        // extraction as live mirror initialization. Never serialize a padded
        // or truncated image after a malformed host export.
        let image = extract_bridge_memory_image(py, &self.memory)?;
        let fallback_ranges = image.python_ranges().to_vec();
        let readonly_ranges = image.readonly_ranges().to_vec();
        let bound = self.memory.bind(py);

        let temps: std::collections::HashMap<String, u32> = (0..u32::from(NUM_TEMP_REGISTERS))
            .map(|idx| {
                let reg = LlamaRegName::Temp(idx as u8);
                (idx.to_string(), self.state.get_reg(reg) & ADDRESS_MASK)
            })
            .collect();
        let kb_state = self.keyboard.snapshot_state();
        let call_metrics = self.state.snapshot_call_metrics();
        let mut metadata = SnapshotMetadata {
            backend: "llama".to_string(),
            instruction_count: perfetto_last_instr_index(),
            cycle_count: self.cycles,
            pc: self.state.get_reg(LlamaRegName::PC) & ADDRESS_MASK,
            memory_image_size: image.external_len(),
            fallback_ranges,
            readonly_ranges,
            memory_dump_pc: 0,
            memory_reads: self.memory_reads,
            memory_writes: self.memory_writes,
            call_depth: self.state.call_depth(),
            call_sub_level: self.state.call_sub_level(),
            call_stack: call_metrics.call_stack,
            call_page_stack: call_metrics.call_page_stack,
            call_return_widths: call_metrics.call_return_widths,
            temps,
            keyboard: Some(to_value(&kb_state).map_err(|err| {
                PyRuntimeError::new_err(format!("serialize keyboard snapshot: {err}"))
            })?),
            kb_metrics: Some(json!({
                "irq_count": kb_state.irq_count,
                "strobe_count": kb_state.strobe_count,
                "column_hist": kb_state.column_histogram,
                "last_cols": kb_state.active_columns,
                "last_kol": kb_state.kol,
                "last_koh": kb_state.koh,
                "kil_reads": kb_state.kil_read_count,
                "kb_irq_enabled": self.timer.kb_irq_enabled,
            })),
            ..SnapshotMetadata::default()
        };
        metadata.power_state = self.state.power_state();
        metadata.onk_level = image
            .read_internal_byte(IMEM_SSR_OFFSET)
            .is_some_and(|ssr| ssr & 0x08 != 0);
        // Capture the complete live timer/interrupt scheduler state. IMR/ISR
        // come from the host-authored image because it is the persisted memory
        // authority; all remaining scheduler fields stay exactly as reported
        // by TimerContext.
        populate_live_timer_snapshot_metadata(&mut metadata, &self.timer, &image);

        let (lcd_meta, lcd_payload) = match capture_lcd_snapshot(py, bound)? {
            Some(pair) => (Some(pair.0), Some(pair.1)),
            None => (None, None),
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

    fn restore_keyboard_snapshot(&mut self, py: Python<'_>, snapshot: PyObject) -> PyResult<()> {
        self.keyboard = keyboard_snapshot_candidate(py, snapshot, &self.keyboard)?;
        Ok(())
    }

    fn validate_keyboard_snapshot(&self, py: Python<'_>, snapshot: PyObject) -> PyResult<()> {
        let _ = keyboard_snapshot_candidate(py, snapshot, &self.keyboard)?;
        Ok(())
    }

    fn restore_scheduler_snapshot(
        &mut self,
        py: Python<'_>,
        timer: PyObject,
        interrupts: PyObject,
        current_cycle: u64,
    ) -> PyResult<()> {
        // Decode and validate both metadata objects before mutating either the
        // native scheduler or its IMR/ISR memory mirror.
        let (candidate, interrupt_candidate) =
            scheduler_snapshot_candidate(py, timer, interrupts, current_cycle, &self.timer)?;
        self.timer = candidate;
        self.mirror.sync_committed_host_write(
            INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
            interrupt_candidate.imr,
        );
        self.mirror.sync_committed_host_write(
            INTERNAL_MEMORY_START + IMEM_ISR_OFFSET,
            interrupt_candidate.isr,
        );
        Ok(())
    }

    fn validate_scheduler_snapshot(
        &self,
        py: Python<'_>,
        timer: PyObject,
        interrupts: PyObject,
        current_cycle: u64,
    ) -> PyResult<()> {
        let _ = scheduler_snapshot_candidate(py, timer, interrupts, current_cycle, &self.timer)?;
        Ok(())
    }

    // The explicit fields form the stable Python snapshot bridge API.
    #[allow(clippy::too_many_arguments)]
    fn synchronize_host_snapshot_state(
        &mut self,
        py: Python<'_>,
        timer: PyObject,
        interrupts: PyObject,
        keyboard: PyObject,
        instruction_count: u64,
        cycle_count: u64,
        memory_reads: u64,
        memory_writes: u64,
    ) -> PyResult<()> {
        // PCE500Emulator owns periodic timer advancement, interrupt delivery,
        // and keyboard scanning when this native CPU is used through the
        // Python facade.  Decode every host-owned shadow field before changing
        // native state so a malformed snapshot boundary cannot leave a
        // partially synchronized executor behind.
        let keyboard_candidate = keyboard_snapshot_candidate(py, keyboard, &self.keyboard)?;
        let (timer_candidate, interrupt_candidate) =
            scheduler_snapshot_candidate(py, timer, interrupts, cycle_count, &self.timer)?;

        self.keyboard = keyboard_candidate;
        self.timer = timer_candidate;
        self.mirror.sync_committed_host_write(
            INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
            interrupt_candidate.imr,
        );
        self.mirror.sync_committed_host_write(
            INTERNAL_MEMORY_START + IMEM_ISR_OFFSET,
            interrupt_candidate.isr,
        );
        self.cycles = cycle_count;
        self.memory_reads = memory_reads;
        self.memory_writes = memory_writes;
        set_perf_instr_counter(instruction_count);
        Ok(())
    }

    // The explicit fields form the stable Python snapshot bridge API.
    #[allow(clippy::too_many_arguments)]
    fn validate_runtime_snapshot(
        &self,
        _cycles: u64,
        _memory_reads: u64,
        _memory_writes: u64,
        call_depth: u32,
        call_sub_level: u32,
        call_stack: Vec<u32>,
        call_page_stack: Vec<u32>,
        call_return_widths: Vec<u8>,
        power_state: &str,
    ) -> PyResult<()> {
        let _ = parse_power_state(power_state)?;
        let _ = validate_call_metrics(
            call_depth,
            call_sub_level,
            call_stack,
            call_page_stack,
            call_return_widths,
        )?;
        Ok(())
    }

    // The explicit fields form the stable Python snapshot bridge API.
    #[allow(clippy::too_many_arguments)]
    fn restore_runtime_snapshot(
        &mut self,
        cycles: u64,
        memory_reads: u64,
        memory_writes: u64,
        call_depth: u32,
        call_sub_level: u32,
        call_stack: Vec<u32>,
        call_page_stack: Vec<u32>,
        call_return_widths: Vec<u8>,
        power_state: &str,
    ) -> PyResult<()> {
        // Parse every fallible field before changing live execution state.
        let power_state = parse_power_state(power_state)?;
        let call_metrics = validate_call_metrics(
            call_depth,
            call_sub_level,
            call_stack,
            call_page_stack,
            call_return_widths,
        )?;
        self.cycles = cycles;
        self.memory_reads = memory_reads;
        self.memory_writes = memory_writes;
        self.state.restore_call_metrics(call_metrics);
        self.call_sub_level = call_sub_level as i32;
        self.state.set_power_state(power_state);
        Ok(())
    }

    fn notify_host_write(&mut self, address: u32, value: u8) -> PyResult<()> {
        self.sync_authoritative_host_write(address, value);
        Ok(())
    }

    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("backend", "llama")?;
        Ok(dict.into_py(py))
    }

    #[getter]
    fn call_depth(&self) -> u32 {
        self.state.call_depth()
    }

    #[setter]
    fn set_call_depth(&mut self, value: u32) {
        self.state.set_call_depth(value);
    }

    #[getter]
    fn call_sub_level(&self) -> i32 {
        self.call_sub_level
    }

    #[setter]
    fn set_call_sub_level(&mut self, value: i32) {
        self.call_sub_level = value;
        self.state.set_call_sub_level(value.max(0) as u32);
    }

    #[getter]
    fn halted(&self) -> bool {
        self.state.is_halted()
    }

    #[setter]
    fn set_halted(&mut self, value: bool) {
        self.state.set_halted(value);
    }

    #[getter]
    fn power_state(&self) -> &'static str {
        match self.state.power_state() {
            PowerState::Running => "running",
            PowerState::Halted => "halted",
            PowerState::Off => "off",
        }
    }

    #[setter]
    fn set_power_state(&mut self, value: &str) -> PyResult<()> {
        let state = parse_power_state(value)?;
        self.state.set_power_state(state);
        Ok(())
    }

    #[pyo3(signature = (path=None))]
    fn set_perfetto_trace(&mut self, path: Option<&str>) -> PyResult<()> {
        if let Some(p) = path {
            reset_perf_counters();
            let tracer = PerfettoTracer::new(PathBuf::from(p));
            let mut guard = PERFETTO_TRACER.enter();
            guard.replace(Some(tracer));
            println!("[perfetto-tracer] started at {}", p);
        } else {
            let mut guard = PERFETTO_TRACER.enter();
            guard.replace(None);
            println!("[perfetto-tracer] cleared");
        }
        Ok(())
    }

    fn set_perf_instr_counter(&self, value: u64) -> PyResult<()> {
        set_perf_instr_counter(value);
        Ok(())
    }

    fn flush_perfetto(&mut self) -> PyResult<()> {
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.take() {
            tracer
                .finish()
                .map_err(|err| PyRuntimeError::new_err(format!("flush perfetto trace: {err}")))?;
        }
        Ok(())
    }
}

impl LlamaCpu {
    fn capture_callback_rollback_state(&self) -> CallbackRollbackState {
        CallbackRollbackState {
            state: self.state.clone(),
            timer: self.timer.clone(),
            keyboard: self.keyboard.clone(),
            cycles: self.cycles,
            memory_reads: self.memory_reads,
            memory_writes: self.memory_writes,
            call_sub_level: self.call_sub_level,
            temps: self.temps.clone(),
        }
    }

    fn restore_after_callback_failure(&mut self, rollback: CallbackRollbackState) {
        self.state = rollback.state;
        self.timer = rollback.timer;
        self.keyboard = rollback.keyboard;
        self.cycles = rollback.cycles;
        self.memory_reads = rollback.memory_reads;
        self.memory_writes = rollback.memory_writes;
        self.call_sub_level = rollback.call_sub_level;
        self.temps = rollback.temps;
        self.mirror.rollback_capture();
    }

    fn poison_callback_error<T>(&mut self, err: PyErr) -> PyResult<T> {
        // The first callback failure is the useful root cause. Recovery work
        // can itself fail, but must not erase that original diagnostic.
        if self.poisoned.is_none() {
            self.poisoned = Some(err.to_string());
        }
        Err(err)
    }

    fn ensure_not_poisoned(&self) -> PyResult<()> {
        if let Some(reason) = &self.poisoned {
            Err(PyRuntimeError::new_err(format!(
                "LLAMA CPU is poisoned after a Python callback failure; reset required: {reason}"
            )))
        } else {
            Ok(())
        }
    }

    fn canonical_host_address(address: u32) -> u32 {
        let address = address & ADDRESS_MASK;
        if address >= INTERNAL_MEMORY_START {
            INTERNAL_MEMORY_START + ((address - INTERNAL_MEMORY_START) & 0xFF)
        } else {
            address & (EXTERNAL_SPACE as u32 - 1)
        }
    }

    fn sync_authoritative_host_write(&mut self, address: u32, value: u8) {
        let address = Self::canonical_host_address(address);
        if let Some(offset) = MemoryImage::internal_offset(address) {
            if MemoryImage::is_keyboard_offset(offset) {
                // Host-side keyboard overlays have already accepted this
                // write. Update the native device state, then mark the host's
                // value as committed so the generated mirror write is not
                // sent back through Python a second time.
                self.keyboard.handle_write(offset, value, &mut self.mirror);
            }
            self.mirror.sync_committed_host_write(address, value);
            match offset {
                IMEM_IMR_OFFSET => self.timer.irq_imr = value,
                IMEM_ISR_OFFSET => self.timer.irq_isr = value,
                _ => {}
            }
        } else {
            self.mirror.sync_committed_host_write(address, value);
        }
    }

    fn record_uncertain_host_writes<I>(&mut self, addresses: I)
    where
        I: IntoIterator<Item = u32>,
    {
        for address in addresses {
            let address = Self::canonical_host_address(address);
            if !self.uncertain_host_writes.contains(&address) {
                self.uncertain_host_writes.push(address);
            }
        }
        if !self.uncertain_host_writes.is_empty() {
            self.memory_synced = false;
        }
    }

    fn resync_uncertain_host_writes(&mut self, py: Python<'_>) -> PyResult<()> {
        // Read and validate the full candidate set before changing native
        // state. If any host read fails, the mirror remains untouched and the
        // CPU keeps its original poison and uncertainty set.
        let addresses = self.uncertain_host_writes.clone();
        let mut committed = Vec::with_capacity(addresses.len());
        for address in addresses {
            let result = if self.read_with_pc {
                self.read_callback
                    .bind(py)
                    .call1((address, self.state.pc() & ADDRESS_MASK))
            } else {
                self.read_callback.bind(py).call1((address,))
            };
            committed.push((address, result?.extract::<u8>()?));
        }
        for (address, value) in committed {
            self.sync_authoritative_host_write(address, value);
        }
        Ok(())
    }

    fn sync_mirror_tracking(
        &mut self,
        py: Python<'_>,
        write_attempts: &mut Vec<u32>,
    ) -> PyResult<()> {
        flush_mirror_to_python(
            py,
            &self.write_callback,
            self.write_with_pc,
            self.state.pc() & ADDRESS_MASK,
            &mut self.mirror,
            Some(write_attempts),
        )
    }

    fn transactional_key_mutation<T, F>(&mut self, py: Python<'_>, mutate: F) -> PyResult<T>
    where
        F: FnOnce(&mut Self) -> T,
    {
        self.ensure_not_poisoned()?;
        let timer_before = self.timer.clone();
        let keyboard_before = self.keyboard.clone();
        self.mirror.begin_rollback_capture();
        let result = mutate(self);
        let mut write_attempts = Vec::new();
        if let Err(err) = self.sync_mirror_tracking(py, &mut write_attempts) {
            self.record_uncertain_host_writes(write_attempts);
            self.timer = timer_before;
            self.keyboard = keyboard_before;
            self.mirror.rollback_capture();
            return self.poison_callback_error(err);
        }
        self.mirror.commit_rollback_capture();
        Ok(result)
    }

    fn emit_irq_trace(
        &self,
        py: Python<'_>,
        name: &str,
        payload: HashMap<&'static str, u32>,
    ) -> PyResult<()> {
        let dict = PyDict::new_bound(py);
        for (k, v) in payload {
            dict.set_item(k, v)?;
        }
        if let Some(hook) = optional_callable_attr(py, &self.memory, "trace_irq_from_rust")? {
            hook.bind(py).call1((name, dict))?;
        }
        Ok(())
    }

    fn read_irq_registers(&self, py: Python<'_>, bus: &mut LlamaPyBus) -> PyResult<(u8, u8)> {
        let imr = bus.read_byte_with_gil(py, INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
        let isr = bus.read_byte_with_gil(py, INTERNAL_MEMORY_START + IMEM_ISR_OFFSET);
        if let Some(err) = bus.take_callback_error() {
            Err(err)
        } else {
            Ok((imr, isr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_python_error_contains(err: PyErr, expected: &str) {
        let rendered = err.to_string();
        assert!(
            rendered.contains(expected),
            "expected {rendered:?} to contain {expected:?}"
        );
    }

    #[test]
    fn contract_bus_tick_timers_sets_mti_and_isr() {
        let mut bus = LlamaContractBus::new();
        bus.configure_timer(1, 0, true);
        bus.tick_timers(1);
        let isr = bus.memory.read_internal_byte(0xFC).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01, "MTI bit should be set in ISR");
        assert!(bus.timer.irq_pending, "MTI should mark irq_pending");
    }

    #[test]
    fn llama_python_bridge_snapshots_temp15() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "temp15.py", "temp15").expect("mem module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
            cpu.state.set_reg(LlamaRegName::Temp(15), 0x12_3456);
            cpu.sync_temps_from_state();

            assert_eq!(cpu.read_register("TEMP15").unwrap(), 0x12_3456);
            let snapshot = cpu.snapshot_cpu_registers(py).expect("snapshot");
            let temps = snapshot
                .bind(py)
                .getattr("temps")
                .expect("temps")
                .extract::<HashMap<u32, u32>>()
                .expect("temp mapping");
            assert_eq!(temps.get(&15), Some(&0x12_3456));
            assert!(!temps.contains_key(&16));
        });
    }

    #[test]
    fn initialise_rust_memory_rejects_malformed_exports_atomically() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self, mode):
        self.mode = mode

    def read_byte(self, addr, pc=None):
        return 0

    def write_byte(self, addr, val, pc=None):
        return None

    def export_flat_memory(self):
        if self.mode == "export_error":
            raise RuntimeError("export boom")
        size = 0x0FFFFF if self.mode == "short_external" else 0x100000
        external = bytes([0xA5]) * size
        if self.mode == "shape":
            return external, ()
        fallback = ((0x100, 0x1FF),)
        if self.mode == "malformed_ranges":
            fallback = ("not-a-range",)
        elif self.mode == "inverted_range":
            fallback = ((0x200, 0x100),)
        return external, fallback, ((0xE0000, 0xFFFFF),)

    def get_internal_memory_bytes(self):
        if self.mode == "internal_error":
            raise RuntimeError("imem boom")
        size = 0xFF if self.mode == "short_internal" else 0x100
        return bytes([0x5A]) * size
"#;
            let module = PyModule::from_code_bound(py, code, "bridge_init.py", "bridge_init")
                .expect("memory module");
            let cls = module.getattr("Mem").expect("Mem class");

            for (mode, expected) in [
                ("export_error", "export boom"),
                ("short_external", "flattened memory length mismatch"),
                ("shape", "expected exactly 3"),
                ("malformed_ranges", "fallback ranges"),
                ("inverted_range", "inverted range"),
                ("internal_error", "imem boom"),
                ("short_internal", "internal memory length mismatch"),
            ] {
                let mem = cls.call1((mode,)).expect("memory instance");
                let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
                cpu.mirror.sync_committed_host_write(0x12345, 0x11);
                cpu.mirror
                    .sync_committed_host_write(INTERNAL_MEMORY_START + 0x42, 0x22);
                cpu.mirror.set_python_ranges(vec![(0x10, 0x20)]);
                cpu.mirror.set_readonly_ranges(vec![(0x30, 0x40)]);
                cpu.memory_synced = false;

                let err = cpu._initialise_rust_memory(py).unwrap_err();

                assert_python_error_contains(err, expected);
                assert_eq!(cpu.mirror.external_slice()[0x12345], 0x11, "{mode}");
                assert_eq!(
                    cpu.mirror.read_internal_byte_silent(0x42),
                    Some(0x22),
                    "{mode}"
                );
                assert_eq!(cpu.mirror.python_ranges(), &[(0x10, 0x20)], "{mode}");
                assert_eq!(cpu.mirror.readonly_ranges(), &[(0x30, 0x40)], "{mode}");
                assert!(!cpu.memory_synced, "{mode}");
            }
        });
    }

    #[test]
    fn initialise_rust_memory_commits_only_a_complete_candidate() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        return 0

    def write_byte(self, addr, val, pc=None):
        return None

    def export_flat_memory(self):
        return (
            bytes([0xA5]) * 0x100000,
            ((0x100, 0x1FF), (0x1000F0, 0x1000F2)),
            ((0xE0000, 0xFFFFF),),
        )

    def get_internal_memory_bytes(self):
        return bytes([0x5A]) * 0x100
"#;
            let module = PyModule::from_code_bound(py, code, "bridge_init_ok.py", "bridge_init_ok")
                .expect("memory module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
            cpu.mirror.sync_committed_host_write(0x12345, 0x11);
            cpu.memory_synced = false;

            cpu._initialise_rust_memory(py).expect("initialize mirror");

            assert_eq!(cpu.mirror.external_slice()[0x12345], 0xA5);
            assert_eq!(cpu.mirror.read_internal_byte_silent(0x42), Some(0x5A));
            assert_eq!(
                cpu.mirror.python_ranges(),
                &[(0x100, 0x1FF), (0x1000F0, 0x1000F2)]
            );
            assert_eq!(cpu.mirror.readonly_ranges(), &[(0xE0000, 0xFFFFF)]);
            assert!(cpu.mirror.drain_dirty().is_empty());
            assert!(cpu.mirror.drain_dirty_internal().is_empty());
            assert!(cpu.memory_synced);
        });
    }

    #[test]
    fn save_snapshot_rejects_inexact_bridge_images_before_writing() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self, mode):
        self.mode = mode

    def read_byte(self, addr, pc=None):
        return 0

    def write_byte(self, addr, val, pc=None):
        return None

    def export_flat_memory(self):
        base = (bytes(0x100000), (), ())
        if self.mode == "extra_item":
            return base + ("unexpected",)
        return base

    def get_internal_memory_bytes(self):
        if self.mode == "short_internal":
            return bytes(0xFF)
        return bytes(0x100)
"#;
            let module = PyModule::from_code_bound(
                py,
                code,
                "snapshot_bridge_shape.py",
                "snapshot_bridge_shape",
            )
            .expect("memory module");
            let cls = module.getattr("Mem").expect("Mem class");

            for (mode, expected) in [
                ("short_internal", "internal memory length mismatch"),
                ("extra_item", "expected exactly 3"),
            ] {
                let mem = cls.call1((mode,)).expect("memory instance");
                let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
                let err = cpu
                    .save_snapshot(py, "/__sc62015_missing__/snapshot.pcsnap")
                    .unwrap_err();
                assert_python_error_contains(err, expected);
            }
        });
    }

    #[test]
    fn save_snapshot_propagates_present_broken_lcd_controller() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Snapshot:
    def __init__(self):
        self.chips = []

class Controller:
    def __init__(self, mode):
        self.mode = mode
    def get_snapshot(self):
        if self.mode == "snapshot":
            raise RuntimeError("lcd snapshot boom")
        return Snapshot()

class Mem:
    def __init__(self, mode):
        self.mode = mode
    @property
    def _lcd_controller(self):
        if self.mode == "getter":
            raise RuntimeError("lcd controller getter boom")
        return Controller(self.mode)
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        return None
    def export_flat_memory(self):
        return (bytes(0x100000), (), ())
    def get_internal_memory_bytes(self):
        return bytes(0x100)
"#;
            let module = PyModule::from_code_bound(
                py,
                code,
                "broken_lcd_snapshot.py",
                "broken_lcd_snapshot",
            )
            .expect("memory module");
            let cls = module.getattr("Mem").expect("Mem class");

            for (mode, expected) in [
                ("getter", "lcd controller getter boom"),
                ("snapshot", "lcd snapshot boom"),
                ("empty", "must contain at least one chip"),
            ] {
                let mem = cls.call1((mode,)).expect("memory instance");
                let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");

                let err = cpu
                    .save_snapshot(py, "/__sc62015_missing__/snapshot.pcsnap")
                    .unwrap_err();

                assert_python_error_contains(err, expected);
            }
        });
    }

    #[test]
    fn restore_keyboard_snapshot_roundtrips_native_matrix_state_atomically() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "keyboard_snapshot.py", "keyboard_snapshot")
                    .expect("memory module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
            cpu.keyboard
                .inject_matrix_event(0x11, false, &mut cpu.mirror, true);
            cpu.keyboard.handle_write(0xF0, 0x04, &mut cpu.mirror);
            cpu.keyboard.handle_write(0xF1, 0x02, &mut cpu.mirror);
            let expected = cpu.keyboard.snapshot_state();
            let encoded = serde_json::to_string(&expected).expect("serialize keyboard");
            let py_snapshot = py
                .import_bound("json")
                .unwrap()
                .call_method1("loads", (encoded,))
                .unwrap();

            cpu.keyboard.reset(&mut cpu.mirror);
            let before_validation = to_value(cpu.keyboard.snapshot_state()).unwrap();
            cpu.validate_keyboard_snapshot(py, py_snapshot.to_object(py))
                .expect("validate native keyboard");
            assert_eq!(
                to_value(cpu.keyboard.snapshot_state()).unwrap(),
                before_validation,
                "validation must not mutate the native matrix"
            );
            cpu.restore_keyboard_snapshot(py, py_snapshot.into())
                .expect("restore native keyboard");

            assert_eq!(
                to_value(cpu.keyboard.snapshot_state()).unwrap(),
                to_value(&expected).unwrap()
            );

            let mut python_matrix = to_value(&expected).unwrap();
            let python_matrix_obj = python_matrix.as_object_mut().unwrap();
            python_matrix_obj.remove("fifo_len");
            python_matrix_obj.remove("active_columns");
            let python_handler = json!({
                "matrix": python_matrix,
                "last_kol": expected.kol,
                "last_koh": expected.koh,
                "last_kil": expected.kil_latch,
                "scan_enabled": expected.scan_enabled,
            });
            let py_handler = py
                .import_bound("json")
                .unwrap()
                .call_method1("loads", (python_handler.to_string(),))
                .unwrap();
            cpu.keyboard.reset(&mut cpu.mirror);
            cpu.restore_keyboard_snapshot(py, py_handler.into())
                .expect("restore Python KeyboardHandler snapshot");
            assert_eq!(
                to_value(cpu.keyboard.snapshot_state()).unwrap(),
                to_value(&expected).unwrap()
            );

            let before_bad_restore = to_value(cpu.keyboard.snapshot_state()).unwrap();
            let malformed = PyDict::new_bound(py);
            malformed.set_item("kol", 1).unwrap();
            let validation_err = cpu
                .validate_keyboard_snapshot(py, malformed.to_object(py))
                .unwrap_err();
            assert_python_error_contains(validation_err, "keyboard snapshot");
            let err = cpu
                .restore_keyboard_snapshot(py, malformed.into())
                .unwrap_err();
            assert_python_error_contains(err, "keyboard snapshot");
            assert_eq!(
                to_value(cpu.keyboard.snapshot_state()).unwrap(),
                before_bad_restore,
                "invalid metadata must not partially mutate the native matrix"
            );
        });
    }

    #[test]
    fn restore_scheduler_snapshot_applies_complete_native_state_atomically() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "scheduler_snapshot.py", "scheduler_snapshot")
                    .expect("memory module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
            let timer = json!({
                "enabled": true,
                "mti_period": 101,
                "sti_period": 202,
                "next_mti": 303,
                "next_sti": 404,
                "kb_irq_enabled": false,
                "instruction_start_cycle": 77,
                "last_mti_fire_cycle": null,
                "last_sti_fire_cycle": null,
                "fired_mti_since_boundary": false,
                "fired_sti_since_boundary": false,
                "preserve_phase": true,
            });
            let interrupts = json!({
                "pending": true,
                "in_interrupt": true,
                "key_irq_latched": true,
                "source": "KEY",
                "stack": [0x100001, 0x100002],
                "next_id": 0x100003,
                "imr": 0xA1,
                "isr": 0x04,
                "irq_counts": {"total": 9, "KEY": 5, "MTI": 3, "STI": 1},
                "last_irq": {"src": "KEY", "pc": 0x34567, "vector": 0xFFFFA},
                "irq_bit_watch": null,
                "delivered_masks": [0x81, 0x84],
            });
            let json_module = py.import_bound("json").unwrap();
            let py_timer = json_module
                .call_method1("loads", (timer.to_string(),))
                .unwrap();
            let py_interrupts = json_module
                .call_method1("loads", (interrupts.to_string(),))
                .unwrap();

            let timer_before_validation = cpu.timer.clone();
            cpu.validate_scheduler_snapshot(
                py,
                py_timer.to_object(py),
                py_interrupts.to_object(py),
                77,
            )
            .expect("validate scheduler");
            assert_eq!(cpu.timer.next_mti, timer_before_validation.next_mti);
            assert_eq!(cpu.timer.irq_pending, timer_before_validation.irq_pending);
            cpu.restore_scheduler_snapshot(py, py_timer.into(), py_interrupts.into(), 77)
                .expect("restore scheduler");

            assert!(cpu.timer.enabled);
            assert_eq!(cpu.timer.mti_period, 101);
            assert_eq!(cpu.timer.sti_period, 202);
            assert_eq!(cpu.timer.next_mti, 303);
            assert_eq!(cpu.timer.next_sti, 404);
            assert!(!cpu.timer.kb_irq_enabled);
            assert!(cpu.timer.irq_pending);
            assert!(cpu.timer.in_interrupt);
            assert_eq!(cpu.timer.irq_source.as_deref(), Some("KEY"));
            assert!(cpu.timer.key_irq_latched);
            assert_eq!(cpu.timer.interrupt_stack, vec![0x100001, 0x100002]);
            assert_eq!(cpu.timer.next_interrupt_id, 0x100003);
            assert_eq!(cpu.timer.irq_imr, 0xA1);
            assert_eq!(cpu.timer.irq_isr, 0x04);
            assert_eq!(cpu.timer.irq_total, 9);
            assert_eq!(cpu.timer.irq_key, 5);
            assert_eq!(cpu.timer.irq_mti, 3);
            assert_eq!(cpu.timer.irq_sti, 1);
            assert_eq!(cpu.timer.last_irq_src.as_deref(), Some("KEY"));
            assert_eq!(cpu.timer.last_irq_pc, Some(0x34567));
            assert_eq!(cpu.timer.last_irq_vector, Some(0xFFFFA));
            assert_eq!(cpu.timer.delivered_masks, vec![0x81, 0x84]);
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_IMR_OFFSET),
                Some(0xA1)
            );
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_ISR_OFFSET),
                Some(0x04)
            );

            let before = cpu.timer.clone();
            let mut invalid = interrupts.clone();
            invalid["isr"] = json!(0);
            let py_timer = json_module
                .call_method1("loads", (timer.to_string(),))
                .unwrap();
            let py_invalid = json_module
                .call_method1("loads", (invalid.to_string(),))
                .unwrap();
            let validation_err = cpu
                .validate_scheduler_snapshot(
                    py,
                    py_timer.to_object(py),
                    py_invalid.to_object(py),
                    77,
                )
                .unwrap_err();
            assert_python_error_contains(validation_err, "cannot be pending with ISR == 0");
            let err = cpu
                .restore_scheduler_snapshot(py, py_timer.into(), py_invalid.into(), 77)
                .unwrap_err();
            assert_python_error_contains(err, "cannot be pending with ISR == 0");
            assert_eq!(cpu.timer.next_mti, before.next_mti);
            assert_eq!(cpu.timer.irq_pending, before.irq_pending);
            assert_eq!(cpu.timer.irq_source, before.irq_source);
            assert_eq!(cpu.timer.irq_imr, before.irq_imr);
            assert_eq!(cpu.timer.irq_isr, before.irq_isr);

            for malformed in [
                json!({}),
                {
                    let mut value = interrupts.clone();
                    value["in_interrupt"] = json!(false);
                    value["source"] = json!("KEY");
                    value["isr"] = json!(0x01);
                    value
                },
                {
                    let mut value = interrupts.clone();
                    value["irq_bit_watch"] = json!({"IMR": {}, "ISR": {}});
                    value
                },
            ] {
                let py_timer = json_module
                    .call_method1("loads", (timer.to_string(),))
                    .unwrap();
                let py_bad = json_module
                    .call_method1("loads", (malformed.to_string(),))
                    .unwrap();
                assert!(cpu
                    .validate_scheduler_snapshot(
                        py,
                        py_timer.to_object(py),
                        py_bad.to_object(py),
                        77,
                    )
                    .is_err());
                assert_eq!(cpu.timer.next_mti, before.next_mti);
                assert_eq!(cpu.timer.irq_pending, before.irq_pending);
            }
        });
    }

    #[test]
    fn snapshot_metadata_preserves_live_interrupt_scheduler_state() {
        let mut timer = TimerContext::new(true, 11, 17);
        timer.next_mti = 123;
        timer.next_sti = 456;
        timer.irq_pending = true;
        timer.in_interrupt = true;
        timer.key_irq_latched = true;
        timer.irq_source = Some("KEY".to_string());
        timer.interrupt_stack = vec![7, 8];
        timer.next_interrupt_id = 9;
        timer.delivered_masks = vec![0x81, 0x84];
        timer.irq_imr = 0x01;
        timer.irq_isr = 0x02;

        let mut image = MemoryImage::new();
        image.sync_committed_host_write(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 0xA1);
        image.sync_committed_host_write(INTERNAL_MEMORY_START + IMEM_ISR_OFFSET, 0x24);
        let mut metadata = SnapshotMetadata::default();

        populate_live_timer_snapshot_metadata(&mut metadata, &timer, &image);

        assert_eq!(metadata.timer.next_mti, 123);
        assert_eq!(metadata.timer.next_sti, 456);
        assert!(metadata.interrupts.pending);
        assert!(metadata.interrupts.in_interrupt);
        assert!(metadata.interrupts.key_irq_latched);
        assert_eq!(metadata.interrupts.source.as_deref(), Some("KEY"));
        assert_eq!(metadata.interrupts.stack, vec![7, 8]);
        assert_eq!(metadata.interrupts.next_id, 9);
        assert_eq!(metadata.interrupts.delivered_masks, vec![0x81, 0x84]);
        assert_eq!(metadata.interrupts.imr, 0xA1);
        assert_eq!(metadata.interrupts.isr, 0x24);
    }

    #[test]
    fn runtime_snapshot_validation_is_non_mutating_and_restore_is_exact() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "runtime_snapshot.py", "runtime_snapshot")
                    .expect("memory module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
            cpu.cycles = 7;
            cpu.memory_reads = 8;
            cpu.memory_writes = 9;
            cpu.state.set_call_depth(2);
            cpu.state.set_power_state(PowerState::Halted);

            cpu.validate_runtime_snapshot(
                1 << 40,
                101,
                202,
                3,
                2,
                vec![0x12345],
                vec![0x10000],
                vec![16],
                "off",
            )
            .expect("validate runtime snapshot");
            assert_eq!(cpu.cycles, 7);
            assert_eq!(cpu.memory_reads, 8);
            assert_eq!(cpu.memory_writes, 9);
            assert_eq!(cpu.state.call_depth(), 2);
            assert_eq!(cpu.state.power_state(), PowerState::Halted);

            cpu.restore_runtime_snapshot(
                1 << 40,
                101,
                202,
                3,
                2,
                vec![0x12345],
                vec![0x10000],
                vec![16],
                "off",
            )
            .expect("restore runtime snapshot");
            assert_eq!(cpu.cycles, 1 << 40);
            assert_eq!(cpu.memory_reads, 101);
            assert_eq!(cpu.memory_writes, 202);
            assert_eq!(cpu.state.call_depth(), 3);
            assert_eq!(cpu.state.call_sub_level(), 2);
            assert_eq!(cpu.state.call_stack(), &[0x12345]);
            assert_eq!(cpu.state.power_state(), PowerState::Off);

            let invalid = cpu
                .restore_runtime_snapshot(
                    0,
                    0,
                    0,
                    0,
                    0,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    "experiment",
                )
                .unwrap_err();
            assert_python_error_contains(invalid, "unknown power state");
            assert_eq!(cpu.cycles, 1 << 40);
            assert_eq!(cpu.state.call_depth(), 3);
            assert_eq!(cpu.state.power_state(), PowerState::Off);
        });
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
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.data[addr & 0xFFFFFF] = val & 0xFF
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "mem_mod.py", "mem_mod").expect("mem module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();

            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");
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
            assert!(
                cpu.timer.irq_pending,
                "timer should pend after WAIT fallback"
            );
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

    #[test]
    fn wide_internal_stores_run_kio_and_irq_hooks_per_byte() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0:4] = bytes([
            0xA4, 0xF0,  # MV (KOL),X -- writes KOL/KOH/KIL
            0xA2, 0xFA,  # MV (TXD),BA -- writes TXD/IMR
        ])
        self.writes = []
        self.kio_events = []
        self.irq_events = []
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.data[addr] = val
        self.writes.append((addr, val))
    def trace_kio_from_rust(self, offset, value, pc):
        self.kio_events.append((offset, value, pc))
    def trace_irq_from_rust(self, name, payload):
        self.irq_events.append((name, payload["offset"], payload["value"]))
"#;
            let module = PyModule::from_code_bound(py, code, "wide_hooks.py", "wide_hooks")
                .expect("memory module");
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).expect("cpu init");

            cpu.state.set_reg(LlamaRegName::X, 0x02_3456);
            cpu.execute_instruction(py, 0).expect("wide KIO store");
            cpu.state.set_reg(LlamaRegName::BA, 0xBBAA);
            cpu.execute_instruction(py, 2).expect("word spanning IMR");

            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap(),
                vec![
                    (INTERNAL_MEMORY_START + IMEM_KOL_OFFSET, 0x56),
                    (INTERNAL_MEMORY_START + IMEM_KOH_OFFSET, 0x34),
                    (INTERNAL_MEMORY_START + IMEM_KIL_OFFSET, 0x02),
                    (INTERNAL_MEMORY_START + 0xFA, 0xAA),
                    (INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 0xBB),
                ]
            );
            assert_eq!(
                mem.getattr("kio_events")
                    .unwrap()
                    .extract::<Vec<(u32, u8, u32)>>()
                    .unwrap(),
                vec![
                    (IMEM_KOL_OFFSET, 0x56, 0),
                    (IMEM_KOH_OFFSET, 0x34, 0),
                    (IMEM_KIL_OFFSET, 0x02, 0),
                ]
            );
            assert_eq!(
                mem.getattr("irq_events")
                    .unwrap()
                    .extract::<Vec<(String, u32, u8)>>()
                    .unwrap(),
                vec![("IMR_Write".to_string(), IMEM_IMR_OFFSET, 0xBB)]
            );
            let keyboard = cpu.keyboard.snapshot_state();
            assert_eq!(keyboard.kol, 0x56);
            assert_eq!(keyboard.koh, 0x34);
            assert_eq!(cpu.timer.irq_imr, 0xBB);
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_IMR_OFFSET),
                Some(0xBB)
            );
            assert!(
                cpu.mirror.drain_dirty_internal().is_empty(),
                "host-committed KIO/IMR bytes must not remain queued for replay"
            );
        });
    }

    #[test]
    fn llama_python_bus_propagates_read_callback_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def read_byte(self, addr, pc=None):
        raise RuntimeError("read boom")
    def peek_byte_for_preflight(self, addr, pc=None):
        return 0x38 if (addr & 0xFFFFFF) == 0 else 0
    def write_byte(self, addr, val, pc=None):
        return None
"#;
            let module =
                PyModule::from_code_bound(py, code, "read_error.py", "read_error").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

            let err = cpu.execute_instruction(py, 0).unwrap_err();

            assert_python_error_contains(err, "read boom");
        });
    }

    #[test]
    fn llama_python_bus_propagates_write_callback_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0] = 0xA0
        self.data[1] = 0x10
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        raise RuntimeError("write boom")
"#;
            let module =
                PyModule::from_code_bound(py, code, "write_error.py", "write_error").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

            let err = cpu.execute_instruction(py, 0).unwrap_err();

            assert_python_error_contains(err, "write boom");
        });
    }

    #[test]
    fn late_lcd_callback_failure_rolls_back_read_and_write_accounting() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0:4] = bytes([0xA8, 0x00, 0x20, 0x00])  # MV ($002000),A
        self.writes = []
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.data[addr] = val
        self.writes.append((addr, val))
    def _llama_lcd_write(self, addr, value, pc):
        raise RuntimeError("lcd callback failed after host commit")
"#;
            let module = PyModule::from_code_bound(py, code, "lcd_error.py", "lcd_error").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.state.set_reg(LlamaRegName::A, 0x5A);
            cpu.memory_reads = 31;
            cpu.memory_writes = 37;

            let err = cpu.execute_instruction(py, 0).unwrap_err();

            assert_python_error_contains(err, "lcd callback failed after host commit");
            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap(),
                vec![(0x2000, 0x5A)],
                "host write must have completed before the LCD hook failed"
            );
            assert_eq!(
                (cpu.memory_reads, cpu.memory_writes),
                (31, 37),
                "native rollback must include both callback accounting counters"
            );
            assert_eq!(cpu.state.pc(), 0, "architectural state must roll back too");
        });
    }

    #[test]
    fn failed_reset_flush_preserves_global_perf_instruction_index() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        if addr == 0x1234:
            raise RuntimeError("late mirror flush failed")
        self.data[addr] = val & 0xFF
"#;
            let module =
                PyModule::from_code_bound(py, code, "reset_flush_error.py", "reset_flush_error")
                    .unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.mirror.write_external_byte(0x1234, 0xAA);
            cpu.prepare_immediate_machine_reset_transfer(py)
                .expect("prepare recovery reset");
            set_perf_instr_counter(0x1234_5678);

            let err = cpu.power_on_reset().unwrap_err();

            assert_python_error_contains(err, "late mirror flush failed");
            assert_eq!(
                perfetto_last_instr_index(),
                0x1234_5678,
                "a failed recovery must not reset process-global trace ordering"
            );
            reset_perf_counters();
        });
    }

    #[test]
    fn reset_vector_rejection_is_atomic_and_does_not_poison_native_cpu() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0xFFFFD:0x100000] = bytes([0x78, 0x56, 0xF4])
        self.reads = []
        self.peeks = []
        self.writes = []
    def read_byte(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.reads.append(addr)
        return self.data[addr]
    def peek_byte_for_preflight(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.peeks.append(addr)
        return self.data[addr]
    def instruction_byte_is_callback_free(self, addr):
        return True
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.writes.append((addr, val))
        self.data[addr] = val
    def make_vector_valid(self):
        self.data[0xFFFFD:0x100000] = bytes([0x00, 0x00, 0x00])
"#;
            let module = PyModule::from_code_bound(
                py,
                code,
                "reset_vector_atomic.py",
                "reset_vector_atomic",
            )
            .unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.state.set_pc(0x012345);
            cpu.state.set_reg(LlamaRegName::S, 0x000240);
            cpu.state.set_reg(LlamaRegName::F, 0x03);
            cpu.state.set_reg(LlamaRegName::IMR, 0xA5);
            cpu.state.halt();
            cpu.cycles = 123;

            let err = cpu.power_on_reset().unwrap_err();

            assert_python_error_contains(err, "requires a prepared SC62015 machine_reset");
            assert_eq!(cpu.state.pc(), 0x012345);
            assert_eq!(cpu.state.get_reg(LlamaRegName::S), 0x000240);
            assert_eq!(cpu.state.get_reg(LlamaRegName::F), 0x03);
            assert_eq!(cpu.state.get_reg(LlamaRegName::IMR), 0xA5);
            assert!(cpu.state.is_halted());
            assert_eq!(cpu.cycles, 123);
            assert!(cpu.poisoned.is_none());
            assert!(cpu.uncertain_host_writes.is_empty());
            assert_eq!(
                mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap(),
                Vec::<u32>::new(),
                "rejected vector must not reach architectural reads"
            );
            assert_eq!(
                mem.getattr("peeks").unwrap().extract::<Vec<u32>>().unwrap(),
                Vec::<u32>::new(),
                "unprepared reset must not even enter vector preflight"
            );
            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap(),
                Vec::<(u32, u8)>::new()
            );

            let invalid = cpu
                .prepare_vector_transfer(
                    py,
                    ROM_RESET_VECTOR_ADDR,
                    0x012345,
                    false,
                    "machine_reset",
                )
                .unwrap_err();
            assert_python_error_contains(
                invalid,
                sc62015_core::llama::eval::VECTOR_UPPER_NIBBLE_ERROR,
            );
            assert!(cpu.poisoned.is_none());
            assert_eq!(
                mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap(),
                Vec::<u32>::new(),
                "safe-peek rejection remains unpoisoned and non-architectural"
            );
            assert_eq!(
                mem.getattr("peeks").unwrap().extract::<Vec<u32>>().unwrap(),
                vec![0xFFFFD, 0xFFFFE, 0xFFFFF]
            );

            mem.call_method0("make_vector_valid").unwrap();
            cpu.prepare_vector_transfer(
                py,
                ROM_RESET_VECTOR_ADDR,
                0x012345,
                false,
                "machine_reset",
            )
            .expect("prepare valid recovery reset");
            cpu.power_on_reset()
                .expect("semantic rejection must not poison recovery reset");
            assert!(cpu.poisoned.is_none());
        });
    }

    #[test]
    fn direct_vector_opcodes_require_a_prepared_transfer_before_vector_reads() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self, opcode):
        self.data = bytearray(0x100100)
        self.data[0] = opcode
        self.data[0xFFFFA:0x100000] = bytes([0x00, 0x02, 0x00, 0x00, 0x02, 0x00])
        self.reads = []
        self.peeks = []
        self.writes = []
    def read_byte(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.reads.append(addr)
        return self.data[addr]
    def peek_byte_for_preflight(self, addr, pc=None):
        self.peeks.append(addr & 0xFFFFFF)
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.writes.append((addr & 0xFFFFFF, val & 0xFF))
        self.data[addr & 0xFFFFFF] = val & 0xFF
"#;
            let module = PyModule::from_code_bound(
                py,
                code,
                "unprepared_vector_opcode.py",
                "unprepared_vector_opcode",
            )
            .unwrap();

            for opcode in [0xFEu8, 0xFFu8] {
                let mem = module.getattr("Mem").unwrap().call1((opcode,)).unwrap();
                let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

                for expected_peeks in [vec![0u32], vec![0u32, 0u32]] {
                    let err = cpu.execute_instruction(py, 0).unwrap_err();
                    assert_python_error_contains(err, "requires a prepared instruction vector");
                    assert_eq!(
                        mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap(),
                        Vec::<u32>::new(),
                        "an unprepared vector opcode must fail before its architectural fetch"
                    );
                    assert_eq!(
                        mem.getattr("peeks").unwrap().extract::<Vec<u32>>().unwrap(),
                        expected_peeks,
                        "each retry may use only the declared side-effect-free reader"
                    );
                    assert!(cpu.poisoned.is_none());
                    assert!(mem
                        .getattr("writes")
                        .unwrap()
                        .extract::<Vec<(u32, u8)>>()
                        .unwrap()
                        .is_empty());
                }
            }
        });
    }

    #[test]
    fn direct_opcode_change_after_silent_preflight_poisoned_after_one_read() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.reads = []
        self.peeks = []
        self.writes = []
    def read_byte(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.reads.append(addr)
        return 0xFE if addr == 0 else 0
    def peek_byte_for_preflight(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.peeks.append(addr)
        return 0x00
    def write_byte(self, addr, val, pc=None):
        self.writes.append((addr & 0xFFFFFF, val & 0xFF))
"#;
            let module = PyModule::from_code_bound(
                py,
                code,
                "direct_opcode_change.py",
                "direct_opcode_change",
            )
            .unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

            let first = cpu.execute_instruction(py, 0).unwrap_err();
            assert_python_error_contains(first, "changed after silent preflight");
            assert!(cpu.poisoned.is_some());
            assert_eq!(
                mem.getattr("peeks").unwrap().extract::<Vec<u32>>().unwrap(),
                vec![0]
            );
            assert_eq!(
                mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap(),
                vec![0],
                "the uncertain architectural opcode fetch occurs exactly once"
            );

            let retry = cpu.execute_instruction(py, 0).unwrap_err();
            assert_python_error_contains(retry, "LLAMA CPU is poisoned");
            assert_eq!(
                mem.getattr("peeks").unwrap().extract::<Vec<u32>>().unwrap(),
                vec![0]
            );
            assert_eq!(
                mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap(),
                vec![0],
                "poisoning must block a second observable fetch"
            );
            assert!(mem
                .getattr("writes")
                .unwrap()
                .extract::<Vec<(u32, u8)>>()
                .unwrap()
                .is_empty());
        });
    }

    #[test]
    fn prefetched_native_reset_does_not_read_vector_or_target_again() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0xFFFFD:0x100000] = bytes((0x00, 0x02, 0x00))
        self.reads = []
        self.peeks = []
        self.writes = []
        self.forbidden = False
    def vector_transfer_provenance(self):
        return (id(self), 0)
    def instruction_byte_is_callback_free(self, addr):
        return True
    def forbid_vector_access(self):
        self.forbidden = True
        self.reads.clear()
        self.peeks.clear()
    def read_byte(self, addr, pc=None):
        addr &= 0xFFFFFF
        if self.forbidden and (0xFFFFD <= addr <= 0xFFFFF or addr == 0x200):
            raise RuntimeError("prefetched reset performed a forbidden read")
        self.reads.append(addr)
        return self.data[addr]
    def peek_byte_for_preflight(self, addr, pc=None):
        addr &= 0xFFFFFF
        self.peeks.append(addr)
        if self.forbidden:
            raise RuntimeError("prefetched reset performed a forbidden peek")
        return self.data[addr]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.writes.append((addr, val))
        self.data[addr] = val
"#;
            let module =
                PyModule::from_code_bound(py, code, "prefetched_reset.py", "prefetched_reset")
                    .unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.state.set_pc(0x12345);
            cpu.state.halt();

            cpu.prepare_vector_transfer(py, 0xFFFFD, 0x12345, false, "machine_reset")
                .expect("prepare native RESET");
            mem.call_method0("forbid_vector_access").unwrap();
            cpu.power_on_reset().expect("prepared native RESET");

            assert_eq!(cpu.state.pc(), 0x00200);
            assert!(!cpu.state.is_halted());
            assert!(mem
                .getattr("peeks")
                .unwrap()
                .extract::<Vec<u32>>()
                .unwrap()
                .is_empty());
            let reads = mem.getattr("reads").unwrap().extract::<Vec<u32>>().unwrap();
            assert!(!reads.contains(&0xFFFFD));
            assert!(!reads.contains(&0xFFFFE));
            assert!(!reads.contains(&0xFFFFF));
            assert!(!reads.contains(&0x00200));
            assert!(!mem
                .getattr("writes")
                .unwrap()
                .extract::<Vec<(u32, u8)>>()
                .unwrap()
                .is_empty());
        });
    }

    #[test]
    fn multibyte_callback_failure_stops_and_rolls_back_native_state() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0:2] = bytes([0xA4, 0xF0])  # MV (KOL),X
        self.writes = []
        self.fail = True
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.data[addr] = val
        self.writes.append((addr, val))
        if self.fail:
            raise RuntimeError("first byte committed then failed")
"#;
            let module =
                PyModule::from_code_bound(py, code, "multibyte_error.py", "multibyte_error")
                    .unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.notify_host_write(INTERNAL_MEMORY_START + IMEM_KOL_OFFSET, 0x77)
                .unwrap();
            cpu.state.set_reg(LlamaRegName::PC, 0);
            cpu.state.set_reg(LlamaRegName::X, 0x02_3456);
            cpu.memory_reads = 17;
            cpu.memory_writes = 23;

            let err = cpu.execute_instruction(py, 0).unwrap_err();

            assert_python_error_contains(err, "first byte committed then failed");
            assert_eq!(
                (cpu.memory_reads, cpu.memory_writes),
                (17, 23),
                "failed execution must roll back runtime accounting"
            );
            let writes = mem
                .getattr("writes")
                .unwrap()
                .extract::<Vec<(u32, u8)>>()
                .unwrap();
            assert_eq!(
                writes,
                vec![(INTERNAL_MEMORY_START + IMEM_KOL_OFFSET, 0x56)]
            );
            let data = mem.getattr("data").unwrap();
            assert_eq!(
                data.get_item(INTERNAL_MEMORY_START + IMEM_KOL_OFFSET)
                    .unwrap()
                    .extract::<u8>()
                    .unwrap(),
                0x56
            );
            assert_eq!(
                data.get_item(INTERNAL_MEMORY_START + IMEM_KOL_OFFSET + 1)
                    .unwrap()
                    .extract::<u8>()
                    .unwrap(),
                0
            );
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_KOL_OFFSET),
                Some(0x77),
                "native mirror must roll back the pre-callback KIO mutation"
            );
            assert!(cpu.mirror.drain_dirty_internal().is_empty());
            assert_eq!(cpu.state.get_reg(LlamaRegName::PC), 0);
            assert_eq!(cpu.state.get_reg(LlamaRegName::X), 0x02_3456);

            let poison = cpu.execute_instruction(py, 0).unwrap_err();
            assert_python_error_contains(poison, "CPU is poisoned");
            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap()
                    .len(),
                1,
                "poisoned execution must not call the host again"
            );

            // The failed callback committed 0x56 to the authoritative host,
            // while native rollback correctly restored 0x77. A successful
            // RESET must read the uncertain byte back before clearing poison.
            mem.setattr("fail", false).unwrap();
            cpu.prepare_immediate_machine_reset_transfer(py)
                .expect("prepare recovery reset");
            cpu.power_on_reset().expect("recovery reset");
            assert!(cpu.poisoned.is_none());
            assert!(cpu.uncertain_host_writes.is_empty());
            assert!(cpu.memory_synced);
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_KOL_OFFSET),
                Some(0x56),
                "recovery reset must reconcile the host-committed byte"
            );
        });
    }

    #[test]
    fn key_callback_failure_rolls_back_and_poison_gates_all_key_mutations() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.writes = []
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        addr &= 0xFFFFFF
        val &= 0xFF
        self.data[addr] = val
        self.writes.append((addr, val))
        if len(self.writes) == 1:
            raise RuntimeError("matrix byte committed then failed")
        raise RuntimeError("later reset callback failed")
"#;
            let module = PyModule::from_code_bound(py, code, "key_error.py", "key_error").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            let timer_before = cpu.timer.clone();

            let err = cpu.keyboard_press_matrix_code(py, 0x11).unwrap_err();

            assert_python_error_contains(err, "matrix byte committed then failed");
            let writes = mem
                .getattr("writes")
                .unwrap()
                .extract::<Vec<(u32, u8)>>()
                .unwrap();
            assert_eq!(writes.len(), 1, "stop after the first failed callback");
            assert_eq!(writes[0].0, INTERNAL_MEMORY_START + 0xF2);
            assert_eq!(cpu.keyboard.fifo_len(), 0);
            assert!(cpu.keyboard.snapshot_state().pressed_keys.is_empty());
            assert_eq!(cpu.timer.irq_pending, timer_before.irq_pending);
            assert_eq!(cpu.timer.irq_source, timer_before.irq_source);
            assert_eq!(cpu.mirror.read_internal_byte_silent(0xF2), Some(0));
            assert_eq!(
                cpu.poisoned.as_deref(),
                Some("RuntimeError: matrix byte committed then failed")
            );

            for err in [
                cpu.keyboard_press_matrix_code(py, 0x12).unwrap_err(),
                cpu.keyboard_release_matrix_code(py, 0x11).unwrap_err(),
                cpu.keyboard_press_on_key(py).unwrap_err(),
                cpu.keyboard_release_on_key(py).unwrap_err(),
            ] {
                assert_python_error_contains(err, "matrix byte committed then failed");
            }
            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap()
                    .len(),
                1,
                "poisoned key APIs must not mutate or call the host"
            );
            assert_eq!(cpu.keyboard.fifo_len(), 0);
            assert!(cpu.keyboard.snapshot_state().pressed_keys.is_empty());
            assert_eq!(cpu.mirror.read_internal_byte_silent(0xF2), Some(0));

            cpu.prepare_immediate_machine_reset_transfer(py)
                .expect("prepare recovery reset");
            let reset_err = cpu.power_on_reset().unwrap_err();
            assert_python_error_contains(reset_err, "later reset callback failed");
            assert_eq!(
                cpu.poisoned.as_deref(),
                Some("RuntimeError: matrix byte committed then failed"),
                "recovery failure must not replace the original poison reason"
            );
            assert_eq!(
                mem.getattr("writes")
                    .unwrap()
                    .extract::<Vec<(u32, u8)>>()
                    .unwrap()
                    .len(),
                2,
                "failed RESET must also stop after its first callback"
            );
        });
    }

    #[test]
    fn on_key_release_clears_only_level_and_preserves_latched_status() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.data[addr & 0xFFFFFF] = val & 0xFF
"#;
            let module = PyModule::from_code_bound(py, code, "on_key.py", "on_key").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

            assert!(cpu.keyboard_press_on_key(py).unwrap());
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(0xFF).unwrap() & 0x08,
                0x08
            );
            assert_eq!(
                cpu.mirror
                    .read_internal_byte_silent(IMEM_ISR_OFFSET)
                    .unwrap()
                    & 0x08,
                0x08
            );
            assert!(cpu.timer.irq_pending);
            assert_eq!(cpu.timer.irq_source.as_deref(), Some("ONK"));

            cpu.keyboard_release_on_key(py).unwrap();
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(0xFF).unwrap() & 0x08,
                0
            );
            assert_eq!(
                cpu.mirror
                    .read_internal_byte_silent(IMEM_ISR_OFFSET)
                    .unwrap()
                    & 0x08,
                0x08
            );
            assert!(cpu.timer.irq_pending);
            assert_eq!(cpu.timer.irq_source.as_deref(), Some("ONK"));

            cpu.mirror.write_internal_byte(IMEM_ISR_OFFSET, 0x08 | 0x01);
            cpu.timer.irq_isr = 0x08 | 0x01;
            cpu.timer.irq_pending = true;
            cpu.timer.irq_source = Some("ONK".to_string());
            cpu.keyboard_release_on_key(py).unwrap();
            assert_eq!(cpu.timer.irq_isr, 0x09);
            assert!(cpu.timer.irq_pending);
            assert_eq!(cpu.timer.irq_source.as_deref(), Some("ONK"));
        });
    }

    #[test]
    fn notify_host_write_updates_mirror_without_repeating_python_write() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.write_calls = 0
    def read_byte(self, addr, pc=None):
        return 0
    def write_byte(self, addr, val, pc=None):
        self.write_calls += 1
        raise RuntimeError("notify must not call host write_byte")
"#;
            let module = PyModule::from_code_bound(py, code, "host_sync.py", "host_sync").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();

            cpu.notify_host_write(0x20, 0x42).unwrap();
            cpu.notify_host_write(0x200020, 0x99).unwrap();

            assert_eq!(cpu.mirror.load(0x20, 8), Some(0x42));
            assert_eq!(cpu.mirror.read_internal_byte_silent(0x20), Some(0x99));
            assert!(cpu.mirror.drain_dirty().is_empty());
            assert_eq!(
                mem.getattr("write_calls")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
        });
    }

    #[test]
    fn failed_keyboard_write_rolls_back_mirror_and_stale_dirty_write() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0:2] = bytes([0xA0, 0xF0])  # MV (KOL),A
        self.fail = True
        self.writes = []
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.writes.append((addr & 0xFFFFFF, val & 0xFF))
        if self.fail:
            raise RuntimeError("keyboard write boom")
        self.data[addr & 0xFFFFFF] = val & 0xFF
"#;
            let module = PyModule::from_code_bound(py, code, "rollback.py", "rollback").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.notify_host_write(INTERNAL_MEMORY_START + IMEM_KOL_OFFSET, 0x77)
                .unwrap();
            cpu.state.set_reg(LlamaRegName::A, 0x12);

            let err = cpu.execute_instruction(py, 0).unwrap_err();
            assert_python_error_contains(err, "keyboard write boom");
            assert_eq!(
                cpu.mirror.read_internal_byte_silent(IMEM_KOL_OFFSET),
                Some(0x77)
            );
            assert!(cpu.mirror.drain_dirty_internal().is_empty());

            // A recovery RESET may perform its own documented IMEM writes,
            // but it must not replay the failed KOL value from the mirror.
            mem.setattr("fail", false).unwrap();
            mem.setattr("writes", pyo3::types::PyList::empty_bound(py))
                .unwrap();
            cpu.prepare_immediate_machine_reset_transfer(py)
                .expect("prepare recovery reset");
            cpu.power_on_reset().unwrap();
            let writes = mem
                .getattr("writes")
                .unwrap()
                .extract::<Vec<(u32, u8)>>()
                .unwrap();
            assert!(!writes
                .iter()
                .any(|(address, _)| { *address == INTERNAL_MEMORY_START + IMEM_KOL_OFFSET }));
        });
    }

    #[test]
    fn llama_python_bus_propagates_wait_callback_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let code = r#"
class Mem:
    def __init__(self):
        self.data = bytearray(0x100100)
        self.data[0] = 0xEF
    def read_byte(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def peek_byte_for_preflight(self, addr, pc=None):
        return self.data[addr & 0xFFFFFF]
    def write_byte(self, addr, val, pc=None):
        self.data[addr & 0xFFFFFF] = val & 0xFF
    def wait_cycles(self, cycles):
        raise RuntimeError("wait boom")
"#;
            let module =
                PyModule::from_code_bound(py, code, "wait_error.py", "wait_error").unwrap();
            let mem = module.getattr("Mem").unwrap().call0().unwrap();
            let mut cpu = LlamaCpu::new(mem.to_object(py), false, 1.0).unwrap();
            cpu.state.set_reg(LlamaRegName::I, 1);

            let err = cpu.execute_instruction(py, 0).unwrap_err();

            assert_python_error_contains(err, "wait boom");
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
    guard.with_some(|tracer| {
        let mut converted = HashMap::new();
        for (k, v) in payload {
            converted.insert(k, AnnotationValue::UInt(v));
        }
        tracer.record_irq_event(name, converted);
    });
    Ok(())
}
