#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyModule};
use pyo3::Bound;
use sc62015_core::{
    llama::{
        eval::{LlamaBus, LlamaExecutor},
        opcodes::RegName as LlamaRegName,
        state::LlamaState,
    },
    PerfettoTracer, ADDRESS_MASK, INTERNAL_MEMORY_START, PERFETTO_TRACER,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

const IMEM_KOL_OFFSET: u32 = 0xF0;
const IMEM_KOH_OFFSET: u32 = 0xF1;
const IMEM_KIL_OFFSET: u32 = 0xF2;

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

struct LlamaPyBus {
    memory: Py<PyAny>,
    pc: u32,
}

impl LlamaPyBus {
    fn new(py: Python<'_>, memory: &Py<PyAny>, pc: u32) -> Self {
        Self {
            memory: memory.clone_ref(py),
            pc,
        }
    }

    fn read_byte(&self, addr: u32) -> u8 {
        Python::with_gil(|py| {
            self.memory
                .bind(py)
                .call_method1("read_byte", (addr,))
                .and_then(|obj| obj.extract::<u8>())
                .unwrap_or(0)
        })
    }

    fn write_byte(&self, addr: u32, value: u8) {
        Python::with_gil(|py| {
            let _ = self
                .memory
                .bind(py)
                .call_method1("write_byte", (addr, value));
        });
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
                    if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                        if let Some(tracer) = guard.as_mut() {
                            tracer.record_kio_read(Some(self.pc), offset as u8, byte as u8);
                            tracer_ok = true;
                        }
                    }
                    // Mirror into Python's dispatcher so the main Perfetto trace sees KIO reads.
                    Python::with_gil(|py| {
                        let _ = self.memory.bind(py).call_method1(
                            "trace_kio_from_rust",
                            (offset as u32, byte as u32, self.pc),
                        );
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

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
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
            .map_or(false, |v| v != "0")
    })
}

#[pyclass(name = "LlamaCPU")]
struct LlamaCpu {
    state: LlamaState,
    executor: LlamaExecutor,
    memory: Py<PyAny>,
    call_sub_level: u32,
    temps: HashMap<u32, u32>,
}

#[pymethods]
impl LlamaCpu {
    /// One-shot helper to deliver an IRQ from Python when the LLAMA bus has pending bits set.
    /// This now mirrors the normal IRQ delivery flow (push PC/F/IMR, clear IRM) and emits
    /// explicit perfetto markers for IRQ_Enter-equivalent visibility.
    fn _deliver_irq_once(&mut self) -> PyResult<()> {
        let pc = self.state.get_reg(LlamaRegName::PC) & sc62015_core::ADDRESS_MASK;
        let sp_before = self.state.get_reg(LlamaRegName::S);
        let mut bus = Python::with_gil(|py| LlamaPyBus::new(py, &self.memory, pc));
        let imr_addr = sc62015_core::INTERNAL_MEMORY_START + 0xFB;
        let isr_addr = sc62015_core::INTERNAL_MEMORY_START + 0xFC;
        let imr = bus.load(imr_addr, 8) as u8;
        let isr = bus.load(isr_addr, 8) as u8;
        // Push PC, F, IMR
        let sp = self.state.get_reg(LlamaRegName::S);
        let mut new_sp = sp.wrapping_sub(3) & sc62015_core::ADDRESS_MASK;
        bus.store(new_sp, 8, (pc & 0xFF) as u32);
        bus.store(new_sp + 1, 8, ((pc >> 8) & 0xFF) as u32);
        bus.store(new_sp + 2, 8, ((pc >> 16) & 0xFF) as u32);
        self.state.set_reg(LlamaRegName::S, new_sp);
        let f = self.state.get_reg(LlamaRegName::F) & 0xFF;
        new_sp = self.state.get_reg(LlamaRegName::S).wrapping_sub(1) & sc62015_core::ADDRESS_MASK;
        bus.store(new_sp, 8, f);
        self.state.set_reg(LlamaRegName::S, new_sp);
        new_sp = self.state.get_reg(LlamaRegName::S).wrapping_sub(1) & sc62015_core::ADDRESS_MASK;
        bus.store(new_sp, 8, imr as u32);
        self.state.set_reg(LlamaRegName::S, new_sp);
        // Clear IRM (bit7)
        let cleared_imr = imr & 0x7F;
        bus.store(imr_addr, 8, cleared_imr as u32);
        self.state.set_reg(LlamaRegName::IMR, cleared_imr as u32);
        // Jump directly to known KEY handler (diagnostic)
        let vec = 0x0F1D56;
        self.state
            .set_reg(LlamaRegName::PC, vec & sc62015_core::ADDRESS_MASK);
        self.state.set_halted(false);
        // Mirror IRQ delivery into the main Python perfetto trace.
        Python::with_gil(|py| -> PyResult<()> {
            let bound = self.memory.bind(py);
            let key_payload = PyDict::new_bound(py);
            key_payload.set_item("from", pc)?;
            key_payload.set_item("imr", imr)?;
            key_payload.set_item("isr", isr)?;
            key_payload.set_item("s", sp_before)?;
            key_payload.set_item("vector", vec & sc62015_core::ADDRESS_MASK)?;
            let _ = bound.call_method1(
                "trace_irq_from_rust",
                ("KeyDeliver", key_payload, "irq.key"),
            );

            let irq_payload = PyDict::new_bound(py);
            irq_payload.set_item("from", pc)?;
            irq_payload.set_item("vector", vec & sc62015_core::ADDRESS_MASK)?;
            irq_payload.set_item("imr_before", imr)?;
            irq_payload.set_item("imr_after", cleared_imr)?;
            irq_payload.set_item("isr", isr)?;
            irq_payload.set_item("y", self.state.get_reg(LlamaRegName::Y))?;
            irq_payload.set_item("s", sp_before)?;
            let _ =
                bound.call_method1("trace_irq_from_rust", ("IRQ_Enter", irq_payload, "irq.key"));
            Ok(())
        })?;
        // Emit a perfetto instant so forced delivery is visible even if the caller doesn’t execute the IRQ loop.
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                tracer.record_kio_read(Some(pc), 0xFF, (vec & 0xFF) as u8);
                tracer.record_kio_read(Some(pc), 0xFB, imr);
                tracer.record_kio_read(Some(pc), 0xFC, isr);
                tracer.record_kio_read(Some(pc), 0xFE, 1);
            }
        }
        println!(
            "[force-deliver] pc=0x{pc:05X} imr=0x{imr:02X} isr=0x{isr:02X} vec=0x{vec:05X}",
            pc = pc,
            imr = imr,
            isr = isr,
            vec = vec & sc62015_core::ADDRESS_MASK
        );
        Ok(())
    }
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let mut cpu = Self {
            state: LlamaState::new(),
            executor: LlamaExecutor::new(),
            memory,
            call_sub_level: 0,
            temps: HashMap::new(),
        };
        if reset_on_init {
            cpu.power_on_reset()?;
        }
        Ok(cpu)
    }

    fn power_on_reset(&mut self) -> PyResult<()> {
        self.state.reset();
        self.state.set_pc(0);
        self.state.set_halted(false);
        self.call_sub_level = 0;
        self.temps.clear();
        Ok(())
    }

    fn execute_instruction(&mut self, py: Python<'_>, address: u32) -> PyResult<(u8, u8)> {
        let opcode = self
            .memory
            .bind(py)
            .call_method1("read_byte", (address,))
            .and_then(|obj| obj.extract::<u8>())
            .unwrap_or(0);
        self.state.set_pc(address & ADDRESS_MASK);
        let mut bus = LlamaPyBus::new(py, &self.memory, self.state.get_reg(LlamaRegName::PC));
        let len = self
            .executor
            .execute(opcode, &mut self.state, &mut bus)
            .map_err(|e| PyRuntimeError::new_err(format!("llama execute: {e}")))?;
        self.call_sub_level = self.state.call_depth();
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
        for (idx, value) in self.temps.iter() {
            temps.set_item(idx, value)?;
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
            }
        }
        if let Ok(call_depth) = snap
            .getattr("call_sub_level")
            .and_then(|obj| obj.extract::<u32>())
        {
            self.call_sub_level = call_depth;
        }
        Ok(())
    }

    /// Convenience hook to run a single pending IRQ check/delivery with logging.
    fn _deliver_irq_logged(&mut self) -> PyResult<()> {
        // Minimal inline copy of the standalone IRQ check/deliver path.
        let cur_pc = self.state.get_reg(LlamaRegName::PC);
        let mut bus = Python::with_gil(|py| LlamaPyBus::new(py, &self.memory, cur_pc));
        let imr_addr = sc62015_core::INTERNAL_MEMORY_START + 0xFB;
        let isr_addr = sc62015_core::INTERNAL_MEMORY_START + 0xFC;
        let imr = bus.load(imr_addr, 8) as u8;
        let isr = bus.load(isr_addr, 8) as u8;
        let pending = (imr & 0x80 != 0) && (imr & isr != 0);
        if pending {
            let vec = 0x0F1D56;
            let pc = cur_pc;
            let sp_before = self.state.get_reg(LlamaRegName::S);
            if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
                    // Record a full IRQ enter event so the main Perfetto trace sees the forced delivery.
                    tracer.record_kio_read(Some(pc), 0xFB, imr);
                    tracer.record_kio_read(Some(pc), 0xFC, isr);
                    tracer.record_kio_read(Some(pc), 0xFF, (vec & 0xFF) as u8);
                    tracer.record_kio_read(Some(pc), 0xFE, 1);
                }
            }
            // Push PC/F/IMR and jump to handler, same as _deliver_irq_once.
            let sp = self.state.get_reg(LlamaRegName::S);
            let pc_val = pc & sc62015_core::ADDRESS_MASK;
            let mut new_sp = sp.wrapping_sub(3) & sc62015_core::ADDRESS_MASK;
            bus.store(new_sp, 8, (pc_val & 0xFF) as u32);
            bus.store(new_sp + 1, 8, ((pc_val >> 8) & 0xFF) as u32);
            bus.store(new_sp + 2, 8, ((pc_val >> 16) & 0xFF) as u32);
            self.state.set_reg(LlamaRegName::S, new_sp);
            let f = self.state.get_reg(LlamaRegName::F) & 0xFF;
            new_sp =
                self.state.get_reg(LlamaRegName::S).wrapping_sub(1) & sc62015_core::ADDRESS_MASK;
            bus.store(new_sp, 8, f);
            self.state.set_reg(LlamaRegName::S, new_sp);
            new_sp =
                self.state.get_reg(LlamaRegName::S).wrapping_sub(1) & sc62015_core::ADDRESS_MASK;
            bus.store(new_sp, 8, imr as u32);
            self.state.set_reg(LlamaRegName::S, new_sp);
            let cleared_imr = imr & 0x7F;
            bus.store(imr_addr, 8, cleared_imr as u32);
            self.state.set_reg(LlamaRegName::IMR, cleared_imr as u32);
            self.state
                .set_reg(LlamaRegName::PC, vec & sc62015_core::ADDRESS_MASK);
            self.state.set_halted(false);
            // Mirror IRQ delivery into the main Python perfetto trace.
            Python::with_gil(|py| -> PyResult<()> {
                let bound = self.memory.bind(py);
                let key_payload = PyDict::new_bound(py);
                key_payload.set_item("from", pc)?;
                key_payload.set_item("imr", imr)?;
                key_payload.set_item("isr", isr)?;
                key_payload.set_item("s", sp_before)?;
                key_payload.set_item("vector", vec & sc62015_core::ADDRESS_MASK)?;
                let _ = bound.call_method1(
                    "trace_irq_from_rust",
                    ("KeyDeliver", key_payload, "irq.key"),
                );

                let irq_payload = PyDict::new_bound(py);
                irq_payload.set_item("from", pc)?;
                irq_payload.set_item("vector", vec & sc62015_core::ADDRESS_MASK)?;
                irq_payload.set_item("imr_before", imr)?;
                irq_payload.set_item("imr_after", cleared_imr)?;
                irq_payload.set_item("isr", isr)?;
                irq_payload.set_item("y", self.state.get_reg(LlamaRegName::Y))?;
                irq_payload.set_item("s", sp_before)?;
                let _ = bound
                    .call_method1("trace_irq_from_rust", ("IRQ_Enter", irq_payload, "irq.key"));
                Ok(())
            })?;
        }
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
            let tracer = PerfettoTracer::new(PathBuf::from(p));
            if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                *guard = Some(tracer);
            }
            println!("[perfetto-tracer] started at {}", p);
        } else if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            *guard = None;
            println!("[perfetto-tracer] cleared");
        }
        Ok(())
    }

    fn flush_perfetto(&mut self) -> PyResult<()> {
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.take() {
                let _ = tracer.finish();
            }
        }
        Ok(())
    }
}

#[pymodule]
fn _sc62015_rustcore(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("HAS_CPU_IMPLEMENTATION", false)?;
    m.add("HAS_LLAMA_IMPLEMENTATION", true)?;
    m.add_class::<LlamaCpu>()?;
    Ok(())
}
