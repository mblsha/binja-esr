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
    ADDRESS_MASK,
};
use std::collections::HashMap;

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
}

impl LlamaPyBus {
    fn new(py: Python<'_>, memory: &Py<PyAny>) -> Self {
        Self {
            memory: memory.clone_ref(py),
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
        let mut value = 0u32;
        for i in 0..bytes {
            let byte = self.read_byte(addr.wrapping_add(i as u32)) as u32;
            value |= byte << (8 * i);
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
                self.write_byte(addr, (value & 0xFF) as u8);
                self.write_byte(addr.wrapping_add(1), ((value >> 8) & 0xFF) as u8);
            }
            24 => {
                self.write_byte(addr, (value & 0xFF) as u8);
                self.write_byte(addr.wrapping_add(1), ((value >> 8) & 0xFF) as u8);
                self.write_byte(addr.wrapping_add(2), ((value >> 16) & 0xFF) as u8);
            }
            _ => {
                let bytes = bits.div_ceil(8);
                for i in 0..bytes {
                    let byte = ((value >> (8 * i)) & 0xFF) as u8;
                    self.write_byte(addr.wrapping_add(i as u32), byte);
                }
            }
        }
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }
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
        let mut bus = LlamaPyBus::new(py, &self.memory);
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
        let _ = path;
        Ok(())
    }

    fn flush_perfetto(&mut self) -> PyResult<()> {
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
