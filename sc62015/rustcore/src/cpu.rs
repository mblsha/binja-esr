use std::cell::RefCell;
use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::constants::NUM_TEMP_REGISTERS;
use crate::decode::decode_opcode;
use crate::executor::{ExecutionError, LlilRuntime};
use crate::memory::MemoryBus;
use crate::state::{Flag, RegisterError, Registers};
use crate::OPCODE_HANDLERS;

fn exec_error_to_py(err: ExecutionError) -> PyErr {
    match err {
        ExecutionError::UnsupportedOpcode => {
            PyRuntimeError::new_err("unsupported opcode in rustcore backend")
        }
        ExecutionError::Unimplemented(name) => {
            PyRuntimeError::new_err(format!("unimplemented feature in rustcore backend: {name}"))
        }
        ExecutionError::InvalidOperand(reason) => {
            PyValueError::new_err(format!("invalid operand: {reason}"))
        }
        ExecutionError::MissingValue(context) => {
            PyRuntimeError::new_err(format!("missing value while executing {context}"))
        }
        ExecutionError::Register(err) => register_error_to_py(err),
        ExecutionError::Python(err) => err,
    }
}

fn register_error_to_py(err: RegisterError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn register_name_table() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names = vec![
            "A".to_string(),
            "B".to_string(),
            "IL".to_string(),
            "IH".to_string(),
            "I".to_string(),
            "BA".to_string(),
            "X".to_string(),
            "Y".to_string(),
            "U".to_string(),
            "S".to_string(),
            "PC".to_string(),
            "F".to_string(),
            "FC".to_string(),
            "FZ".to_string(),
        ];
        for index in 0..NUM_TEMP_REGISTERS {
            names.push(format!("TEMP{index}"));
        }
        names
    })
}

#[pyclass(name = "CPU")]
pub struct Cpu {
    runtime: RefCell<LlilRuntime>,
}

impl Cpu {
    fn new_runtime(py: Python<'_>, memory: PyObject) -> PyResult<LlilRuntime> {
        let memory_bus = MemoryBus::new(py, memory)?;
        Ok(LlilRuntime::new(Registers::new(), memory_bus))
    }

    fn with_runtime<R>(
        &self,
        mut f: impl FnMut(&mut LlilRuntime) -> Result<R, ExecutionError>,
    ) -> PyResult<R> {
        let mut runtime = self.runtime.borrow_mut();
        f(&mut runtime).map_err(exec_error_to_py)
    }

    fn with_runtime_ro<R>(&self, mut f: impl FnMut(&LlilRuntime) -> R) -> R {
        let runtime = self.runtime.borrow();
        f(&runtime)
    }
}

#[pymethods]
impl Cpu {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    pub fn new(py: Python<'_>, memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let mut runtime = Self::new_runtime(py, memory)?;
        if reset_on_init {
            runtime
                .invoke_intrinsic("RESET")
                .map_err(exec_error_to_py)?;
        }
        Ok(Self {
            runtime: RefCell::new(runtime),
        })
    }

    pub fn execute_instruction(&self, address: u32) -> PyResult<(u8, u8)> {
        let opcode = self.with_runtime(|runtime| {
            runtime.write_named_register("PC", address as i64, Some(3))?;
            runtime.read_memory_value(address as i64, 1)
        })? as u8;

        let length = decode_opcode(opcode)
            .map(|desc| desc.length())
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown opcode 0x{opcode:02X}")))?;

        self.with_runtime(|runtime| {
            let handler = OPCODE_HANDLERS[opcode as usize];
            handler(runtime)?;
            Ok(())
        })?;

        Ok((opcode, length))
    }

    pub fn power_on_reset(&self) -> PyResult<()> {
        self.with_runtime(|runtime| runtime.invoke_intrinsic("RESET"))?;
        Ok(())
    }

    pub fn read_register(&self, name: &str) -> PyResult<u32> {
        self.with_runtime_ro(|runtime| {
            runtime
                .registers()
                .get_by_name(name)
                .map_err(register_error_to_py)
        })
    }

    pub fn write_register(&self, name: &str, value: u32) -> PyResult<()> {
        self.with_runtime(|runtime| {
            runtime
                .registers_mut()
                .set_by_name(name, value)
                .map_err(ExecutionError::Register)?;
            Ok(())
        })
    }

    pub fn read_flag(&self, name: &str) -> PyResult<u8> {
        let ch = name
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("flag name must not be empty"))?;
        let flag = Flag::from_char(ch).map_err(register_error_to_py)?;
        Ok(self.with_runtime_ro(|runtime| runtime.registers().get_flag(flag)))
    }

    pub fn write_flag(&self, name: &str, value: u8) -> PyResult<()> {
        let ch = name
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("flag name must not be empty"))?;
        let flag = Flag::from_char(ch).map_err(register_error_to_py)?;
        self.with_runtime(|runtime| {
            runtime.registers_mut().set_flag(flag, value);
            Ok(())
        })
    }

    pub fn read_memory_value(&self, address: u32, width: u8) -> PyResult<i64> {
        if width == 0 {
            return Err(PyValueError::new_err("width must be >= 1"));
        }
        self.with_runtime(|runtime| runtime.read_memory_value(address as i64, width))
    }

    pub fn write_memory_value(&self, address: u32, width: u8, value: i64) -> PyResult<()> {
        if width == 0 {
            return Err(PyValueError::new_err("width must be >= 1"));
        }
        self.with_runtime(|runtime| {
            runtime.write_memory_value(address as i64, width, value)?;
            Ok(())
        })
    }

    #[allow(deprecated)]
    pub fn snapshot_registers(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        let runtime = self.runtime.borrow();
        for name in register_name_table() {
            if let Ok(value) = runtime.registers().get_by_name(name) {
                dict.set_item(name.as_str(), value)?;
            }
        }
        Ok(dict.into())
    }

    #[getter]
    pub fn call_sub_level(&self) -> PyResult<u32> {
        Ok(self.with_runtime_ro(|runtime| runtime.registers().call_sub_level()))
    }

    #[setter]
    pub fn set_call_sub_level(&self, level: u32) -> PyResult<()> {
        self.with_runtime(|runtime| {
            runtime.registers_mut().set_call_sub_level(level);
            Ok(())
        })
    }

    #[getter]
    pub fn halted(&self) -> PyResult<bool> {
        Ok(self.with_runtime_ro(|runtime| runtime.state().halted()))
    }

    #[setter]
    pub fn set_halted(&self, value: bool) -> PyResult<()> {
        self.with_runtime(|runtime| {
            runtime.state_mut().set_halted(value);
            Ok(())
        })
    }
}
