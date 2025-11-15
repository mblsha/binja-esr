mod generated {
    pub mod types;
    pub mod handlers {
        include!("../generated/handlers.rs");
    }
}

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use rust_scil::{
    ast::{Binder, Instr, PreLatch},
    bus::Space,
    eval,
    state::State as RsState,
};
use serde_json;

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

fn space_to_str(space: &Space) -> &'static str {
    match space {
        Space::Int => "int",
        Space::Ext => "ext",
        Space::Code => "code",
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

#[pymodule]
fn _sc62015_rustcore(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Cpu>()?;
    m.add("__backend_name__", backend_name())?;
    m.add("HAS_CPU_IMPLEMENTATION", true)?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_ready, m)?)?;
    m.add_function(wrap_pyfunction!(scil_step_json, m)?)?;
    Ok(())
}
