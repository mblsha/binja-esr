mod generated {
    pub mod types {
        include!("../generated/types.rs");
    }
    pub mod payload {
        include!("../generated/handlers.rs");
    }
}

use generated::types::{BoundInstrRepr, ManifestEntry, PreInfo};
use once_cell::sync::Lazy;
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
use serde_json::{self, Map, Value};
use std::collections::HashMap;

static MANIFEST: Lazy<Vec<ManifestEntry>> = Lazy::new(|| {
    serde_json::from_str(generated::payload::PAYLOAD).expect("manifest json")
});

const INTERNAL_MEMORY_START: u32 = 0x100000;
const ADDRESS_MASK: u32 = 0x00FF_FFFF;
const INTERNAL_ADDR_MASK: u32 = 0xFF;
const DEFAULT_REG_WIDTH: u8 = 24;

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
}

#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let mut runtime = Self {
            state: RsState::default(),
            memory: memory.into(),
        };
        if reset_on_init {
            runtime.power_on_reset();
        }
        Ok(runtime)
    }

    fn power_on_reset(&mut self) {
        self.state = RsState::default();
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
