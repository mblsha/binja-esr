use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/opcode_table.rs"));
}

mod generated_handlers {
    include!(concat!(env!("OUT_DIR"), "/opcode_handlers.rs"));
}

pub mod constants;
pub mod decode;
pub mod executor;
pub mod memory;
pub mod state;

pub use generated::{LlilExpr, LlilNode, LlilOperand, LlilProgram, OpcodeMetadata, OPCODES};
pub use generated_handlers::{OpcodeHandler, OPCODE_HANDLERS};
pub use memory::MemoryBus;
pub use state::Registers;

#[pyclass(name = "CPU")]
pub struct CpuStub {}

#[pymethods]
impl CpuStub {
    #[new]
    #[pyo3(signature = (memory, *, reset_on_init = true))]
    fn new(memory: PyObject, reset_on_init: bool) -> PyResult<Self> {
        let _ = memory;
        let _ = reset_on_init;
        Err(PyNotImplementedError::new_err(
            "The Rust SC62015 core is not implemented yet. \
             Builds succeed so the Python shim can detect availability.",
        ))
    }
}

#[pyfunction]
fn backend_name() -> &'static str {
    "rust"
}

#[pyfunction]
fn is_ready() -> bool {
    false
}

#[pymodule]
fn _sc62015_rustcore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CpuStub>()?;
    m.add("__backend_name__", backend_name())?;
    m.add("HAS_CPU_IMPLEMENTATION", is_ready())?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_ready, m)?)?;
    Ok(())
}
