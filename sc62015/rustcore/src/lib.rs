#[cfg(not(feature = "enable_rust_cpu"))]
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/opcode_table.rs"));
}

mod generated_handlers {
    include!(concat!(env!("OUT_DIR"), "/opcode_handlers.rs"));
}

#[cfg(feature = "enable_rust_cpu")]
mod cpu;

pub mod constants;
pub mod decode;
pub mod executor;
pub mod lowering;
pub mod memory;
pub mod state;

pub use generated::{
    LlilExpr, LlilNode, LlilOperand, LlilProgram, OpcodeMetadata, CALL_STACK_EFFECTS, OPCODES,
};
pub use generated_handlers::{OpcodeHandler, OPCODE_HANDLERS};
pub use memory::MemoryBus;
pub use state::Registers;

#[cfg(not(feature = "enable_rust_cpu"))]
#[pyclass(name = "CPU")]
pub struct CpuStub {}

#[cfg(not(feature = "enable_rust_cpu"))]
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

#[cfg(feature = "enable_rust_cpu")]
#[pyfunction]
fn is_ready() -> bool {
    true
}

#[cfg(not(feature = "enable_rust_cpu"))]
#[pyfunction]
fn is_ready() -> bool {
    false
}

#[pymodule]
fn _sc62015_rustcore(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "enable_rust_cpu")]
    {
        log_lowering_coverage();
        m.add_class::<cpu::Cpu>()?;
    }
    #[cfg(not(feature = "enable_rust_cpu"))]
    {
        m.add_class::<CpuStub>()?;
    }
    m.add("__backend_name__", backend_name())?;
    m.add("HAS_CPU_IMPLEMENTATION", is_ready())?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_ready, m)?)?;
    Ok(())
}

#[cfg(feature = "enable_rust_cpu")]
fn log_lowering_coverage() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "sc62015-rustcore lowering coverage: {} specialized, {} LLIL fallback",
            generated_handlers::OPCODE_LOWERING_SPECIALIZED,
            generated_handlers::OPCODE_LOWERING_FALLBACK
        );
        if generated_handlers::OPCODE_LOWERING_FALLBACK > 0 {
            eprintln!(
                "LLIL fallback opcodes: {:?}",
                generated_handlers::OPCODE_LLIL_FALLBACKS
            );
        }
    });
}
