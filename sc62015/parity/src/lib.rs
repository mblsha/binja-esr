use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn run_parity_impl(py: Python<'_>, _seed: u64, _cases: usize) -> PyResult<()> {
    let module = py.import_bound("_sc62015_rustcore").map_err(|err| {
        PyErr::new::<PyImportError, _>(format!(
            "unable to import _sc62015_rustcore: {err}"
        ))
    })?;
    let _ready: bool = module.getattr("HAS_CPU_IMPLEMENTATION")?.extract()?;
    // Placeholder: once the Rust core is ready, hook real comparisons here.
    Ok(())
}

#[pyfunction]
fn run_parity(seed: u64, cases: usize) -> PyResult<()> {
    Python::with_gil(|py| run_parity_impl(py, seed, cases))
}

#[pymodule]
fn sc62015_parity(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_parity, m)?)?;
    Ok(())
}

#[no_mangle]
pub extern "C" fn sc62015_run_parity(seed: u64, cases: usize) -> i32 {
    match run_parity(seed, cases) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("rust/python parity error: {err}");
            1
        }
    }
}
