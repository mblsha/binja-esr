use proptest::prelude::*;
use pyo3::prelude::*;

fn rust_backend_ready(py: Python<'_>) -> PyResult<bool> {
    match py.import("_sc62015_rustcore") {
        Ok(module) => module.getattr("HAS_CPU_IMPLEMENTATION")?.extract(),
        Err(_) => Ok(false),
    }
}

proptest! {
    #[test]
    fn parity_with_python_is_deferred(_bytes in proptest::collection::vec(any::<u8>(), 1..4)) {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            if !rust_backend_ready(py)? {
                return Ok(());
            }

            // TODO: hook into the Python reference emulator once the Rust core is implemented.
            Ok(())
        }).unwrap();
    }
}
