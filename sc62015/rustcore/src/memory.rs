use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Wrapper around the Python-side `binja_test_mocks.eval_llil.Memory` helper.
#[derive(Debug, Clone)]
pub struct MemoryBus {
    inner: Py<PyAny>,
}

impl MemoryBus {
    /// Construct a new memory bus from a Python object implementing the required methods.
    pub fn new(py: Python<'_>, memory: PyObject) -> PyResult<Self> {
        let bound = memory.bind(py);
        for method in ["read_byte", "write_byte", "read_bytes", "write_bytes"] {
            if !bound.hasattr(method)? {
                return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                    format!("memory object is missing required method '{method}'"),
                ));
            }
        }

        Ok(Self {
            inner: memory.into_py(py),
        })
    }

    pub fn read_byte(&self, py: Python<'_>, address: u32) -> PyResult<u8> {
        self.inner
            .call_method1(py, "read_byte", (address,))
            .and_then(|obj| obj.extract::<u8>(py))
    }

    pub fn write_byte(&self, py: Python<'_>, address: u32, value: u8) -> PyResult<()> {
        self.inner
            .call_method1(py, "write_byte", (address, value))
            .map(|_| ())
    }

    pub fn read_bytes(&self, py: Python<'_>, address: u32, size: usize) -> PyResult<u32> {
        self.inner
            .call_method1(py, "read_bytes", (address, size))
            .and_then(|obj| obj.extract::<u32>(py))
    }

    pub fn write_bytes(
        &self,
        py: Python<'_>,
        address: u32,
        size: usize,
        value: u32,
    ) -> PyResult<()> {
        self.inner
            .call_method1(py, "write_bytes", (address, size, value))
            .map(|_| ())
    }

    /// Return the `_perf_tracer` attribute if present.
    pub fn perf_tracer(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .bind(py)
            .getattr("_perf_tracer")
            .ok()
            .map(|obj| obj.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyModule};

    #[test]
    fn memory_bus_round_trips_bytes() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            py.run(
                r#"
data = {}
def read(addr):
    return data.get(addr, 0)
def write(addr, value):
    data[addr] = value & 0xFF
"#,
                None,
                Some(locals),
            )?;

            let read_fn = locals.get_item("read").unwrap();
            let write_fn = locals.get_item("write").unwrap();
            let module = PyModule::import(py, "binja_test_mocks.eval_llil")?;
            let memory_cls = module.getattr("Memory")?;
            let memory_obj = memory_cls.call1((read_fn, write_fn))?;

            let bus = MemoryBus::new(py, memory_obj.into())?;
            assert_eq!(bus.read_byte(py, 0x10)?, 0);
            bus.write_byte(py, 0x10, 0xAB)?;
            assert_eq!(bus.read_byte(py, 0x10)?, 0xAB);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn memory_bus_multibyte_roundtrip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            py.run(
                r#"
data = {}
def read(addr):
    return data.get(addr, 0)
def write(addr, value):
    data[addr] = value & 0xFF
"#,
                None,
                Some(locals),
            )?;

            let read_fn = locals.get_item("read").unwrap();
            let write_fn = locals.get_item("write").unwrap();
            let module = PyModule::import(py, "binja_test_mocks.eval_llil")?;
            let memory_cls = module.getattr("Memory")?;
            let memory_obj = memory_cls.call1((read_fn, write_fn))?;

            let bus = MemoryBus::new(py, memory_obj.into())?;
            bus.write_bytes(py, 0x20, 2, 0xABCD)?;
            assert_eq!(bus.read_bytes(py, 0x20, 2)?, 0xABCD);
            bus.write_bytes(py, 0x24, 3, 0x00FFEE)?;
            assert_eq!(bus.read_bytes(py, 0x24, 3)?, 0x00FFEE);
            Ok(())
        })
        .unwrap();
    }
}
