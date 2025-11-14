use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::{PyAssertionError, PyImportError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const MEMORY_SNAPSHOT_LEN: usize = 64;
const PROGRAM_MEMORY_LEN: usize = 0x200;
const MAX_SEQUENCE_STEPS: usize = 16;

#[derive(Clone)]
struct RegisterSpec {
    name: String,
    mask: u32,
}

struct ParityContext {
    cpu_cls: Py<PyAny>,
    registers: Vec<RegisterSpec>,
}

impl ParityContext {
    fn new(py: Python<'_>) -> PyResult<Self> {
        let pysc = PyModule::import(py, "sc62015.pysc62015")?;
        let cpu_cls = pysc.getattr("CPU")?.into();
        let emulator = PyModule::import(py, "sc62015.pysc62015.emulator")?;
        let register_size: HashMap<String, usize> = emulator.getattr("REGISTER_SIZE")?.extract()?;
        let mut registers = Vec::with_capacity(register_size.len());
        for (name, bytes) in register_size {
            let bits = (bytes * 8).min(32);
            let mask = if bits >= 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            registers.push(RegisterSpec { name, mask });
        }
        registers.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { cpu_cls, registers })
    }

    fn instantiate_cpu(
        &self,
        py: Python<'_>,
        memory: &PyAny,
        backend: &str,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("reset_on_init", false)?;
        kwargs.set_item("backend", backend)?;
        let cpu = self
            .cpu_cls
            .bind(py)
            .call((memory,), Some(kwargs))?
            .into_py(py);
        Ok(cpu)
    }
}

fn run_parity_impl(py: Python<'_>, seed: u64, cases: usize) -> PyResult<()> {
    ensure_rust_backend(py)?;
    let ctx = ParityContext::new(py)?;
    let mut rng = StdRng::seed_from_u64(seed);
    run_single_opcode_pass(py, &ctx, &mut rng)?;
    run_sequence_pass(py, &ctx, &mut rng, cases)?;
    Ok(())
}

fn ensure_rust_backend(py: Python<'_>) -> PyResult<()> {
    let module = py.import_bound("_sc62015_rustcore").map_err(|err| {
        PyErr::new::<PyImportError, _>(format!("unable to import _sc62015_rustcore: {err}"))
    })?;
    let ready: bool = module.getattr("HAS_CPU_IMPLEMENTATION")?.extract()?;
    if !ready {
        return Err(PyErr::new::<PyRuntimeError, _>(
            "Rust backend is built without enable_rust_cpu feature",
        ));
    }
    Ok(())
}

fn run_single_opcode_pass(py: Python<'_>, ctx: &ParityContext, rng: &mut StdRng) -> PyResult<()> {
    for opcode in 0u8..=0xFF {
        let memory_image = random_program(opcode, PROGRAM_MEMORY_LEN, rng);
        let mem_py = make_memory(py, &memory_image)?;
        let mem_rs = make_memory(py, &memory_image)?;
        let cpu_python = ctx.instantiate_cpu(py, mem_py.as_ref(py), "python")?;
        let cpu_rust = ctx.instantiate_cpu(py, mem_rs.as_ref(py), "rust")?;
        randomize_registers(py, cpu_python.as_ref(py), &ctx.registers, rng)?;
        randomize_registers(py, cpu_rust.as_ref(py), &ctx.registers, rng)?;
        set_pc(py, cpu_python.as_ref(py), 0)?;
        set_pc(py, cpu_rust.as_ref(py), 0)?;
        execute_and_compare(
            py,
            cpu_python.as_ref(py),
            cpu_rust.as_ref(py),
            &ctx.registers,
            0,
            opcode,
        )?;
    }
    Ok(())
}

fn run_sequence_pass(
    py: Python<'_>,
    ctx: &ParityContext,
    rng: &mut StdRng,
    cases: usize,
) -> PyResult<()> {
    for case in 0..cases.max(1) {
        let memory_image = random_program(rng.gen(), PROGRAM_MEMORY_LEN, rng);
        let mem_py = make_memory(py, &memory_image)?;
        let mem_rs = make_memory(py, &memory_image)?;
        let cpu_python = ctx.instantiate_cpu(py, mem_py.as_ref(py), "python")?;
        let cpu_rust = ctx.instantiate_cpu(py, mem_rs.as_ref(py), "rust")?;
        randomize_registers(py, cpu_python.as_ref(py), &ctx.registers, rng)?;
        randomize_registers(py, cpu_rust.as_ref(py), &ctx.registers, rng)?;
        set_pc(py, cpu_python.as_ref(py), 0)?;
        set_pc(py, cpu_rust.as_ref(py), 0)?;
        let mut step = 0usize;
        while step < MAX_SEQUENCE_STEPS {
            let halted_py = cpu_python
                .as_ref(py)
                .getattr("state")?
                .getattr("halted")?
                .extract::<bool>()?;
            let halted_rs = cpu_rust
                .as_ref(py)
                .getattr("state")?
                .getattr("halted")?
                .extract::<bool>()?;
            if halted_py && halted_rs {
                break;
            }
            let pc = read_pc(py, cpu_python.as_ref(py))?;
            let opcode = read_opcode(py, cpu_python.as_ref(py), pc)?;
            execute_and_compare(
                py,
                cpu_python.as_ref(py),
                cpu_rust.as_ref(py),
                &ctx.registers,
                case,
                opcode,
            )?;
            step += 1;
        }
    }
    Ok(())
}

fn execute_and_compare(
    py: Python<'_>,
    cpu_python: &PyAny,
    cpu_rust: &PyAny,
    registers: &[RegisterSpec],
    case: usize,
    opcode: u8,
) -> PyResult<()> {
    let pc_py = read_pc(py, cpu_python)?;
    let pc_rs = read_pc(py, cpu_rust)?;
    if pc_py != pc_rs {
        return Err(PyAssertionError::new_err(format!(
            "PC mismatch before executing opcode 0x{opcode:02X}: python=0x{pc_py:05X}, rust=0x{pc_rs:05X}"
        )));
    }
    cpu_python.call_method1("execute_instruction", (pc_py,))?;
    cpu_rust.call_method1("execute_instruction", (pc_rs,))?;
    let snap_py = snapshot(py, cpu_python, registers)?;
    let snap_rs = snapshot(py, cpu_rust, registers)?;
    compare_snapshots(opcode, case, &snap_py, &snap_rs, registers)
}

#[derive(Debug)]
struct Snapshot {
    registers: HashMap<String, u32>,
    call_depth: u32,
    halted: bool,
    memory: Vec<u8>,
}

fn snapshot(py: Python<'_>, cpu: &PyAny, registers: &[RegisterSpec]) -> PyResult<Snapshot> {
    let regs_obj = cpu.getattr("regs")?;
    let mut values = HashMap::with_capacity(registers.len());
    for spec in registers {
        let value: u32 = regs_obj
            .call_method1("get_by_name", (spec.name.as_str(),))?
            .extract()?;
        values.insert(spec.name.clone(), value & spec.mask);
    }
    let call_depth: u32 = regs_obj.getattr("call_sub_level")?.extract()?;
    let halted: bool = cpu.getattr("state")?.getattr("halted")?.extract()?;
    let memory = cpu.getattr("memory")?.getattr("_raw")?;
    let raw: Vec<u8> = memory.extract()?;
    let len = MEMORY_SNAPSHOT_LEN.min(raw.len());
    Ok(Snapshot {
        registers: values,
        call_depth,
        halted,
        memory: raw.into_iter().take(len).collect(),
    })
}

fn compare_snapshots(
    opcode: u8,
    case: usize,
    lhs: &Snapshot,
    rhs: &Snapshot,
    layout: &[RegisterSpec],
) -> PyResult<()> {
    for spec in layout {
        let a = lhs.registers.get(&spec.name).copied().unwrap_or_default();
        let b = rhs.registers.get(&spec.name).copied().unwrap_or_default();
        if a != b {
            return Err(PyAssertionError::new_err(format!(
                "register mismatch for opcode 0x{opcode:02X} (case {case}): {} python=0x{a:06X} rust=0x{b:06X}",
                spec.name,
            )));
        }
    }
    if lhs.call_depth != rhs.call_depth {
        return Err(PyAssertionError::new_err(format!(
            "call depth mismatch for opcode 0x{opcode:02X} (case {case}): python={} rust={}",
            lhs.call_depth, rhs.call_depth
        )));
    }
    if lhs.halted != rhs.halted {
        return Err(PyAssertionError::new_err(format!(
            "halted flag mismatch for opcode 0x{opcode:02X} (case {case}): python={} rust={}",
            lhs.halted, rhs.halted
        )));
    }
    if lhs.memory != rhs.memory {
        return Err(PyAssertionError::new_err(format!(
            "memory mismatch for opcode 0x{opcode:02X} (case {case}) detected in first {MEMORY_SNAPSHOT_LEN} bytes"
        )));
    }
    Ok(())
}

fn random_program(opcode: u8, len: usize, rng: &mut StdRng) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    for byte in &mut bytes {
        *byte = rng.gen();
    }
    bytes[0] = opcode;
    bytes
}

fn randomize_registers(
    py: Python<'_>,
    cpu: &PyAny,
    registers: &[RegisterSpec],
    rng: &mut StdRng,
) -> PyResult<()> {
    let regs_obj = cpu.getattr("regs")?;
    for spec in registers {
        let value: u32 = rng.gen::<u32>() & spec.mask;
        regs_obj.call_method1("set_by_name", (spec.name.as_str(), value))?;
    }
    regs_obj.setattr("call_sub_level", rng.gen_range(0..4))?;
    cpu.getattr("state")?.setattr("halted", false)?;
    Ok(())
}

fn set_pc(py: Python<'_>, cpu: &PyAny, value: u32) -> PyResult<()> {
    cpu.getattr("regs")?
        .call_method1("set_by_name", ("PC", value))?;
    Ok(())
}

fn read_pc(py: Python<'_>, cpu: &PyAny) -> PyResult<u32> {
    cpu.getattr("regs")?
        .call_method1("get_by_name", ("PC",))?
        .extract()
}

fn read_opcode(py: Python<'_>, cpu: &PyAny, pc: u32) -> PyResult<u8> {
    cpu.getattr("memory")?
        .call_method1("read_byte", (pc,))?
        .extract()
}

fn make_memory(py: Python<'_>, contents: &[u8]) -> PyResult<Py<PyAny>> {
    static FACTORY: OnceLock<Py<PyAny>> = OnceLock::new();
    let callable = FACTORY.get_or_try_init(|| {
        let module = PyModule::from_code(
            py,
            r#"
from binja_test_mocks.eval_llil import Memory

def make_memory(initial_bytes):
    raw = bytearray(initial_bytes)
    def read(addr):
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"read address {addr:#x} out of bounds")
        return raw[addr]
    def write(addr, value):
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF
    mem = Memory(read, write)
    setattr(mem, "_raw", raw)
    return mem
"#,
            "parity_memory.py",
            "parity_memory",
        )?;
        Ok(module.getattr("make_memory")?.into())
    })?;
    let obj = callable
        .bind(py)
        .call1((PyBytes::new(py, contents),))?
        .into_py(py);
    Ok(obj)
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
