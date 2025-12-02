//! Parity harness scaffold for LLAMA vs Python emulator.
//!
//! This stays SCIL/LLIL-free: the intent is to drive the LLAMA evaluator and a
//! Python oracle independently and compare register/flag/memory effects.
//! `compare_snapshots` provides the diffing layer; the caller is responsible
//! for invoking the Python emulator (e.g., via a subprocess) and producing
//! snapshots to feed into the comparer.
// PY_SOURCE: tools/llama_parity_runner.py

#[cfg(feature = "llama-tests")]
use crate::{INTERNAL_MEMORY_START, INTERNAL_SPACE};
#[cfg(feature = "llama-tests")]
use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
use std::collections::{HashMap, HashSet};
#[cfg(feature = "llama-tests")]
use std::path::Path;
#[cfg(feature = "llama-tests")]
use std::path::{Path as StdPath, PathBuf};

#[cfg(feature = "llama-tests")]
use super::eval::{LlamaBus, LlamaExecutor};
use super::opcodes::RegName;
use super::state::LlamaState;
#[cfg(feature = "llama-tests")]
use std::process::{Command, Output, Stdio};

/// Result of invoking the Python oracle.
#[cfg(feature = "llama-tests")]
pub struct OracleResult {
    pub snapshot: Snapshot,
    pub perfetto_path: Option<String>,
    pub process_output: Output,
}

/// Minimal state snapshot used for parity checks.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub regs: HashMap<RegName, u32>,
    pub mem_writes: Vec<MemWrite>,
}

/// Memory space classification for parity traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Internal,
    External,
}

impl MemorySpace {
    pub fn as_str(self) -> &'static str {
        match self {
            MemorySpace::Internal => "internal",
            MemorySpace::External => "external",
        }
    }
}

/// Memory write annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemWrite {
    pub addr: u32,
    pub bits: u8,
    pub value: u32,
    pub space: MemorySpace,
}

/// Differences detected during parity comparison.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParityDiff {
    pub reg_mismatches: Vec<(RegName, u32, u32)>, // reg, lhs, rhs
    pub mem_mismatches: Vec<((u32, u8), u32, u32)>, // (addr,bits), lhs, rhs
}

impl Snapshot {
    pub fn from_llama(state: &LlamaState) -> Self {
        let mut regs = HashMap::new();
        // Enumerate known GPR/flag registers; Unknown is intentionally excluded.
        for reg in [
            RegName::A,
            RegName::B,
            RegName::BA,
            RegName::IL,
            RegName::IH,
            RegName::I,
            RegName::X,
            RegName::Y,
            RegName::U,
            RegName::S,
            RegName::PC,
            RegName::F,
            RegName::FC,
            RegName::FZ,
            RegName::IMR,
        ] {
            regs.insert(reg, state.get_reg(reg));
        }
        Self {
            regs,
            mem_writes: Vec::new(),
        }
    }
}

/// Compare two snapshots and return a diff if any mismatches are detected.
pub fn compare_snapshots(lhs: &Snapshot, rhs: &Snapshot) -> Option<ParityDiff> {
    let mut diff = ParityDiff::default();
    let all_regs: HashSet<_> = lhs.regs.keys().chain(rhs.regs.keys()).copied().collect();
    for reg in all_regs {
        let l = lhs.regs.get(&reg).copied().unwrap_or(0);
        let r = rhs.regs.get(&reg).copied().unwrap_or(0);
        if l != r {
            diff.reg_mismatches.push((reg, l, r));
        }
    }

    // Index memory writes by (addr,bits) and compare last-write value.
    let mut lhs_mem: HashMap<(u32, u8), u32> = HashMap::new();
    for write in &lhs.mem_writes {
        lhs_mem.insert((write.addr, write.bits), write.value);
    }
    let mut rhs_mem: HashMap<(u32, u8), u32> = HashMap::new();
    for write in &rhs.mem_writes {
        rhs_mem.insert((write.addr, write.bits), write.value);
    }
    let all_addrs: HashSet<_> = lhs_mem.keys().chain(rhs_mem.keys()).copied().collect();
    for key in all_addrs {
        let l = lhs_mem.get(&key).copied().unwrap_or(0);
        let r = rhs_mem.get(&key).copied().unwrap_or(0);
        if l != r {
            diff.mem_mismatches.push((key, l, r));
        }
    }

    if diff.reg_mismatches.is_empty() && diff.mem_mismatches.is_empty() {
        None
    } else {
        Some(diff)
    }
}

#[cfg(feature = "llama-tests")]
fn reg_from_name(name: &str) -> Option<RegName> {
    match name {
        "A" => Some(RegName::A),
        "B" => Some(RegName::B),
        "BA" => Some(RegName::BA),
        "IL" => Some(RegName::IL),
        "IH" => Some(RegName::IH),
        "I" => Some(RegName::I),
        "X" => Some(RegName::X),
        "Y" => Some(RegName::Y),
        "U" => Some(RegName::U),
        "S" => Some(RegName::S),
        "PC" => Some(RegName::PC),
        "F" => Some(RegName::F),
        "FC" => Some(RegName::FC),
        "FZ" => Some(RegName::FZ),
        "IMR" => Some(RegName::IMR),
        _ => None,
    }
}

/// Invoke the Python emulator once via the helper script, returning a snapshot.
///
/// This keeps the Rust side free of Python bindings; the helper lives at
/// `tools/llama_parity_runner.py` and emits a JSON blob with registers and
/// memory writes.
#[cfg(feature = "llama-tests")]
pub fn run_python_oracle(
    bytes: &[u8],
    regs: &[(RegName, u32)],
    pc: u32,
    cwd: Option<&Path>,
    mem: Option<&[(u32, u8)]>,
) -> Result<OracleResult, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("tools").join("llama_parity_runner.py"))
        .ok_or("could not resolve llama_parity_runner.py")?;
    if !script.exists() {
        return Err(format!("{} not found", script.display()));
    }
    let mut payload = serde_json::json!({
        "bytes": bytes,
        "pc": pc,
    });
    if !regs.is_empty() {
        let mut reg_map = serde_json::Map::new();
        for (reg, value) in regs {
            let name = format!("{reg:?}")
                .replace("Unknown(\"", "")
                .replace("\")", "");
            reg_map.insert(name, serde_json::json!(*value));
        }
        payload
            .as_object_mut()
            .expect("payload map")
            .insert("regs".to_string(), serde_json::Value::Object(reg_map));
    }
    if let Some(mem_seed) = mem {
        let arr = mem_seed
            .iter()
            .map(|(addr, val)| serde_json::json!([addr, val]))
            .collect::<Vec<_>>();
        payload
            .as_object_mut()
            .expect("payload map")
            .insert("mem".to_string(), serde_json::Value::Array(arr));
    }
    // Prefer uv to ensure binja_test_mocks is installed in the virtualenv.
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("python")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .env("FORCE_BINJA_MOCK", "1");
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn python oracle: {e}"))?;
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().ok_or("stdin not available")?;
        stdin
            .write_all(payload.to_string().as_bytes())
            .map_err(|e| format!("write oracle stdin: {e}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("oracle wait: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "oracle exited with {}",
            output.status.code().unwrap_or(-1)
        ));
    }
    let parsed: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("oracle json: {e}"))?;
    let mut regs_out = HashMap::new();
    if let Some(map) = parsed.get("regs").and_then(|v| v.as_object()) {
        for (name, value) in map {
            if let Some(reg) = reg_from_name(name) {
                let val = value.as_u64().unwrap_or(0) as u32;
                regs_out.insert(reg, val);
            }
        }
    }
    let mut writes = Vec::new();
    if let Some(list) = parsed.get("mem_writes").and_then(|v| v.as_array()) {
        for entry in list {
            if let Some(tuple) = entry.as_array() {
                if tuple.len() == 3 || tuple.len() == 4 {
                    let addr = tuple[0].as_u64().unwrap_or(0) as u32;
                    let bits = tuple[1].as_u64().unwrap_or(0) as u8;
                    let value = tuple[2].as_u64().unwrap_or(0) as u32;
                    let space = tuple.get(3).and_then(|v| v.as_str()).unwrap_or("internal");
                    let space = match space {
                        "internal" => MemorySpace::Internal,
                        "external" => MemorySpace::External,
                        _ => MemorySpace::Internal,
                    };
                    writes.push(MemWrite {
                        addr,
                        bits,
                        value,
                        space,
                    });
                }
            }
        }
    }
    Ok(OracleResult {
        snapshot: Snapshot {
            regs: regs_out,
            mem_writes: writes,
        },
        perfetto_path: None,
        process_output: output,
    })
}

#[cfg(feature = "llama-tests")]
fn reg_to_key(reg: RegName) -> &'static str {
    match reg {
        RegName::A => "reg_a",
        RegName::B => "reg_b",
        RegName::BA => "reg_ba",
        RegName::IL => "reg_il",
        RegName::IH => "reg_ih",
        RegName::I => "reg_i",
        RegName::X => "reg_x",
        RegName::Y => "reg_y",
        RegName::U => "reg_u",
        RegName::S => "reg_s",
        RegName::PC => "reg_pc",
        RegName::F => "reg_f",
        RegName::FC => "flag_c",
        RegName::FZ => "flag_z",
        RegName::IMR => "reg_imr",
        RegName::Temp(_) => "reg_temp",
        RegName::Unknown(_) => "reg_unknown",
    }
}

/// Trace event describing one executed instruction for Perfetto export.
#[cfg(feature = "llama-tests")]
pub struct TraceEvent {
    pub backend: String,
    pub instr_index: u64,
    pub pc: u32,
    pub opcode: u8,
    pub regs: HashMap<RegName, u32>,
    pub mem_imr: u8,
    pub mem_isr: u8,
    pub mem_writes: Vec<MemWrite>,
}

/// Write a binary Perfetto trace (protobuf) containing instruction/memory events.
#[cfg(feature = "llama-tests")]
pub fn write_perfetto_trace(events: &[TraceEvent], path: &Path) -> Result<(), String> {
    let mut writer = PerfettoTraceWriter::new(path, 10_000);
    for ev in events {
        writer.record_instr(
            &ev.backend,
            ev.instr_index,
            ev.pc,
            ev.opcode,
            &ev.regs,
            ev.mem_imr,
            ev.mem_isr,
        );
        for write in &ev.mem_writes {
            writer.record_mem_write(
                &ev.backend,
                ev.instr_index,
                ev.pc,
                write.addr,
                write.bits,
                write.value,
                write.space,
            );
        }
    }
    writer.finish()
}

/// Streaming Perfetto writer that avoids buffering events; timestamps are based
/// on the instruction index so multiple tracks stay aligned.
#[cfg(feature = "llama-tests")]
pub struct PerfettoTraceWriter {
    builder: PerfettoTraceBuilder,
    instr_track: TrackId,
    mem_track: TrackId,
    units_per_instr: u64,
    path: PathBuf,
}

#[cfg(feature = "llama-tests")]
impl PerfettoTraceWriter {
    pub fn new(path: impl Into<PathBuf>, units_per_instr: u64) -> Self {
        let mut builder = PerfettoTraceBuilder::new("LLAMA Parity");
        let instr_track = builder.add_thread("InstructionTrace");
        let mem_track = builder.add_thread("MemoryWrites");
        Self {
            builder,
            instr_track,
            mem_track,
            units_per_instr: units_per_instr.max(1),
            path: path.into(),
        }
    }

    fn ts(&self, instr_index: u64, substep: u64) -> i64 {
        (instr_index
            .saturating_mul(self.units_per_instr)
            .saturating_add(substep)) as i64
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_instr(
        &mut self,
        backend: &str,
        instr_index: u64,
        pc: u32,
        opcode: u8,
        regs: &HashMap<RegName, u32>,
        mem_imr: u8,
        mem_isr: u8,
    ) {
        let mut inst = self.builder.add_instant_event(
            self.instr_track,
            format!("Exec@0x{pc:06X}"),
            self.ts(instr_index, 0),
        );
        inst.add_annotations([
            ("backend", AnnotationValue::Str(backend.to_string())),
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("opcode", AnnotationValue::UInt(opcode as u64)),
            ("op_index", AnnotationValue::UInt(instr_index)),
            ("mem_imr", AnnotationValue::UInt(mem_imr as u64)),
            ("mem_isr", AnnotationValue::UInt(mem_isr as u64)),
        ]);
        for (reg, value) in regs {
            inst.add_annotation(reg_to_key(*reg), (*value & 0xFF_FFFF) as u64);
        }
        inst.finish();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_mem_write(
        &mut self,
        backend: &str,
        instr_index: u64,
        pc: u32,
        addr: u32,
        bits: u8,
        value: u32,
        space: MemorySpace,
    ) {
        let mut mem_ev = self.builder.add_instant_event(
            self.mem_track,
            format!("Write@0x{addr:06X}"),
            self.ts(instr_index, 1),
        );
        mem_ev.add_annotations([
            ("backend", AnnotationValue::Str(backend.to_string())),
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt((value & 0xFF_FFFF) as u64)),
            ("size", AnnotationValue::UInt(bits as u64)),
            ("op_index", AnnotationValue::UInt(instr_index)),
            // match compare_perfetto_traces expectations for space detection
            ("space", AnnotationValue::Str(space.as_str().to_string())),
        ]);
        mem_ev.finish();
    }

    pub fn finish(self) -> Result<(), String> {
        self.builder
            .save(&self.path)
            .map_err(|e| format!("perfetto save: {e}"))
    }
}

#[cfg(feature = "llama-tests")]
fn space_for_addr(addr: u32) -> MemorySpace {
    let int_end = INTERNAL_MEMORY_START + INTERNAL_SPACE as u32;
    if INTERNAL_MEMORY_START <= addr && addr < int_end {
        MemorySpace::Internal
    } else {
        MemorySpace::External
    }
}

#[cfg(feature = "llama-tests")]
fn state_snapshot_regs() -> impl Iterator<Item = RegName> {
    [
        RegName::A,
        RegName::B,
        RegName::BA,
        RegName::IL,
        RegName::IH,
        RegName::I,
        RegName::X,
        RegName::Y,
        RegName::U,
        RegName::S,
        RegName::PC,
        RegName::F,
        RegName::FC,
        RegName::FZ,
        RegName::IMR,
    ]
    .into_iter()
}

#[cfg(feature = "llama-tests")]
#[derive(Default)]
struct RecordingBus {
    mem: HashMap<u32, u8>,
    writes: Vec<MemWrite>,
}

#[cfg(feature = "llama-tests")]
impl RecordingBus {
    fn preload(&mut self, base: u32, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            self.mem.insert(base + i as u32, *b);
        }
    }

    fn load_bits(&self, addr: u32, bits: u8) -> u32 {
        let mut value = 0u32;
        let bytes = bits.div_ceil(8);
        for i in 0..bytes {
            let byte = *self.mem.get(&(addr + i as u32)).unwrap_or(&0) as u32;
            value |= byte << (8 * i);
        }
        let mask = if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        value & mask
    }

    fn store_bits(&mut self, addr: u32, bits: u8, value: u32) {
        let bytes = bits.div_ceil(8);
        for i in 0..bytes {
            let byte = ((value >> (8 * i)) & 0xFF) as u8;
            self.mem.insert(addr + i as u32, byte);
        }
        self.writes.push(MemWrite {
            addr,
            bits,
            value,
            space: space_for_addr(addr),
        });
    }
}

#[cfg(feature = "llama-tests")]
impl LlamaBus for RecordingBus {
    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        self.load_bits(addr, bits)
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        self.store_bits(addr, bits, value);
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }

    fn wait_cycles(&mut self, _cycles: u32) {}
}

/// Execute one instruction in LLAMA and produce a trace event plus snapshot.
#[cfg(feature = "llama-tests")]
pub fn run_llama_step(
    bytes: &[u8],
    regs: &[(RegName, u32)],
    pc: u32,
    instr_index: u64,
    mem: &[(u32, u8)],
) -> Result<(TraceEvent, Snapshot), String> {
    if bytes.is_empty() {
        return Err("no opcode bytes provided".into());
    }
    let mut bus = RecordingBus::default();
    bus.preload(pc, bytes);
    for (addr, val) in mem {
        bus.mem.insert(*addr & 0xFF_FFFF, *val);
    }
    let mut state = LlamaState::new();
    state.set_pc(pc);
    for (reg, value) in regs {
        state.set_reg(*reg, *value);
    }
    let mut exec = LlamaExecutor::new();
    let opcode = bytes[0];
    exec.execute(opcode, &mut state, &mut bus)
        .map_err(|e| e.to_string())?;

    let mut regs_out = HashMap::new();
    for reg in state_snapshot_regs() {
        regs_out.insert(reg, state.get_reg(reg));
    }
    let mem_imr = state.get_reg(RegName::IMR) as u8;
    // ISR is not represented as a register; capture what the bus saw (default 0).
    let mem_isr = bus.load_bits(INTERNAL_MEMORY_START + 0xFC, 8) as u8;

    let event = TraceEvent {
        backend: "llama".to_string(),
        instr_index,
        pc,
        opcode,
        regs: regs_out.clone(),
        mem_imr,
        mem_isr,
        mem_writes: bus.writes.clone(),
    };
    let snap = Snapshot {
        regs: regs_out,
        mem_writes: bus.writes,
    };
    Ok((event, snap))
}

/// High-level parity runner: execute both LLAMA and Python for a single instruction,
/// emit Perfetto traces, and return the snapshots plus trace paths.
#[cfg(feature = "llama-tests")]
pub fn run_parity_once(
    bytes: &[u8],
    regs: &[(RegName, u32)],
    pc: u32,
    instr_index: u64,
    cwd: &StdPath,
    mem: &[(u32, u8)],
) -> Result<(Snapshot, Snapshot, PathBuf, PathBuf, Output), String> {
    let (llama_event, llama_snap) = run_llama_step(bytes, regs, pc, instr_index, mem)?;
    let llama_trace = cwd.join("llama_parity.pftrace");
    write_perfetto_trace(&[llama_event], &llama_trace).map_err(|e| format!("llama trace: {e}"))?;

    let py_output =
        run_python_oracle(bytes, regs, pc, Some(cwd), Some(mem)).map_err(|e| e.to_string())?;
    let py_snap = py_output.snapshot;
    let py_trace = cwd.join("python_parity.pftrace");
    // If the oracle already emitted a trace, prefer that; otherwise serialize our own.
    if let Some(path) = py_output.perfetto_path.as_ref().map(PathBuf::from) {
        std::fs::copy(&path, &py_trace).map_err(|e| format!("copy python trace: {e}"))?;
    } else {
        write_perfetto_trace(
            &[TraceEvent {
                backend: "python".to_string(),
                instr_index,
                pc,
                opcode: bytes.first().copied().unwrap_or(0),
                regs: py_snap
                    .regs
                    .iter()
                    .filter_map(|(name, val)| reg_from_name(&format!("{name:?}")).zip(Some(*val)))
                    .collect(),
                mem_imr: *py_snap.regs.get(&RegName::IMR).unwrap_or(&0) as u8,
                // Python snapshots don’t expose ISR directly; default to 0.
                mem_isr: 0,
                mem_writes: py_snap.mem_writes.clone(),
            }],
            &py_trace,
        )
        .map_err(|e| format!("python trace: {e}"))?;
    }

    Ok((
        llama_snap,
        py_snap,
        llama_trace,
        py_trace,
        py_output.process_output,
    ))
}

/// Run the bundled compare_perfetto_traces.py script on two traces. Returns the raw process output.
#[cfg(feature = "llama-tests")]
pub fn compare_traces(trace_a: &Path, trace_b: &Path) -> Result<Output, String> {
    // Resolve script relative to workspace root (two levels up from this crate).
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("scripts").join("compare_perfetto_traces.py"))
        .ok_or("could not resolve scripts dir")?;
    if !script.exists() {
        return Err(format!("{} not found", script.display()));
    }
    Command::new("python3")
        .arg(script)
        .arg(trace_a)
        .arg(trace_b)
        .output()
        .map_err(|e| format!("compare traces failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "llama-tests")]
    use crate::INTERNAL_MEMORY_START;
    #[cfg(feature = "llama-tests")]
    use std::{env, fs};

    #[test]
    fn snapshot_roundtrip_matches() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x12);
        state.set_reg(RegName::BA, 0x3456);
        state.set_reg(RegName::FC, 1);
        let snap_a = Snapshot::from_llama(&state);
        let snap_b = Snapshot {
            regs: snap_a.regs.clone(),
            mem_writes: snap_a.mem_writes.clone(),
        };
        assert!(compare_snapshots(&snap_a, &snap_b).is_none());
    }

    #[test]
    fn detect_reg_and_mem_differences() {
        let mut lhs = Snapshot::default();
        lhs.regs.insert(RegName::A, 1);
        lhs.mem_writes.push(MemWrite {
            addr: 0x10,
            bits: 8,
            value: 0xAA,
            space: MemorySpace::Internal,
        });
        let mut rhs = Snapshot::default();
        rhs.regs.insert(RegName::A, 2);
        rhs.mem_writes.push(MemWrite {
            addr: 0x10,
            bits: 8,
            value: 0xBB,
            space: MemorySpace::Internal,
        });

        let diff = compare_snapshots(&lhs, &rhs).expect("diff expected");
        assert_eq!(diff.reg_mismatches.len(), 1);
        assert_eq!(diff.mem_mismatches.len(), 1);
    }

    /// Feature-gated integration that exercises the parity path end-to-end for NOP.
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_traces_align_for_nop() {
        // Workspace root is two levels up from this crate.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity");
        let _ = fs::create_dir_all(&workdir);
        let bytes = [0x00u8]; // NOP
        let regs: &[(RegName, u32)] = &[];
        let (llama_snap, py_snap, llama_trace, py_trace, compare_output) =
            run_parity_once(&bytes, regs, 0, 0, &workdir, &[]).expect("parity run");
        assert!(llama_snap.mem_writes.is_empty());
        assert!(py_snap.mem_writes.is_empty());
        // Compare traces with the helper script.
        let output = compare_traces(&py_trace, &llama_trace).expect("compare traces");
        if !output.status.success() {
            panic!(
                "compare traces failed: {}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        // Also assert the uv-run python invocation succeeded.
        assert!(
            compare_output.status.success(),
            "python oracle failed: {}",
            String::from_utf8_lossy(&compare_output.stdout)
        );
    }

    /// Feature-gated parity check for WAIT semantics: drains I to zero and leaves flags untouched.
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_wait_matches_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-wait");
        let _ = fs::create_dir_all(&workdir);
        let regs = &[(RegName::I, 5), (RegName::FC, 1), (RegName::FZ, 1)];
        assert_parity_mem(&[0xEF], regs, &[], &workdir);
    }

    /// Helper to run a single-instruction parity check and panic on mismatch.
    #[cfg(feature = "llama-tests")]
    fn assert_parity(bytes: &[u8], regs: &[(RegName, u32)], workdir: &Path) {
        assert_parity_mem(bytes, regs, &[], workdir);
    }

    #[cfg(feature = "llama-tests")]
    fn assert_parity_mem(bytes: &[u8], regs: &[(RegName, u32)], mem: &[(u32, u8)], workdir: &Path) {
        let (llama_snap, py_snap, _llama_trace, _py_trace, compare_output) =
            run_parity_once(bytes, regs, 0, 0, workdir, mem).expect("parity run");
        assert!(
            compare_output.status.success(),
            "python oracle failed: {}",
            String::from_utf8_lossy(&compare_output.stdout)
        );
        if let Some(diff) = compare_snapshots(&llama_snap, &py_snap) {
            panic!("parity mismatch: {:?}", diff);
        }
    }

    /// ADD A, imm8 parity check.
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_add_a_imm() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-add-imm");
        let _ = fs::create_dir_all(&workdir);
        // Opcode 0x40: ADD A, #imm8
        assert_parity(&[0x40, 0x05], &[(RegName::A, 1)], &workdir);
    }

    /// EX A,B parity check.
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_ex_ab() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-ex-ab");
        let _ = fs::create_dir_all(&workdir);
        // Opcode 0xDD: EX A,B
        assert_parity(&[0xDD], &[(RegName::A, 0x12), (RegName::B, 0x34)], &workdir);
    }

    /// SWAP nibbles parity check.
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_swap_nibbles() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-swap");
        let _ = fs::create_dir_all(&workdir);
        // Opcode 0xEE: SWAP
        assert_parity(&[0xEE], &[(RegName::A, 0xA5)], &workdir);
    }

    /// ADD A, [IMem8] parity check (0x42, offset 0x10, mem=0x05).
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_add_a_imem() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-add-imem");
        let _ = fs::create_dir_all(&workdir);
        let offset = 0x10u32;
        let mem = &[(INTERNAL_MEMORY_START + offset, 0x05)];
        assert_parity_mem(&[0x42, offset as u8], &[(RegName::A, 1)], mem, &workdir);
    }

    /// EX [IMem8], [IMem8] parity check (0xC0).
    #[cfg(feature = "llama-tests")]
    #[test]
    #[ignore = "Requires python perfetto tooling; skip in CI"]
    fn parity_ex_mem_mem() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let workdir = root.join("target").join("llama-parity-ex-mem");
        let _ = fs::create_dir_all(&workdir);
        let off1 = 0x20u32;
        let off2 = 0x21u32;
        let mem = &[
            (INTERNAL_MEMORY_START + off1, 0x11),
            (INTERNAL_MEMORY_START + off2, 0x22),
        ];
        // Opcode 0xC0: EX [IMem8],[IMem8] with two offsets following.
        assert_parity_mem(&[0xC0, off1 as u8, off2 as u8], &[], mem, &workdir);
    }
}
