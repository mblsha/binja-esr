// PY_SOURCE: pce500/tracing/perfetto_tracing.py:PerfettoTracer

use crate::llama::eval::{perfetto_instr_context, perfetto_last_instr_index, perfetto_last_pc};
use crate::Result;
pub(crate) use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
use std::collections::HashMap;
use std::path::PathBuf;

/// Perfetto protobuf trace writer powered by `retrobus-perfetto`.
pub struct PerfettoTracer {
    builder: PerfettoTraceBuilder,
    exec_track: TrackId,
    timeline_track: TrackId,
    mem_track: TrackId,
    instr_counter_track: TrackId,
    imr_track: TrackId,
    imr_zero_counter: TrackId,
    imr_nonzero_counter: TrackId,
    irq_timer_track: TrackId,
    irq_key_track: TrackId,
    irq_misc_track: TrackId,
    units_per_instr: u64,
    path: PathBuf,
    imr_seq: u64,
    imr_read_zero: u64,
    imr_read_nonzero: u64,
    irq_seq: u64,
    mem_seq: u64,
    call_depth_counter: TrackId,
    mem_read_counter: TrackId,
    mem_write_counter: TrackId,
}

impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        let mut builder = PerfettoTraceBuilder::new("SC62015");
        // Match Python trace naming so compare_perfetto_traces.py can ingest directly.
        let exec_track = builder.add_thread("InstructionTrace");
        // Optional visual parity: emit a parallel Execution track like the Python tracer does.
        let timeline_track = builder.add_thread("Execution");
        let mem_track = builder.add_thread("MemoryWrites");
        let instr_counter_track = builder.add_counter_track("instructions", Some("count"), None);
        let imr_track = builder.add_thread("IMR");
        let imr_zero_counter = builder.add_counter_track("IMR_ReadZero", Some("count"), None);
        let imr_nonzero_counter = builder.add_counter_track("IMR_ReadNonZero", Some("count"), None);
        // Align IRQ tracks with Python tracer.
        let irq_timer_track = builder.add_thread("irq.timer");
        let irq_key_track = builder.add_thread("irq.key");
        let irq_misc_track = builder.add_thread("irq.misc");
        let call_depth_counter = builder.add_counter_track("call_depth", Some("depth"), None);
        let mem_read_counter = builder.add_counter_track("read_ops", Some("count"), None);
        let mem_write_counter = builder.add_counter_track("write_ops", Some("count"), None);
        Self {
            builder,
            exec_track,
            timeline_track,
            mem_track,
            instr_counter_track,
            imr_track,
            imr_zero_counter,
            imr_nonzero_counter,
            irq_timer_track,
            irq_key_track,
            irq_misc_track,
            units_per_instr: 10_000,
            path,
            imr_seq: 0,
            imr_read_zero: 0,
            imr_read_nonzero: 0,
            irq_seq: 0,
            mem_seq: 0,
            call_depth_counter,
            mem_read_counter,
            mem_write_counter,
        }
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    fn ts(&self, instr_index: u64, substep: u64) -> i64 {
        (instr_index
            .saturating_mul(self.units_per_instr)
            .saturating_add(substep)) as i64
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_regs(
        &mut self,
        instr_index: u64,
        pc: u32,
        reg_pc: u32,
        opcode: u8,
        regs: &HashMap<String, u32>,
        mem_imr: u8,
        mem_isr: u8,
    ) {
        {
            // Encode key fields in the event name so readers without interned annotations
            // can still recover pc/opcode/op_index.
            let mut ev = self.builder.add_instant_event(
                self.exec_track,
                format!("Exec@0x{pc:06X}/op=0x{opcode:02X}/idx={instr_index}"),
                self.ts(instr_index, 0),
            );
            // Duplicate annotations with explicit, non-interned keys to keep compatibility with Python proto parsers.
            ev.add_annotations([
                ("backend", AnnotationValue::Str("rust".to_string())),
                ("pc", AnnotationValue::Pointer(reg_pc as u64)),
                ("opcode", AnnotationValue::UInt(opcode as u64)),
                ("op_index", AnnotationValue::UInt(instr_index)),
                ("mem_imr", AnnotationValue::UInt(mem_imr as u64)),
                ("mem_isr", AnnotationValue::UInt(mem_isr as u64)),
                // Redundant, uninferred keys to bypass interned name lookups.
                ("pc_raw", AnnotationValue::Pointer(reg_pc as u64)),
                ("opcode_raw", AnnotationValue::UInt(opcode as u64)),
                ("op_index_raw", AnnotationValue::UInt(instr_index)),
            ]);
            for (name, value) in regs {
                let key = format!("reg_{}", name.to_ascii_lowercase());
                let masked = if name == "BA" {
                    *value & 0xFFFF
                } else if name == "F" {
                    *value & 0xFF
                } else {
                    *value & 0xFF_FFFF
                };
                ev.add_annotation(key, masked as u64);
            }
            ev.finish();
        }

        // Duplicate on the Execution track for UI parity with Python traces.
        {
            let mut exec = self.builder.add_instant_event(
                self.timeline_track,
                "Execution".to_string(),
                self.ts(instr_index, 0),
            );
            exec.add_annotations([
                ("pc", AnnotationValue::Pointer(reg_pc as u64)),
                ("opcode", AnnotationValue::UInt(opcode as u64)),
                ("op_index", AnnotationValue::UInt(instr_index)),
            ]);
            exec.finish();
        }

        // Keep an explicit instruction counter aligned to InstructionTrace.
        self.builder.update_counter(
            self.instr_counter_track,
            instr_index as f64 + 1.0,
            self.ts(instr_index, 0),
        );
    }

    pub fn record_mem_write(
        &mut self,
        instr_index: u64,
        pc: u32,
        addr: u32,
        value: u32,
        space: &str,
        size: u8,
    ) {
        let masked_value = if size == 0 || size >= 32 {
            value
        } else {
            value & ((1u32 << size) - 1)
        };
        let mut ev = self.builder.add_instant_event(
            self.mem_track,
            format!("Write@0x{addr:06X}"),
            self.ts(instr_index, 1),
        );
        ev.add_annotations([
            ("backend", AnnotationValue::Str("rust".to_string())),
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt(masked_value as u64)),
            ("space", AnnotationValue::Str(space.to_string())),
            ("size", AnnotationValue::UInt(size as u64)),
            ("op_index", AnnotationValue::UInt(instr_index)),
        ]);
        ev.finish();
    }

    /// Record a memory write at a specific manual-clock cycle (used for host/applied writes outside executor).
    pub fn record_mem_write_at_cycle(
        &mut self,
        cycle: u64,
        pc: Option<u32>,
        addr: u32,
        value: u32,
        space: &str,
        size: u8,
    ) {
        let masked_value = if size == 0 || size >= 32 {
            value
        } else {
            value & ((1u32 << size) - 1)
        };
        let (ctx_op, ctx_pc) = perfetto_instr_context()
            .unwrap_or_else(|| (perfetto_last_instr_index(), perfetto_last_pc()));
        let ts = if ctx_op != u64::MAX {
            self.ts(ctx_op, 1)
        } else {
            // Fallback: maintain monotonic ordering even without instruction context.
            self.mem_seq = self.mem_seq.saturating_add(1);
            self.mem_seq as i64
        };
        let mut ev =
            self.builder
                .add_instant_event(self.mem_track, format!("Write@0x{addr:06X}"), ts);
        ev.add_annotations([
            ("backend", AnnotationValue::Str("rust".to_string())),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt(masked_value as u64)),
            ("space", AnnotationValue::Str(space.to_string())),
            ("size", AnnotationValue::UInt(size as u64)),
        ]);
        if let Some(pc_val) = pc.or(if ctx_op != u64::MAX { Some(ctx_pc) } else { None }) {
            ev.add_annotation("pc", AnnotationValue::Pointer(pc_val as u64));
        }
        if ctx_op != u64::MAX {
            ev.add_annotation("op_index", AnnotationValue::UInt(ctx_op));
        }
        ev.add_annotation("cycle", AnnotationValue::UInt(cycle));
        ev.finish();
    }

    pub fn update_counters(
        &mut self,
        instr_index: u64,
        call_depth: u32,
        mem_reads: u64,
        mem_writes: u64,
    ) {
        let ts = self.ts(instr_index, 0);
        self.builder
            .update_counter(self.call_depth_counter, call_depth as f64, ts);
        self.builder
            .update_counter(self.mem_read_counter, mem_reads as f64, ts);
        self.builder
            .update_counter(self.mem_write_counter, mem_writes as f64, ts);
    }

    pub fn finish(self) -> Result<()> {
        self.builder
            .save(&self.path)
            .map_err(|e| crate::CoreError::Other(format!("perfetto save: {e}")))
    }

    /// IMR read diagnostics: log each read and keep running zero/non-zero counters.
    /// If an instruction index is available, align the timestamp to that index to match Python.
    pub fn record_imr_read(&mut self, pc: Option<u32>, value: u8, instr_index: Option<u64>) {
        let zero = value == 0;
        if zero {
            self.imr_read_zero = self.imr_read_zero.saturating_add(1);
        } else {
            self.imr_read_nonzero = self.imr_read_nonzero.saturating_add(1);
        }
        let ts = if let Some(op) = instr_index {
            self.ts(op, 0)
        } else {
            self.imr_seq as i64
        };
        self.imr_seq = self.imr_seq.saturating_add(1);

        // Mirror Python tracer: update IMR counters.
        if zero {
            self.builder.update_counter(
                self.imr_zero_counter,
                self.imr_read_zero as f64,
                ts as i64,
            );
        } else {
            self.builder.update_counter(
                self.imr_nonzero_counter,
                self.imr_read_nonzero as f64,
                ts as i64,
            );
        }

        let mut ev = self
            .builder
            .add_instant_event(self.imr_track, "IMR_Read".to_string(), ts);
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", pc_val as u64);
        }
        ev.add_annotation("value", value as u64);
        ev.add_annotation("zero", zero as u64);
        ev.add_annotation("count_zero", self.imr_read_zero);
        ev.add_annotation("count_nonzero", self.imr_read_nonzero);
        ev.finish();
    }

    /// IMEM effective address diagnostics to align with Python tracing.
    pub fn record_imem_addr(
        &mut self,
        mode: &str,
        base: u32,
        bp: u32,
        px: u32,
        py: u32,
        op_index: Option<u64>,
        pc: Option<u32>,
    ) {
        let ts = op_index.map(|idx| self.ts(idx, 0)).unwrap_or(0);
        let mut ev =
            self.builder
                .add_instant_event(self.exec_track, "IMEM_EffectiveAddr".to_string(), ts);
        ev.add_annotation("mode", mode.to_string());
        ev.add_annotation("base", base as u64);
        ev.add_annotation("bp", bp as u64);
        ev.add_annotation("px", px as u64);
        ev.add_annotation("py", py as u64);
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", pc_val as u64);
        }
        if let Some(idx) = op_index {
            ev.add_annotation("op_index", idx);
        }
        ev.finish();
    }

    /// Lightweight instant for IMEM/ISR diagnostics (used by test hooks).
    pub fn record_keyi_set(
        &mut self,
        addr: u32,
        value: u8,
        op_index: Option<u64>,
        pc: Option<u32>,
    ) {
        let ts = op_index.map(|idx| self.ts(idx, 0)).unwrap_or(0);
        let mut ev = self
            .builder
            .add_instant_event(self.exec_track, "KEYI_Set".to_string(), ts);
        ev.add_annotation("offset", addr as u64);
        ev.add_annotation("value", value as u64);
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", pc_val as u64);
        }
        if let Some(idx) = op_index {
            ev.add_annotation("op_index", idx);
        }
        ev.finish();
    }

    /// Generic KIO read hook for KOL/KOH/KIL visibility.
    pub fn record_kio_read(
        &mut self,
        pc: Option<u32>,
        offset: u8,
        value: u8,
        op_index: Option<u64>,
    ) {
        let ts = op_index.map(|idx| self.ts(idx, 0)).unwrap_or(0);
        let mut ev = self
            .builder
            .add_instant_event(self.exec_track, "read@KIO".to_string(), ts);
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", pc_val as u64);
        }
        ev.add_annotation("offset", offset as u64);
        ev.add_annotation("value", value as u64);
        if let Some(idx) = op_index {
            ev.add_annotation("op_index", idx);
        }
        ev.finish();
    }

    /// Timer/IRQ events for parity tracing (MTI/STI/KEYI etc.).
    pub fn record_irq_event(&mut self, name: &str, mut payload: HashMap<String, AnnotationValue>) {
        // Prefer explicit cycle payload (manual clock) when present to align with Python traces.
        let cycle_ts = payload.get("cycle").and_then(|v| match v {
            AnnotationValue::UInt(c) => Some(*c as i64),
            _ => None,
        });

        // Align IRQ/key events to the instruction index when available and when no manual cycle is provided.
        let (op_idx, pc) = perfetto_instr_context()
            .unwrap_or_else(|| (perfetto_last_instr_index(), perfetto_last_pc()));
        let have_ctx = op_idx != u64::MAX && cycle_ts.is_none();
        if have_ctx {
            payload
                .entry("op_index".to_string())
                .or_insert(AnnotationValue::UInt(op_idx));
            payload
                .entry("pc".to_string())
                .or_insert(AnnotationValue::Pointer(pc as u64));
        }

        let ts = cycle_ts.unwrap_or_else(|| {
            if have_ctx {
                self.ts(op_idx, 0)
            } else if op_idx != u64::MAX {
                self.ts(op_idx, 0)
            } else {
                self.irq_seq as i64
            }
        });
        self.irq_seq = self.irq_seq.saturating_add(1);
        let track = match name {
            "TimerFired" | "Timer" | "MTI" | "STI" => self.irq_timer_track,
            "KEYI_Set" | "KeyScan" | "KeyScanEvent" | "KeyScanEmpty" | "KEYI" | "KeyDeliver" => {
                self.irq_key_track
            }
            "IRQ_Enter" | "IRQ_Exit" | "IRQ_Delivered" => {
                // Derive track from src payload when available; default to misc.
                let src_track = payload
                    .get("src")
                    .and_then(|v| match v {
                        AnnotationValue::Str(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .map(|s| s.to_ascii_uppercase());
                match src_track.as_deref() {
                    Some("KEY") | Some("ONK") => self.irq_key_track,
                    Some("MTI") | Some("STI") => self.irq_timer_track,
                    _ => self.irq_misc_track,
                }
            }
            _ => self.irq_misc_track,
        };
        let mut ev = self.builder.add_instant_event(track, name.to_string(), ts);
        for (k, v) in payload {
            ev.add_annotation(k, v);
        }
        ev.finish();
    }

    /// Diagnostic instants mirroring Python IRQ pending checks (IRQ_Check/IRQ_PendingCheck/IMR_ReadZero).
    pub fn record_irq_check(
        &mut self,
        name: &str,
        pc: u32,
        imr: u8,
        isr: u8,
        pending: bool,
        in_interrupt: bool,
        pending_src: Option<&str>,
    ) {
        let mut payload = HashMap::new();
        payload.insert("pc".to_string(), AnnotationValue::Pointer(pc as u64));
        payload.insert("imr".to_string(), AnnotationValue::UInt(imr as u64));
        payload.insert("isr".to_string(), AnnotationValue::UInt(isr as u64));
        payload.insert("pending".to_string(), AnnotationValue::UInt(pending as u64));
        payload.insert(
            "in_interrupt".to_string(),
            AnnotationValue::UInt(in_interrupt as u64),
        );
        if let Some(src) = pending_src {
            payload.insert(
                "pending_src".to_string(),
                AnnotationValue::Str(src.to_string()),
            );
        }
        self.record_irq_event(name, payload);
    }

    /// Function enter/exit markers to mirror Python call/return tracing.
    pub fn record_call_flow(&mut self, name: &str, pc_from: u32, pc_to: u32, depth: u32) {
        let mut payload = HashMap::new();
        payload.insert("from".to_string(), AnnotationValue::Pointer(pc_from as u64));
        payload.insert("to".to_string(), AnnotationValue::Pointer(pc_to as u64));
        payload.insert("depth".to_string(), AnnotationValue::UInt(depth as u64));
        self.record_irq_event(name, payload);
    }
}
