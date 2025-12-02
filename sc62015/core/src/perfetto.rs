// PY_SOURCE: pce500/tracing/perfetto_tracing.py:PerfettoTracer

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
    imr_track: TrackId,
    irq_timer_track: TrackId,
    irq_key_track: TrackId,
    irq_misc_track: TrackId,
    units_per_instr: u64,
    path: PathBuf,
    imr_seq: u64,
    imr_read_zero: u64,
    imr_read_nonzero: u64,
    irq_seq: u64,
}

impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        let mut builder = PerfettoTraceBuilder::new("SC62015");
        // Match Python trace naming so compare_perfetto_traces.py can ingest directly.
        let exec_track = builder.add_thread("InstructionTrace");
        // Optional visual parity: emit a parallel Execution track like the Python tracer does.
        let timeline_track = builder.add_thread("Execution");
        let mem_track = builder.add_thread("MemoryWrites");
        let imr_track = builder.add_thread("IMR");
        // Align IRQ tracks with Python tracer.
        let irq_timer_track = builder.add_thread("irq.timer");
        let irq_key_track = builder.add_thread("irq.key");
        let irq_misc_track = builder.add_thread("irq.misc");
        Self {
            builder,
            exec_track,
            timeline_track,
            mem_track,
            imr_track,
            irq_timer_track,
            irq_key_track,
            irq_misc_track,
            units_per_instr: 10_000,
            path,
            imr_seq: 0,
            imr_read_zero: 0,
            imr_read_nonzero: 0,
            irq_seq: 0,
        }
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
            let mut ev = self.builder.add_instant_event(
                self.exec_track,
                format!("Exec@0x{pc:06X}"),
                self.ts(instr_index, 0),
            );
            ev.add_annotations([
                ("backend", AnnotationValue::Str("rust".to_string())),
                ("pc", AnnotationValue::Pointer(reg_pc as u64)),
                ("opcode", AnnotationValue::UInt(opcode as u64)),
                ("op_index", AnnotationValue::UInt(instr_index)),
                ("mem_imr", AnnotationValue::UInt(mem_imr as u64)),
                ("mem_isr", AnnotationValue::UInt(mem_isr as u64)),
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
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt(masked_value as u64)),
            ("space", AnnotationValue::Str(space.to_string())),
            ("size", AnnotationValue::UInt(size as u64)),
            ("op_index", AnnotationValue::UInt(instr_index)),
        ]);
        ev.finish();
    }

    pub fn finish(self) -> Result<()> {
        self.builder
            .save(&self.path)
            .map_err(|e| crate::CoreError::Other(format!("perfetto save: {e}")))
    }

    /// IMR read diagnostics: log each read and keep running zero/non-zero counters.
    pub fn record_imr_read(&mut self, pc: Option<u32>, value: u8) {
        let zero = value == 0;
        if zero {
            self.imr_read_zero = self.imr_read_zero.saturating_add(1);
        } else {
            self.imr_read_nonzero = self.imr_read_nonzero.saturating_add(1);
        }
        let ts = self.imr_seq;
        self.imr_seq = self.imr_seq.saturating_add(1);

        let mut ev =
            self.builder
                .add_instant_event(self.imr_track, "IMR_Read".to_string(), ts as i64);
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
    pub fn record_imem_addr(&mut self, mode: &str, base: u32, bp: u32, px: u32, py: u32) {
        let mut ev =
            self.builder
                .add_instant_event(self.exec_track, "IMEM_EffectiveAddr".to_string(), 0);
        ev.add_annotation("mode", mode.to_string());
        ev.add_annotation("base", base as u64);
        ev.add_annotation("bp", bp as u64);
        ev.add_annotation("px", px as u64);
        ev.add_annotation("py", py as u64);
        ev.finish();
    }

    /// Lightweight instant for IMEM/ISR diagnostics (used by test hooks).
    pub fn record_keyi_set(&mut self, addr: u32, value: u8) {
        let mut ev = self
            .builder
            .add_instant_event(self.exec_track, "KEYI_Set".to_string(), 0);
        ev.add_annotation("offset", addr as u64);
        ev.add_annotation("value", value as u64);
        ev.finish();
    }

    /// Generic KIO read hook for KOL/KOH/KIL visibility.
    pub fn record_kio_read(&mut self, pc: Option<u32>, offset: u8, value: u8) {
        let mut ev = self
            .builder
            .add_instant_event(self.exec_track, "read@KIO".to_string(), 0);
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", pc_val as u64);
        }
        ev.add_annotation("offset", offset as u64);
        ev.add_annotation("value", value as u64);
        ev.finish();
    }

    /// Timer/IRQ events for parity tracing (MTI/STI/KEYI etc.).
    pub fn record_irq_event(&mut self, name: &str, payload: HashMap<String, u64>) {
        let ts = self.irq_seq as i64;
        self.irq_seq = self.irq_seq.saturating_add(1);
        let track = match name {
            "TimerFired" | "Timer" | "MTI" | "STI" => self.irq_timer_track,
            "KEYI_Set" | "KeyScan" | "KEYI" | "KeyDeliver" => self.irq_key_track,
            _ => self.irq_misc_track,
        };
        let mut ev = self
            .builder
            .add_instant_event(track, name.to_string(), ts);
        for (k, v) in payload {
            ev.add_annotation(k, v);
        }
        ev.finish();
    }
}
