use crate::Result;
use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
use std::collections::HashMap;
use std::path::PathBuf;

/// Perfetto protobuf trace writer powered by `retrobus-perfetto`.
pub struct PerfettoTracer {
    builder: PerfettoTraceBuilder,
    exec_track: TrackId,
    mem_track: TrackId,
    units_per_instr: u64,
    path: PathBuf,
}

impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        let mut builder = PerfettoTraceBuilder::new("SC62015");
        // Match Python trace naming so compare_perfetto_traces.py can ingest directly.
        let exec_track = builder.add_thread("InstructionTrace");
        let mem_track = builder.add_thread("MemoryWrites");
        Self {
            builder,
            exec_track,
            mem_track,
            units_per_instr: 10_000,
            path,
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
            match name.as_str() {
                "BA" => {
                    let ba = *value & 0xFFFF;
                    ev.add_annotation("reg_BA", ba as u64);
                    ev.add_annotation("reg_A", (ba & 0xFF) as u64);
                    ev.add_annotation("reg_B", ((ba >> 8) & 0xFF) as u64);
                }
                "I" => {
                    ev.add_annotation("reg_i", (*value & 0xFFFF) as u64);
                }
                "F" => {
                    let f = *value & 0xFF;
                    ev.add_annotation("reg_f", f as u64);
                    ev.add_annotation("flag_c", f & 0x01);
                    ev.add_annotation("flag_z", (f >> 1) & 0x01);
                }
                _ => {
                    ev.add_annotation(
                        format!("reg_{}", name.to_ascii_lowercase()),
                        (*value & 0xFF_FFFF) as u64,
                    );
                }
            }
        }
        ev.finish();
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
        let mut ev = self.builder.add_instant_event(
            self.mem_track,
            format!("Write@0x{addr:06X}"),
            self.ts(instr_index, 1),
        );
        ev.add_annotations([
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt((value & 0xFF) as u64)),
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

    /// Lightweight instant for IMEM/ISR diagnostics (used by test hooks).
    pub fn record_keyi_set(&mut self, addr: u32, value: u8) {
        let mut ev = self
            .builder
            .add_instant_event(self.exec_track, "KEYI_Set".to_string(), 0);
        ev.add_annotation("offset", addr as u64);
        ev.add_annotation("value", value as u64);
        ev.finish();
    }
}
