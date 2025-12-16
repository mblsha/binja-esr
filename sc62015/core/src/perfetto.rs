// PY_SOURCE: pce500/tracing/perfetto_tracing.py:PerfettoTracer

use crate::Result;
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "perfetto")]
use crate::CoreError;
#[cfg(feature = "perfetto")]
use crate::llama::eval::{perfetto_instr_context, perfetto_last_instr_index, perfetto_last_pc};
#[cfg(feature = "perfetto")]
pub(crate) use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
#[cfg(all(test, feature = "perfetto"))]
use std::cell::RefCell;

#[cfg(not(feature = "perfetto"))]
#[derive(Clone, Debug)]
pub enum AnnotationValue {
    Pointer(u64),
    UInt(u64),
    Str(String),
}

/// Perfetto protobuf trace writer powered by `retrobus-perfetto`.
#[cfg(feature = "perfetto")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfettoLayout {
    /// Python-compatible track names/events used by parity tooling.
    PythonCompat,
    /// UI-oriented call traces used by the web Function Runner `e.call(..., { trace: true })`.
    CallUi,
}

/// Perfetto protobuf trace writer powered by `retrobus-perfetto`.
#[cfg(feature = "perfetto")]
pub struct PerfettoTracer {
    builder: PerfettoTraceBuilder,
    layout: PerfettoLayout,
    exec_track: TrackId,
    timeline_track: TrackId,
    mem_track: TrackId,
    memory_track: TrackId,
    instr_counter_track: TrackId,
    imr_track: TrackId,
    imr_zero_counter: TrackId,
    imr_nonzero_counter: TrackId,
    irq_timer_track: TrackId,
    irq_key_track: TrackId,
    irq_misc_track: TrackId,
    display_track: TrackId,
    units_per_instr: u64,
    path: PathBuf,
    imr_seq: u64,
    imr_read_zero: u64,
    imr_read_nonzero: u64,
    irq_seq: u64,
    display_seq: u64,
    call_depth_counter: TrackId,
    mem_read_counter: TrackId,
    mem_write_counter: TrackId,
    cpu_track: TrackId,
    functions_track: TrackId,
    instructions_track: TrackId,
    ewrites_track: TrackId,
    iwrites_track: TrackId,
    call_ui_root_open: bool,
    call_ui_root_addr: u32,
    #[cfg(test)]
    test_exec_events: RefCell<Vec<(u32, u8, u64)>>, // pc, opcode, op_index
    #[cfg(test)]
    test_timestamps: RefCell<Vec<i64>>,
    #[cfg(test)]
    pub(crate) test_mem_write_pcs: RefCell<Vec<Option<u32>>>,
    #[cfg(test)]
    pub(crate) test_counters: RefCell<Vec<(u64, u32, u64, u64)>>, // (instr_index, call_depth, reads, writes)
}

#[cfg(feature = "perfetto")]
impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        Self::new_with_layout(path, PerfettoLayout::PythonCompat)
    }

    pub fn new_call_ui(path: PathBuf) -> Self {
        Self::new_with_layout(path, PerfettoLayout::CallUi)
    }

    fn new_with_layout(path: PathBuf, layout: PerfettoLayout) -> Self {
        let mut builder = PerfettoTraceBuilder::new("SC62015");
        // Default to Python-compatible layout/timestamps.
        let compat_python = true;

        // Match Python trace naming so compare_perfetto_traces.py can ingest directly.
        let exec_track = builder.add_thread("InstructionTrace");
        // Optional visual parity: emit parallel Execution/CPU tracks regardless of layout flag.
        let timeline_track = builder.add_thread("Execution");
        let mem_track = if compat_python {
            builder.add_thread("Memory")
        } else {
            builder.add_thread("MemoryWrites")
        };
        // Track aliases to match Python tracer naming.
        let memory_track = builder.add_thread("Memory");
        let cpu_track = builder.add_thread("CPU");
        let instr_counter_track = builder.add_counter_track("instructions", Some("count"), None);
        let imr_track = builder.add_thread("IMR");
        let imr_zero_counter = builder.add_counter_track("IMR_ReadZero", Some("count"), None);
        let imr_nonzero_counter = builder.add_counter_track("IMR_ReadNonZero", Some("count"), None);
        // Align IRQ tracks with Python tracer.
        let irq_timer_track = builder.add_thread("irq.timer");
        let irq_key_track = builder.add_thread("irq.key");
        let irq_misc_track = builder.add_thread("irq.misc");
        let display_track = builder.add_thread("Display");
        let functions_track = builder.add_thread("Functions");
        let instructions_track = builder.add_thread("Instructions");
        let ewrites_track = builder.add_thread("EWrites");
        let iwrites_track = builder.add_thread("IWrites");
        let call_depth_counter = builder.add_counter_track("call_depth", Some("depth"), None);
        let mem_read_counter = builder.add_counter_track("read_ops", Some("count"), None);
        let mem_write_counter = builder.add_counter_track("write_ops", Some("count"), None);
        let units_per_instr = match layout {
            // retrobus-perfetto stores timestamps in microseconds; it converts our input
            // nanoseconds by dividing by 1000. Use >=1000ns so short traces don't collapse to 0.
            PerfettoLayout::CallUi => 1_000, // 1 "tick" (instruction) = 1µs
            PerfettoLayout::PythonCompat => 1,
        };
        Self {
            builder,
            layout,
            exec_track,
            timeline_track,
            mem_track,
            memory_track,
            instr_counter_track,
            imr_track,
            imr_zero_counter,
            imr_nonzero_counter,
            irq_timer_track,
            irq_key_track,
            irq_misc_track,
            display_track,
            units_per_instr,
            path,
            imr_seq: 0,
            imr_read_zero: 0,
            imr_read_nonzero: 0,
            irq_seq: 0,
            display_seq: 0,
            call_depth_counter,
            mem_read_counter,
            mem_write_counter,
            cpu_track,
            functions_track,
            instructions_track,
            ewrites_track,
            iwrites_track,
            call_ui_root_open: false,
            call_ui_root_addr: 0,
            #[cfg(test)]
            test_exec_events: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_timestamps: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_mem_write_pcs: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_counters: RefCell::new(Vec::new()),
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

    /// Begin a fake top-level function slice for web `e.call` traces.
    ///
    /// This provides a stable "root" slice on the `Functions` track named by the
    /// target address, even though the call harness enters the function by setting
    /// `PC=addr` rather than executing a real CALL instruction.
    pub fn call_ui_begin_root(&mut self, addr: u32) {
        if self.layout != PerfettoLayout::CallUi || self.call_ui_root_open {
            return;
        }
        let addr = addr & 0x000f_ffff;
        let mut ev = self.builder.begin_slice(
            self.functions_track,
            format!("root@0x{addr:06X}"),
            self.ts(0, 0),
        );
        ev.add_annotations([
            ("to", AnnotationValue::Pointer(addr as u64)),
            ("depth", AnnotationValue::UInt(0)),
        ]);
        ev.finish();
        self.call_ui_root_open = true;
        self.call_ui_root_addr = addr;
    }

    fn call_ui_end_root(&mut self) {
        if self.layout != PerfettoLayout::CallUi || !self.call_ui_root_open {
            return;
        }
        let end_ts = self.ts(perfetto_last_instr_index(), 0);
        self.builder.end_slice(self.functions_track, end_ts);
        self.call_ui_root_open = false;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_regs(
        &mut self,
        instr_index: u64,
        pc: u32,
        reg_pc: u32,
        opcode: u8,
        mnemonic: Option<&str>,
        regs: &HashMap<String, u32>,
        mem_imr: u8,
        mem_isr: u8,
    ) {
        if self.layout == PerfettoLayout::CallUi {
            let ts_start = self.ts(instr_index, 0);
            let ts_end = self.ts(instr_index.saturating_add(1), 0);
            let ts_counter = ts_start;
            let name = if let Some(m) = mnemonic {
                format!("{m} @0x{pc:06X}")
            } else {
                format!("op=0x{opcode:02X} @0x{pc:06X}")
            };
            {
                let mut ev = self
                    .builder
                    .begin_slice(self.instructions_track, name, ts_start);
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
            self.builder.end_slice(self.instructions_track, ts_end);

            self.builder.update_counter(
                self.instr_counter_track,
                instr_index as f64 + 1.0,
                ts_counter,
            );
            return;
        }

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

        // Duplicate on the CPU track for parity with Python tracer naming.
        {
            let mut cpu_ev = self.builder.add_instant_event(
                self.cpu_track,
                "Exec".to_string(),
                self.ts(instr_index, 0),
            );
            cpu_ev.add_annotations([
                ("pc", AnnotationValue::Pointer(reg_pc as u64)),
                ("opcode", AnnotationValue::UInt(opcode as u64)),
                ("op_index", AnnotationValue::UInt(instr_index)),
                ("backend", AnnotationValue::Str("rust".to_string())),
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
                cpu_ev.add_annotation(key, masked as u64);
            }
            cpu_ev.finish();
        }

        // Keep an explicit instruction counter aligned to InstructionTrace.
        self.builder.update_counter(
            self.instr_counter_track,
            instr_index as f64 + 1.0,
            self.ts(instr_index, 0),
        );
        #[cfg(test)]
        {
        self.test_exec_events
            .borrow_mut()
            .push((pc, opcode, instr_index));
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
        let substep = 1;
        self.record_mem_write_with_substep(instr_index, pc, addr, value, space, size, substep);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_mem_write_with_substep(
        &mut self,
        instr_index: u64,
        pc: u32,
        addr: u32,
        value: u32,
        space: &str,
        size: u8,
        substep: u64,
    ) {
        let masked_value = if size == 0 || size >= 32 {
            value
        } else {
            value & ((1u32 << size) - 1)
        };
        let ts = self.ts(instr_index, substep.max(1));
        if self.layout == PerfettoLayout::CallUi {
            let track = if space == "internal" {
                self.iwrites_track
            } else {
                self.ewrites_track
            };
            let mut ev =
                self.builder
                    .add_instant_event(track, format!("Write@0x{addr:06X}"), ts);
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
            return;
        }
        {
            let mut ev =
                self.builder
                    .add_instant_event(self.mem_track, format!("Write@0x{addr:06X}"), ts);
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
        #[cfg(test)]
        {
            self.test_timestamps.borrow_mut().push(ts);
        }

        // Also mirror to "Memory" track to match Python tracer naming.
        let mut mem_alias =
            self.builder
                .add_instant_event(self.memory_track, format!("Write@0x{addr:06X}"), ts);
        mem_alias.add_annotations([
            ("backend", AnnotationValue::Str("rust".to_string())),
            ("pc", AnnotationValue::Pointer(pc as u64)),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt(masked_value as u64)),
            ("space", AnnotationValue::Str(space.to_string())),
            ("size", AnnotationValue::UInt(size as u64)),
            ("op_index", AnnotationValue::UInt(instr_index)),
        ]);
        mem_alias.finish();
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
        let ctx = perfetto_instr_context();
        let pc_effective = pc
            .or(ctx.map(|(_, pc)| pc))
            .or_else(|| Some(perfetto_last_pc()));
        let ts = ctx
            .map(|(op, _)| self.ts(op, 1))
            // Align to provided manual cycle when no live instruction context.
            .unwrap_or(cycle as i64);
        {
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
            if let Some(pc_val) = pc_effective {
                ev.add_annotation("pc", AnnotationValue::Pointer(pc_val as u64));
            }
            if let Some((op_idx, _)) = ctx {
                ev.add_annotation("op_index", AnnotationValue::UInt(op_idx));
            }
            ev.add_annotation("cycle", AnnotationValue::UInt(cycle));
            ev.finish();
        }

        // Mirror to Memory track for Python parity.
        let mut mem_alias =
            self.builder
                .add_instant_event(self.memory_track, format!("Write@0x{addr:06X}"), ts);
        mem_alias.add_annotations([
            ("backend", AnnotationValue::Str("rust".to_string())),
            ("address", AnnotationValue::Pointer(addr as u64)),
            ("value", AnnotationValue::UInt(masked_value as u64)),
            ("space", AnnotationValue::Str(space.to_string())),
            ("size", AnnotationValue::UInt(size as u64)),
            ("cycle", AnnotationValue::UInt(cycle)),
        ]);
        if let Some(pc_val) = pc_effective {
            mem_alias.add_annotation("pc", AnnotationValue::Pointer(pc_val as u64));
        }
        if let Some((op_idx, _)) = ctx {
            mem_alias.add_annotation("op_index", AnnotationValue::UInt(op_idx));
        }
        mem_alias.finish();

        #[cfg(test)]
        self.test_mem_write_pcs.borrow_mut().push(pc_effective);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_lcd_event(
        &mut self,
        name: &str,
        addr: u32,
        value: u8,
        chip: usize,
        page: u8,
        column: u8,
        pc: Option<u32>,
        op_index: Option<u64>,
    ) {
        let ts = op_index
            .map(|idx| self.ts(idx, 0))
            .unwrap_or_else(|| self.display_seq as i64);
        self.display_seq = self.display_seq.saturating_add(1);
        let mut ev = self
            .builder
            .add_instant_event(self.display_track, name.to_string(), ts);
        ev.add_annotation("address", AnnotationValue::Pointer(addr as u64));
        ev.add_annotation("value", AnnotationValue::UInt(value as u64));
        ev.add_annotation("chip", AnnotationValue::UInt(chip as u64));
        ev.add_annotation("page", AnnotationValue::UInt(page as u64));
        ev.add_annotation("column", AnnotationValue::UInt(column as u64));
        if let Some(pc_val) = pc {
            ev.add_annotation("pc", AnnotationValue::Pointer(pc_val as u64));
        }
        if let Some(idx) = op_index {
            ev.add_annotation("op_index", AnnotationValue::UInt(idx));
        }
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
        #[cfg(test)]
        self.test_counters
            .borrow_mut()
            .push((instr_index, call_depth, mem_reads, mem_writes));
    }

    pub fn finish(self) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = self;
            Err(CoreError::Other(
                "PerfettoTracer::finish is not supported on wasm32; use serialize()".to_string(),
            ))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut this = self;
            this.call_ui_end_root();
            this.builder
                .save(&this.path)
                .map_err(|e| CoreError::Other(format!("perfetto save: {e}")))
        }
    }

    pub fn serialize(self) -> Result<Vec<u8>> {
        let mut this = self;
        this.call_ui_end_root();
        this.builder
            .serialize()
            .map_err(|e| CoreError::Other(format!("perfetto serialize: {e}")))
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
            self.builder
                .update_counter(self.imr_zero_counter, self.imr_read_zero as f64, ts);
        } else {
            self.builder
                .update_counter(self.imr_nonzero_counter, self.imr_read_nonzero as f64, ts);
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
    #[allow(clippy::too_many_arguments)]
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
        let fallback = {
            let last = perfetto_last_instr_index();
            (last != u64::MAX).then_some(last)
        };
        let ts = op_index
            .or(fallback)
            .map(|idx| self.ts(idx, 0))
            .unwrap_or(self.imr_seq as i64);
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
        let fallback = {
            let last = perfetto_last_instr_index();
            (last != u64::MAX).then_some(last)
        };
        let ts = op_index
            .or(fallback)
            .map(|idx| self.ts(idx, 0))
            .unwrap_or(self.imr_seq as i64);
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
        let fallback = {
            let last = perfetto_last_instr_index();
            (last != u64::MAX).then_some(last)
        };
        let ts = op_index
            .or(fallback)
            .map(|idx| self.ts(idx, 0))
            .unwrap_or(self.imr_seq as i64);
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
    pub fn record_irq_event(&mut self, name: &str, payload: HashMap<String, AnnotationValue>) {
        // Prefer explicit cycle payload (manual clock) when present to align with Python traces.
        let cycle_ts = payload.get("cycle").and_then(|v| match v {
            AnnotationValue::UInt(c) => Some(*c as i64),
            _ => None,
        });

        // Align IRQ/key events to the instruction index when available and when no manual cycle is provided.
        let (ctx_idx, _pc) = perfetto_instr_context()
            .unwrap_or_else(|| (perfetto_last_instr_index(), perfetto_last_pc()));
        let ctx_idx = (ctx_idx != u64::MAX).then_some(ctx_idx);
        let last_idx = {
            let last = perfetto_last_instr_index();
            (last != u64::MAX).then_some(last)
        };

        let ts = irq_timestamp(
            cycle_ts,
            ctx_idx,
            last_idx,
            self.units_per_instr,
            self.irq_seq,
        );
        self.irq_seq = self.irq_seq.saturating_add(1);
        let track = match classify_irq_track(name, &payload) {
            IrqTrack::Timer => self.irq_timer_track,
            IrqTrack::Key => self.irq_key_track,
            IrqTrack::Misc => self.irq_misc_track,
        };
        let mut ev = self.builder.add_instant_event(track, name.to_string(), ts);
        for (k, v) in payload {
            ev.add_annotation(k, v);
        }
        ev.finish();
    }

    /// Diagnostic instants mirroring Python IRQ pending checks (IRQ_Check/IRQ_PendingCheck/IMR_ReadZero).
    #[allow(clippy::too_many_arguments)]
    pub fn record_irq_check(
        &mut self,
        name: &str,
        pc: u32,
        imr: u8,
        isr: u8,
        pending: bool,
        in_interrupt: bool,
        pending_src: Option<&str>,
        kil: Option<u8>,
        imr_reg: Option<u8>,
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
        if let Some(k) = kil {
            payload.insert("kil".to_string(), AnnotationValue::UInt(k as u64));
        }
        if let Some(imr_reg_val) = imr_reg {
            payload.insert(
                "imr_reg".to_string(),
                AnnotationValue::UInt(imr_reg_val as u64),
            );
        }
        self.record_irq_event(name, payload);
    }

    /// Function enter/exit markers to mirror Python call/return tracing.
    pub fn record_call_flow(&mut self, name: &str, pc_from: u32, pc_to: u32, depth: u32) {
        if self.layout == PerfettoLayout::CallUi {
            let ctx = perfetto_instr_context();
            let op = ctx.map(|(idx, _)| idx).unwrap_or_else(perfetto_last_instr_index);
            let ts = self.ts(op, 0);
            if name == "CALL" {
                let mut ev = self.builder.begin_slice(
                    self.functions_track,
                    format!("fn@0x{pc_to:06X}"),
                    ts,
                );
                ev.add_annotations([
                    ("from", AnnotationValue::Pointer(pc_from as u64)),
                    ("to", AnnotationValue::Pointer(pc_to as u64)),
                    ("depth", AnnotationValue::UInt(depth as u64)),
                    ("op_index", AnnotationValue::UInt(op)),
                ]);
                ev.finish();
            } else if name == "RET" || name == "RETF" {
                self.builder.end_slice(
                    self.functions_track,
                    ts.saturating_add(self.units_per_instr as i64),
                );
            }
            return;
        }
        let mut payload = HashMap::new();
        payload.insert("from".to_string(), AnnotationValue::Pointer(pc_from as u64));
        payload.insert("to".to_string(), AnnotationValue::Pointer(pc_to as u64));
        payload.insert("depth".to_string(), AnnotationValue::UInt(depth as u64));
        self.record_irq_event(name, payload);
    }

    #[cfg(test)]
    pub fn test_exec_events(&self) -> Vec<(u32, u8, u64)> {
        self.test_exec_events.borrow().clone()
    }
}

#[cfg(not(feature = "perfetto"))]
pub struct PerfettoTracer {
    _path: PathBuf,
}

#[cfg(not(feature = "perfetto"))]
impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        Self { _path: path }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_regs(
        &mut self,
        _instr_index: u64,
        _pc: u32,
        _reg_pc: u32,
        _opcode: u8,
        _regs: &HashMap<String, u32>,
        _mem_imr: u8,
        _mem_isr: u8,
    ) {
    }

    pub fn record_mem_write(
        &mut self,
        _instr_index: u64,
        _pc: u32,
        _addr: u32,
        _value: u32,
        _space: &str,
        _size: u8,
    ) {
    }

    pub fn record_mem_write_with_substep(
        &mut self,
        _instr_index: u64,
        _pc: u32,
        _addr: u32,
        _value: u32,
        _space: &str,
        _size: u8,
        _substep: u64,
    ) {
    }

    pub fn record_mem_write_at_cycle(
        &mut self,
        _cycle: u64,
        _pc: Option<u32>,
        _addr: u32,
        _value: u32,
        _space: &str,
        _size: u8,
    ) {
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_lcd_event(
        &mut self,
        _name: &str,
        _addr: u32,
        _value: u8,
        _chip: usize,
        _page: u8,
        _column: u8,
        _pc: Option<u32>,
        _op_index: Option<u64>,
    ) {
    }

    pub fn update_counters(
        &mut self,
        _instr_index: u64,
        _call_depth: u32,
        _mem_reads: u64,
        _mem_writes: u64,
    ) {
    }

    pub fn finish(self) -> Result<()> {
        Ok(())
    }

    pub fn record_imr_read(&mut self, _pc: Option<u32>, _value: u8, _instr_index: Option<u64>) {}

    #[allow(clippy::too_many_arguments)]
    pub fn record_imem_addr(
        &mut self,
        _mode: &str,
        _base: u32,
        _bp: u32,
        _px: u32,
        _py: u32,
        _op_index: Option<u64>,
        _pc: Option<u32>,
    ) {
    }

    pub fn record_keyi_set(
        &mut self,
        _addr: u32,
        _value: u8,
        _op_index: Option<u64>,
        _pc: Option<u32>,
    ) {
    }

    pub fn record_kio_read(
        &mut self,
        _pc: Option<u32>,
        _offset: u8,
        _value: u8,
        _op_index: Option<u64>,
    ) {
    }

    pub fn record_irq_event(&mut self, _name: &str, _payload: HashMap<String, AnnotationValue>) {}

    #[allow(clippy::too_many_arguments)]
    pub fn record_irq_check(
        &mut self,
        _name: &str,
        _pc: u32,
        _imr: u8,
        _isr: u8,
        _pending: bool,
        _in_interrupt: bool,
        _pending_src: Option<&str>,
        _kil: Option<u8>,
        _imr_reg: Option<u8>,
    ) {
    }

    pub fn record_call_flow(&mut self, _name: &str, _pc_from: u32, _pc_to: u32, _depth: u32) {}
}

/// Compute a Perfetto timestamp for IRQ/key events, honoring a manual clock when present,
/// otherwise falling back to the current/last instruction index with a small substep to
/// avoid collapsing multiple host events into the same instant.
#[cfg(feature = "perfetto")]
fn irq_timestamp(
    cycle_ts: Option<i64>,
    instr_idx: Option<u64>,
    last_idx: Option<u64>,
    units_per_instr: u64,
    seq: u64,
) -> i64 {
    if let Some(cycle) = cycle_ts {
        return cycle;
    }
    if let Some(idx) = instr_idx {
        return (idx.saturating_mul(units_per_instr)) as i64;
    }
    if let Some(last) = last_idx {
        return (last.saturating_mul(units_per_instr).saturating_add(1)) as i64;
    }
    seq as i64
}

#[cfg(feature = "perfetto")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IrqTrack {
    Timer,
    Key,
    Misc,
}

/// Decide which Perfetto track to use for IRQ/key events based on the name and src payload.
#[cfg(feature = "perfetto")]
fn classify_irq_track(name: &str, payload: &HashMap<String, AnnotationValue>) -> IrqTrack {
    let upper_name = name.to_ascii_uppercase();
    let src = payload
        .get("src")
        .or_else(|| payload.get("source"))
        .and_then(|v| match v {
            AnnotationValue::Str(s) => Some(s.to_ascii_uppercase()),
            _ => None,
        });

    // Primary routing on event name (matches Python tracer defaults).
    if matches!(upper_name.as_str(), "TIMERFIRED" | "TIMER" | "MTI" | "STI") {
        return IrqTrack::Timer;
    }
    if matches!(
        upper_name.as_str(),
        "KEYI_SET" | "KEYSCAN" | "KEYSCANEVENT" | "KEYSCANEMPTY" | "KEYI" | "KEYDELIVER" | "KEYIRQ"
    ) {
        return IrqTrack::Key;
    }

    // Secondary routing based on src tag.
    if let Some(src) = src.as_deref() {
        if src.contains("MTI") || src.contains("STI") {
            return IrqTrack::Timer;
        }
        if src == "KEY" || src == "ONK" || src.contains("KEY") {
            return IrqTrack::Key;
        }
    }

    IrqTrack::Misc
}

#[cfg(all(test, feature = "perfetto"))]
mod tests {
    use super::*;

    #[test]
    fn keyirq_routes_to_key_track_by_name() {
        let track = classify_irq_track("KeyIRQ", &HashMap::new());
        assert_eq!(track, IrqTrack::Key);
    }

    #[test]
    fn keyirq_routes_to_key_track_by_src() {
        let mut payload = HashMap::new();
        payload.insert("src".to_string(), AnnotationValue::Str("KEY".to_string()));
        let track = classify_irq_track("PythonOverlayMissing", &payload);
        assert_eq!(track, IrqTrack::Key);
    }

    #[test]
    fn timer_src_routes_to_timer_track() {
        let mut payload = HashMap::new();
        payload.insert("src".to_string(), AnnotationValue::Str("MTI".to_string()));
        let track = classify_irq_track("IRQ_Delivered", &payload);
        assert_eq!(track, IrqTrack::Timer);
    }

    #[test]
    fn unknown_name_timer_source_routes_to_timer() {
        let mut payload = HashMap::new();
        payload.insert(
            "source".to_string(),
            AnnotationValue::Str("STI".to_string()),
        );
        let track = classify_irq_track("PythonOverlayMissing", &payload);
        assert_eq!(track, IrqTrack::Timer);
    }

    #[test]
    fn unknown_name_onk_source_routes_to_key() {
        let mut payload = HashMap::new();
        payload.insert(
            "source".to_string(),
            AnnotationValue::Str("ONK".to_string()),
        );
        let track = classify_irq_track("PythonOverlayMissing", &payload);
        assert_eq!(track, IrqTrack::Key);
    }

    #[test]
    fn unknown_defaults_to_misc() {
        let track = classify_irq_track("PythonOverlayMissing", &HashMap::new());
        assert_eq!(track, IrqTrack::Misc);
    }

    #[test]
    fn irq_timestamp_prefers_cycle() {
        let ts = irq_timestamp(Some(1234), Some(5), Some(4), 1, 7);
        assert_eq!(ts, 1234);
    }

    #[test]
    fn irq_timestamp_uses_instr_index_when_present() {
        let ts = irq_timestamp(None, Some(10), None, 1, 3);
        assert_eq!(ts, 10);
    }

    #[test]
    fn irq_timestamp_falls_back_to_last_index_with_substep() {
        let ts = irq_timestamp(None, None, Some(8), 1, 5);
        assert_eq!(ts, 9);
    }

    #[test]
    fn irq_timestamp_uses_seq_when_no_context() {
        let ts = irq_timestamp(None, None, None, 1, 42);
        assert_eq!(ts, 42);
    }

    #[test]
    fn record_mem_write_uses_substep_for_timestamp() {
        let path = std::env::temp_dir().join("perfetto_substep_test.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut tracer = PerfettoTracer::new(path);
        tracer.record_mem_write_with_substep(0, 0, 0x10, 0xAA, "internal", 8, 1);
        tracer.record_mem_write_with_substep(0, 0, 0x11, 0xBB, "internal", 8, 2);
        let ts = tracer.test_timestamps.borrow().clone();
        assert_eq!(ts.len(), 2);
        assert!(ts[0] < ts[1], "substeps should advance timestamps");
    }
}
