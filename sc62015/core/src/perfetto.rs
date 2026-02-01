// PY_SOURCE: pce500/tracing/perfetto_tracing.py:PerfettoTracer

use crate::Result;
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "perfetto")]
use crate::llama::eval::{perfetto_instr_context, perfetto_last_instr_index, perfetto_last_pc};
#[cfg(feature = "perfetto")]
use crate::CoreError;
#[cfg(feature = "perfetto")]
pub(crate) use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
#[cfg(all(test, feature = "perfetto"))]
use std::cell::RefCell;
#[cfg(any(feature = "perfetto", test))]
use std::sync::OnceLock;
#[cfg(feature = "perfetto")]
use std::sync::RwLock;
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

#[cfg(not(feature = "perfetto"))]
#[derive(Clone, Debug)]
pub enum AnnotationValue {
    Bool(bool),
    Int(i64),
    Pointer(u64),
    UInt(u64),
    Double(f64),
    Str(String),
}

#[cfg(feature = "perfetto")]
static CALL_UI_FUNCTION_NAMES: OnceLock<RwLock<HashMap<u32, String>>> = OnceLock::new();

#[cfg(test)]
static PERFETTO_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
pub(crate) fn perfetto_test_guard() -> MutexGuard<'static, ()> {
    PERFETTO_TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Install an address→name mapping for Call-UI Perfetto traces.
///
/// This mapping is used to label the "Functions" track in traces emitted by the Web Function Runner
/// (`e.call(..., { trace: true })` / IOCS helpers with `{ trace: true }`). Missing entries fall back
/// to `sub_XXXXXX` labels.
#[cfg(feature = "perfetto")]
pub fn set_call_ui_function_names(symbols: HashMap<u32, String>) {
    let lock = CALL_UI_FUNCTION_NAMES.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(mut guard) = lock.write() {
        *guard = symbols;
    }
}

#[cfg(not(feature = "perfetto"))]
pub fn set_call_ui_function_names(_symbols: HashMap<u32, String>) {}

#[cfg(feature = "perfetto")]
fn lookup_call_ui_function_name(addr: u32) -> Option<String> {
    CALL_UI_FUNCTION_NAMES
        .get()
        .and_then(|lock| lock.read().ok())
        .and_then(|map| map.get(&addr).cloned())
}

/// Perfetto protobuf trace writer powered by `retrobus-perfetto`.
#[cfg(feature = "perfetto")]
pub struct PerfettoTracer {
    builder: PerfettoTraceBuilder,
    exec_track: TrackId,
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
    lcd_chars_track: TrackId,
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
    functions_track: TrackId,
    instructions_track: TrackId,
    control_flow_track: TrackId,
    ewrites_track: TrackId,
    iwrites_track: TrackId,
    call_ui_functions_depth: u32,
    #[cfg(test)]
    test_exec_events: RefCell<Vec<(u32, u8, u64)>>, // pc, opcode, op_index
    #[cfg(test)]
    test_timestamps: RefCell<Vec<i64>>,
    #[cfg(test)]
    test_function_slices: RefCell<Vec<String>>,
    #[cfg(test)]
    pub(crate) test_mem_write_pcs: RefCell<Vec<Option<u32>>>,
    #[cfg(test)]
    pub(crate) test_counters: RefCell<Vec<(u64, u32, u64, u64)>>, // (instr_index, call_depth, reads, writes)
    #[cfg(test)]
    test_owner: std::thread::ThreadId,
}

#[cfg(feature = "perfetto")]
impl PerfettoTracer {
    pub fn new(path: PathBuf) -> Self {
        let mut builder = PerfettoTraceBuilder::new("SC62015");
        // Single Perfetto format (shared by Rust + Python):
        // - `Functions`: slices for CALL/RET spanning full function duration
        // - `Instructions`: per-instruction slices with pre-state regs/IMR/ISR, dur=1 tick
        // - `EWrites`/`IWrites`: instants for external/internal memory writes
        //
        // Timestamps are deterministic instruction-index ticks. retrobus-perfetto stores
        // timestamps in microseconds; it converts input nanoseconds by dividing by 1000.
        // Use 1000ns per tick so each instruction advances Perfetto time by 1us.

        let exec_track = builder.add_thread("InstructionTrace");
        let _timeline_track = builder.add_thread("Execution");
        let mem_track = builder.add_thread("Memory");
        let memory_track = builder.add_thread("Memory");
        let _cpu_track = builder.add_thread("CPU");
        let instr_counter_track = builder.add_counter_track("instructions", Some("count"), None);
        let imr_track = builder.add_thread("IMR");
        let imr_zero_counter = builder.add_counter_track("IMR_ReadZero", Some("count"), None);
        let imr_nonzero_counter = builder.add_counter_track("IMR_ReadNonZero", Some("count"), None);
        // Align IRQ tracks with Python tracer.
        let irq_timer_track = builder.add_thread("irq.timer");
        let irq_key_track = builder.add_thread("irq.key");
        let irq_misc_track = builder.add_thread("irq.misc");
        let display_track = builder.add_thread("Display");
        let lcd_chars_track = builder.add_thread("LCD Characters");
        let functions_track = builder.add_thread("Functions");
        let instructions_track = builder.add_thread("Instructions");
        let control_flow_track = builder.add_thread("ControlFlow");
        let ewrites_track = builder.add_thread("EWrites");
        let iwrites_track = builder.add_thread("IWrites");
        let call_depth_counter = builder.add_counter_track("call_depth", Some("depth"), None);
        let mem_read_counter = builder.add_counter_track("read_ops", Some("count"), None);
        let mem_write_counter = builder.add_counter_track("write_ops", Some("count"), None);
        Self {
            builder,
            exec_track,
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
            lcd_chars_track,
            units_per_instr: 1_000,
            path,
            imr_seq: 0,
            imr_read_zero: 0,
            imr_read_nonzero: 0,
            irq_seq: 0,
            display_seq: 0,
            call_depth_counter,
            mem_read_counter,
            mem_write_counter,
            functions_track,
            instructions_track,
            control_flow_track,
            ewrites_track,
            iwrites_track,
            call_ui_functions_depth: 0,
            #[cfg(test)]
            test_exec_events: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_timestamps: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_function_slices: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_mem_write_pcs: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_counters: RefCell::new(Vec::new()),
            #[cfg(test)]
            test_owner: std::thread::current().id(),
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

    #[cfg(test)]
    fn test_owner_matches(&self) -> bool {
        self.test_owner == std::thread::current().id()
    }

    fn call_ui_close_open_function_slices(&mut self) {
        // Close any unbalanced nested slices (e.call can stop mid-call).
        let end_ts = self.ts(perfetto_last_instr_index(), 0);
        while self.call_ui_functions_depth > 0 {
            self.builder.end_slice(self.functions_track, end_ts);
            self.call_ui_functions_depth = self.call_ui_functions_depth.saturating_sub(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_regs(
        &mut self,
        instr_index: u64,
        _pc: u32,
        reg_pc: u32,
        opcode: u8,
        mnemonic: Option<&str>,
        regs: &HashMap<String, u32>,
        mem_imr: u8,
        mem_isr: u8,
    ) {
        let ts_start = self.ts(instr_index, 0);
        let ts_end = self.ts(instr_index.saturating_add(1), 0);
        let name = if let Some(m) = mnemonic {
            m.to_string()
        } else {
            format!("op=0x{opcode:02X}")
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

        self.builder
            .update_counter(self.instr_counter_track, instr_index as f64 + 1.0, ts_start);

        #[cfg(test)]
        {
            if self.test_owner_matches() {
                self.test_exec_events
                    .borrow_mut()
                    .push((reg_pc, opcode, instr_index));
                self.test_timestamps.borrow_mut().push(ts_start);
            }
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
        let track = if space == "internal" {
            self.iwrites_track
        } else {
            self.ewrites_track
        };
        {
            let mut ev = self
                .builder
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
        }
        #[cfg(test)]
        let allow_test = self.test_owner_matches();
        #[cfg(test)]
        if allow_test {
            self.test_timestamps.borrow_mut().push(ts);
        }
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

        #[cfg(test)]
        let allow_test = self.test_owner_matches();

        // Mirror to Memory track for Python parity.
        {
            let mut mem_alias = self.builder.add_instant_event(
                self.memory_track,
                format!("Write@0x{addr:06X}"),
                ts,
            );
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
        }

        #[cfg(test)]
        if allow_test {
            self.test_mem_write_pcs.borrow_mut().push(pc_effective);
        }
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

    pub fn record_lcd_character_slice(
        &mut self,
        ch: char,
        start_op_index: u64,
        end_op_index: u64,
        x: u16,
        y: u8,
    ) {
        let label = if ch.is_ascii_graphic() {
            ch.to_string()
        } else {
            let code = ch as u32;
            if code <= 0xFF {
                format!("0x{code:02X}")
            } else {
                format!("U+{code:04X}")
            }
        };

        let ts_start = self.ts(start_op_index, 0);
        let ts_end = self
            .ts(end_op_index, 0)
            .saturating_add(self.units_per_instr as i64);

        {
            let mut ev = self
                .builder
                .begin_slice(self.lcd_chars_track, label, ts_start);
            ev.add_annotations([
                ("x", AnnotationValue::UInt(x as u64)),
                ("y", AnnotationValue::UInt(y as u64)),
                ("ch", AnnotationValue::UInt(ch as u32 as u64)),
                ("start_op_index", AnnotationValue::UInt(start_op_index)),
                ("end_op_index", AnnotationValue::UInt(end_op_index)),
            ]);
            ev.finish();
        }
        self.builder.end_slice(self.lcd_chars_track, ts_end);
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
        if self.test_owner_matches() {
            self.test_counters
                .borrow_mut()
                .push((instr_index, call_depth, mem_reads, mem_writes));
        }
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
            this.call_ui_close_open_function_slices();
            this.builder
                .save(&this.path)
                .map_err(|e| CoreError::Other(format!("perfetto save: {e}")))
        }
    }

    pub fn serialize(self) -> Result<Vec<u8>> {
        let mut this = self;
        this.call_ui_close_open_function_slices();
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
        // Always timestamp against the instruction timeline. Some callers use `cycle=0` as a
        // placeholder; relying on it collapses start times to 0.
        let cycle_ts = None;

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

    /// Control-flow instants aligned to instruction timestamps.
    pub fn record_control_flow(
        &mut self,
        name: &str,
        instr_index: u64,
        pc: u32,
        payload: HashMap<String, AnnotationValue>,
    ) {
        let ts = self.ts(instr_index, 0);
        let mut ev = self
            .builder
            .add_instant_event(self.control_flow_track, name.to_string(), ts);
        ev.add_annotation("backend", AnnotationValue::Str("rust".to_string()));
        ev.add_annotation("pc", AnnotationValue::Pointer(pc as u64));
        ev.add_annotation("op_index", AnnotationValue::UInt(instr_index));
        for (k, v) in payload {
            ev.add_annotation(k, v);
        }
        ev.finish();
    }

    /// Function enter/exit markers to mirror Python call/return tracing.
    pub fn record_call_flow(&mut self, name: &str, pc_from: u32, pc_to: u32, depth: u32) {
        let ctx = perfetto_instr_context();
        let op = ctx
            .map(|(idx, _)| idx)
            .unwrap_or_else(perfetto_last_instr_index);
        let ts = self.ts(op, 0);
        if name == "CALL" || name == "CALLF" {
            let label = lookup_call_ui_function_name(pc_to & 0x000f_ffff)
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("sub_{pc_to:06X}"));
            #[cfg(test)]
            if self.test_owner_matches() {
                self.test_function_slices.borrow_mut().push(label.clone());
            }
            let mut ev = self.builder.begin_slice(self.functions_track, label, ts);
            ev.add_annotations([
                ("from", AnnotationValue::Pointer(pc_from as u64)),
                ("to", AnnotationValue::Pointer(pc_to as u64)),
                ("depth", AnnotationValue::UInt(depth as u64)),
                ("op_index", AnnotationValue::UInt(op)),
            ]);
            ev.finish();
            self.call_ui_functions_depth = self.call_ui_functions_depth.saturating_add(1);
        } else if name == "RET" || name == "RETF" {
            self.builder.end_slice(
                self.functions_track,
                ts.saturating_add(self.units_per_instr as i64),
            );
            self.call_ui_functions_depth = self.call_ui_functions_depth.saturating_sub(1);
        }
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
        _mnemonic: Option<&str>,
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

    pub fn record_lcd_character_slice(
        &mut self,
        _ch: char,
        _start_op_index: u64,
        _end_op_index: u64,
        _x: u16,
        _y: u8,
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

    pub fn record_control_flow(
        &mut self,
        _name: &str,
        _instr_index: u64,
        _pc: u32,
        _payload: HashMap<String, AnnotationValue>,
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

    #[test]
    fn call_ui_function_slices_use_symbol_map_when_present() {
        let mut symbols = HashMap::new();
        symbols.insert(0x000E_07D0, "pne".to_string());
        set_call_ui_function_names(symbols);

        let path = std::env::temp_dir().join("perfetto_call_symbols_test.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut tracer = PerfettoTracer::new(path);
        tracer.record_call_flow("CALLF", 0x000E_0000, 0x000E_07D0, 1);
        let names = tracer.test_function_slices.borrow().clone();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "pne");
    }

    #[test]
    fn lcd_character_slices_create_lcd_characters_track() {
        let path = std::env::temp_dir().join("perfetto_lcd_chars_test.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut tracer = PerfettoTracer::new(path);
        tracer.record_lcd_character_slice('A', 10, 12, 5, 1);
        let bytes = tracer.serialize().expect("serialize");
        let haystack = String::from_utf8_lossy(&bytes);
        assert!(
            haystack.contains("LCD Characters"),
            "expected LCD Characters track name in serialized trace"
        );
    }
}
