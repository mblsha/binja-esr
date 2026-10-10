// PY_SOURCE: pce500/oz9600/boundary.py
//! Machine peripherals attached to the shared runtime, independent of report
//! collection. Native batch stepping and cooperative slices use this path too.
use super::SharedHardware;
use crate::{
    lcd::{LcdDisplayWrite, LcdHal, LcdKind, LcdStats, LcdWriteTrace},
    lcd_frame::LcdFrame,
    lcd_snapshot::LcdSnapshotMetadata,
    memory::{MemoryImage, MemoryOverlay},
    runtime_device::{BoundaryContext, BoundaryDevice},
    CoreError, Result,
};

#[derive(Default, serde::Serialize, serde::Deserialize, Clone)]
pub struct ExecutionState {
    #[serde(skip)]
    mapped_selector: Option<u8>,
    /// Opt-in ONKI contact-edge hypothesis; SSR still reports the held level.
    pub diagnostic_on_irq_edge_only: bool,
    /// Opt-in IRQ acceptance by architectural IMR rather than handler metadata.
    pub diagnostic_irq_imr_only: bool,
    pub diagnostic_lcc7_halt_main_timer: bool,
    pub halt_main_timer_units: u64,
    pub diagnostic_rtc_a2_period: Option<u64>,
    pub rtc_a2_phase_units: u64,
    pub rtc_a2_ticks: u64,
    pub experimental_rtc_second_period: Option<u64>,
    pub rtc_second_phase_units: u64,
    pub rtc_elapsed_seconds: u64,
    pub rtc_held_units: u64,
    pub rtc_timing_units: u64,
    pub rtc_off_elapsed_units: u64,
}

impl ExecutionState {
    pub(super) fn session_policy(&self) -> (bool, bool, bool, Option<u64>, Option<u64>) {
        (
            self.diagnostic_on_irq_edge_only,
            self.diagnostic_irq_imr_only,
            self.diagnostic_lcc7_halt_main_timer,
            self.diagnostic_rtc_a2_period,
            self.experimental_rtc_second_period,
        )
    }
    pub(super) fn invalidate_bank_view(&mut self) {
        self.mapped_selector = None;
    }
    /// Read-only diagnostics for the opt-in profile; absent in historical
    /// profile reports so their observation contract stays unchanged.
    pub fn rtc_progression_report(&self) -> Option<serde_json::Value> {
        self.experimental_rtc_second_period.map(|period| serde_json::json!({
            "second_period_units":period,
            "second_phase_units":self.rtc_second_phase_units,
            "elapsed_seconds":self.rtc_elapsed_seconds,
            "held_units":self.rtc_held_units,
            "elapsed_timing_units":self.rtc_timing_units,
            "off_elapsed_units":self.rtc_off_elapsed_units,
            "half_second_period_units":self.diagnostic_rtc_a2_period,
            "half_second_phase_units":self.rtc_a2_phase_units,
            "half_second_ticks":self.rtc_a2_ticks,
            "qualification":"Experimental nominal scheduler rate, freeze-on-HOLD and inferred cyclic/deadline/seconds-3F policy; not physical calibration or complete wildcard support"
        }))
    }
}

pub(super) fn sync_bank_view(hardware: &SharedHardware, memory: &mut MemoryImage) {
    let mut hw = hardware.borrow_mut();
    if hw.execution.mapped_selector == Some(hw.selector) {
        return;
    }
    memory.remove_overlay("oz9600_banked_rom");
    if let Some(bank) = hw.banks.get(&hw.selector) {
        // A new static view advances the shared mapping epoch. Instruction and
        // vector preflight never reuse a proof for another ROM selector.
        memory.add_rom_overlay(0xc0000, bank, "oz9600_banked_rom");
    } else {
        let reader = hardware.clone();
        memory.add_overlay(MemoryOverlay {
            start: 0xc0000,
            end: 0xdffff,
            name: "oz9600_banked_rom".into(),
            data: None,
            read_only: true,
            read_handler: Some(Box::new(move |address, pc| {
                Some(reader.borrow_mut().architectural_read(address, pc))
            })),
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: None,
        });
    }
    hw.execution.mapped_selector = Some(hw.selector);
}

pub(super) struct PeripheralBoundary {
    hardware: SharedHardware,
    halt_main_deadline: Option<u64>,
    elapsed_before: u64,
    audio_scr_before: u8,
    audio_off_before: bool,
}

impl PeripheralBoundary {
    pub fn new(hardware: SharedHardware) -> Self {
        Self {
            hardware,
            halt_main_deadline: None,
            elapsed_before: 0,
            audio_scr_before: 0,
            audio_off_before: false,
        }
    }
}

impl BoundaryDevice for PeripheralBoundary {
    fn before_boundary(&mut self, context: &mut BoundaryContext<'_>) -> Result<Option<bool>> {
        if context.state.pc() == 0 {
            return Err(CoreError::Other("OZ-9600 reached an unqualified zero code target; empty-RAM reset/ESR-P policies remain unresolved".into()));
        }
        sync_bank_view(&self.hardware, context.memory);
        self.elapsed_before = context.elapsed_timing_units;
        let mut hw = self.hardware.borrow_mut();
        if hw.audio.enabled() {
            self.audio_scr_before = context.memory.read_internal_byte_silent(0xfd).unwrap_or(0);
            self.audio_off_before = context.state.is_off();
        }
        if let Some(card) = &hw.card {
            let ssr = context.memory.read_internal_byte_silent(0xff).unwrap_or(0);
            context
                .memory
                .write_internal_byte(0xff, card.presence_ssr(ssr));
        }
        let irq = hw.refresh_irq_inputs();
        hw.cycle = context.cycles;
        self.halt_main_deadline = (hw.execution.diagnostic_lcc7_halt_main_timer
            && context.state.is_halted()
            && !context.state.is_off()
            && context.memory.read_internal_byte_silent(0xfe).unwrap_or(0) & 0x80 != 0)
            .then_some(context.timer.next_mti);
        Ok(Some(irq))
    }

    fn after_boundary(&mut self, context: &mut BoundaryContext<'_>, elapsed: u64) -> Result<()> {
        let mut hw = self.hardware.borrow_mut();
        // The crystal is independent of CPU execution. Only the explicit RTC
        // profile consumes scheduler OFF time; historical A2 experiments retain
        // their CPU-cycle-only policy. This is not an alarm-to-power-wake model.
        let rtc_elapsed = if hw.execution.experimental_rtc_second_period.is_some() {
            context
                .elapsed_timing_units
                .wrapping_sub(self.elapsed_before)
        } else {
            elapsed
        };
        if hw.execution.experimental_rtc_second_period.is_some() {
            hw.execution.rtc_timing_units = hw.execution.rtc_timing_units.wrapping_add(rtc_elapsed);
            hw.execution.rtc_off_elapsed_units = hw
                .execution
                .rtc_off_elapsed_units
                .wrapping_add(rtc_elapsed.wrapping_sub(elapsed));
        }
        if let Some(deadline) = self.halt_main_deadline.take() {
            // Existing explicit experiment only. The shared default freezes
            // MTI in HALT. Reconcile that same timer's deadline here; normal
            // CPU IRQ delivery and wake behavior remain in CoreRuntime.
            context.timer.next_mti = deadline;
            context.timer.tick_timers_selected(
                context.memory,
                context.cycles,
                Some(context.state.pc()),
                true,
                false,
            );
            hw.execution.halt_main_timer_units += elapsed;
        }
        if let Some(period) = hw.execution.diagnostic_rtc_a2_period {
            let total = u128::from(hw.execution.rtc_a2_phase_units) + u128::from(rtc_elapsed);
            let ticks = (total / u128::from(period)) as u64;
            hw.execution.rtc_a2_phase_units = (total % u128::from(period)) as u64;
            if ticks != 0 {
                hw.execution.rtc_a2_ticks = hw.execution.rtc_a2_ticks.wrapping_add(ticks);
                hw.rtc.raise_causes(4, 0);
                hw.cycle = context.cycles;
                hw.log("diagnostic_rtc_cause", 0x800a, 4, None);
            }
        }
        if let Some(period) = hw.execution.experimental_rtc_second_period {
            // Boundary-time experiment only. HOLD freezes calendar phase;
            // half-second diagnostic causes keep their separate latch path.
            // No host clock, backlog jump, or per-read time progression.
            if hw.rtc.read(super::rtc::CONTROL) & super::rtc::HOLD != 0 {
                hw.execution.rtc_held_units = hw.execution.rtc_held_units.wrapping_add(rtc_elapsed);
            } else {
                let total =
                    u128::from(hw.execution.rtc_second_phase_units) + u128::from(rtc_elapsed);
                let seconds = (total / u128::from(period)) as u64;
                if seconds != 0 {
                    hw.rtc
                        .advance_experimental_seconds(seconds)
                        .map_err(CoreError::Other)?;
                    hw.execution.rtc_elapsed_seconds =
                        hw.execution.rtc_elapsed_seconds.wrapping_add(seconds);
                }
                hw.execution.rtc_second_phase_units = (total % u128::from(period)) as u64;
            }
        }
        if let Some(error) = &hw.fault {
            return Err(CoreError::Other(error.clone()));
        }
        let audio_ci_input = hw.audio_ci_input;
        hw.audio.advance(
            context
                .elapsed_timing_units
                .wrapping_sub(self.elapsed_before),
            self.audio_scr_before,
            self.audio_off_before,
            audio_ci_input,
        );
        Ok(())
    }
}

/// Observes the same controller written by the memory overlays. Address
/// routing remains there so PC/cycle diagnostics and read side effects retain
/// one owner. This view performs no bus operations and holds no pixel copy.
pub(super) struct LcdView(pub SharedHardware);

impl LcdHal for LcdView {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    // A shared typed LH1553F snapshot/model profile has not been introduced.
    fn kind(&self) -> LcdKind {
        LcdKind::Unknown
    }
    fn reset(&mut self) {
        self.0.borrow_mut().lcd = Default::default();
    }
    fn handles(&self, _address: u32) -> bool {
        false
    }
    fn fixed_windows(&self) -> Option<[(u32, u32); 2]> {
        Some([(1, 0), (1, 0)])
    }
    fn may_handle_span(&self, _start: u32, _end: u32) -> bool {
        false
    }
    fn read(&mut self, _address: u32) -> Option<u8> {
        None
    }
    fn write(&mut self, _address: u32, _value: u8) {
        panic!("OZ LCD is routed by its bus overlay");
    }
    fn read_placeholder(&self, _address: u32) -> u32 {
        panic!("OZ LCD is routed by its bus overlay");
    }
    fn begin_display_write_capture(&mut self) {
        panic!("use the OZ LCD bus observation API");
    }
    fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite> {
        panic!("use the OZ LCD bus observation API");
    }
    fn matrix_frame(&self) -> LcdFrame {
        self.0.borrow().lcd.matrix_frame()
    }
    fn display_buffer(&self) -> [[u8; 240]; 32] {
        panic!("use matrix_frame for the full OZ LCD");
    }
    fn chip_display_buffer(&self, _chip_index: usize) -> [[u8; 64]; 64] {
        panic!("OZ LCD has no HD61202 chips");
    }
    fn display_vram_bytes(&self) -> [[u8; 240]; 8] {
        panic!("use matrix_frame for the full OZ LCD");
    }
    fn display_trace_buffer(&self) -> [[LcdWriteTrace; 240]; 8] {
        panic!("use the OZ LCD bus observation API");
    }
    fn stats(&self) -> LcdStats {
        LcdStats {
            chip_on: [false; 2],
            instruction_counts: [0; 2],
            data_write_counts: [0; 2],
            cs_both_count: 0,
            cs_left_count: 0,
            cs_right_count: 0,
        }
    }
    fn snapshot_state(&self) -> (LcdSnapshotMetadata, Vec<u8>) {
        panic!("full OZ LCD snapshots are not represented by v4");
    }
    fn restore_state(
        &mut self,
        _metadata: &LcdSnapshotMetadata,
        _payload: &[u8],
    ) -> std::result::Result<(), String> {
        Err("full OZ LCD snapshots are not represented by v4".into())
    }
}
