// PY_SOURCE: pce500/pi3000/hardware.py
//! PI-3000 board adapter. Shared CPU/LCD/RTC, fresh RAM and strict unknown I/O.
//! RAM decoding, auxiliary absence and compatibility handoff remain explicit
//! firmware-derived bootstrap policies, not physical hardware qualification.
use crate::{
    memory::{MemoryImage, MemoryOverlay},
    oz9600::{lcd::LcdController, rtc::Rtc, tablet::Tablet},
    runtime_device::{BoundaryContext, BoundaryDevice},
    CoreError, CoreRuntime, DeviceModel, KeyboardMatrix, TimerContext,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
pub mod auxiliary;
pub mod bundle;
pub mod compatibility;
pub mod input;
mod lcd_view;
pub mod retained;
#[cfg(test)]
mod tests;

pub const FIXED_SHA256: &str = "855205ac2655e02d670c4cf7c3107241dd3fbef8ec056e4766f97074c5c90009";
pub const BANKS_SHA256: &str = "a31c087974e5a5a96d45bb25f9f12bcb8e19a075cfe39af5f2074b683d5641b4";
pub const WIDTH: usize = 240;
pub const HEIGHT: usize = 336;

/// The two ADC branches selected by FE:DBE41's 08B1B.bit0 read.
/// Physical board selection remains unqualified; callers opt in explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabletAdcRoute {
    Integrated,
    External,
}

/// Explicit contact wiring candidates; neither has physical qualification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TabletInterruptSource {
    /// FD859 interrupt sampler, conditional on BFF20.bit7.
    #[default]
    Gate3,
    /// FD6B0 posts the contact flag consumed by the foreground raw sampler.
    Gate1,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootstrapProfile {
    /// Retain the native-only control and unknown auxiliary status trap.
    #[default]
    Strict,
    Compatibility,
    /// No auxiliary completion/replies; firmware must handle its own timeout.
    AuxiliaryAbsent,
    /// Stateful startup control/ack candidate; query/recognition data pending.
    AuxiliaryControl,
    /// Runnable firmware-derived startup candidate, retaining strict unknown I/O.
    Firmware,
}

#[derive(Clone)]
pub struct SharedHardware(Arc<Mutex<Hardware>>);
impl SharedHardware {
    pub fn borrow(&self) -> std::sync::MutexGuard<'_, Hardware> {
        self.0.lock().expect("PI-3000 hardware lock poisoned")
    }
}

pub struct Hardware {
    pub selector: u8,
    pub mode: u8,
    pub gate: [u8; 64],
    pub ram: Vec<u8>,
    pub scratch: Vec<u8>,
    pub banks: BTreeMap<u8, Vec<u8>>,
    pub lcd: LcdController,
    pub rtc: Rtc,
    pub tablet: Tablet,
    pub tablet_route: Option<TabletAdcRoute>,
    pub tablet_interrupt_source: TabletInterruptSource,
    pub halt_main_timer_candidate: bool,
    pub halt_main_timer_units: u64,
    /// Explicit periodic A2 clock candidate; calendar advancement is separate.
    pub rtc_tick_period: Option<u64>,
    pub rtc_tick_phase: u64,
    pub rtc_ticks: u64,
    /// Opt-in packed-calendar candidate, independent of periodic A2 causes.
    pub rtc_second_period: Option<u64>,
    pub rtc_second_phase: u64,
    pub rtc_seconds: u64,
    pub rtc_held_units: u64,
    /// Separate experimental cyclic/alarm policy; calendar-only control stays available.
    pub rtc_event_candidate: bool,
    /// Explicit read-only hypothesis for the unmapped 05000..07FFF bus window.
    /// A sequential hardware capture returned E0; electrical behavior is open.
    pub unmapped_low_read: Option<u8>,
    pub gpio: [u8; 18],
    pub clock_control: u8,
    pub startup_latches: [u8; 3],
    pub mapping_unlock: u8,
    pub io_8a: [u8; 9],
    pub auxiliary_request: [u8; 32],
    pub auxiliary_commands: Vec<Vec<u8>>,
    pub auxiliary_command_count: u64,
    pub absent_auxiliary: bool,
    pub auxiliary: Option<auxiliary::Auxiliary>,
    pub auxiliary_sample_irq_candidate: bool,
    pub auxiliary_sample_period: u64,
    auxiliary_sample_deadline: Option<u64>,
    pub cycle: u64,
    pub fault: Option<String>,
    pub lcd_writes: u64,
    mapped: Option<(u8, u8)>,
}
impl Hardware {
    pub fn advance_auxiliary_sample_irq(&mut self, cycle: u64) {
        if self
            .auxiliary_sample_deadline
            .is_some_and(|deadline| cycle >= deadline)
        {
            if self.auxiliary_sample_irq_candidate && self.io_8a[6] & 0x80 != 0 {
                self.gate[0x12] |= 8;
                self.auxiliary_sample_deadline =
                    Some(cycle.saturating_add(self.auxiliary_sample_period));
            } else {
                self.auxiliary_sample_deadline = None;
            }
        }
    }

    pub fn external_irq(&mut self) -> bool {
        // F15C4 consumes the RTC mask/cause pair through gate bit0. It clears
        // the RTC cause before acknowledging the gate latch (F1860/F14C5).
        if self.rtc.interrupt_asserted() {
            self.gate[0x12] |= 1;
        }
        self.gate[0x10] & self.gate[0x12] & 0x7f != 0
    }
    pub fn advance_rtc_ticks(&mut self, elapsed: u64) {
        if let Some(period) = self.rtc_tick_period {
            let total = u128::from(self.rtc_tick_phase) + u128::from(elapsed);
            let ticks = (total / u128::from(period)) as u64;
            self.rtc_tick_phase = (total % u128::from(period)) as u64;
            if ticks != 0 {
                self.rtc_ticks = self.rtc_ticks.wrapping_add(ticks);
                // A2 -> BFE15 -> F1B83 advances the firmware time counter.
                // Masked ticks coalesce in the device latch, never guest RAM.
                self.rtc.raise_causes(4, 0);
            }
        }
    }
    pub fn advance_rtc_calendar(&mut self, elapsed: u64) -> Result<(), String> {
        if let Some(period) = self.rtc_second_period {
            // F8210 reads the five packed bytes under CONTROL.HOLD. Freeze
            // the calendar phase while held; crystal/hold timing is provisional.
            if self.rtc.read(crate::oz9600::rtc::CONTROL) & crate::oz9600::rtc::HOLD != 0 {
                self.rtc_held_units = self.rtc_held_units.wrapping_add(elapsed);
            } else {
                let total = u128::from(self.rtc_second_phase) + u128::from(elapsed);
                let seconds = (total / u128::from(period)) as u64;
                if seconds != 0 {
                    // PI FC:C1938/C196A use the shared packed fields; the
                    // fixed A3..A6 callbacks consume second/minute/hour/day
                    // events. C1A09 programs A0 ordinary deadlines. Keep the
                    // source policy opt-in; silicon timing and other wildcard
                    // alarm forms are still unqualified.
                    if self.rtc_event_candidate {
                        self.rtc.advance_experimental_seconds(seconds)?;
                    } else {
                        self.rtc.advance_seconds(seconds)?;
                    }
                    self.rtc_seconds = self.rtc_seconds.wrapping_add(seconds);
                }
                self.rtc_second_phase = (total % u128::from(period)) as u64;
            }
        }
        Ok(())
    }
    pub fn new(banks: BTreeMap<u8, Vec<u8>>, profile: BootstrapProfile) -> Self {
        let mut lcd = LcdController::with_geometry(WIDTH, HEIGHT);
        // FA47E inserts foreground contexts before existing ones; FA592
        // removes them, and FB2BA programs descriptors in that order from 2.
        // This candidate protects the alarm popup from Clock's minute redraw.
        lcd.ordered_window_clipping_candidate = profile == BootstrapProfile::Firmware;
        Self {
            selector: 0xfd,
            mode: 0,
            gate: [0; 64],
            ram: vec![0; 0x40000],
            scratch: vec![0; 0x4000],
            banks,
            lcd,
            rtc: Rtc::default(),
            tablet: Tablet::default(),
            tablet_route: None,
            tablet_interrupt_source: TabletInterruptSource::default(),
            halt_main_timer_candidate: false,
            halt_main_timer_units: 0,
            rtc_tick_period: (profile == BootstrapProfile::Firmware).then_some(512_000),
            rtc_tick_phase: 0,
            rtc_ticks: 0,
            rtc_second_period: None,
            rtc_second_phase: 0,
            rtc_seconds: 0,
            rtc_held_units: 0,
            rtc_event_candidate: false,
            unmapped_low_read: (profile == BootstrapProfile::Firmware).then_some(0xe0),
            gpio: [0; 18],
            clock_control: 0,
            startup_latches: [0; 3],
            mapping_unlock: 0,
            io_8a: [0; 9],
            auxiliary_request: [0; 32],
            auxiliary_commands: Vec::new(),
            auxiliary_command_count: 0,
            absent_auxiliary: profile == BootstrapProfile::AuxiliaryAbsent,
            auxiliary: matches!(
                profile,
                BootstrapProfile::AuxiliaryControl | BootstrapProfile::Firmware
            )
            .then(auxiliary::Auxiliary::default),
            auxiliary_sample_irq_candidate: profile == BootstrapProfile::Firmware,
            auxiliary_sample_period: 4096,
            auxiliary_sample_deadline: None,
            cycle: 0,
            fault: None,
            lcd_writes: 0,
            mapped: None,
        }
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0..=0x3fff => Some(self.scratch[address as usize]),
            0x5000..=0x7fff => self.unmapped_low_read,
            0x4021 => Some(self.selector),
            0x4000..=0x403f => Some(self.gate[(address - 0x4000) as usize]),
            0x8000..=0x801f => Some(self.rtc.read((address - 0x8000) as usize)),
            0x8400..=0x841f | 0x8438..=0x843f => {
                Some(self.lcd.peek((address & 31) as usize, self.cycle))
            }
            0x8a00 if self.absent_auxiliary => Some(2),
            0x8a00 => self.auxiliary.as_ref().map(|a| a.status()),
            0x8a04 | 0x8a05 | 0x8a09 if self.auxiliary.is_some() => Some(0),
            0x8a40..=0x8a5f => self
                .auxiliary
                .as_ref()
                .and_then(|a| a.response((address - 0x8a40) as usize)),
            0x8b1b => self
                .tablet_route
                .map(|r| u8::from(r == TabletAdcRoute::External)),
            0x8800 if self.tablet_route == Some(TabletAdcRoute::External) => {
                Some(self.tablet.conversion_control)
            }
            0x8b80 if self.tablet_route == Some(TabletAdcRoute::Integrated) => {
                Some(self.tablet.conversion_control)
            }
            0x8c00 if self.tablet_route == Some(TabletAdcRoute::External) => {
                Some(self.tablet.peek_data())
            }
            0x8bc0 if self.tablet_route == Some(TabletAdcRoute::Integrated) => {
                Some(self.tablet.peek_data())
            }
            0x8b1a => Some(self.gpio[2]),
            0x8b1c => Some(self.clock_control),
            0x8b1d => Some(self.gpio[5]),
            0xf01c..=0xf01e => Some(self.startup_latches[(address - 0xf01c) as usize]),
            0x20000..=0x7ffff => Some(255),
            0x80000..=0xbffff => Some(self.ram[(address - 0x80000) as usize]),
            0xc0000..=0xdffff if self.mode == 0 => self
                .banks
                .get(&self.selector)
                .map(|b| b[(address - 0xc0000) as usize]),
            _ => None,
        }
    }
    pub fn architectural_read(&mut self, address: u32, pc: Option<u32>) -> u8 {
        let value = if (address == 0x8c00 && self.tablet_route == Some(TabletAdcRoute::External))
            || (address == 0x8bc0 && self.tablet_route == Some(TabletAdcRoute::Integrated))
        {
            self.tablet.read_data()
        } else if (0x8400..=0x841f).contains(&address) {
            self.lcd.read((address & 31) as usize, self.cycle)
        } else {
            self.read(address)
                .ok_or_else(|| format!("unqualified PI read {address:05X}, PC {pc:?}"))
        };
        match value {
            Ok(v) => v,
            Err(e) => {
                self.fault.get_or_insert(e);
                255
            }
        }
    }
    pub fn write(&mut self, address: u32, value: u8, pc: Option<u32>) {
        match address {
            0..=0x3fff => self.scratch[address as usize] = value,
            0x4021 => self.selector = value,
            0x4000..=0x403f => self.gate[(address - 0x4000) as usize] = value,
            0x8000..=0x801f => self.rtc.write((address - 0x8000) as usize, value),
            0x8420..=0x843f => {
                if let Err(e) = self.lcd.write((address & 31) as usize, value) {
                    self.fault.get_or_insert(e);
                }
                self.lcd_writes += 1;
            }
            0x8b01 => self.mode = value,
            0x8b00 => self.clock_control = value,
            0x8b04 => {
                self.gpio[4] = value;
                self.tablet.drive_control = value;
            }
            0x8b02..=0x8b05 | 0x8b08..=0x8b11 => self.gpio[(address - 0x8b00) as usize] = value,
            0x8800 if self.tablet_route == Some(TabletAdcRoute::External) => {
                if let Err(e) = self.tablet.write_conversion_control(value) {
                    self.fault.get_or_insert(e);
                }
            }
            0x8b80 if self.tablet_route == Some(TabletAdcRoute::Integrated) => {
                if let Err(e) = self.tablet.write_conversion_control(value) {
                    self.fault.get_or_insert(e);
                }
            }
            0x8a01 | 0x8a06..=0x8a08 => {
                self.io_8a[(address - 0x8a00) as usize] = value;
                // The editor enables 08A06.bit7 before any 03FC capture
                // command. The explicit clock candidate follows this latch,
                // independently of the auxiliary transport's capture session.
                if address == 0x8a06 && value & 0x80 != 0 && self.auxiliary_sample_irq_candidate {
                    self.gate[0x12] |= 8;
                    self.auxiliary_sample_deadline =
                        Some(self.cycle.saturating_add(self.auxiliary_sample_period));
                } else if address == 0x8a06 {
                    self.auxiliary_sample_deadline = None;
                }
            }
            0x8a02..=0x8a03 if self.absent_auxiliary || self.auxiliary.is_some() => {
                self.io_8a[(address - 0x8a00) as usize] = value;
                if address == 0x8a02 && value == 0x80 {
                    self.auxiliary_command_count += 1;
                    if self.auxiliary_commands.len() < 256 {
                        self.auxiliary_commands
                            .push(self.auxiliary_request.to_vec());
                    }
                }
                if let Some(auxiliary) = &mut self.auxiliary {
                    let result = if address == 0x8a02 && value == 0x80 {
                        auxiliary.start(&self.auxiliary_request, self.cycle)
                    } else if address == 0x8a03 {
                        auxiliary.acknowledge(value)
                    } else {
                        Err(format!(
                            "unimplemented PI auxiliary start value {value:02X}"
                        ))
                    };
                    if let Err(error) = result {
                        self.fault.get_or_insert(error);
                    }
                }
            }
            0x8a20..=0x8a3f if self.absent_auxiliary || self.auxiliary.is_some() => {
                let offset = (address - 0x8a20) as usize;
                self.auxiliary_request[offset] = value;
                if let Some(auxiliary) = &mut self.auxiliary {
                    auxiliary.mark_written(offset);
                }
            }
            0xf00c..=0xf00e => self.startup_latches[(address - 0xf00c) as usize] = value,
            0xf103 => self.mapping_unlock = value,
            0x20000..=0x7ffff => {}
            0x80000..=0xbffff => self.ram[(address - 0x80000) as usize] = value,
            _ => {
                self.fault.get_or_insert_with(|| {
                    format!("unqualified PI write {address:05X}, PC {pc:?}")
                });
            }
        }
    }
}

fn sync_bank(hw: &SharedHardware, memory: &mut MemoryImage) {
    let mut h = hw.borrow();
    if h.mapped == Some((h.selector, h.mode)) {
        return;
    }
    memory.remove_overlay("pi3000_rom_bank");
    if let Some(bank) = h.banks.get(&h.selector).filter(|_| h.mode == 0) {
        memory.add_rom_overlay(0xc0000, bank, "pi3000_rom_bank");
    } else {
        // Missing bank/mode must reject preflight rather than expose stale ROM.
        let read = hw.clone();
        memory.add_overlay(MemoryOverlay {
            start: 0xc0000,
            end: 0xdffff,
            name: "pi3000_rom_bank".into(),
            data: None,
            read_only: true,
            read_handler: Some(Box::new(move |a, pc| {
                Some(read.borrow().architectural_read(a, pc))
            })),
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: None,
        });
    }
    h.mapped = Some((h.selector, h.mode));
}

struct Boundary(SharedHardware, Option<u64>, u64);
impl BoundaryDevice for Boundary {
    fn before_boundary(&mut self, c: &mut BoundaryContext<'_>) -> crate::Result<Option<bool>> {
        sync_bank(&self.0, c.memory);
        let mut h = self.0.borrow();
        if let Some(e) = &h.fault {
            return Err(CoreError::Other(e.clone()));
        }
        h.cycle = c.cycles;
        self.2 = c.elapsed_timing_units;
        self.1 = (h.halt_main_timer_candidate
            && c.state.is_halted()
            && !c.state.is_off()
            && c.memory.read_internal_byte_silent(0xfe).unwrap_or(0) & 0x80 != 0)
            .then_some(c.timer.next_mti);
        Ok(Some(h.external_irq()))
    }
    fn after_boundary(&mut self, c: &mut BoundaryContext<'_>, elapsed: u64) -> crate::Result<()> {
        let mut h = self.0.borrow();
        if let Some(deadline) = self.1.take() {
            // Same explicit LCC7/HALT clock experiment as the related board.
            // CoreRuntime owns wake/IRQ delivery; no guest flags are seeded.
            c.timer.next_mti = deadline;
            c.timer
                .tick_timers_selected(c.memory, c.cycles, Some(c.state.pc()), true, false);
            h.halt_main_timer_units += elapsed;
        }
        h.cycle = c.cycles;
        let rtc_elapsed = c.elapsed_timing_units.wrapping_sub(self.2);
        h.advance_rtc_ticks(rtc_elapsed);
        h.advance_rtc_calendar(rtc_elapsed)
            .map_err(CoreError::Other)?;
        if let Some(auxiliary) = &mut h.auxiliary {
            auxiliary.advance(c.cycles);
        }
        h.advance_auxiliary_sample_irq(c.cycles);
        if let Some(e) = &h.fault {
            return Err(CoreError::Other(e.clone()));
        }
        Ok(())
    }
}

/// Component construction; normal consumers must validate their ROM bundle.
pub fn configure(
    fixed: &[u8],
    banks: BTreeMap<u8, Vec<u8>>,
    profile: BootstrapProfile,
) -> Result<CoreRuntime, String> {
    if fixed.len() != 0x20000 || banks.values().any(|b| b.len() != 0x20000) {
        return Err("PI ROM windows must be 128 KiB".into());
    }
    let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(banks, profile))));
    let mut rt = CoreRuntime::new();
    rt.set_device_model(DeviceModel::Pi3000)
        .map_err(|e| e.to_string())?;
    rt.lcd = None;
    rt.sio = None;
    rt.pce500_peripherals = None;
    let mut keyboard = KeyboardMatrix::new();
    DeviceModel::Pi3000.configure_keyboard(&mut keyboard);
    rt.keyboard = Some(keyboard);
    rt.memory.set_internal_ram_mirror(false);
    rt.memory.set_keyboard_bridge(false);
    *rt.timer = TimerContext::new(false, 0, 0);
    rt.load_rom(fixed, 0xe0000);
    rt.memory.set_readonly_ranges(vec![(0xe0000, 0xfffff)]);
    let mut known = vec![
        (0, 0x3fff, true),
        (0x4000, 0x403f, true),
        (0x8000, 0x801f, true),
        (0x8400, 0x841f, true),
        (0x8420, 0x8437, false),
        (0x8438, 0x843f, true),
        (0x8800, 0x8800, true),
        (
            0x8a00,
            0x8a00,
            matches!(
                profile,
                BootstrapProfile::AuxiliaryAbsent | BootstrapProfile::AuxiliaryControl
            ),
        ),
        (0x8a01, 0x8a03, false),
        (0x8a04, 0x8a05, true),
        (0x8a06, 0x8a08, false),
        (0x8a09, 0x8a09, true),
        (0x8a20, 0x8a3f, false),
        (0x8a40, 0x8a5f, true),
        (0x8b00, 0x8b11, false),
        (0x8b1a, 0x8b1b, true),
        (0x8b1c, 0x8b1d, true),
        (0x8b80, 0x8b80, true),
        (0x8bc0, 0x8bc0, true),
        (0x8c00, 0x8c00, true),
        (0xf00c, 0xf00e, false),
        (0xf01c, 0xf01e, true),
        (0xf103, 0xf103, false),
        (0x20000, 0x7ffff, true),
        (0x80000, 0xbffff, true),
    ];
    known.sort_unstable();
    let mut spans = Vec::new();
    let mut previous = 0;
    for (start, end, safe) in known {
        if previous < start {
            spans.push((previous, start - 1, false));
        }
        spans.push((start, end, safe));
        previous = end + 1;
    }
    for (start, end, safe) in spans {
        let read = hardware.clone();
        let peek = hardware.clone();
        let write = hardware.clone();
        rt.memory.add_overlay(MemoryOverlay {
            start,
            end,
            name: format!("pi3000_bus_{start:05X}"),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(move |a, pc| {
                Some(read.borrow().architectural_read(a, pc))
            })),
            preflight_read_handler: if safe {
                Some(Box::new(move |a, _| peek.borrow().read(a)))
            } else {
                None
            },
            write_handler: Some(Box::new(move |a, v, pc| {
                write.borrow().write(a, v, pc);
                true
            })),
            perfetto_thread: None,
        });
    }
    sync_bank(&hardware, &mut rt.memory);
    rt.power_on_reset().map_err(|e| e.to_string())?;
    rt.install_boundary_device(Box::new(Boundary(hardware.clone(), None, 0)))
        .map_err(|e| e.to_string())?;
    rt.lcd = Some(Box::new(lcd_view::LcdView(hardware.clone())));
    rt.pi3000_hardware = Some(hardware.clone());
    if profile != BootstrapProfile::Strict {
        rt.pi3000_compatibility = Some(compatibility::Coordinator::new(hardware.clone()));
    }
    if profile == BootstrapProfile::Firmware {
        use crate::llama::state::{
            BlockTransferPolicy, ByteArithmeticSourcePolicy, IsrSoftwareWritePolicy,
        };
        rt.memory.write_internal_byte(0xec, 0xd0);
        // F06CA requests ISR.MTI after its masked keyboard HALT. F13A9/
        // F1BC5 dispatch that request to F1BEB and the ordinary matrix scanner.
        // Retain clear-only writes for other sources; silicon timing remains
        // unqualified, as with the rest of this firmware startup profile.
        rt.state
            .set_isr_software_write_policy(IsrSoftwareWritePolicy::ClearOnlyExceptMti);
        rt.state
            .set_block_transfer_policy(BlockTransferPolicy::CoupledPredecrement);
        // FD:CECA9 uses SUB A,I when constructing Schedule's entry controls.
        // Reuse the existing ESR-P low-byte source hypothesis; Strict retains
        // the mixed-width decode rejection and silicon semantics remain open.
        rt.state
            .set_byte_arithmetic_source_policy(ByteArithmeticSourcePolicy::LowByte);
        let timing = DeviceModel::Pi3000.timer_profile();
        rt.timer.enabled = true;
        rt.timer.configure_scr_periods(
            timing.mti_period,
            timing.mti_long_period,
            timing.sti_period,
            timing.sti_long_period,
            rt.memory.read_internal_byte_silent(0xfd).unwrap(),
            rt.cycle_count(),
        );
        let mut h = hardware.borrow();
        h.tablet_route = Some(TabletAdcRoute::Integrated);
        h.tablet_interrupt_source = TabletInterruptSource::Gate1;
        h.halt_main_timer_candidate = true;
    }
    Ok(rt)
}

impl CoreRuntime {
    pub fn pi3000_retained_state(&self) -> Result<Vec<u8>, String> {
        let h = self.pi3000_hardware.as_ref().ok_or("not a PI runtime")?;
        Ok(retained::retained_state(self, h))
    }
    /// Ordered window exclusion inferred from PI window lifecycle and consumers.
    /// Configure before CPU or HALT time advances; hardware flags remain open.
    pub fn configure_pi3000_ordered_window_clipping_candidate(
        &mut self,
        enabled: bool,
    ) -> Result<(), String> {
        if self.instruction_count() != 0 || self.elapsed_timing_units() != 0 {
            return Err("PI LCD window policy must be configured before execution".into());
        }
        self.pi3000_hardware
            .as_ref()
            .ok_or("not a PI runtime")?
            .borrow()
            .lcd
            .ordered_window_clipping_candidate = enabled;
        Ok(())
    }
    pub fn restore_pi3000_retained_state(&mut self, image: &[u8]) -> Result<(), String> {
        let h = self.pi3000_hardware.as_ref().ok_or("not a PI runtime")?;
        retained::restore_retained_state(self, h, image)
    }
    /// A2 periodic source candidate. Physical crystal/period remain unqualified.
    /// Configure before execution. Strict leaves it off; Firmware uses the
    /// related board's nominal half-second period (512,000 timing units).
    pub fn configure_pi3000_rtc_tick_candidate(&mut self, period: u64) -> Result<(), String> {
        if period == 0 || self.instruction_count() != 0 || self.elapsed_timing_units() != 0 {
            return Err("PI RTC tick requires a nonzero period before execution".into());
        }
        self.pi3000_hardware
            .as_ref()
            .ok_or("not a PI runtime")?
            .borrow()
            .rtc_tick_period = Some(period);
        Ok(())
    }
    /// Opt-in calendar clock; does not enable alarm/cyclic causes or seed a date.
    pub fn configure_pi3000_rtc_calendar_candidate(&mut self, period: u64) -> Result<(), String> {
        if period == 0 || self.instruction_count() != 0 || self.elapsed_timing_units() != 0 {
            return Err("PI RTC calendar requires a nonzero period before execution".into());
        }
        self.pi3000_hardware
            .as_ref()
            .ok_or("not a PI runtime")?
            .borrow()
            .rtc_second_period = Some(period);
        Ok(())
    }
    /// Select alarm/cyclic causes separately from the calendar, before execution.
    pub fn configure_pi3000_rtc_event_candidate(&mut self, enabled: bool) -> Result<(), String> {
        if self.instruction_count() != 0 || self.elapsed_timing_units() != 0 {
            return Err("PI RTC events must be selected before execution".into());
        }
        let mut h = self
            .pi3000_hardware
            .as_ref()
            .ok_or("not a PI runtime")?
            .borrow();
        if enabled && h.rtc_second_period.is_none() {
            return Err("PI RTC events require the calendar candidate".into());
        }
        h.rtc_event_candidate = enabled;
        Ok(())
    }
    pub fn configure_pi3000_unmapped_low_read_candidate(
        &mut self,
        value: u8,
    ) -> Result<(), String> {
        if self.instruction_count() != 0 || self.elapsed_timing_units() != 0 {
            return Err("PI unmapped read candidate must be selected before execution".into());
        }
        self.pi3000_hardware
            .as_ref()
            .ok_or("not a PI runtime")?
            .borrow()
            .unmapped_low_read = Some(value);
        Ok(())
    }
    pub fn configure_pi3000_halt_main_timer_candidate(
        &mut self,
        enabled: bool,
    ) -> Result<(), String> {
        self.pi3000_hardware
            .as_ref()
            .ok_or("PI hardware is unavailable")?
            .borrow()
            .halt_main_timer_candidate = enabled;
        Ok(())
    }
    pub fn configure_pi3000_tablet_route(&mut self, route: TabletAdcRoute) -> Result<(), String> {
        let mut h = self
            .pi3000_hardware
            .as_ref()
            .ok_or("PI hardware is unavailable")?
            .borrow();
        if h.tablet_route.is_some() && h.tablet_route != Some(route) {
            return Err("PI tablet ADC route cannot change after selection".into());
        }
        h.tablet_route = Some(route);
        Ok(())
    }
    pub fn configure_pi3000_tablet_interrupt_source(
        &mut self,
        source: TabletInterruptSource,
    ) -> Result<(), String> {
        let mut h = self
            .pi3000_hardware
            .as_ref()
            .ok_or("PI hardware is unavailable")?
            .borrow();
        if h.tablet.pressed || h.tablet.data_reads != 0 {
            return Err("PI contact source must be selected before input/conversion".into());
        }
        h.tablet_interrupt_source = source;
        Ok(())
    }
    /// Raw ten-bit physical samples only. Interrupt wiring is still pending;
    /// this does not inject firmware event queues or calibration coordinates.
    /// AuxiliaryControl routes press edges through the selected gate candidate.
    pub fn set_pi3000_raw_contact(&mut self, x: u16, y: u16, pressed: bool) -> Result<(), String> {
        let mut h = self
            .pi3000_hardware
            .as_ref()
            .ok_or("PI hardware is unavailable")?
            .borrow();
        if h.tablet_route.is_none() {
            return Err("PI tablet ADC route is unqualified; select an explicit candidate".into());
        }
        if h.tablet.set_contact(x, y, pressed)? && h.auxiliary.is_some() {
            h.gate[0x12] |= match h.tablet_interrupt_source {
                TabletInterruptSource::Gate3 => 8,
                TabletInterruptSource::Gate1 => 2,
            };
        }
        Ok(())
    }
    pub fn pi3000_hardware(&self) -> Option<&SharedHardware> {
        self.pi3000_hardware.as_ref()
    }
    pub fn pi3000_status(&self) -> Result<Value, String> {
        let h = self
            .pi3000_hardware
            .as_ref()
            .ok_or("PI hardware is unavailable")?
            .borrow();
        Ok(
            json!({"model":"pi-3000", "native_instructions":self.instruction_count(),
            "cycles":self.cycle_count(), "pc":self.get_reg("PC"),"selector":h.selector,"mode":h.mode,
            "BP":self.memory.read_internal_byte_silent(0xec),"lcd_writes":h.lcd_writes,
            "lcd_ordered_window_clipping_candidate":h.lcd.ordered_window_clipping_candidate,
            "absent_auxiliary":h.absent_auxiliary,"fault":h.fault,
            "auxiliary_control":h.auxiliary.is_some(),
            "auxiliary_accepted":h.auxiliary.as_ref().map(|a|a.accepted.clone()),
            "auxiliary_accepted_count":h.auxiliary.as_ref().map(|a|a.accepted_count),
            "auxiliary_command_count":h.auxiliary_command_count,"auxiliary_commands":h.auxiliary_commands,
            "tablet_route":h.tablet_route.map(|r|format!("{r:?}")),
            "tablet_interrupt_source":format!("{:?}",h.tablet_interrupt_source),
            "halt_main_timer_candidate":h.halt_main_timer_candidate,"halt_main_timer_units":h.halt_main_timer_units,
            "rtc_tick_period":h.rtc_tick_period,"rtc_ticks":h.rtc_ticks,"rtc_tick_phase":h.rtc_tick_phase,
            "rtc_second_period":h.rtc_second_period,"rtc_second_phase":h.rtc_second_phase,
            "rtc_seconds":h.rtc_seconds,"rtc_held_units":h.rtc_held_units,
            "rtc_event_candidate":h.rtc_event_candidate,
            "unmapped_low_read":h.unmapped_low_read,
            "tablet":[u64::from(h.tablet.x),u64::from(h.tablet.y),u64::from(h.tablet.pressed),u64::from(h.tablet.conversion_control),u64::from(h.tablet.drive_control),h.tablet.data_reads],
            "compatibility":self.pi3000_compatibility.as_ref().map(|c|c.status())}),
        )
    }
}
