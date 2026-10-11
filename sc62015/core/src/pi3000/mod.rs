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
    pub cycle: u64,
    pub fault: Option<String>,
    pub lcd_writes: u64,
    mapped: Option<(u8, u8)>,
}
impl Hardware {
    pub fn external_irq(&self) -> bool {
        self.auxiliary.is_some() && self.gate[0x10] & self.gate[0x12] & 0x7f != 0
    }
    pub fn new(banks: BTreeMap<u8, Vec<u8>>, profile: BootstrapProfile) -> Self {
        Self {
            selector: 0xfd,
            mode: 0,
            gate: [0; 64],
            ram: vec![0; 0x40000],
            scratch: vec![0; 0x4000],
            banks,
            lcd: LcdController::with_geometry(WIDTH, HEIGHT),
            rtc: Rtc::default(),
            tablet: Tablet::default(),
            tablet_route: None,
            tablet_interrupt_source: TabletInterruptSource::default(),
            halt_main_timer_candidate: false,
            halt_main_timer_units: 0,
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
            cycle: 0,
            fault: None,
            lcd_writes: 0,
            mapped: None,
        }
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0..=0x3fff => Some(self.scratch[address as usize]),
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
            0x8a01 | 0x8a06..=0x8a08 => self.io_8a[(address - 0x8a00) as usize] = value,
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

struct Boundary(SharedHardware, Option<u64>);
impl BoundaryDevice for Boundary {
    fn before_boundary(&mut self, c: &mut BoundaryContext<'_>) -> crate::Result<Option<bool>> {
        sync_bank(&self.0, c.memory);
        let mut h = self.0.borrow();
        if let Some(e) = &h.fault {
            return Err(CoreError::Other(e.clone()));
        }
        h.cycle = c.cycles;
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
        if let Some(auxiliary) = &mut h.auxiliary {
            auxiliary.advance(c.cycles);
        }
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
    rt.install_boundary_device(Box::new(Boundary(hardware.clone(), None)))
        .map_err(|e| e.to_string())?;
    rt.lcd = Some(Box::new(lcd_view::LcdView(hardware.clone())));
    rt.pi3000_hardware = Some(hardware.clone());
    if profile != BootstrapProfile::Strict {
        rt.pi3000_compatibility = Some(compatibility::Coordinator::new(hardware.clone()));
    }
    if profile == BootstrapProfile::Firmware {
        use crate::llama::state::{BlockTransferPolicy, IsrSoftwareWritePolicy};
        rt.memory.write_internal_byte(0xec, 0xd0);
        rt.state
            .set_isr_software_write_policy(IsrSoftwareWritePolicy::ClearOnly);
        rt.state
            .set_block_transfer_policy(BlockTransferPolicy::CoupledPredecrement);
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
            "absent_auxiliary":h.absent_auxiliary,"fault":h.fault,
            "auxiliary_control":h.auxiliary.is_some(),
            "auxiliary_accepted":h.auxiliary.as_ref().map(|a|a.accepted.clone()),
            "auxiliary_accepted_count":h.auxiliary.as_ref().map(|a|a.accepted_count),
            "auxiliary_command_count":h.auxiliary_command_count,"auxiliary_commands":h.auxiliary_commands,
            "tablet_route":h.tablet_route.map(|r|format!("{r:?}")),
            "tablet_interrupt_source":format!("{:?}",h.tablet_interrupt_source),
            "halt_main_timer_candidate":h.halt_main_timer_candidate,"halt_main_timer_units":h.halt_main_timer_units,
            "tablet":[u64::from(h.tablet.x),u64::from(h.tablet.y),u64::from(h.tablet.pressed),u64::from(h.tablet.conversion_control),u64::from(h.tablet.drive_control),h.tablet.data_reads],
            "compatibility":self.pi3000_compatibility.as_ref().map(|c|c.status())}),
        )
    }
}
