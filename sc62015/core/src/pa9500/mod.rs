// PY_SOURCE: pce500/pa9500/bootstrap.py
//! Bounded PA-9500 reset exploration on the shared CPU/runtime.
//! Fresh 256 KiB high RAM is an explicit wiring hypothesis. Unimplemented bus
//! regions fail preflight; no captured RAM, fallback ROM banks or guest repair.
use crate::{
    memory::{MemoryImage, MemoryOverlay},
    oz9600::{lcd::LcdController, rtc::Rtc},
    runtime_device::{BoundaryContext, BoundaryDevice},
    CoreError, CoreRuntime, DeviceModel, TimerContext,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub const FIXED_ROM_HASH: &str = "4e61da23861f8a007cd83116dd26b03db5843bdd6daa033aba36a68e52d6c1aa";
pub const SELECTORS: [u8; 16] = [
    0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255,
];
pub mod card;
pub mod compat_probe;
pub mod compat_runtime;
mod lcd_view;
pub mod retained;
const MAGIC: &[u8; 8] = b"PAROM01\0";

#[derive(Clone, Copy, Debug, Default)]
pub enum BootstrapProfile {
    #[default]
    TrapPeripherals,
    /// Explicit OZ controller semantics with PA ROM window endpoint geometry.
    ControllerHypothesis,
    /// Same board hypothesis with initial BP=D0, before any guest instruction.
    ControllerBpD0,
}

/// Explicit PA ON-contact hypothesis: preserve physical SSR.ONK while held,
/// but latch ISR.ONKI only on the initial contact edge. Physical assertion and
/// re-latch timing remain unmeasured. The strict factory keeps level behavior.
pub fn install_on_contact_edge_candidate(rt: &mut CoreRuntime) -> Result<(), String> {
    if rt.device_model() != DeviceModel::Pa9500
        || rt.instruction_count() != 0
        || rt.elapsed_timing_units() != 0
    {
        return Err("PA ON contact candidate requires a fresh PA runtime".into());
    }
    rt.pa9500_on_contact_edge_candidate = true;
    Ok(())
}

/// E098A/E097F branches into the OFF continuation without RETI; F43D5
/// restores IMR and the original stack. In this explicit candidate, IMR.IRM
/// and source masks govern IRQ acceptance despite stale handler bookkeeping.
/// Strict PA and other models retain their existing metadata guard.
pub fn install_irq_imr_candidate(rt: &mut CoreRuntime, hw: &SharedHardware) -> Result<(), String> {
    if rt.device_model() != DeviceModel::Pa9500
        || rt.instruction_count() != 0
        || rt.elapsed_timing_units() != 0
        || rt.pa9500_irq_imr_only_candidate
    {
        return Err("PA IRQ IMR candidate requires a fresh PA runtime".into());
    }
    rt.pa9500_irq_imr_only_candidate = true;
    hw.borrow().irq_imr_only_candidate = true;
    Ok(())
}

/// Opt-in register-only COM transport. All OPEN/PRINT, conversion, buffering
/// and flow control remain acquired firmware work. Nominal clock/divisor are
/// functional hypotheses; the strict PA constructor leaves UART unavailable.
// PY_SOURCE: pce500/pa9500/bootstrap.py:register_uart_candidate
pub fn install_uart_candidate(rt: &mut CoreRuntime) -> Result<(), String> {
    if rt.device_model() != DeviceModel::Pa9500
        || rt.elapsed_timing_units() != 0
        || rt.sio.is_some()
    {
        return Err("PA UART candidate requires a fresh PA runtime without UART".into());
    }
    let mut uart = crate::sio::SioStub::register_only(1_024_000, 2);
    uart.init(&mut rt.memory);
    rt.sio = Some(uart);
    Ok(())
}

/// F2FAC/F2FB1 restore Y words through the LCD read-side addresses. Accept only
/// those four coordinate bytes as an explicit candidate, before execution.
// PY_SOURCE: pce500/pa9500/bootstrap.py:Hardware.install_lcd_coordinate_restore_candidate
pub fn install_lcd_coordinate_restore_candidate(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
) -> Result<(), String> {
    if rt.device_model() != DeviceModel::Pa9500
        || rt.elapsed_timing_units() != 0
        || hw.borrow().lcd.is_none()
        || hw.borrow().lcd_coordinate_restore_candidate
    {
        return Err("LCD coordinate restore candidate requires a fresh PA controller".into());
    }
    hw.borrow().lcd_coordinate_restore_candidate = true;
    Ok(())
}

#[derive(Clone)]
pub struct SharedHardware(Arc<Mutex<Hardware>>);
impl SharedHardware {
    pub fn borrow(&self) -> std::sync::MutexGuard<'_, Hardware> {
        self.0.lock().expect("PA hardware lock poisoned")
    }
}

/// Raw tablet electrical candidate, before firmware calibration or hit testing.
/// Channel 06/07 samples and EIL bit5 are inferred from F07F5/F087D. Physical
/// wiring, conversion timing and sample-to-panel calibration remain unqualified.
#[derive(Clone, Copy, Debug)]
pub struct TabletInput {
    pub x: u16,
    pub y: u16,
    pub pressure: u8,
    pub contact: bool,
}

pub struct Hardware {
    pub selector: u8,
    pub gate: [u8; 64],
    pub ram: Vec<u8>,
    pub startup_latches: [u8; 2],
    /// Explicit compatibility exploration only; ordinary profiles reject F00C.
    pub compatibility_command: Option<u8>,
    pub external_media_enabled: bool,
    pub native_zero_data: Option<u8>,
    pub observed_low_pair: Option<[u8; 2]>,
    pub observed_low_triad: Option<[u8; 3]>,
    pub observed_field_words: Option<[u8; 4]>,
    pub observed_power_pair: Option<[u8; 2]>,
    pub irq_imr_only_candidate: bool,
    /// Separate optional 4 KiB candidate inferred from F28CA/F4146. No alias
    /// wiring is established by the captured open-bus low-address data.
    pub scratch_candidate: Option<Vec<u8>>,
    scratch_mapped_active: bool,
    pub card: Option<card::Pa9c3Card>,
    /// Opt-in candidate: native 30000 SRAM writes reach the same 32 KiB chip.
    pub card_lower_write_alias: bool,
    /// Opt-in F483B consumer candidate: EOL bit2 strobes card presence on EIL
    /// bit7. Physical pin level/timing and EIL bit6 remain unqualified.
    pub card_eil_presence_candidate: bool,
    /// Unwired PA ADC/contact inputs in the explicit peripheral hypothesis.
    pub adc_control: Option<u8>,
    pub tablet: Option<TabletInput>,
    adc_low_next: bool,
    adc_latched: u16,
    pub adc_data_reads: u64,
    pub banks: BTreeMap<u8, Vec<u8>>,
    pub events: Vec<Value>,
    pub cycle: u64,
    pub fault: Option<String>,
    pub lcd: Option<LcdController>,
    pub lcd_coordinate_restore_candidate: bool,
    pub rtc: Option<Rtc>,
    mapped_selector: Option<u8>,
}
impl Hardware {
    pub fn new(banks: BTreeMap<u8, Vec<u8>>, profile: BootstrapProfile) -> Self {
        let peripherals = !matches!(profile, BootstrapProfile::TrapPeripherals);
        Self {
            selector: 7,
            gate: [0; 64],
            startup_latches: [0; 2],
            compatibility_command: None,
            external_media_enabled: false,
            native_zero_data: None,
            observed_low_pair: None,
            observed_low_triad: None,
            observed_field_words: None,
            observed_power_pair: None,
            irq_imr_only_candidate: false,
            scratch_candidate: None,
            scratch_mapped_active: false,
            card: None,
            card_lower_write_alias: false,
            card_eil_presence_candidate: false,
            adc_control: peripherals.then_some(0),
            tablet: None,
            adc_low_next: false,
            adc_latched: 0,
            adc_data_reads: 0,
            ram: vec![0; 0x40000],
            banks,
            events: Vec::new(),
            cycle: 0,
            fault: None,
            mapped_selector: None,
            lcd: peripherals.then(|| LcdController::with_geometry(195, 145)),
            lcd_coordinate_restore_candidate: false,
            rtc: peripherals.then(Rtc::default),
        }
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0 if self.native_zero_data.is_some() => self.native_zero_data,
            1..=3 if self.observed_low_triad.is_some() => {
                Some(self.observed_low_triad.unwrap()[(address - 1) as usize])
            }
            5..=8 if self.observed_field_words.is_some() => {
                Some(self.observed_field_words.unwrap()[(address - 5) as usize])
            }
            0xc..=0xd if self.observed_low_pair.is_some() => {
                Some(self.observed_low_pair.unwrap()[(address - 0xc) as usize])
            }
            0x4012 if self.tablet.is_some() => {
                let status = self.gate[18] & !2;
                Some(status | if self.tablet.unwrap().contact { 2 } else { 0 })
            }
            0x4021 => Some(self.selector),
            0xf01d..=0xf01e if self.observed_power_pair.is_some() => {
                Some(self.observed_power_pair.unwrap()[(address - 0xf01d) as usize])
            }
            0x4000..=0x403f => Some(self.gate[(address - 0x4000) as usize]),
            0x8800 => self.adc_control,
            0x8c00 if self.adc_control.is_some() => Some(if self.adc_low_next {
                (self.adc_latched & 3) as u8
            } else {
                (self.adc_sample() >> 2) as u8
            }),
            0xe000..=0xefff if self.compatibility_command == Some(1) => self
                .scratch_candidate
                .as_ref()
                .map(|bytes| bytes[(address - 0xe000) as usize]),
            0x80000..=0xbffff => Some(self.ram[(address - 0x80000) as usize]),
            0x20000..=0x7ffff if self.external_media_enabled => Some(
                self.card
                    .as_ref()
                    .and_then(|card| card.read(address))
                    .unwrap_or(0xff),
            ),
            0xc0000..=0xdffff => self
                .banks
                .get(&self.selector)
                .map(|b| b[(address - 0xc0000) as usize]),
            0x8000..=0x801f => self.rtc.as_ref().map(|r| r.read((address & 31) as usize)),
            0x8400..=0x843f => self
                .lcd
                .as_ref()
                .map(|l| l.peek((address & 31) as usize, self.cycle)),
            _ => None,
        }
    }
    pub fn card_presence_input(&self, eol: u8) -> Option<bool> {
        (self.card_eil_presence_candidate && self.external_media_enabled)
            .then_some(self.card.is_some() && eol & 4 != 0)
    }
    fn adc_sample(&self) -> u16 {
        self.tablet.map_or(0, |sample| {
            if !sample.contact {
                return 0;
            }
            match self.adc_control {
                Some(6) => sample.x,
                Some(7) => sample.y,
                Some(0x16) => u16::from(sample.pressure) << 2,
                _ => 0,
            }
        })
    }
    fn event(&mut self, kind: &str, address: u32, value: u8, pc: Option<u32>) {
        if let Some(last) = self.events.last_mut() {
            if last["kind"] == kind
                && last["address"] == address
                && last["value"] == value
                && last["pc"] == serde_json::json!(pc)
            {
                last["repeats"] = serde_json::json!(last["repeats"].as_u64().unwrap_or(1) + 1);
                last["last_cycle"] = serde_json::json!(self.cycle);
                return;
            }
        }
        if self.events.len() < 1024 {
            self.events.push(serde_json::json!({"kind":kind,"address":address,"value":value,"pc":pc,"cycle":self.cycle}));
        }
    }
    pub fn architectural_read(&mut self, address: u32, pc: Option<u32>) -> u8 {
        let value = if address == 0x8c00 && self.adc_control.is_some() {
            let value = self.read(address).unwrap();
            if !self.adc_low_next {
                self.adc_latched = self.adc_sample();
            }
            self.adc_low_next = !self.adc_low_next;
            self.adc_data_reads += 1;
            Ok(value)
        } else if (0x8400..=0x841f).contains(&address) {
            self.lcd
                .as_mut()
                .ok_or("LCD contract disabled".to_string())
                .and_then(|l| l.read((address & 31) as usize, self.cycle))
        } else {
            self.read(address)
                .ok_or_else(|| format!("unqualified PA read {address:05X}"))
        };
        match value {
            Ok(value) => {
                if address < 0x10000 {
                    self.event("read", address, value, pc);
                }
                value
            }
            Err(error) => {
                self.fault = Some(format!("{error}, PC {pc:?}"));
                0xff
            }
        }
    }
    pub fn write(&mut self, address: u32, value: u8, pc: Option<u32>) {
        match address {
            0x8800 if self.adc_control.is_some() && matches!(value, 6 | 7 | 0x0e | 0x16 | 0x1e) => {
                self.adc_control = Some(value);
                self.adc_low_next = false;
                self.event("write", address, value, pc);
            }
            0x4021 => {
                self.selector = value;
                self.event("select", address, value, pc);
            }
            0x4000..=0x403f => {
                self.gate[(address - 0x4000) as usize] = value;
                self.event("write", address, value, pc);
            }
            // F3A71/F3A75 clear two startup outputs. Only writes are qualified.
            0xf00d..=0xf00e => {
                self.startup_latches[(address - 0xf00d) as usize] = value;
                self.event("write", address, value, pc);
            }
            0xf00c if self.compatibility_command.is_some() && value == 1 => {
                self.compatibility_command = Some(value);
                self.event("compatibility_command", address, value, pc);
            }
            0x80000..=0xbffff => self.ram[(address - 0x80000) as usize] = value,
            0xe000..=0xefff
                if self.compatibility_command == Some(1) && self.scratch_candidate.is_some() =>
            {
                self.scratch_candidate.as_mut().unwrap()[(address - 0xe000) as usize] = value;
            }
            0x20000..=0x7ffff if self.external_media_enabled => {
                if let Some(card) = self.card.as_mut() {
                    let target =
                        if self.card_lower_write_alias && (0x30000..=0x37fff).contains(&address) {
                            address + 0x8000
                        } else {
                            address
                        };
                    card.write(target, value);
                }
            }
            0x8000..=0x801f if self.rtc.is_some() => {
                self.rtc
                    .as_mut()
                    .unwrap()
                    .write((address & 31) as usize, value);
                self.event("write", address, value, pc);
            }
            0x8420..=0x843f | 0x8412..=0x8413 | 0x8416..=0x8417
                if self.lcd.is_some()
                    && (address >= 0x8420 || self.lcd_coordinate_restore_candidate) =>
            {
                if let Err(error) = self
                    .lcd
                    .as_mut()
                    .unwrap()
                    .write((address & 31) as usize, value)
                {
                    self.fault = Some(error);
                }
                self.event("write", address, value, pc);
            }
            _ => self.fault = Some(format!("unqualified PA write {address:05X}, PC {pc:?}")),
        }
    }
}

fn sync_bank(hardware: &SharedHardware, memory: &mut MemoryImage) -> Result<(), String> {
    let mut hw = hardware.borrow();
    if hw.mapped_selector == Some(hw.selector) {
        return Ok(());
    }
    let bank = hw
        .banks
        .get(&hw.selector)
        .ok_or_else(|| format!("PA selector {:02X} was not acquired", hw.selector))?;
    memory.remove_overlay("pa9500_bank");
    memory.add_rom_overlay(0xc0000, bank, "pa9500_bank");
    hw.mapped_selector = Some(hw.selector);
    Ok(())
}
struct Boundary(SharedHardware);
impl BoundaryDevice for Boundary {
    fn before_boundary(&mut self, c: &mut BoundaryContext<'_>) -> crate::Result<Option<bool>> {
        sync_bank(&self.0, c.memory).map_err(CoreError::Other)?;
        let scratch = {
            let hw = self.0.borrow();
            let active = hw.compatibility_command == Some(1);
            (hw.scratch_candidate.is_some() && active != hw.scratch_mapped_active).then_some(active)
        };
        if let Some(active) = scratch {
            c.memory.remove_overlay("pa9500_scratch_candidate");
            add_memory_view(
                c.memory,
                &self.0,
                0xe000,
                0xefff,
                active,
                "pa9500_scratch_candidate",
            );
            self.0.borrow().scratch_mapped_active = active;
        }
        self.0.borrow().cycle = c.cycles;
        let hw = self.0.borrow();
        if hw.external_media_enabled {
            let ssr = c.memory.read_internal_byte_silent(0xff).unwrap_or(0);
            c.memory
                .write_internal_byte(0xff, if hw.card.is_some() { ssr | 2 } else { ssr & !2 });
        }
        let eol = c.memory.read_internal_byte_silent(0xf3).unwrap_or(0);
        if let Some(present) = hw.card_presence_input(eol) {
            let eil = c.memory.read_internal_byte_silent(0xf5).unwrap_or(0);
            c.memory
                .write_internal_byte(0xf5, if present { eil | 0x80 } else { eil & !0x80 });
        }
        if let Some(sample) = hw.tablet {
            let eil = c.memory.read_internal_byte_silent(0xf5).unwrap_or(0);
            c.memory.write_internal_byte(
                0xf5,
                if sample.contact {
                    eil | 0x20
                } else {
                    eil & !0x20
                },
            );
        }
        // E0A71 intersects gate 4010 enables with 4012 causes; bit1 calls
        // F0AA1, which masks that enable until F0A94 rearms after release.
        // This electrical gate/EXI coupling remains an explicit candidate.
        Ok(Some(
            hw.tablet.is_some_and(|sample| sample.contact) && hw.gate[16] & 2 != 0,
        ))
    }
    fn after_boundary(&mut self, c: &mut BoundaryContext<'_>, _: u64) -> crate::Result<()> {
        let mut hw = self.0.borrow();
        hw.cycle = c.cycles;
        if let Some(error) = &hw.fault {
            return Err(CoreError::Other(error.clone()));
        }
        Ok(())
    }
}

/// Explicit external-media profile, installed before any guest boundary.
pub fn install_media(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    card: Option<card::Pa9c3Card>,
) -> Result<(), String> {
    if rt.instruction_count() != 0 || rt.cycle_count() != 0 || hw.borrow().external_media_enabled {
        return Err("configure PA media exactly once before execution".into());
    }
    let rom = card.as_ref().map(|card| card.rom().repeat(2));
    hw.borrow().card = card;
    hw.borrow().external_media_enabled = true;
    let reader = hw.clone();
    let peeker = hw.clone();
    let writer = hw.clone();
    rt.memory.remove_overlay("pa9500_bus_0F00F");
    add_bus_view(rt, hw, 0xf00f, 0x1ffff, false, "pa9500_bus_0F00F");
    rt.memory.add_overlay(MemoryOverlay {
        start: 0x20000,
        end: if rom.is_some() { 0x3ffff } else { 0x7ffff },
        name: "pa9500_external_media".into(),
        data: None,
        read_only: false,
        read_handler: Some(Box::new(move |address, pc| {
            Some(reader.borrow().architectural_read(address, pc))
        })),
        preflight_read_handler: Some(Box::new(move |address, _| peeker.borrow().read(address))),
        write_handler: Some(Box::new(move |address, value, pc| {
            writer.borrow().write(address, value, pc);
            true
        })),
        perfetto_thread: None,
    });
    if let Some(rom) = rom {
        rt.memory.add_rom_overlay(0x40000, &rom, "pa9500_card_rom");
    }
    Ok(())
}

fn add_bus_view(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    start: u32,
    end: u32,
    safe: bool,
    name: &str,
) {
    add_memory_view(&mut rt.memory, hw, start, end, safe, name);
}

fn add_memory_view(
    memory: &mut MemoryImage,
    hw: &SharedHardware,
    start: u32,
    end: u32,
    safe: bool,
    name: &str,
) {
    let reader = hw.clone();
    let peeker = hw.clone();
    let writer = hw.clone();
    memory.add_overlay(MemoryOverlay {
        start,
        end,
        name: name.into(),
        data: None,
        read_only: false,
        read_handler: Some(Box::new(move |address, pc| {
            Some(reader.borrow().architectural_read(address, pc))
        })),
        preflight_read_handler: if safe {
            Some(Box::new(move |address, _| peeker.borrow().read(address)))
        } else {
            None
        },
        write_handler: Some(Box::new(move |address, value, pc| {
            writer.borrow().write(address, value, pc);
            true
        })),
        perfetto_thread: None,
    });
}

/// Explicit bounded scratch mapping experiment, before any guest execution.
/// Candidate visibility after F00C=1 is not an established hardware effect.
pub fn install_scratch_candidate(rt: &mut CoreRuntime, hw: &SharedHardware) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.cycle_count() != 0
        || hw.borrow().scratch_candidate.is_some()
        || hw.borrow().adc_control.is_none()
    {
        return Err(
            "configure scratch candidate once before execution in a peripheral profile".into(),
        );
    }
    hw.borrow().scratch_candidate = Some(vec![0; 4096]);
    rt.memory.remove_overlay("pa9500_bus_08C01");
    add_bus_view(rt, hw, 0x8c01, 0xdfff, false, "pa9500_bus_08C01");
    // The next ordinary boundary publishes a safe view only after F00C=1.
    // A disabled view must have no safe callback to prevent shadow fallback.
    add_bus_view(rt, hw, 0xe000, 0xefff, false, "pa9500_scratch_candidate");
    add_bus_view(rt, hw, 0xf000, 0xf00c, false, "pa9500_bus_0F000");
    Ok(())
}

/// Single observed data-bus byte; no instruction alias or wider fallback.
pub fn install_zero_data(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    value: u8,
) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.cycle_count() != 0
        || hw.borrow().native_zero_data.is_some()
    {
        return Err("configure the observed zero-address byte before execution".into());
    }
    hw.borrow().native_zero_data = Some(value);
    rt.memory.remove_overlay("pa9500_bus_00000");
    add_bus_view(rt, hw, 1, 0x3fff, false, "pa9500_bus_00001");
    let reader = hw.clone();
    let peeker = hw.clone();
    let writer = hw.clone();
    rt.memory.add_overlay(MemoryOverlay {
        start: 0,
        end: 0,
        name: "pa9500_zero_data".into(),
        data: None,
        read_only: false,
        read_handler: Some(Box::new(move |address, pc| {
            Some(reader.borrow().architectural_read(address, pc))
        })),
        preflight_read_handler: Some(Box::new(move |address, _| peeker.borrow().read(address))),
        write_handler: Some(Box::new(move |address, value, pc| {
            writer.borrow().write(address, value, pc);
            true
        })),
        perfetto_thread: None,
    });
    Ok(())
}

/// Opt-in observed data pair. The physical dump returned E0 at C/D, but the
/// electrical source, writable behavior and mapping under other modes are
/// unknown. No adjacent low-address fallback is installed.
pub fn install_observed_low_pair(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    bytes: [u8; 2],
) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.cycle_count() != 0
        || hw.borrow().native_zero_data.is_none()
        || hw.borrow().observed_low_pair.is_some()
    {
        return Err(
            "configure the bounded observed low pair once after the zero byte, before execution"
                .into(),
        );
    }
    hw.borrow().observed_low_pair = Some(bytes);
    rt.memory.remove_overlay("pa9500_bus_00001");
    add_bus_view(rt, hw, 1, 0xb, false, "pa9500_bus_00001");
    add_bus_view(rt, hw, 0xc, 0xd, true, "pa9500_observed_low_pair");
    add_bus_view(rt, hw, 0xe, 0x3fff, false, "pa9500_bus_0000E");
    Ok(())
}

/// Opt-in native data at 1..3 for the empty F72D0 callback selection. The
/// acquisition supplies the bytes; physical decode under other modes is open.
/// Requires the earlier pair candidate, keeps all writes/neighbors strict.
pub fn install_observed_low_triad(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    bytes: [u8; 3],
) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.cycle_count() != 0
        || hw.borrow().observed_low_pair.is_none()
        || hw.borrow().observed_low_triad.is_some()
    {
        return Err(
            "configure the bounded observed triad once after the pair, before execution".into(),
        );
    }
    hw.borrow().observed_low_triad = Some(bytes);
    rt.memory.remove_overlay("pa9500_bus_00001");
    add_bus_view(rt, hw, 1, 3, true, "pa9500_observed_low_triad");
    add_bus_view(rt, hw, 4, 0xb, false, "pa9500_bus_00004");
    Ok(())
}

/// Native data words consumed by the ROM's null NAPR format pointer. Only
/// 5..8 are readable; no legacy alias, adjacent fill or writable low RAM.
/// Acquired E0 bytes remain a board candidate, not a physical decode contract.
pub fn install_observed_field_words(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    bytes: [u8; 4],
) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.cycle_count() != 0
        || hw.borrow().observed_low_triad.is_none()
        || hw.borrow().observed_field_words.is_some()
    {
        return Err("configure observed field words once after the triad, before execution".into());
    }
    hw.borrow().observed_field_words = Some(bytes);
    rt.memory.remove_overlay("pa9500_bus_00004");
    add_bus_view(rt, hw, 4, 4, false, "pa9500_bus_00004");
    add_bus_view(rt, hw, 5, 8, true, "pa9500_observed_field_words");
    add_bus_view(rt, hw, 9, 0xb, false, "pa9500_bus_00009");
    Ok(())
}

/// Only the two acquired power-context input bytes become readable. They
/// remain distinct from F00D/E output latches; physical readback is unproven.
pub fn install_observed_power_pair(
    rt: &mut CoreRuntime,
    hw: &SharedHardware,
    bytes: [u8; 2],
) -> Result<(), String> {
    if rt.instruction_count() != 0
        || rt.elapsed_timing_units() != 0
        || !hw.borrow().external_media_enabled
        || hw.borrow().observed_field_words.is_none()
        || hw.borrow().observed_power_pair.is_some()
    {
        return Err("power context read pair requires a fresh field-read bootstrap".into());
    }
    hw.borrow().observed_power_pair = Some(bytes);
    rt.memory.remove_overlay("pa9500_bus_0F00F");
    add_bus_view(rt, hw, 0xf00f, 0xf01c, false, "pa9500_bus_0F00F");
    add_bus_view(rt, hw, 0xf01d, 0xf01e, true, "pa9500_observed_power_pair");
    add_bus_view(rt, hw, 0xf01f, 0x1ffff, false, "pa9500_bus_0F01F");
    Ok(())
}

/// Component constructor; acquisition loading validates identity separately.
pub fn configure(
    fixed: &[u8],
    banks: BTreeMap<u8, Vec<u8>>,
    profile: BootstrapProfile,
) -> Result<(CoreRuntime, SharedHardware), String> {
    if fixed.len() != 0x20000 || banks.values().any(|b| b.len() != 0x20000) {
        return Err("PA ROM window length mismatch".into());
    }
    let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(banks, profile))));
    let mut rt = CoreRuntime::new();
    rt.set_device_model(DeviceModel::Pa9500)
        .map_err(|e| e.to_string())?;
    rt.lcd = None;
    rt.keyboard = None;
    rt.sio = None;
    rt.pce500_peripherals = None;
    rt.memory.set_internal_ram_mirror(false);
    rt.memory.set_keyboard_bridge(false);
    *rt.timer = TimerContext::new(false, 0, 0);
    rt.load_rom(fixed, 0xe0000);
    rt.memory.set_readonly_ranges(vec![(0xe0000, 0xfffff)]);
    // Disjoint regions prevent an unknown read from falling through to the
    // generic external shadow. Executable C/D ROM is a separate static overlay.
    let peripherals = !matches!(profile, BootstrapProfile::TrapPeripherals);
    let mut ranges = vec![
        (0x4000, 0x403f, true),
        (0xf00d, 0xf00e, false),
        (0x80000, 0xbffff, true),
    ];
    if peripherals {
        ranges.extend([
            (0x8000, 0x801f, true),
            (0x8400, 0x843f, true),
            (0x8800, 0x8800, true),
            (0x8c00, 0x8c00, true),
        ]);
    }
    ranges.sort_unstable();
    let mut complete = Vec::new();
    let mut previous = 0;
    for (start, end, safe) in ranges {
        if previous < start {
            complete.push((previous, start - 1, false));
        }
        complete.push((start, end, safe));
        previous = end + 1;
    }
    for (start, end, safe) in complete {
        let reader = hardware.clone();
        let peeker = hardware.clone();
        let writer = hardware.clone();
        rt.memory.add_overlay(MemoryOverlay {
            start,
            end,
            name: format!("pa9500_bus_{start:05X}"),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(move |a, pc| {
                Some(reader.borrow().architectural_read(a, pc))
            })),
            preflight_read_handler: if safe {
                Some(Box::new(move |a, _| peeker.borrow().read(a)))
            } else {
                None
            },
            write_handler: Some(Box::new(move |a, v, pc| {
                writer.borrow().write(a, v, pc);
                true
            })),
            perfetto_thread: None,
        });
    }
    sync_bank(&hardware, &mut rt.memory)?;
    rt.power_on_reset().map_err(|e| e.to_string())?;
    if matches!(profile, BootstrapProfile::ControllerBpD0) {
        rt.memory.write_internal_byte(0xec, 0xd0);
    }
    rt.install_boundary_device(Box::new(Boundary(hardware.clone())))
        .map_err(|e| e.to_string())?;
    Ok((rt, hardware))
}

pub fn from_verified_images(
    fixed: &[u8],
    manifest: &[u8],
    mut provider: impl FnMut(&str) -> Result<Vec<u8>, String>,
    profile: BootstrapProfile,
) -> Result<(CoreRuntime, SharedHardware), String> {
    if fixed.len() != 0x20000 || format!("{:x}", Sha256::digest(fixed)) != FIXED_ROM_HASH {
        return Err("PA fixed ROM identity mismatch".into());
    }
    let manifest: Value = serde_json::from_slice(manifest).map_err(|e| e.to_string())?;
    if manifest["schema"] != "pa9500-rom-bank-set-1"
        || manifest["bank_window_start"] != 0xc0000
        || manifest["bank_view_bytes"] != 0x20000
    {
        return Err("PA acquisition manifest mismatch".into());
    }
    let mut banks = BTreeMap::new();
    for frame in manifest["banks"].as_array().ok_or("missing banks")? {
        let selector = frame["selector"]
            .as_u64()
            .filter(|s| *s <= 255)
            .ok_or("invalid selector")? as u8;
        let name = frame["file"].as_str().ok_or("missing bank file")?;
        if !SELECTORS.contains(&selector)
            || name != format!("banks/bank-{selector:02X}-C0000.bin")
            || frame["bytes"] != 0x20000
            || frame["cpu_start"] != 0xc0000
            || frame["cpu_end_inclusive"] != 0xdffff
            || frame["selector_restoration_verified"] != true
        {
            return Err("PA bank metadata mismatch".into());
        }
        let data = provider(name)?;
        if data.len() != 0x20000
            || frame["sha256"].as_str() != Some(&format!("{:x}", Sha256::digest(&data)))
            || banks.insert(selector, data).is_some()
        {
            return Err("PA bank hash/length/duplicate mismatch".into());
        }
    }
    if banks.len() != 16 {
        return Err("PA bundle requires all sixteen acquired selectors".into());
    }
    configure(fixed, banks, profile)
}

/// Immutable fixed ROM + manifest + banks, never the captured live memory.
pub fn from_rom_bundle(image: &[u8]) -> Result<(CoreRuntime, SharedHardware), String> {
    from_rom_bundle_with_profile(image, BootstrapProfile::TrapPeripherals)
}

fn from_rom_bundle_with_profile(
    image: &[u8],
    profile: BootstrapProfile,
) -> Result<(CoreRuntime, SharedHardware), String> {
    if image.len() < 52 || &image[..8] != MAGIC {
        return Err("PA ROM bundle header mismatch".into());
    }
    let word = |n| u32::from_le_bytes(image[n..n + 4].try_into().unwrap()) as usize;
    let (fixed_len, manifest_len, banks_len) = (word(8), word(12), word(16));
    if fixed_len != 0x20000
        || manifest_len == 0
        || manifest_len > 0x40000
        || banks_len != 16 * 0x20000
        || image.len() != 52 + fixed_len + manifest_len + banks_len
    {
        return Err("PA ROM bundle length mismatch".into());
    }
    let payload = &image[52..];
    if Sha256::digest(payload).as_slice() != &image[20..52] {
        return Err("PA ROM bundle checksum mismatch".into());
    }
    let mut offset = 0x20000 + manifest_len;
    from_verified_images(
        &payload[..0x20000],
        &payload[0x20000..offset],
        |_| {
            let data = payload[offset..offset + 0x20000].to_vec();
            offset += 0x20000;
            Ok(data)
        },
        profile,
    )
}

/// Explicit firmware-qualified board candidate, with fresh organizer backing.
/// Physical bus decode, compatibility timing and auxiliary LCD pixels remain
/// hypotheses. This does not change the strict model factory.
pub fn from_bootstrap_bundle(
    image: &[u8],
    card: Option<card::Pa9c3Card>,
) -> Result<(CoreRuntime, SharedHardware), String> {
    let (mut rt, hw) = from_rom_bundle_with_profile(image, BootstrapProfile::ControllerBpD0)?;
    install_media(&mut rt, &hw, card)?;
    install_zero_data(&mut rt, &hw, 0xe0)?;
    hw.borrow().card_eil_presence_candidate = true;
    install_scratch_candidate(&mut rt, &hw)?;
    rt.state
        .set_block_transfer_policy(crate::llama::state::BlockTransferPolicy::CoupledPredecrement);
    install_on_contact_edge_candidate(&mut rt)?;
    install_lcd_coordinate_restore_candidate(&mut rt, &hw)?;
    rt.install_pa9500_compatibility(
        hw.clone(),
        compat_runtime::LegacyDataPolicy::AbsentMediaAndPagedRom,
    )?;
    *rt.timer = DeviceModel::Pa9500.timer_profile().new_context(true);
    rt.press_on_key();
    rt.lcd = Some(Box::new(lcd_view::LcdView(hw.clone())));
    Ok((rt, hw))
}

/// Versioned low-data candidate, backed by the native 1 MiB acquisition
/// 5b532a995ba21ec3145a207d11ab8c586451df4799e624ba472732496729e351.
/// Only bytes 1..3 and C/D are added, read-only; other unknown reads/writes
/// retain their traps. Physical decode in other modes remains unqualified.
pub fn from_bootstrap_bundle_with_observed_low_reads(
    image: &[u8],
    card: Option<card::Pa9c3Card>,
) -> Result<(CoreRuntime, SharedHardware), String> {
    let (mut rt, hw) = from_bootstrap_bundle(image, card)?;
    install_observed_low_pair(&mut rt, &hw, [0xe0; 2])?;
    install_observed_low_triad(&mut rt, &hw, [0xe0; 3])?;
    Ok((rt, hw))
}

/// bootstrap-v3 adds four acquired native bytes for the optional NAPR format
/// read. All older profiles and unknown low reads/writes retain their traps.
pub fn from_bootstrap_bundle_with_observed_field_reads(
    image: &[u8],
    card: Option<card::Pa9c3Card>,
) -> Result<(CoreRuntime, SharedHardware), String> {
    let (mut rt, hw) = from_bootstrap_bundle_with_observed_low_reads(image, card)?;
    install_observed_field_words(&mut rt, &hw, [0xe0; 4])?;
    Ok((rt, hw))
}

/// bootstrap-v4 adds bounded acquired power-context reads and the explicit
/// IMR-based IRQ candidate for the firmware's OFF continuation without RETI.
pub fn from_bootstrap_bundle_with_power_resume(
    image: &[u8],
    card: Option<card::Pa9c3Card>,
) -> Result<(CoreRuntime, SharedHardware), String> {
    let (mut rt, hw) = from_bootstrap_bundle_with_observed_field_reads(image, card)?;
    install_observed_power_pair(&mut rt, &hw, [0xe0; 2])?;
    install_irq_imr_candidate(&mut rt, &hw)?;
    Ok((rt, hw))
}

/// Read-only cross-runner observation. Native counters exclude H boundaries;
/// legacy time is uncalibrated and is not charged as fabricated CPU cycles.
pub fn bootstrap_status(rt: &CoreRuntime) -> Result<Value, String> {
    let c = rt
        .pa9500_compatibility()
        .ok_or("PA bootstrap coordinator unavailable")?;
    let hw = c.hardware.borrow();
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let mut status = serde_json::json!({
        "schema": "pa9500-bootstrap-status-1", "profile": if hw.observed_power_pair.is_some() { "bootstrap-v4" } else if hw.observed_field_words.is_some() { "bootstrap-v3" } else if hw.observed_low_triad.is_some() { "bootstrap-v2" } else { "bootstrap-v1" },
        "active_isa": if c.in_legacy { "esr-h" } else { "native" },
        "native_registers": crate::collect_registers(&rt.state),
        "native_instructions": rt.instruction_count(), "native_cycles": rt.cycle_count(),
        "legacy_boundaries": c.legacy_boundaries,
        "legacy": compat_runtime::legacy_state(&c.legacy),
        "transitions": c.transitions, "legacy_trace": c.legacy_trace,
        "internal_sha256": hash(&(0..256).map(|a| rt.memory.read_internal_byte_silent(a).unwrap()).collect::<Vec<_>>()),
        "main_ram_sha256": hash(&hw.ram), "scratch_sha256": hw.scratch_candidate.as_ref().map(|b| hash(b)),
        "selector": hw.selector, "gate": hw.gate.to_vec(),
        "card_sram_sha256": hw.card.as_ref().map(|card| hash(&card.sram)),
        "physical_switch_timing_qualified": false
    });
    if hw.observed_power_pair.is_some() {
        status["power_context"] = serde_json::json!({
            "state":format!("{:?}", rt.state.power_state()).to_ascii_lowercase(),
            "input_pair":hw.observed_power_pair,"output_latches":hw.startup_latches,
            "irq_IMR_candidate":hw.irq_imr_only_candidate,
            "elapsed_timing_units":rt.elapsed_timing_units(),
            "physical_readback_qualified":false
        });
    }
    Ok(status)
}

#[cfg(test)]
mod tests;
