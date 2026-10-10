//! Provisional OZ-9600 machine hardware, shared by native and WASM runtimes.
//! Firmware-derived bus policies remain experimental; no captured RAM is loaded.
// PY_SOURCE: pce500/oz9600/hardware.py:Hardware
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
pub mod audio;
mod boundary;
pub mod bundle;
pub mod card;
pub mod input;
pub mod lcd;
#[cfg(test)]
mod parity;
mod profile;
pub mod retained;
pub mod rtc;
pub mod session;
pub mod tablet;
#[cfg(test)]
mod tests;
use crate::{
    keyboard::KeyboardMatrix, memory::MemoryOverlay, timer::TimerContext, CoreRuntime, DeviceModel,
};
use lcd::LcdController;
pub use profile::ExecutionProfile;
use rtc::Rtc;
use tablet::Tablet;
pub const FIXED_ROM_HASH: &str = "05fa315246a13d7d33650be7c628d803157d79c94e620aca920ab7473e98e519";
/// F0D6B/F0D6E samples this live ON level, separate from SSR.1 card presence.
// PY_SOURCE: pce500/oz9600/hardware.py:ON_KEY_SSR_MASK
pub const ON_KEY_SSR_MASK: u8 = 8;
/// FE591 reserves upper RAM at [1FF38]+A0000. Thus the low 64 KiB
/// workspace shares the last quarter of the reported 256 KiB SRAM.
/// This is a firmware-derived normal-view policy; mapper modes/pages and
/// physical chip-enable decoding remain unqualified.
// PY_SOURCE: pce500/oz9600/hardware.py:WORKSPACE_RAM_OFFSET
pub const WORKSPACE_RAM_OFFSET: usize = 0x30000;

/// E-port input buffers are read-only to the guest. Their backing can still
/// be updated by a device/host input source. The related ESR-L manual names
/// F5/F6 as EIL/EIH; cold OZ firmware otherwise turns allocator data written
/// here into an asserted EI6 event. Physical input wiring remains provisional.
// PY_SOURCE: pce500/oz9600/hardware.py:guest_internal_write_is_allowed
pub fn guest_internal_write_is_allowed(address: u32) -> bool {
    !matches!(address, 0x1000f5 | 0x1000f6)
}

/// One shared device state for memory overlays, boundary work and LCD capture.
#[derive(Clone)]
pub struct SharedHardware(Arc<RwLock<Hardware>>);

impl SharedHardware {
    pub fn borrow(&self) -> RwLockReadGuard<'_, Hardware> {
        self.0.read().expect("OZ-9600 hardware lock poisoned")
    }

    pub fn borrow_mut(&self) -> RwLockWriteGuard<'_, Hardware> {
        self.0.write().expect("OZ-9600 hardware lock poisoned")
    }
}

pub struct Hardware {
    pub execution: boundary::ExecutionState,
    pub audio: audio::AudioCapture,
    /// Unwired provisional CO passthrough source; deliberately no SSR alias.
    pub audio_ci_input: bool,
    pub selector: u8,
    pub(super) retained_loaded: bool,
    pub gate: [u8; 64],
    pub rtc: Rtc,
    pub lcd: LcdController,
    pub tablet: Tablet,
    pub gpio: [u8; 256],
    pub ram: Vec<u8>,
    pub card: Option<card::Oz707Card>,
    pub banks: BTreeMap<u8, Vec<u8>>,
    pub cycle: u64,
    pub fault: Option<String>,
    pub events: Vec<Value>,
    pub lcd_pixel_watch: Option<(usize, usize)>,
    pub lcd_pixel_changes: Vec<Value>,
    pub ram_access_watch: Option<(u32, u32)>,
    pub ram_access_samples: Vec<Value>,
    pub ram_access_count: u64,
}

impl Default for Hardware {
    fn default() -> Self {
        Self {
            execution: boundary::ExecutionState::default(),
            audio: audio::AudioCapture::default(),
            audio_ci_input: false,
            selector: 7,
            retained_loaded: false,
            gate: [0; 64],
            rtc: Rtc::default(),
            lcd: LcdController::default(),
            tablet: Tablet::default(),
            gpio: [0; 256],
            ram: vec![0; 0x40000],
            card: None,
            banks: BTreeMap::new(),
            cycle: 0,
            fault: None,
            events: Vec::new(),
            lcd_pixel_watch: None,
            lcd_pixel_changes: Vec::new(),
            ram_access_watch: None,
            ram_access_samples: Vec::new(),
            ram_access_count: 0,
        }
    }
}

impl Hardware {
    fn record_ram_access(&mut self, kind: &str, address: u32, value: u8, pc: Option<u32>) {
        if !self
            .ram_access_watch
            .is_some_and(|(start, end)| (start..end).contains(&address))
        {
            return;
        }
        self.ram_access_count += 1;
        if self.ram_access_samples.len() < 1024 {
            self.ram_access_samples
                .push(json!({"kind":kind,"address":address,
                "value":value,"pc":pc,"cycle":self.cycle,"selector":self.selector,
                "gate_04020":self.gate[0x20],"page_04023":self.gate[0x23],
                "gpio_write_08b05":self.gpio[5]}));
        }
    }

    fn watched_pixel(&self) -> Option<bool> {
        self.lcd_pixel_watch.map(|(x, y)| self.lcd.pixel(x, y))
    }

    fn record_pixel_change(&mut self, before: Option<bool>, address: u32, pc: Option<u32>) {
        let after = self.watched_pixel();
        if before != after && self.lcd_pixel_changes.len() < 1024 {
            self.lcd_pixel_changes
                .push(json!({"address":address,"pc":pc,"cycle":self.cycle,
                "before":before,"after":after,"pixel":self.lcd_pixel_watch,
                "registers":self.lcd.registers.to_vec()}));
        }
    }

    pub(super) fn log(&mut self, kind: &str, address: u32, value: u8, pc: Option<u32>) {
        if let Some(last) = self.events.last_mut() {
            if last["kind"] == kind
                && last["address"] == address
                && last["value"] == value
                && last["pc"] == json!(pc)
            {
                last["repeats"] = json!(last["repeats"].as_u64().unwrap_or(1) + 1);
                last["last_cycle"] = json!(self.cycle);
                return;
            }
        }
        if self.events.len() < 2048 {
            self.events
                .push(json!({"kind":kind,"address":address,"value":value,
                                    "pc":pc,"cycle":self.cycle}));
        }
    }

    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0x04021 => Some(self.selector),
            0x04000..=0x0403f => Some(self.gate[(address - 0x4000) as usize]),
            // The full capture repeats the same 32-byte RTC view through 083FF.
            0x08000..=0x083ff => Some(self.rtc.read((address & 31) as usize)),
            0x08400..=0x0843f => Some(self.lcd.peek((address & 31) as usize, self.cycle)),
            0x08800 => Some(self.tablet.conversion_control),
            // Native RMW sequences pair these read ports with distinct write
            // addresses. Only these firmware-qualified output readbacks are
            // modeled; input pin muxes and remaining 08Bxx aliases are pending.
            0x08b1a => Some(self.gpio[2]),
            0x08b1c => Some(self.gpio[0]),
            0x08b1d => Some(self.gpio[5]),
            0x08b00..=0x08bff => Some(self.gpio[(address & 255) as usize]),
            0x08c00 => Some(self.tablet.peek_data()),
            0x08801..=0x088ff | 0x08c01..=0x08cff => Some(0),
            0x10000..=0x1ffff => {
                Some(self.ram[WORKSPACE_RAM_OFFSET + (address - 0x10000) as usize])
            }
            // The 256 KiB population in these two slots is a provisional boot
            // configuration, not established physical page wiring.
            0x80000..=0xbffff => Some(self.ram[(address - 0x80000) as usize]),
            0x30000..=0x3ffff => self.card.as_ref().and_then(|card| card.read(address)),
            0x40000..=0x7ffff => Some(
                self.card
                    .as_ref()
                    .and_then(|card| card.read(address))
                    .unwrap_or(0xff),
            ),
            0xc0000..=0xdffff => self
                .banks
                .get(&self.selector)
                .map(|bank| bank[(address - 0xc0000) as usize]),
            _ => None,
        }
    }

    pub fn architectural_read(&mut self, address: u32, pc: Option<u32>) -> u8 {
        if address == 0x08c00 {
            return match self.tablet.read_data() {
                Ok(value) => {
                    self.log("read", address, value, pc);
                    value
                }
                Err(error) => {
                    self.fault = Some(format!("{error}, read {address:05X}, PC {pc:?}"));
                    0xff
                }
            };
        }
        if (0x08400..=0x0841f).contains(&address) {
            let before = self.watched_pixel();
            return match self.lcd.read((address & 31) as usize, self.cycle) {
                Ok(value) => {
                    self.record_pixel_change(before, address, pc);
                    self.log("read", address, value, pc);
                    value
                }
                Err(error) => {
                    self.fault = Some(format!("{error}, read {address:05X}, PC {pc:?}"));
                    0xff
                }
            };
        }
        match self.read(address) {
            Some(value) => {
                self.record_ram_access("read", address, value, pc);
                if address < 0x10000 {
                    self.log("read", address, value, pc);
                }
                value
            }
            None => {
                self.fault = Some(if (0xc0000..=0xdffff).contains(&address) {
                    format!(
                        "unavailable ROM selector {:02X}, read {:05X}, PC {:?}",
                        self.selector, address, pc
                    )
                } else {
                    format!("unimplemented peripheral read {:05X}, PC {:?}", address, pc)
                });
                0xff
            }
        }
    }

    pub fn write(&mut self, address: u32, value: u8, pc: Option<u32>) {
        match address {
            0x04021 => {
                self.selector = value;
                self.log("select", address, value, pc);
            }
            0x04000..=0x0403f => {
                self.gate[(address - 0x4000) as usize] = value;
                self.log("write", address, value, pc);
            }
            0x08000..=0x083ff => {
                self.rtc.write((address & 31) as usize, value);
                self.log("write", address, value, pc);
            }
            0x08420..=0x0843f => {
                let before = self.watched_pixel();
                if let Err(error) = self.lcd.write((address & 31) as usize, value) {
                    self.fault = Some(format!("{error}, write {address:05X}, PC {pc:?}"));
                }
                self.record_pixel_change(before, address, pc);
                self.log("write", address, value, pc);
            }
            0x08800 => {
                if let Err(error) = self.tablet.write_conversion_control(value) {
                    self.fault = Some(format!("{error}, write {address:05X}, PC {pc:?}"));
                }
                self.log("write", address, value, pc);
            }
            0x08b00..=0x08bff => {
                self.gpio[(address & 255) as usize] = value;
                if address == 0x08b04 {
                    self.tablet.drive_control = value;
                }
                self.log("write", address, value, pc);
            }
            0x80000..=0xbffff => {
                self.ram[(address - 0x80000) as usize] = value;
                self.record_ram_access("write", address, value, pc);
            }
            0x10000..=0x1ffff => {
                self.ram[WORKSPACE_RAM_OFFSET + (address - 0x10000) as usize] = value;
                self.record_ram_access("write", address, value, pc);
            }
            0x30000..=0x3ffff => {
                if let Some(card) = &mut self.card {
                    card.write(address, value);
                } else {
                    self.log("unimplemented_write", address, value, pc);
                }
            }
            0x40000..=0x7ffff | 0xc0000..=0xdffff => {} // card / system ROM
            _ => self.log("unimplemented_write", address, value, pc),
        }
    }

    pub fn refresh_irq_inputs(&mut self) -> bool {
        // Firmware acknowledges RTC causes, then the GA latch. Reassert the
        // latch while a masked RTC cause remains; electrical edge/level timing
        // is a provisional device policy, not qualified silicon behavior.
        if self.rtc.interrupt_asserted() {
            self.gate[0x12] |= 1;
        }
        self.gate[0x12] & self.gate[0x10] & 0x7f != 0
    }
}

use sha2::{Digest, Sha256};
/// Shared validation for file-backed captures and browser ROM bundles.
/// The bank provider runs only after each frame's filename is checked.
pub fn from_verified_images(
    mapped: &[u8],
    manifest_bytes: &[u8],
    mut bank_provider: impl FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<CoreRuntime, String> {
    if mapped.len() != 0x100000
        || format!("{:x}", Sha256::digest(&mapped[0xe0000..])) != FIXED_ROM_HASH
    {
        return Err("fixed system ROM does not match the verified acquisition".into());
    }
    let manifest: Value = serde_json::from_slice(manifest_bytes).map_err(|e| e.to_string())?;
    if manifest["schema"] != "oz9600-bank-stream-snapshot-1" {
        return Err("requires an inspect_bank_stream verified snapshot".into());
    }
    let mut hw = Hardware::default();
    for frame in manifest["frames"].as_array().ok_or("missing frames")? {
        let selector = frame["selector"].as_u64().ok_or("missing selector")?;
        let original = frame["original_selector"].as_u64().filter(|v| *v <= 255);
        let restored = frame["restored_selector"].as_u64().filter(|v| *v <= 255);
        if !(0xf0..=0xff).contains(&selector)
            || frame["start"] != 0xc0000
            || frame["length"] != 0x20000
            || original.is_none()
            || original != restored
        {
            return Err("invalid bank metadata".into());
        }
        let name = frame["file"].as_str().ok_or("missing bank name")?;
        if name != format!("bank-{selector:02X}-C0000.bin") {
            return Err("unexpected bank filename".into());
        }
        let bank = bank_provider(name)?;
        if bank.len() != 0x20000
            || frame["sha256"].as_str() != Some(&format!("{:x}", Sha256::digest(&bank)))
            || frame["sum16"].as_u64()
                != Some(bank.iter().map(|b| u64::from(*b)).sum::<u64>() & 0xffff)
            || hw.banks.contains_key(&(selector as u8))
        {
            return Err("bank length/hash/checksum or duplicate selector mismatch".into());
        }
        hw.banks.insert(selector as u8, bank);
    }
    configure_hardware(&mapped[0xe0000..], hw)
}

/// Low-level construction for component fixtures. This entry point does not
/// validate capture identity; normal model construction uses `from_rom_bundle`.
pub fn configure_hardware(fixed: &[u8], hw: Hardware) -> Result<CoreRuntime, String> {
    if fixed.len() != 0x20000 || hw.banks.values().any(|bank| bank.len() != 0x20000) {
        return Err("OZ fixed and bank ROM windows must be 128 KiB".into());
    }
    let hardware = SharedHardware(Arc::new(RwLock::new(hw)));
    let mut runtime = CoreRuntime::new();
    runtime
        .set_device_model(DeviceModel::Oz9600)
        .map_err(|e| e.to_string())?;
    runtime.lcd = None;
    let mut keyboard = KeyboardMatrix::new();
    keyboard.set_raw_kil(true);
    keyboard.disable_fifo_mirroring();
    runtime.keyboard = Some(keyboard);
    // Nominal scheduler clock and half the related ESR-L baud table align
    // with the ROM's displayed Terminal rates. This is a functional hypothesis,
    // not a measured OZ clock or connector waveform.
    let mut uart = crate::sio::SioStub::register_only(1_024_000, 2);
    uart.init(&mut runtime.memory);
    runtime.sio = Some(uart);
    runtime.pce500_peripherals = None;
    runtime.memory.set_internal_ram_mirror(false);
    runtime.memory.set_keyboard_bridge(false);
    // Keep the allocation stable: CoreRuntime's IMR/ISR hook points into it.
    *runtime.timer = TimerContext::new(false, 0, 0);
    runtime.load_rom(fixed, 0xe0000);
    runtime.memory.set_readonly_ranges(vec![(0xe0000, 0xfffff)]);
    for (start, end, name) in [
        (0x04000, 0x0403f, "oz9600_gate_array"),
        (0x08000, 0x08fff, "oz9600_peripherals"),
        (0x10000, 0x1ffff, "oz9600_workspace"),
        (0x40000, 0xbffff, "oz9600_main_memory"),
    ] {
        let reader = hardware.clone();
        let peeker = hardware.clone();
        let writer = hardware.clone();
        runtime.memory.add_overlay(MemoryOverlay {
            start,
            end,
            name: name.into(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(move |address, pc| {
                Some(reader.borrow_mut().architectural_read(address, pc))
            })),
            // No side effects and no synthetic bytes for missing banks.
            preflight_read_handler: Some(Box::new(move |address, _| peeker.borrow().read(address))),
            write_handler: Some(Box::new(move |address, value, pc| {
                writer.borrow_mut().write(address, value, pc);
                true
            })),
            perfetto_thread: None,
        });
    }
    runtime.power_on_reset().map_err(|e| e.to_string())?;
    boundary::sync_bank_view(&hardware, &mut runtime.memory);
    runtime.lcd = Some(Box::new(boundary::LcdView(hardware.clone())));
    runtime
        .install_boundary_device(Box::new(boundary::PeripheralBoundary::new(
            hardware.clone(),
        )))
        .map_err(|error| error.to_string())?;
    runtime.oz9600_hardware = Some(hardware);
    Ok(runtime)
}
