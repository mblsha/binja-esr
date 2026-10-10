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
pub mod compat_probe;
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

#[derive(Clone)]
pub struct SharedHardware(Arc<Mutex<Hardware>>);
impl SharedHardware {
    pub fn borrow(&self) -> std::sync::MutexGuard<'_, Hardware> {
        self.0.lock().expect("PA hardware lock poisoned")
    }
}

pub struct Hardware {
    pub selector: u8,
    pub gate: [u8; 64],
    pub ram: Vec<u8>,
    pub startup_latches: [u8; 2],
    /// Unwired PA ADC/contact inputs in the explicit peripheral hypothesis.
    pub adc_control: Option<u8>,
    pub banks: BTreeMap<u8, Vec<u8>>,
    pub events: Vec<Value>,
    pub cycle: u64,
    pub fault: Option<String>,
    pub lcd: Option<LcdController>,
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
            adc_control: peripherals.then_some(0),
            ram: vec![0; 0x40000],
            banks,
            events: Vec::new(),
            cycle: 0,
            fault: None,
            mapped_selector: None,
            lcd: peripherals.then(|| LcdController::with_geometry(195, 145)),
            rtc: peripherals.then(Rtc::default),
        }
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0x4021 => Some(self.selector),
            0x4000..=0x403f => Some(self.gate[(address - 0x4000) as usize]),
            0x8800 => self.adc_control,
            0x8c00 if self.adc_control.is_some() => Some(0),
            0x80000..=0xbffff => Some(self.ram[(address - 0x80000) as usize]),
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
        let value = if (0x8400..=0x841f).contains(&address) {
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
            0x80000..=0xbffff => self.ram[(address - 0x80000) as usize] = value,
            0x8000..=0x801f if self.rtc.is_some() => {
                self.rtc
                    .as_mut()
                    .unwrap()
                    .write((address & 31) as usize, value);
                self.event("write", address, value, pc);
            }
            0x8420..=0x843f if self.lcd.is_some() => {
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
        self.0.borrow().cycle = c.cycles;
        Ok(Some(false))
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

/// Component constructor; normal acquisition loading validates identity below.
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
        BootstrapProfile::TrapPeripherals,
    )
}

#[cfg(test)]
mod tests;
