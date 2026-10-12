// PY_SOURCE: pce500/pvf1/hardware.py
//! PV-F1 board routing. Acquired-ROM candidates are distinct from physical
//! qualification; shared CPU, LCD, RTC and ADC have independent owners.
use crate::{
    memory::MemoryOverlay,
    oz9600::{lcd::LcdController, rtc::Rtc},
    runtime_device::{BoundaryContext, BoundaryDevice},
    CoreError, CoreRuntime, DeviceModel, TimerContext,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
pub mod auxiliary;
pub mod bundle;
pub mod compatibility;
mod lcd_view;
mod rtc_clock;
pub const WIDTH: usize = 336;
pub const HEIGHT: usize = 292;
pub const FIXED_SHA256: &str = "4ba7105476b2da9d28f10df6ff7ae41466357dd0acd3915d592de3ebde3b37a9";
pub const BANKS_SHA256: &str = "222a81b9abb0092b6db694cc03eb49114a8c16180728853d12669a2ff3f29a80";
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootstrapProfile {
    #[default]
    Strict,
    AbsentAuxiliary,
    ControlCandidate,
    Firmware,
}
pub struct Hardware {
    pub selector: u8,
    pub gate: [u8; 64],
    pixel_expansion: u16,
    pub lcd: LcdController,
    pub rtc: Rtc,
    rtc_clock: rtc_clock::HalfSecondClock,
    pub ram: Vec<u8>,
    pub scratch: Vec<u8>,
    pub gpio: [u8; 18],
    pub banks: BTreeMap<u8, Vec<u8>>,
    pub cycle: u64,
    pub fault: Option<String>,
    mapped: Option<u8>,
    startup: [u8; 3],
    mapping_unlock: u8,
    auxiliary: Option<auxiliary::Auxiliary>,
    expansion_absent: bool,
    pub card: Option<crate::pa9500::card::Pa9c3Card>,
    pub tablet: Option<crate::oz9600::tablet::Tablet>,
    halt_main_timer_candidate: bool,
    pub halt_main_timer_units: u64,
    start_timers_at_first_halt_candidate: bool,
}
impl Hardware {
    fn new(banks: BTreeMap<u8, Vec<u8>>, profile: BootstrapProfile) -> Self {
        Self {
            selector: 7,
            gate: [0; 64],
            pixel_expansion: 0,
            lcd: LcdController::with_geometry(WIDTH, HEIGHT),
            rtc: Rtc::default(),
            rtc_clock: {
                let mut clock = rtc_clock::HalfSecondClock::default();
                if profile == BootstrapProfile::Firmware {
                    // Native F1BEB/BFE15=F210A owns the A2 half-second
                    // counter. This nominal timebase remains uncalibrated.
                    clock
                        .configure(Some(DeviceModel::PvF1.timer_profile().timebase_hz / 2))
                        .expect("positive PV-F1 RTC clock candidate");
                }
                clock
            },
            ram: vec![0; 0x40000],
            scratch: vec![0; 0x4000],
            gpio: [0; 18],
            banks,
            cycle: 0,
            fault: None,
            mapped: None,
            startup: [0; 3],
            mapping_unlock: 0,
            auxiliary: match profile {
                BootstrapProfile::Strict => None,
                BootstrapProfile::AbsentAuxiliary => Some(auxiliary::Auxiliary::absent()),
                BootstrapProfile::ControlCandidate | BootstrapProfile::Firmware => {
                    let mut aux = auxiliary::Auxiliary::control();
                    if profile == BootstrapProfile::Firmware {
                        aux.configure_sample_period(Some(
                            auxiliary::CAPTURE_SAMPLE_PERIOD_CANDIDATE,
                        ))
                        .expect("positive PV-F1 sample clock candidate");
                    }
                    Some(aux)
                }
            },
            expansion_absent: profile != BootstrapProfile::Strict,
            card: None,
            halt_main_timer_candidate: false,
            halt_main_timer_units: 0,
            start_timers_at_first_halt_candidate: false,
            tablet: (profile != BootstrapProfile::Strict).then(|| {
                crate::oz9600::tablet::Tablet::with_contact_sample(1023)
                    .expect("PV-F1 full-scale pressure candidate")
            }),
        }
    }
    pub fn peek(&self, a: u32) -> Option<u8> {
        match a {
            0..=0x3fff => Some(self.scratch[a as usize]),
            0x4021 => Some(self.selector),
            // F2CDA/F2CFA write a packed column and read the little-endian
            // sixteen-bit result. Each input pixel becomes two adjacent bits.
            0x401e..=0x401f => Some((self.pixel_expansion >> (8 * (a - 0x401e))) as u8),
            0x4000..=0x403f => Some(self.gate[(a - 0x4000) as usize]),
            0x8000..=0x801f => Some(self.rtc.read((a - 0x8000) as usize)),
            // FC:D27B7 read/modify/writes the control latch at its write port.
            0x8400..=0x841f | 0x842e | 0x8438..=0x843f => {
                Some(self.lcd.peek((a & 31) as usize, self.cycle))
            }
            0x8b1a => Some(self.gpio[2]), // FB6E9/FB6EF -> 08B02 read/modify/write
            0x8b1c => Some(self.gpio[0]),
            0x8b1d => Some(self.gpio[5]), // FB6E1 -> 08B05 paired feedback
            0x8a00..=0x8a5f => self.auxiliary.as_ref().and_then(|aux| aux.peek(a)),
            // FE DB71E/DB805 select external ADC 06/07. DB753/DB757
            // consume high8/low2 from 08C00. Peeking cannot advance it.
            0x8800 => self.tablet.as_ref().map(|t| t.conversion_control),
            0x8c00 => self.tablet.as_ref().map(|t| t.peek_data()),
            0xf01c..=0xf01e => Some(self.startup[(a - 0xf01c) as usize]),
            0x20000..=0x7ffff if self.card.is_some() => {
                self.card.as_ref().unwrap().read(a).or(Some(255))
            }
            0x20000..=0x7ffff if self.expansion_absent => Some(255),
            0x80000..=0xbffff => Some(self.ram[(a - 0x80000) as usize]),
            _ => None,
        }
    }
    fn read(&mut self, a: u32, pc: Option<u32>) -> u8 {
        match a {
            0x8c00 if self.tablet.is_some() => self
                .tablet
                .as_mut()
                .unwrap()
                .read_data()
                .unwrap_or_else(|e| {
                    self.fault.get_or_insert(e);
                    0
                }),
            0x8400..=0x841f => self
                .lcd
                .read((a & 31) as usize, self.cycle)
                .unwrap_or_else(|e| {
                    self.fault.get_or_insert(e);
                    0
                }),
            _ => self.peek(a).unwrap_or_else(|| {
                self.fault
                    .get_or_insert_with(|| format!("unresolved PV-F1 read {a:05X}, PC {pc:?}"));
                0
            }),
        }
    }
    fn write(&mut self, a: u32, v: u8, pc: Option<u32>) {
        match a {
            0..=0x3fff => self.scratch[a as usize] = v,
            0x4021 => self.selector = v,
            0x401e => {
                self.gate[0x1e] = v;
                self.pixel_expansion = (0..8).fold(0, |result, bit| {
                    result
                        | (if v & (1 << bit) != 0 {
                            3 << (2 * bit)
                        } else {
                            0
                        })
                });
            }
            0x4000..=0x403f => self.gate[(a - 0x4000) as usize] = v,
            0x8000..=0x801f => self.rtc.write((a - 0x8000) as usize, v),
            0x8420..=0x843f => {
                if let Err(e) = self.lcd.write((a & 31) as usize, v) {
                    self.fault.get_or_insert(e);
                }
            }
            0x8b00..=0x8b11 => {
                self.gpio[(a - 0x8b00) as usize] = v;
                if a == 0x8b04 {
                    if let Some(t) = &mut self.tablet {
                        t.drive_control = v;
                    }
                }
            }
            0x8800 if self.tablet.is_some() => {
                if let Err(e) = self.tablet.as_mut().unwrap().write_conversion_control(v) {
                    self.fault.get_or_insert(e);
                }
            }
            0x8a00..=0x8a5f if self.auxiliary.is_some() => {
                if let Err(e) = self.auxiliary.as_mut().unwrap().write(a, v, self.cycle) {
                    self.fault.get_or_insert(e);
                }
            }
            0xf00c..=0xf00e => self.startup[(a - 0xf00c) as usize] = v,
            // F80A9/F80AF and other PV mapper branches emit A5, 5A here.
            // Keep the latch; candidate absent expansion stays read-only.
            0xf103 if matches!(v, 0xa5 | 0x5a) => self.mapping_unlock = v,
            0x20000..=0x7ffff if self.card.is_some() => self.card.as_mut().unwrap().write(a, v),
            0x20000..=0x7ffff if self.expansion_absent => {}
            0x80000..=0xbffff => self.ram[(a - 0x80000) as usize] = v,
            _ => {
                self.fault.get_or_insert_with(|| {
                    format!("unresolved PV-F1 write {a:05X}={v:02X}, PC {pc:?}")
                });
            }
        }
    }
}
pub type SharedHardware = Arc<Mutex<Hardware>>;
pub fn sync_bank(hardware: &SharedHardware, memory: &mut crate::MemoryImage) -> Result<(), String> {
    let mut h = hardware.lock().unwrap();
    if h.mapped == Some(h.selector) {
        return Ok(());
    }
    let bank = h
        .banks
        .get(&h.selector)
        .ok_or_else(|| format!("unacquired PV-F1 selector {:02X}", h.selector))?;
    memory.remove_overlay("pvf1_bank");
    memory.add_rom_overlay(0xc0000, bank, "pvf1_bank");
    h.mapped = Some(h.selector);
    Ok(())
}
struct Boundary(SharedHardware, Option<u64>, u64);
impl BoundaryDevice for Boundary {
    fn before_boundary(&mut self, c: &mut BoundaryContext<'_>) -> crate::Result<Option<bool>> {
        sync_bank(&self.0, c.memory).map_err(CoreError::Other)?;
        self.2 = c.elapsed_timing_units;
        let mut h = self.0.lock().unwrap();
        if let Some(e) = &h.fault {
            return Err(CoreError::Other(e.clone()));
        }
        h.cycle = c.cycles;
        // Firmware profile starts the provisional clock at the native initial
        // HALT boundary. Strict/control profiles retain caller-owned clocks.
        if h.start_timers_at_first_halt_candidate && c.state.is_halted() && !c.state.is_off() {
            *c.timer = DeviceModel::PvF1.timer_profile().new_context(true);
            c.timer.reset(c.cycles);
            c.timer.sync_scr_selection(
                c.memory.read_internal_byte_silent(0xfd).unwrap_or(0),
                c.cycles,
            );
            h.start_timers_at_first_halt_candidate = false;
        }
        // F0B4F/F0B67 explicitly request LCC.bit7 before HALT while a
        // source remains pending. The oscillator policy is an opt-in
        // experiment; CoreRuntime still owns wake and IRQ delivery.
        self.1 = (h.halt_main_timer_candidate
            && c.state.is_halted()
            && !c.state.is_off()
            && c.memory.read_internal_byte_silent(0xfe).unwrap_or(0) & 0x80 != 0)
            .then_some(c.timer.next_mti);
        if let Some(aux) = &mut h.auxiliary {
            // F19F7 selects BFE06; its native FF575 callback samples the ADC.
            // F1A6D clears GA source3 and acknowledges80 through08A07=7F.
            if aux.advance(c.cycles) {
                h.gate[0x12] |= 8;
            }
        }
        // F19A7/F19AC gate sources through pending & mask; F1AA5
        // consumes RTC mask/cause B. Opt-in profiles model that wiring.
        if h.tablet.is_some() && h.rtc.interrupt_asserted() {
            h.gate[0x12] |= 1;
        }
        Ok(Some(h.tablet.is_some() && h.gate[0x12] & h.gate[0x10] != 0))
    }
    fn after_boundary(&mut self, c: &mut BoundaryContext<'_>, elapsed: u64) -> crate::Result<()> {
        let mut h = self.0.lock().unwrap();
        if h.rtc_clock
            .advance(c.elapsed_timing_units.wrapping_sub(self.2))
        {
            h.rtc.raise_causes(4, 0);
        }
        if let Some(deadline) = self.1.take() {
            c.timer.next_mti = deadline;
            c.timer
                .tick_timers_selected(c.memory, c.cycles, Some(c.state.pc()), true, false);
            h.halt_main_timer_units += elapsed;
        }
        h.cycle = c.cycles;
        if let Some(e) = &h.fault {
            return Err(CoreError::Other(e.clone()));
        }
        Ok(())
    }
}
pub fn configure(
    fixed: &[u8],
    banks: BTreeMap<u8, Vec<u8>>,
    profile: BootstrapProfile,
) -> Result<(CoreRuntime, SharedHardware), String> {
    bundle::validate_images(fixed, &banks)?;
    let hardware = Arc::new(Mutex::new(Hardware::new(banks, profile)));
    let mut rt = CoreRuntime::new();
    rt.set_device_model(DeviceModel::PvF1)
        .map_err(|e| e.to_string())?;
    rt.lcd = None;
    rt.sio = None;
    rt.pce500_peripherals = None;
    DeviceModel::PvF1.configure_keyboard(rt.keyboard.as_mut().unwrap());
    rt.memory.set_internal_ram_mirror(false);
    rt.memory.set_keyboard_bridge(false);
    *rt.timer = TimerContext::new(false, 0, 0);
    rt.load_rom(fixed, 0xe0000);
    rt.memory.set_readonly_ranges(vec![(0xe0000, 0xfffff)]);
    // One closed low bus and one high-RAM overlay. Preflight uses peek;
    // LCD data/block ports mutate only on architectural reads.
    {
        let (start, end) = (0, 0xbffff);
        let read = hardware.clone();
        let peek = hardware.clone();
        let write = hardware.clone();
        rt.memory.add_overlay(MemoryOverlay {
            start,
            end,
            name: "pvf1_bus".into(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(move |a, pc| {
                Some(read.lock().unwrap().read(a, pc))
            })),
            preflight_read_handler: Some(Box::new(move |a, _| peek.lock().unwrap().peek(a))),
            write_handler: Some(Box::new(move |a, v, pc| {
                write.lock().unwrap().write(a, v, pc);
                true
            })),
            perfetto_thread: None,
        });
    }
    sync_bank(&hardware, &mut rt.memory)?;
    rt.power_on_reset().map_err(|e| e.to_string())?;
    rt.install_boundary_device(Box::new(Boundary(hardware.clone(), None, 0)))
        .map_err(|e| e.to_string())?;
    if profile != BootstrapProfile::Strict {
        rt.state.set_block_transfer_policy(
            crate::llama::state::BlockTransferPolicy::CoupledPredecrement,
        );
        // Acquired FD:CEE20 uses4E03 (SUB A,I) during normal New Entry.
        // Keep the ESR-P low-byte source hypothesis in this opt-in board
        // profile; the shared SC62015 strict decoder remains unchanged.
        rt.state.set_byte_arithmetic_source_policy(
            crate::llama::state::ByteArithmeticSourcePolicy::LowByte,
        );
    }

    rt.lcd = Some(Box::new(lcd_view::LcdView(hardware.clone())));
    rt.pvf1_hardware = Some(hardware.clone());
    if profile == BootstrapProfile::Firmware {
        rt.memory.write_internal_byte(0xec, 0xd0);
        hardware.lock().unwrap().halt_main_timer_candidate = true;
        hardware
            .lock()
            .unwrap()
            .start_timers_at_first_halt_candidate = true;
        rt.pvf1_compatibility = Some(compatibility::Coordinator::new(hardware.clone()));
    }
    Ok((rt, hardware))
}
impl CoreRuntime {
    /// Logical LCD pixel center to raw ADC using coefficients owned by native IOCS.
    /// The visible main window is 336x240; 292 is controller storage height.
    pub fn pvf1_lcd_to_raw(&self, x: u16, y: u16) -> Result<[u16; 2], String> {
        if x >= WIDTH as u16 || y >= 240 {
            return Err("LCD contact outside PV-F1 main window".into());
        }
        let h = self
            .pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap();
        let ws = (0..3).fold(0u32, |v, i| {
            v | ((self.memory.read_internal_byte_silent(0xe6 + i).unwrap_or(0) as u32) << (8 * i))
        });
        if !(0x80000..=0xbff66).contains(&ws) {
            return Err("PV-F1 tablet workspace is unavailable".into());
        }
        let mut words = [0i16; 6];
        for (i, w) in words.iter_mut().enumerate() {
            let a = ws + 0x8e + 2 * i as u32;
            *w = i16::from_le_bytes([h.peek(a).unwrap(), h.peek(a + 1).unwrap()]);
        }
        crate::tablet_geometry::TabletCalibration(words)
            .from_quarter([(x * 4 + 2) as i16, (y * 4 + 2) as i16])
    }
    pub fn pvf1_hardware(&self) -> Option<&SharedHardware> {
        self.pvf1_hardware.as_ref()
    }
    pub fn set_pvf1_raw_contact(&mut self, x: u16, y: u16, pressed: bool) -> Result<(), String> {
        let mut h = self
            .pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap();
        if h.tablet
            .as_mut()
            .ok_or("select an explicit PV-F1 tablet candidate")?
            .set_contact(x, y, pressed)?
        {
            h.gate[0x12] |= 2;
        }
        Ok(())
    }
    pub fn configure_pvf1_halt_timer_candidate(&mut self, enabled: bool) -> Result<(), String> {
        let mut h = self
            .pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap();
        if h.tablet.is_none() {
            return Err("HALT timing requires an explicit board candidate".into());
        }
        h.halt_main_timer_candidate = enabled;
        Ok(())
    }
    pub fn configure_pvf1_capture_clock_candidate(
        &mut self,
        period: Option<u64>,
    ) -> Result<(), String> {
        self.pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap()
            .auxiliary
            .as_mut()
            .ok_or("requires an explicit auxiliary candidate")?
            .configure_sample_period(period)
    }
    pub fn configure_pvf1_rtc_half_second_candidate(
        &mut self,
        period: Option<u64>,
    ) -> Result<(), String> {
        let mut h = self
            .pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap();
        if h.tablet.is_none() {
            return Err("RTC timing requires an explicit board candidate".into());
        }
        h.rtc_clock.configure(period)
    }
    pub fn pvf1_status(&self) -> Result<Value, String> {
        let h = self
            .pvf1_hardware
            .as_ref()
            .ok_or("requires a PV-F1 board")?
            .lock()
            .unwrap();
        Ok(
            json!({"model":"pv-f1","pc":self.get_reg("PC"),"instructions":self.instruction_count(),"cycles":self.cycle_count(),"halted":self.state.is_halted(),"selector":h.selector,"gate":h.gate.to_vec(),"lcd_geometry":[WIDTH,HEIGHT],"lcd_counters":[h.lcd.data_reads,h.lcd.data_writes,h.lcd.block_operations],"timer_enabled":self.timer.enabled,"halt_clock_candidate":h.halt_main_timer_candidate,"auxiliary":h.auxiliary.as_ref().map(|a|a.snapshot()),"rtc":h.rtc.registers(),"rtc_clock":h.rtc_clock.snapshot(),"legacy":self.pvf1_compatibility.as_ref().map(|c|c.status()),"physical_verified":false}),
        )
    }
}
