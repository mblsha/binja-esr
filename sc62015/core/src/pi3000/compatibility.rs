// PY_SOURCE: pce500/pi3000/hardware.py
//! PI bus adapter for the shared bounded ESR-H interpreter.
use super::SharedHardware;
use crate::{
    pa9500::compat_probe::{self, StartupBus, StartupState, Transfer},
    CoreRuntime,
};
use serde_json::{json, Value};

pub struct Coordinator {
    hardware: SharedHardware,
    pub legacy: StartupState,
    pub in_legacy: bool,
    pub boundaries: u64,
    pub transitions: Vec<Value>,
}
impl Coordinator {
    pub fn new(hardware: SharedHardware) -> Self {
        Self {
            hardware,
            legacy: StartupState::new(0),
            in_legacy: false,
            boundaries: 0,
            transitions: Vec::new(),
        }
    }
    pub fn status(&self) -> Value {
        json!({"in_legacy":self.in_legacy,"legacy_boundaries":self.boundaries,
            "legacy_pc":self.legacy.pc,"transitions":self.transitions})
    }
    pub fn step(&mut self, native: &mut CoreRuntime) -> Result<(), String> {
        if !self.in_legacy {
            super::sync_bank(&self.hardware, &mut native.memory);
            let pc = native.get_reg("PC");
            let opcode = native.memory.read_byte_for_preflight(pc, Some(pc));
            native.step(1).map_err(|e| e.to_string())?;
            let target = native.get_reg("PC");
            if opcode == Some(3) && target < 0x10000 && self.hardware.borrow().gate[15] & 2 != 0 {
                self.legacy.pc = target as u16;
                for i in 0..128 {
                    self.legacy.internal[i] =
                        native.memory.read_internal_byte_silent(i as u32).unwrap();
                }
                self.legacy.carry = native.get_reg("F") & 1 != 0;
                self.legacy.zero = native.get_reg("F") & 2 != 0;
                self.in_legacy = true;
                self.transitions
                    .push(json!({"from":"native","to":"legacy","source":pc,"target":target}));
            }
            return Ok(());
        }
        let mut bus = Bus {
            hardware: &self.hardware,
        };
        let transfer = compat_probe::step(&mut self.legacy, &mut bus)?;
        self.boundaries += 1;
        for i in 0..128 {
            native
                .memory
                .write_internal_byte(i as u32, self.legacy.internal[i]);
        }
        if let Transfer::Native(target) = transfer {
            if self.hardware.borrow().gate[15] & 2 != 0 {
                return Err("PI legacy transfer did not clear gate bit1".into());
            }
            native.set_reg(
                "F",
                u32::from(self.legacy.carry) | (u32::from(self.legacy.zero) << 1),
            );
            native.set_reg("PC", target);
            self.in_legacy = false;
            self.transitions
                .push(json!({"from":"legacy","to":"native","target":target}));
        }
        Ok(())
    }
}
struct Bus<'a> {
    hardware: &'a SharedHardware,
}
impl StartupBus for Bus<'_> {
    fn fetch(&self, address: u16) -> Result<u8, String> {
        let h = self.hardware.borrow();
        let (selector, offset) = match address {
            0..=0x1fff => (0, 0x10000 + address as usize),
            0x2000..=0x3fff => (0, address as usize),
            0xc000..=0xffff if h.gate[6] <= 0x7f => {
                let page = h.gate[6];
                (
                    page >> 3,
                    (page as usize & 7) * 0x4000 + (address as usize & 0x3fff),
                )
            }
            _ => return Err(format!("unqualified PI legacy fetch {address:04X}")),
        };
        h.banks
            .get(&selector)
            .map(|b| b[offset])
            .ok_or_else(|| format!("PI legacy ROM selector {selector:02X} missing"))
    }
    fn read(&self, address: u16) -> Result<u8, String> {
        if address >= 0xc000 {
            return self.fetch(address);
        }
        let h = self.hardware.borrow();
        match address {
            0x4000..=0x7fff => Ok(h.ram[(address - 0x4000) as usize]),
            0x8000..=0xbfff if h.gate[4] == 1 => Ok(255),
            _ => Err(format!("unqualified PI legacy data {address:04X}")),
        }
    }
    fn write(&mut self, address: u16, value: u8) -> Result<(), String> {
        if !(0x4000..=0x7fff).contains(&address) {
            return Err(format!("unqualified PI legacy write {address:04X}"));
        }
        self.hardware.borrow().ram[(address - 0x4000) as usize] = value;
        Ok(())
    }
    fn fill(&mut self, start: u16, count: u16, value: u8) -> Result<(), String> {
        if count == 0 || (0..count).any(|i| !(0x4000..=0x7fff).contains(&start.wrapping_add(i))) {
            return Err("unqualified PI legacy fill span".into());
        }
        let mut h = self.hardware.borrow();
        for i in 0..count {
            h.ram[(start.wrapping_add(i) - 0x4000) as usize] = value;
        }
        Ok(())
    }
    fn writ(&mut self, dp: u16, p: u8, value: u8) -> Result<(), String> {
        let register = dp as u8;
        let allowed = match register {
            15 => p == 2 && matches!(value, 0 | 2 | 3),
            4 | 5 => matches!(p, 2 | 12 | 13) && value <= 3,
            6 => p == 2 && value <= 0x7f,
            0 => p == 2 && value <= 7,
            7 => p == 2 && value == 1,
            _ => false,
        };
        if !allowed {
            return Err(format!(
                "unqualified PI legacy WRIT {dp:04X}/{p:02X}/{value:02X}"
            ));
        }
        self.hardware.borrow().gate[register as usize] = value;
        Ok(())
    }
    fn outf(&mut self, value: u8) -> Result<(), String> {
        if value != 0 {
            return Err("unqualified PI legacy OUTF".into());
        }
        Ok(())
    }
}
