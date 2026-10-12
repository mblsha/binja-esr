// PY_SOURCE: pce500/pvf1/compatibility.py
//! Opt-in PV-F1 mixed-ISA candidate. Fetch, data and scalar WRIT are separate;
//! acquired bank-00 gateway pairs establish the split fetch view. Electrical
//! page decoding, handoff state and legacy timing remain hypotheses.
use crate::pvf1::Hardware;
use crate::{
    pa9500::compat_probe::{self, StartupBus, StartupState, Transfer},
    CoreRuntime,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

pub struct Coordinator {
    hardware: Arc<Mutex<Hardware>>,
    pub state: StartupState,
    pub active: bool,
    pub boundaries: u64,
    pub transitions: Vec<Value>,
    pub trace: Vec<Value>,
}
impl Coordinator {
    pub fn new(hardware: Arc<Mutex<Hardware>>) -> Self {
        Self {
            hardware,
            state: StartupState::new(0),
            active: false,
            boundaries: 0,
            transitions: vec![],
            trace: vec![],
        }
    }
    pub fn step(&mut self, native: &mut CoreRuntime) -> Result<(), String> {
        if self.active {
            let hardware = self.hardware.clone();
            return self.step_legacy(native, &hardware);
        }
        super::sync_bank(&self.hardware, &mut native.memory)?;
        let source = native.get_reg("PC");
        let opcode = (!native.state.is_halted())
            .then(|| native.memory.read_byte_for_preflight(source, Some(source)))
            .flatten();
        let encoded = (opcode == Some(3))
            .then(|| {
                (0..3).try_fold(0u32, |v, i| {
                    native
                        .memory
                        .read_byte_for_preflight(source + 1 + i, Some(source))
                        .map(|b| v | ((b as u32) << (i * 8)))
                })
            })
            .flatten();
        native
            .step_scheduler_boundaries(1)
            .map_err(|e| e.to_string())?;
        let target = native.get_reg("PC");
        if encoded == Some(target)
            && target < 0x10000
            && self.hardware.lock().unwrap().gate[15] & 2 != 0
        {
            self.enter(native, source, target);
        }
        Ok(())
    }

    pub fn status(&self) -> Value {
        json!({"active":self.active,"boundaries":self.boundaries,"state":crate::pa9500::compat_runtime::legacy_state(&self.state),"transitions":self.transitions,"trace":self.trace,"timing_qualified":false})
    }
    pub fn enter(&mut self, native: &CoreRuntime, source: u32, target: u32) {
        self.state.pc = target as u16;
        for i in 0..128 {
            self.state.internal[i] = native.memory.read_internal_byte_silent(i as u32).unwrap();
        }
        self.state.carry = native.get_reg("F") & 1 != 0;
        self.state.zero = native.get_reg("F") & 2 != 0;
        self.active = true;
        self.transitions.push(json!({"from":"native","to":"legacy","source":source,"target":target,"native_instructions":native.instruction_count()}));
    }
    fn step_legacy(
        &mut self,
        native: &mut CoreRuntime,
        hardware: &Arc<Mutex<Hardware>>,
    ) -> Result<(), String> {
        let before = crate::pa9500::compat_runtime::legacy_state(&self.state);
        let mut bus = Bus(hardware);
        let transfer = compat_probe::step(&mut self.state, &mut bus)?;
        self.boundaries += 1;
        if self.trace.len() < 1024 {
            self.trace.push(json!({"before":before,"after":crate::pa9500::compat_runtime::legacy_state(&self.state)}));
        }
        for i in 0..128 {
            native
                .memory
                .write_internal_byte(i as u32, self.state.internal[i]);
        }
        if let Transfer::Native(target) = transfer {
            if hardware.lock().unwrap().gate[15] & 2 != 0 {
                return Err("PV-F1 legacy return did not clear gate bit1".into());
            }
            native.set_reg(
                "F",
                u32::from(self.state.carry) | (u32::from(self.state.zero) << 1),
            );
            native.set_reg("PC", target);
            self.active = false;
            self.transitions.push(json!({"from":"legacy","to":"native","source":before["pc"],"target":target,"legacy_boundaries":self.boundaries}));
        }
        Ok(())
    }
}
struct Bus<'a>(&'a Arc<Mutex<Hardware>>);
impl StartupBus for Bus<'_> {
    fn fetch(&self, address: u16) -> Result<u8, String> {
        let h = self.0.lock().unwrap();
        let (selector, offset) = match address {
            0..=0x1fff => (0, 0x10000 + address as usize),
            0x2000..=0x3fff => (0, address as usize),
            0xc000..=0xffff if h.gate[6] <= 0x7f => (
                h.gate[6] >> 3,
                (h.gate[6] as usize & 7) * 0x4000 + (address as usize & 0x3fff),
            ),
            _ => {
                return Err(format!(
                    "unqualified PV-F1 legacy fetch {address:04X}; gates={:?}",
                    &h.gate[..16]
                ))
            }
        };
        h.banks
            .get(&selector)
            .map(|b| b[offset])
            .ok_or_else(|| format!("missing PV-F1 legacy selector {selector:02X}"))
    }
    fn read(&self, address: u16) -> Result<u8, String> {
        if address >= 0xc000 {
            return self.fetch(address);
        }
        let h = self.0.lock().unwrap();
        match address {
            0x4000..=0x7fff => Ok(h.ram[(address - 0x4000) as usize]),
            0x8000..=0xbfff if h.gate[4] == 1 => Ok(255),
            _ => Err(format!(
                "unqualified PV-F1 legacy data {address:04X}; gates={:?}",
                &h.gate[..16]
            )),
        }
    }
    fn write(&mut self, address: u16, value: u8) -> Result<(), String> {
        if !(0x4000..=0x7fff).contains(&address) {
            return Err(format!("unqualified PV-F1 legacy write {address:04X}"));
        }
        self.0.lock().unwrap().ram[(address - 0x4000) as usize] = value;
        Ok(())
    }
    fn fill(&mut self, start: u16, count: u16, value: u8) -> Result<(), String> {
        if count == 0 || (0..count).any(|i| !(0x4000..=0x7fff).contains(&start.wrapping_add(i))) {
            return Err("unqualified PV-F1 legacy fill span".into());
        }
        let mut h = self.0.lock().unwrap();
        for i in 0..count {
            h.ram[(start.wrapping_add(i) - 0x4000) as usize] = value;
        }
        Ok(())
    }
    fn writ(&mut self, dp: u16, p: u8, value: u8) -> Result<(), String> {
        let allowed = match dp as u8 {
            15 => p == 2 && matches!(value, 0 | 2 | 3),
            4 | 5 => matches!(p, 2 | 12 | 13) && value <= 3,
            6 => p == 2 && value <= 0x7f,
            0 => p == 2 && value <= 7,
            7 => p == 2 && value == 1,
            _ => false,
        };
        if !allowed {
            return Err(format!(
                "unqualified PV-F1 legacy WRIT {dp:04X}/{p:02X}/{value:02X}"
            ));
        }
        self.0.lock().unwrap().gate[dp as u8 as usize] = value;
        Ok(())
    }
    fn outf(&mut self, value: u8) -> Result<(), String> {
        if value != 0 {
            return Err(format!("unqualified PV-F1 legacy OUTF {value:02X}"));
        }
        Ok(())
    }
}
