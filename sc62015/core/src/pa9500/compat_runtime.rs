// PY_SOURCE: pce500/pa9500/compat_runtime.py
//! Explicit mixed-ISA exploration coordinator over the shared native runtime.
//! Gate-triggered far transfers, 16 KiB legacy pages, scalar WRIT and shared
//! low IRAM/C/Z are hypotheses. Default PA factory profiles remain strict.
use super::{
    compat_probe::{self, StartupBus, StartupState, Transfer},
    SharedHardware,
};
use crate::keyboard::KeyboardMatrix;
use crate::CoreRuntime;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyDataPolicy {
    TrapUnknown,
    /// Only WRIT port04=1's 8000..BFFF aperture is treated as absent media.
    AbsentLegacyMedia,
    /// Additionally read the selected instruction C000 window as ordinary data.
    AbsentMediaAndPagedRom,
}
pub struct CompatibilityCoordinator {
    pub hardware: SharedHardware,
    pub legacy: StartupState,
    pub in_legacy: bool,
    pub legacy_boundaries: u64,
    pub transitions: Vec<Value>,
    pub legacy_trace: Vec<Value>,
    data_policy: LegacyDataPolicy,
}
impl CompatibilityCoordinator {
    fn new(
        native: &mut CoreRuntime,
        hardware: SharedHardware,
        data_policy: LegacyDataPolicy,
    ) -> Self {
        hardware.borrow().compatibility_command = Some(0);
        let mut keyboard = KeyboardMatrix::new();
        keyboard.set_raw_kil(true);
        keyboard.disable_fifo_mirroring();
        keyboard.set_press_threshold(1);
        native.keyboard = Some(keyboard);
        Self {
            hardware,
            legacy: StartupState::new(0),
            in_legacy: false,
            legacy_boundaries: 0,
            transitions: Vec::new(),
            legacy_trace: Vec::new(),
            data_policy,
        }
    }
    pub fn step(&mut self, native: &mut CoreRuntime) -> Result<(), String> {
        if !self.in_legacy {
            let pc = native.get_reg("PC");
            let opcode = native.memory.read_byte_for_preflight(pc, None);
            native.step(1).map_err(|e| e.to_string())?;
            let target = native.get_reg("PC");
            let gate = self.hardware.borrow().gate[15];
            if opcode == Some(3) && target < 0x10000 && gate & 2 != 0 {
                self.legacy.pc = target as u16;
                for i in 0..128 {
                    self.legacy.internal[i] =
                        native.memory.read_internal_byte_silent(i as u32).unwrap();
                }
                self.legacy.carry = native.get_reg("F") & 1 != 0;
                self.legacy.zero = native.get_reg("F") & 2 != 0;
                self.in_legacy = true;
                self.transitions.push(json!({"from":"native","to":"legacy","source_pc":pc,"target_pc":target,
                    "native_instructions":native.instruction_count(),"gate":self.hardware.borrow().gate[15]}));
            }
            return Ok(());
        }
        let before = legacy_state(&self.legacy);
        let mut bus = LegacyBus {
            hardware: &self.hardware,
            policy: self.data_policy,
            pc: self.legacy.pc,
        };
        let transfer = compat_probe::step(&mut self.legacy, &mut bus)?;
        self.legacy_boundaries += 1;
        if self.legacy_trace.len() < 512 {
            self.legacy_trace.push(json!({"before":before,"after":legacy_state(&self.legacy),
                "native_target":match transfer {Transfer::Continue=>None,Transfer::Native(t)=>Some(t)}}));
        }
        for i in 0..128 {
            native
                .memory
                .write_internal_byte(i as u32, self.legacy.internal[i]);
        }
        if let Transfer::Native(target) = transfer {
            if self.hardware.borrow().gate[15] & 2 != 0 {
                return Err("legacy far transfer did not clear gate bit1".into());
            }
            native.set_reg(
                "F",
                u32::from(self.legacy.carry) | (u32::from(self.legacy.zero) << 1),
            );
            native.set_reg("PC", target);
            self.in_legacy = false;
            self.transitions.push(
                json!({"from":"legacy","to":"native","source_pc":bus.pc,"target_pc":target,
                "legacy_boundaries":self.legacy_boundaries,"gate":self.hardware.borrow().gate[15]}),
            );
        }
        Ok(())
    }
}
/// Compatibility wrapper retained for source-bound exploration runners.
pub struct CompatibilityMachine {
    pub native: CoreRuntime,
    coordinator: CompatibilityCoordinator,
}
impl std::ops::Deref for CompatibilityMachine {
    type Target = CompatibilityCoordinator;
    fn deref(&self) -> &Self::Target {
        &self.coordinator
    }
}
impl std::ops::DerefMut for CompatibilityMachine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coordinator
    }
}
impl CompatibilityMachine {
    pub fn new(
        mut native: CoreRuntime,
        hardware: SharedHardware,
        policy: LegacyDataPolicy,
    ) -> Self {
        let coordinator = CompatibilityCoordinator::new(&mut native, hardware, policy);
        Self {
            native,
            coordinator,
        }
    }
    pub fn step(&mut self) -> Result<(), String> {
        self.coordinator.step(&mut self.native)
    }
}

impl CoreRuntime {
    /// Explicit bootstrap candidate; never inferred from the PA model name.
    pub fn install_pa9500_compatibility(
        &mut self,
        hardware: SharedHardware,
        policy: LegacyDataPolicy,
    ) -> Result<(), String> {
        if self.device_model() != crate::DeviceModel::Pa9500
            || self.elapsed_timing_units() != 0
            || self.pa9500_compatibility.is_some()
        {
            return Err("PA compatibility requires a fresh PA runtime, once".into());
        }
        self.pa9500_compatibility = Some(CompatibilityCoordinator::new(self, hardware, policy));
        Ok(())
    }
    pub fn pa9500_compatibility(&self) -> Option<&CompatibilityCoordinator> {
        self.pa9500_compatibility.as_ref()
    }
}

pub fn legacy_state(s: &StartupState) -> Value {
    json!({"pc":s.pc,"dp":s.dp,"p":s.p,"q":s.q,"r":s.r,"table_count":s.table_count,
        "internal":s.internal.to_vec(),"carry":s.carry,"zero":s.zero})
}
pub struct LegacyBus<'a> {
    pub hardware: &'a SharedHardware,
    pub policy: LegacyDataPolicy,
    pub pc: u16,
}
impl StartupBus for LegacyBus<'_> {
    fn fetch(&self, address: u16) -> Result<u8, String> {
        let hw = self.hardware.borrow();
        let (selector, offset) = match address {
            0..=0x1fff => (0, 0x10000 + address as usize),
            0x2000..=0x3fff => (0, address as usize),
            0xc000..=0xffff if hw.gate[6] <= 0x37 => {
                let page = hw.gate[6];
                (
                    page >> 3,
                    (page as usize & 7) * 0x4000 + (address as usize & 0x3fff),
                )
            }
            _ => {
                return Err(format!(
                    "unqualified legacy fetch {address:04X}; mapping={:?}",
                    &hw.gate[..16]
                ))
            }
        };
        hw.banks
            .get(&selector)
            .map(|b| b[offset])
            .ok_or_else(|| format!("legacy ROM page selector {selector:02X} missing"))
    }
    fn read(&self, address: u16) -> Result<u8, String> {
        if address >= 0xc000 && self.policy == LegacyDataPolicy::AbsentMediaAndPagedRom {
            return self.fetch(address);
        }
        let hw = self.hardware.borrow();
        match address {
            // Native and legacy graphics helpers independently refer to
            // 80E20 and 4E20, respectively. The broader 16 KiB alias is a
            // candidate contract; physical page selection remains unqualified.
            0x4000..=0x7fff => Ok(hw.ram[(address - 0x4000) as usize]),
            0x8000..=0xbfff
                if hw.gate[4] == 1
                    && matches!(
                        self.policy,
                        LegacyDataPolicy::AbsentLegacyMedia
                            | LegacyDataPolicy::AbsentMediaAndPagedRom
                    ) =>
            {
                Ok(0xff)
            }
            _ => Err(format!(
                "unqualified legacy data {address:04X}; port04={:02X} port05={:02X}",
                hw.gate[4], hw.gate[5]
            )),
        }
    }
    fn write(&mut self, address: u16, value: u8) -> Result<(), String> {
        let mut hw = self.hardware.borrow();
        if !(0x4000..=0x7fff).contains(&address) {
            return Err(format!("unqualified legacy data write {address:04X}"));
        }
        hw.ram[(address - 0x4000) as usize] = value;
        hw.events.push(json!({"kind":"legacy_data_write","address":address,"native_alias":0x80000+u32::from(address-0x4000),"value":value,"pc":self.pc}));
        Ok(())
    }
    fn fill(&mut self, start: u16, count: u16, value: u8) -> Result<(), String> {
        if count == 0 || (0..count).any(|i| !(0x4000..=0x7fff).contains(&start.wrapping_add(i))) {
            return Err(format!("unqualified legacy fill {start:04X} count={count}"));
        }
        let mut hw = self.hardware.borrow();
        for i in 0..count {
            hw.ram[(start.wrapping_add(i) - 0x4000) as usize] = value;
        }
        hw.events.push(
            json!({"kind":"legacy_fill","start":start,"count":count,"value":value,"pc":self.pc}),
        );
        Ok(())
    }
    fn writ(&mut self, dp: u16, p: u8, value: u8) -> Result<(), String> {
        let register = dp as u8;
        let allowed = match register {
            0x0f => p == 2 && matches!(value, 0 | 2 | 3),
            0x04 | 0x05 => matches!(p, 2 | 0x0c | 0x0d) && value <= 3,
            0x06 => p == 2 && value <= 0x37,
            0x00 => p == 2 && value <= 7,
            0x07 => p == 2 && value == 1,
            _ => false,
        };
        if !allowed {
            return Err(format!(
                "unqualified legacy WRIT dp={dp:04X} p={p:02X} value={value:02X}"
            ));
        }
        let mut hw = self.hardware.borrow();
        hw.gate[register as usize] = value;
        hw.events.push(json!({"kind":"legacy_WRIT","dp":dp,"register":register,"p":p,"value":value,"pc":self.pc}));
        Ok(())
    }
    fn outf(&mut self, value: u8) -> Result<(), String> {
        if value != 0 {
            return Err(format!("unqualified legacy OUTF {value:02X}"));
        }
        self.hardware
            .borrow()
            .events
            .push(json!({"kind":"legacy_OUTF","value":value,"pc":self.pc}));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pa9500::{BootstrapProfile, Hardware};
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    fn switch_fixture() -> (CoreRuntime, SharedHardware) {
        let mut fixed = vec![0; 0x20000];
        fixed[0x1fffd..].copy_from_slice(&[0, 0, 15]);
        fixed[0x10000..0x10004].copy_from_slice(&[3, 0, 0, 0]); // native JPF 00000
        fixed[0x10010..0x10012].copy_from_slice(&[8, 0x5a]); // native MV A,5A
        let mut bank = vec![0; 0x20000];
        // H: LIA 0; LP 2; LIDP 400F; WRIT; far native transfer F0010.
        bank[0x10000..0x1000b]
            .copy_from_slice(&[2, 0, 0x82, 0x10, 0x40, 0x0f, 0xd3, 0x7b, 0x0f, 0, 0x10]);
        let (mut rt, hw) = super::super::configure(
            &fixed,
            BTreeMap::from([(7, bank.clone()), (0, bank)]),
            BootstrapProfile::TrapPeripherals,
        )
        .unwrap();
        hw.borrow().gate[15] = 2;
        rt.memory.write_internal_byte(0x40, 0xa5);
        rt.set_reg("F", 3);
        (rt, hw)
    }

    #[test]
    fn shared_scheduler_and_wrapper_make_identical_isa_transfers() {
        let (native, hw) = switch_fixture();
        let mut wrapper = CompatibilityMachine::new(native, hw, LegacyDataPolicy::TrapUnknown);
        for _ in 0..7 {
            wrapper.step().unwrap();
        }
        for sliced in [false, true] {
            let (mut rt, hw) = switch_fixture();
            rt.install_pa9500_compatibility(hw.clone(), LegacyDataPolicy::TrapUnknown)
                .unwrap();
            assert!(rt
                .install_pa9500_compatibility(hw.clone(), LegacyDataPolicy::TrapUnknown)
                .is_err());
            if sliced {
                rt.run_slice(7, |_| false).unwrap();
            } else {
                rt.step(7).unwrap();
            }
            let c = rt.pa9500_compatibility().unwrap();
            assert!(!c.in_legacy);
            assert_eq!(c.legacy_boundaries, 5);
            assert_eq!(c.transitions, wrapper.transitions);
            assert_eq!(c.legacy_trace, wrapper.legacy_trace);
            assert_eq!(rt.get_reg("PC"), 0xf0012);
            assert_eq!(rt.get_reg("A"), 0x5a);
            assert_eq!(rt.get_reg("F"), 3);
            assert_eq!(rt.instruction_count(), 2);
            assert_eq!(rt.memory.read_internal_byte_silent(0x40), Some(0xa5));
            assert_eq!(hw.borrow().gate[15], 0);
            assert_eq!(hw.borrow().ram, wrapper.hardware.borrow().ram);
            let status = legacy_state(&c.legacy);
            assert!(rt.power_on_reset().is_err());
            assert_eq!(
                legacy_state(&rt.pa9500_compatibility().unwrap().legacy),
                status
            );
            assert_eq!(rt.get_reg("PC"), 0xf0012);
        }
        assert!(CoreRuntime::new()
            .install_pa9500_compatibility(wrapper.hardware.clone(), LegacyDataPolicy::TrapUnknown)
            .is_err());
    }

    #[test]
    fn unsupported_legacy_boundary_keeps_coordinator_and_state() {
        let (mut rt, hw) = switch_fixture();
        hw.borrow().banks.get_mut(&0).unwrap()[0x10000] = 0x14;
        rt.install_pa9500_compatibility(hw.clone(), LegacyDataPolicy::TrapUnknown)
            .unwrap();
        rt.step(1).unwrap();
        let before = legacy_state(&rt.pa9500_compatibility().unwrap().legacy);
        assert!(rt.step(1).is_err());
        let c = rt.pa9500_compatibility().unwrap();
        assert!(c.in_legacy);
        assert_eq!(c.legacy_boundaries, 0);
        assert_eq!(legacy_state(&c.legacy), before);
        assert_eq!(c.transitions.len(), 1);
        assert_eq!(hw.borrow().gate[15], 2);
    }

    #[test]
    fn split_fetch_alias_and_page_selection_reject_unknown_space() {
        let mut banks = BTreeMap::from([(0, vec![0; 0x20000]), (6, vec![0; 0x20000])]);
        banks.get_mut(&0).unwrap()[0x10000] = 0x4d;
        banks.get_mut(&0).unwrap()[0x2000] = 0x79;
        banks.get_mut(&0).unwrap()[0xc000] = 0x7a;
        banks.get_mut(&6).unwrap()[0x1c000] = 0x10;
        let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(
            banks,
            BootstrapProfile::TrapPeripherals,
        ))));
        hardware.borrow().gate[6] = 0x37;
        let mut bus = LegacyBus {
            hardware: &hardware,
            policy: LegacyDataPolicy::TrapUnknown,
            pc: 0,
        };
        assert_eq!(
            [
                bus.fetch(0).unwrap(),
                bus.fetch(0x2000).unwrap(),
                bus.fetch(0xc000).unwrap()
            ],
            [0x4d, 0x79, 0x10]
        );
        assert!(bus.read(0).is_err());
        bus.write(0x4e20, 0xa5).unwrap();
        assert_eq!(hardware.borrow().ram[0xe20], 0xa5);
        bus.writ(0x4006, 2, 3).unwrap();
        assert_eq!(bus.fetch(0xc000).unwrap(), 0x7a);
        assert!(bus.fetch(0x4000).is_err());
        assert!(bus.fetch(0x8000).is_err());
    }

    #[test]
    fn absent_media_requires_declared_policy_and_port04_one() {
        let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(
            BTreeMap::new(),
            BootstrapProfile::TrapPeripherals,
        ))));
        let mut bus = LegacyBus {
            hardware: &hardware,
            policy: LegacyDataPolicy::AbsentLegacyMedia,
            pc: 0,
        };
        assert!(bus.read(0x8000).is_err());
        hardware.borrow().gate[4] = 1;
        assert_eq!(bus.read(0x8000).unwrap(), 255);
        assert_eq!(bus.read(0xbfff).unwrap(), 255);
        assert!(bus.read(0xc000).is_err());
        let before = hardware.borrow().gate;
        assert!(bus.writ(0x4006, 2, 0x38).is_err());
        assert_eq!(hardware.borrow().gate, before);
        bus.policy = LegacyDataPolicy::TrapUnknown;
        assert!(bus.read(0x8000).is_err());
    }

    #[test]
    fn rom_fill_loop_exact_span_and_atomic_rejection() {
        let mut bank = vec![0; 0x20000];
        bank[0x10000..0x10010].copy_from_slice(&[
            0x10, 0x40, 0, 0, 0xff, 2, 0, 3, 0x10, 0x1f, 0xc3, 0x29, 3, 0, 0x0f, 0x1f,
        ]);
        let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(
            BTreeMap::from([(0, bank)]),
            BootstrapProfile::TrapPeripherals,
        ))));
        hardware.borrow().ram.fill(0xa5);
        let mut bus = LegacyBus {
            hardware: &hardware,
            policy: LegacyDataPolicy::TrapUnknown,
            pc: 0,
        };
        let mut state = StartupState::new(0);
        for _ in 0..54 {
            compat_probe::step(&mut state, &mut bus).unwrap();
        }
        assert_eq!(
            (state.pc, state.dp, state.q, state.p, state.internal[3]),
            (16, 0x4fff, 3, 0, 0)
        );
        assert!(hardware.borrow().ram[..0x1000].iter().all(|b| *b == 0));
        assert!(hardware.borrow().ram[0x1000..].iter().all(|b| *b == 0xa5));
        state.pc = 9;
        state.dp = 0x7fff;
        state.internal[0] = 1;
        let before = state.clone();
        let ram = hardware.borrow().ram.clone();
        let events = hardware.borrow().events.len();
        assert!(compat_probe::step(&mut state, &mut bus).is_err());
        assert_eq!(state, before);
        assert_eq!(hardware.borrow().ram, ram);
        assert_eq!(hardware.borrow().events.len(), events);
        state.internal[0] = 0;
        state.carry = true;
        state.zero = false;
        compat_probe::step(&mut state, &mut bus).unwrap();
        assert_eq!(state.dp, 0x7fff);
        assert!(state.carry && !state.zero);
        state.pc = 9;
        state.dp = 0xffff;
        state.internal[0] = 1;
        let before = state.clone();
        assert!(compat_probe::step(&mut state, &mut bus).is_err());
        assert_eq!(state, before);
    }
    #[test]
    fn indexed_load_store_use_little_endian_pairs_and_failed_load_keeps_state() {
        let mut bank = vec![0; 0x20000];
        bank[0x10000..0x10002].copy_from_slice(&[0x24, 0x27]);
        let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(
            BTreeMap::from([(0, bank)]),
            BootstrapProfile::TrapPeripherals,
        ))));
        hardware.borrow().ram[0] = 0x6d;
        let mut bus = LegacyBus {
            hardware: &hardware,
            policy: LegacyDataPolicy::TrapUnknown,
            pc: 0,
        };
        let mut state = StartupState::new(0);
        state.internal[4..6].copy_from_slice(&[0xff, 0x3f]);
        state.carry = true;
        state.zero = false;
        compat_probe::step(&mut state, &mut bus).unwrap();
        assert_eq!(
            (
                state.internal[2],
                state.dp,
                state.q,
                state.carry,
                state.zero
            ),
            (0x6d, 0x4000, 5, true, false)
        );
        assert_eq!(&state.internal[4..6], &[0, 0x40]);
        state.internal[6..8].copy_from_slice(&[0, 0x80]);
        compat_probe::step(&mut state, &mut bus).unwrap();
        assert_eq!(
            (state.dp, state.q, hardware.borrow().ram[0x3fff]),
            (0x7fff, 7, 0x6d)
        );
        state.pc = 0;
        state.internal[4..6].copy_from_slice(&[255, 255]);
        let before = state.clone();
        assert!(compat_probe::step(&mut state, &mut bus).is_err());
        assert_eq!(state, before);
    }
    #[test]
    fn code_window_as_data_requires_explicit_policy() {
        let mut bank = vec![0; 0x20000];
        bank[0xc3cd] = 0x61;
        let hardware = SharedHardware(Arc::new(Mutex::new(Hardware::new(
            BTreeMap::from([(0, bank)]),
            BootstrapProfile::TrapPeripherals,
        ))));
        hardware.borrow().gate[6] = 3;
        let mut bus = LegacyBus {
            hardware: &hardware,
            policy: LegacyDataPolicy::AbsentLegacyMedia,
            pc: 0,
        };
        assert!(bus.read(0xc3cd).is_err());
        bus.policy = LegacyDataPolicy::AbsentMediaAndPagedRom;
        assert_eq!(bus.read(0xc3cd).unwrap(), 0x61);
        hardware.borrow().gate[6] = 0x38;
        assert!(bus.read(0xc3cd).is_err());
    }
}
