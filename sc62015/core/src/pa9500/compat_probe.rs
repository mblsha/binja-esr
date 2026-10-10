// PY_SOURCE: pce500/pa9500/compat_probe.py
//! Strict component executor for the ROM's ESR-H startup signature probe.
//! This is not a complete ESR-H CPU or an automatic machine ISA dispatcher.
//! WRIT is an explicit scalar-output hypothesis supplied by the caller; its
//! board decode and cycle timing are unqualified. Unsupported operations trap.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartupState {
    pub pc: u16,
    pub dp: u16,
    pub p: u8,
    pub internal: [u8; 128],
    pub carry: bool,
    pub zero: bool,
}
impl StartupState {
    pub fn new(pc: u16) -> Self {
        Self {
            pc,
            dp: 0,
            p: 0,
            internal: [0; 128],
            carry: false,
            zero: false,
        }
    }
}

/// Fetch, ordinary data reads and WRIT remain distinct address spaces.
/// A device must reject unqualified accesses rather than substitute a byte.
pub trait StartupBus {
    fn fetch(&self, address: u16) -> Result<u8, String>;
    fn read(&self, address: u16) -> Result<u8, String>;
    fn writ(&mut self, dp: u16, p: u8, value: u8) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Continue,
    /// ROM-evidenced 7B + big-endian 20-bit operand. Silicon mode effects are
    /// deliberately outside this executor and need a qualified device contract.
    Native(u32),
}

/// Predecode on a candidate state: bad fetches/opcodes leave CPU state intact.
/// No timing estimates are mixed into the shared native CPU's cycle count.
pub fn step(state: &mut StartupState, bus: &mut impl StartupBus) -> Result<Transfer, String> {
    let mut next = state.clone();
    let byte = |cpu: &mut StartupState| -> Result<u8, String> {
        let value = bus.fetch(cpu.pc)?;
        cpu.pc = cpu.pc.wrapping_add(1);
        Ok(value)
    };
    let opcode = byte(&mut next)?;
    let mut output = None;
    let transfer = match opcode {
        0x00..=0x03 => {
            next.internal[opcode as usize] = byte(&mut next)?;
            Transfer::Continue
        }
        0x10 => {
            let high = byte(&mut next)?;
            next.dp = ((high as u16) << 8) | byte(&mut next)? as u16;
            Transfer::Continue
        }
        0x4d => Transfer::Continue,
        0x57 => {
            next.internal[2] = bus.read(next.dp)?;
            Transfer::Continue
        }
        0x64 => {
            next.internal[2] &= byte(&mut next)?;
            next.zero = next.internal[2] == 0;
            Transfer::Continue
        }
        0x67 => {
            let value = byte(&mut next)?;
            next.carry = next.internal[2] < value;
            next.zero = next.internal[2] == value;
            Transfer::Continue
        }
        0x7c => {
            let high = byte(&mut next)?;
            let target = ((high as u16) << 8) | byte(&mut next)? as u16;
            if !next.zero {
                next.pc = target;
            }
            Transfer::Continue
        }
        0x80..=0xbf => {
            next.p = opcode & 0x3f;
            Transfer::Continue
        }
        0xd3 => {
            output = Some((next.dp, next.p, next.internal[next.p as usize]));
            Transfer::Continue
        }
        0x7b => {
            let high = byte(&mut next)?;
            let middle = byte(&mut next)?;
            let low = byte(&mut next)?;
            let target = ((high as u32) << 16) | ((middle as u32) << 8) | low as u32;
            if high > 15 || target < 0x10000 {
                return Err(format!(
                    "invalid PA compatibility native transfer at {:04X}",
                    state.pc
                ));
            }
            Transfer::Native(target)
        }
        _ => {
            return Err(format!(
                "unqualified ESR-H startup opcode {opcode:02X} at {:04X}",
                state.pc
            ))
        }
    };
    if let Some((dp, p, value)) = output {
        bus.writ(dp, p, value)?;
    }
    *state = next;
    Ok(transfer)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Bus {
        bytes: Vec<u8>,
    }
    impl StartupBus for Bus {
        fn fetch(&self, a: u16) -> Result<u8, String> {
            self.bytes
                .get(a as usize)
                .copied()
                .ok_or("unmapped fetch".into())
        }
        fn read(&self, _: u16) -> Result<u8, String> {
            Err("unmapped data".into())
        }
        fn writ(&mut self, _: u16, _: u8, _: u8) -> Result<(), String> {
            Err("unqualified WRIT".into())
        }
    }
    #[test]
    fn failure_does_not_advance_or_repair_guest_state() {
        for bytes in [
            vec![0x10, 0x40],
            vec![0x56],
            vec![0xd3],
            vec![0x57],
            vec![0x7b, 0x10, 0, 0],
        ] {
            let mut state = StartupState::new(0);
            state.internal[0] = 0xa5;
            let original = state.clone();
            assert!(step(&mut state, &mut Bus { bytes }).is_err());
            assert_eq!(state, original);
        }
    }
    #[test]
    fn compare_borrow_survives_logical_mask_and_jump_uses_big_endian_target() {
        let mut bus = Bus {
            bytes: vec![2, 0x80, 0x67, 0xa5, 0x64, 0x0f, 0x7c, 0x12, 0x34],
        };
        let mut state = StartupState::new(0);
        step(&mut state, &mut bus).unwrap();
        step(&mut state, &mut bus).unwrap();
        assert!(state.carry && !state.zero);
        step(&mut state, &mut bus).unwrap();
        assert!(state.carry && state.zero);
        step(&mut state, &mut bus).unwrap();
        assert_eq!(state.pc, 9); // zero means JPNZ falls through
        state.pc = 6;
        state.zero = false;
        step(&mut state, &mut bus).unwrap();
        assert_eq!(state.pc, 0x1234);
    }
    #[test]
    fn compatibility_far_transfer_keeps_all_twenty_bits() {
        let mut state = StartupState::new(0);
        let mut bus = Bus {
            bytes: vec![0x7b, 0x0f, 0x3f, 0xc1],
        };
        assert_eq!(
            step(&mut state, &mut bus).unwrap(),
            Transfer::Native(0xf3fc1)
        );
        assert_eq!(state.pc, 4);
    }
}
