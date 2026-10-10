// PY_SOURCE: pce500/pa9500/compat_probe.py
//! Strict component executor for the ROM's ESR-H startup and gateway paths.
//! This is not a complete ESR-H CPU or an automatic machine ISA dispatcher.
//! WRIT is an explicit scalar-output hypothesis supplied by the caller; its
//! board decode and cycle timing are unqualified. Unsupported operations trap.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartupState {
    pub pc: u16,
    pub dp: u16,
    pub p: u8,
    pub q: u8,
    pub r: u8,
    pub table_count: u8,
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
            q: 0,
            r: 0,
            table_count: 0,
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
    fn write(&mut self, address: u16, value: u8) -> Result<(), String> {
        Err(format!(
            "unqualified legacy data write {address:04X}={value:02X}"
        ))
    }
    fn outf(&mut self, value: u8) -> Result<(), String> {
        Err(format!("unqualified legacy OUTF {value:02X}"))
    }
    /// Validate the complete wrapping destination span before any write.
    /// Implementations must reject unsupported spans without bus side effects.
    fn fill(&mut self, start: u16, count: u16, value: u8) -> Result<(), String> {
        Err(format!(
            "unqualified legacy fill {start:04X} count={count} value={value:02X}"
        ))
    }
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
    let mut data_output = None;
    let mut f_output = None;
    let mut fill_output = None;
    let push = |cpu: &mut StartupState, value: u8| {
        cpu.r = cpu.r.wrapping_sub(1) & 0x7f;
        cpu.internal[cpu.r as usize] = value;
    };
    let pop = |cpu: &mut StartupState| {
        let value = cpu.internal[cpu.r as usize];
        cpu.r = cpu.r.wrapping_add(1) & 0x7f;
        value
    };
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
        0x11 => {
            next.dp = (next.dp & 0xff00) | byte(&mut next)? as u16;
            Transfer::Continue
        }
        0x04..=0x07 | 0x24..=0x27 => {
            let reg = if matches!(opcode, 0x04 | 0x05 | 0x24 | 0x25) {
                4
            } else {
                6
            };
            let old = u16::from_le_bytes([next.internal[reg], next.internal[reg + 1]]);
            let value = if opcode & 1 == 0 {
                old.wrapping_add(1)
            } else {
                old.wrapping_sub(1)
            };
            next.internal[reg..reg + 2].copy_from_slice(&value.to_le_bytes());
            next.dp = value;
            next.q = (reg + 1) as u8;
            if matches!(opcode, 0x24 | 0x25) {
                next.internal[2] = bus.read(value)?;
            }
            if matches!(opcode, 0x26 | 0x27) {
                data_output = Some((value, next.internal[2]));
            }
            Transfer::Continue
        }
        0x12 | 0x13 => {
            let value = byte(&mut next)? & 0x7f;
            if opcode == 0x12 {
                next.p = value;
            } else {
                next.q = value;
            }
            Transfer::Continue
        }
        0x28..=0x2d | 0x38..=0x3b => {
            let displacement = byte(&mut next)? as u16;
            let condition = match opcode {
                0x28 | 0x29 => !next.zero,
                0x2a | 0x2b => !next.carry,
                0x38 | 0x39 => next.zero,
                0x3a | 0x3b => next.carry,
                _ => true,
            };
            if condition {
                let base = state.pc.wrapping_add(1);
                next.pc = if opcode & 1 == 0 {
                    base.wrapping_add(displacement)
                } else {
                    base.wrapping_sub(displacement)
                };
            }
            Transfer::Continue
        }
        0x1f => {
            let count = u16::from(next.internal[0]) + 1;
            fill_output = Some((next.dp, count, next.internal[2]));
            next.dp = next.dp.wrapping_add(count - 1);
            Transfer::Continue
        }
        0x08..=0x0b => {
            let count = usize::from(next.internal[if opcode < 0x0a { 0 } else { 1 }]) + 1;
            if usize::from(next.p) + count > 127 || usize::from(next.q) + count > 127 {
                return Err("unqualified legacy internal block pointer range".into());
            }
            for _ in 0..count {
                let p = next.p as usize;
                let q = next.q as usize;
                if opcode & 1 == 0 {
                    next.internal[p] = next.internal[q];
                } else {
                    next.internal.swap(p, q);
                }
                next.p += 1;
                next.q += 1;
            }
            Transfer::Continue
        }
        0x20..=0x22 => {
            next.internal[2] = match opcode {
                0x20 => next.p,
                0x21 => next.q,
                _ => next.r,
            };
            Transfer::Continue
        }
        0x40..=0x43 | 0x48..=0x4b | 0xc0..=0xc3 | 0xc8..=0xcb => {
            let reg = match opcode & 0xfe {
                0x40 => 0,
                0x42 => 2,
                0x48 => 8,
                0x4a => 10,
                0xc0 => 1,
                0xc2 => 3,
                0xc8 => 9,
                _ => 11,
            };
            let old = next.internal[reg];
            next.internal[reg] = if opcode & 1 == 0 {
                old.wrapping_add(1)
            } else {
                old.wrapping_sub(1)
            };
            next.q = reg as u8;
            next.zero = next.internal[reg] == 0;
            next.carry = if opcode & 1 == 0 {
                old == 255
            } else {
                old == 0
            };
            Transfer::Continue
        }
        0x30..=0x32 => {
            let value = next.internal[2] & 0x7f;
            match opcode {
                0x30 => next.p = value,
                0x31 => next.q = value,
                _ => next.r = value,
            };
            Transfer::Continue
        }
        0x34 => {
            let value = next.internal[2];
            push(&mut next, value);
            Transfer::Continue
        }
        0x37 => {
            let low = pop(&mut next);
            next.pc = ((pop(&mut next) as u16) << 8) | low as u16;
            Transfer::Continue
        }
        0x4d | 0xce => Transfer::Continue,
        0x52 | 0x53 => {
            data_output = Some((
                next.dp,
                next.internal[if opcode == 0x52 { 2 } else { next.p as usize }],
            ));
            Transfer::Continue
        }
        0x59 => {
            next.internal[2] = next.internal[next.p as usize];
            Transfer::Continue
        }
        0x55 => {
            next.internal[next.p as usize] = bus.read(next.dp)?;
            Transfer::Continue
        }
        0x50 | 0x51 => {
            next.p = if opcode == 0x50 {
                next.p.checked_add(1).filter(|p| *p < 128)
            } else {
                next.p.checked_sub(1)
            }
            .ok_or("unqualified legacy internal pointer boundary")?;
            Transfer::Continue
        }
        0x58 => {
            next.internal[2] = next.internal[2].rotate_left(4);
            Transfer::Continue
        }
        0x62 | 0x66 => {
            let mask = byte(&mut next)?;
            next.zero = next.internal[if opcode == 0x62 { next.p as usize } else { 2 }] & mask == 0;
            Transfer::Continue
        }
        0x65 => {
            next.internal[2] |= byte(&mut next)?;
            next.zero = next.internal[2] == 0;
            Transfer::Continue
        }
        0x5b => {
            next.internal[2] = pop(&mut next);
            Transfer::Continue
        }
        0x5f => {
            f_output = Some(next.internal[0x5e]);
            next.q = 0x5e;
            Transfer::Continue
        }
        0x60 | 0x61 => {
            let value = byte(&mut next)?;
            let slot = &mut next.internal[next.p as usize];
            *slot = if opcode == 0x60 {
                *slot & value
            } else {
                *slot | value
            };
            next.zero = *slot == 0;
            Transfer::Continue
        }
        0x63 => {
            let value = byte(&mut next)?;
            next.carry = next.internal[next.p as usize] < value;
            next.zero = next.internal[next.p as usize] == value;
            Transfer::Continue
        }
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
        0x69 => {
            let mut selected = None;
            for _ in 0..next.table_count {
                let test = byte(&mut next)?;
                let high = byte(&mut next)?;
                let target = ((high as u16) << 8) | byte(&mut next)? as u16;
                next.zero = next.internal[2] == test;
                if next.zero {
                    selected = Some(target);
                    break;
                }
            }
            next.pc = if let Some(target) = selected {
                target
            } else {
                let high = byte(&mut next)?;
                ((high as u16) << 8) | byte(&mut next)? as u16
            };
            Transfer::Continue
        }
        0x78 | 0x79 | 0x7c | 0x7d | 0x7e | 0x7f => {
            let high = byte(&mut next)?;
            let target = ((high as u16) << 8) | byte(&mut next)? as u16;
            if opcode == 0x78 {
                let return_pc = next.pc;
                push(&mut next, (return_pc >> 8) as u8);
                push(&mut next, return_pc as u8);
            }
            let condition = match opcode {
                0x7c => !next.zero,
                0x7d => !next.carry,
                0x7e => next.zero,
                0x7f => next.carry,
                _ => true,
            };
            if condition {
                next.pc = target;
            }
            Transfer::Continue
        }
        0x7a => {
            next.table_count = byte(&mut next)?;
            let high = byte(&mut next)?;
            let low = byte(&mut next)?;
            push(&mut next, high);
            push(&mut next, low);
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
        0xd4..=0xd6 => {
            let mask = byte(&mut next)?;
            let old = bus.read(next.dp)?;
            let value = if opcode == 0xd5 {
                old | mask
            } else {
                old & mask
            };
            next.zero = value == 0;
            if opcode != 0xd6 {
                data_output = Some((next.dp, value));
            }
            Transfer::Continue
        }
        0xda => {
            next.internal.swap(2, 3);
            Transfer::Continue
        }
        0xdb => {
            next.internal.swap(2, next.p as usize);
            Transfer::Continue
        }
        0xe0..=0xff => {
            let target = (((opcode & 31) as u16) << 8) | byte(&mut next)? as u16;
            let return_pc = next.pc;
            push(&mut next, (return_pc >> 8) as u8);
            push(&mut next, return_pc as u8);
            next.pc = target;
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
    if let Some((address, value)) = data_output {
        bus.write(address, value)?;
    }
    if let Some(value) = f_output {
        bus.outf(value)?;
    }
    if let Some((start, count, value)) = fill_output {
        bus.fill(start, count, value)?;
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
    #[test]
    fn counted_copy_is_sequential_preserves_flags_and_rejects_unknown_pointers() {
        let mut bus = Bus {
            bytes: vec![0x0a, 0x0b],
        };
        let mut state = StartupState::new(0);
        state.p = 8;
        state.q = 2;
        state.internal[1] = 1;
        state.internal[2] = 0xcc;
        state.internal[3] = 0xc3;
        state.carry = true;
        state.zero = false;
        step(&mut state, &mut bus).unwrap();
        assert_eq!(&state.internal[8..10], &[0xcc, 0xc3]);
        assert_eq!(
            (state.p, state.q, state.carry, state.zero),
            (10, 4, true, false)
        );
        state.pc = 0;
        state.p = 3;
        state.q = 2;
        state.internal[1] = 2;
        state.internal[2..6].copy_from_slice(&[1, 2, 3, 4]);
        step(&mut state, &mut bus).unwrap();
        assert_eq!(&state.internal[2..6], &[1, 1, 1, 1]);
        state.pc = 1;
        state.p = 3;
        state.q = 2;
        state.internal[1] = 2;
        state.internal[2..6].copy_from_slice(&[1, 2, 3, 4]);
        step(&mut state, &mut bus).unwrap();
        assert_eq!(&state.internal[2..6], &[2, 3, 4, 1]);
        state.pc = 0;
        state.p = 127;
        state.q = 2;
        let before = state.clone();
        assert!(step(&mut state, &mut bus).is_err());
        assert_eq!(state, before);
    }
    #[test]
    fn register_increment_decrement_set_q_and_zero_borrow_without_changing_p() {
        for (opcode, reg) in [
            (0x40, 0),
            (0x42, 2),
            (0x48, 8),
            (0x4a, 10),
            (0xc0, 1),
            (0xc2, 3),
            (0xc8, 9),
            (0xca, 11),
        ] {
            for (op, old, result, zero, carry) in [
                (opcode, 255, 0, true, true),
                (opcode, 0, 1, false, false),
                (opcode + 1, 0, 255, false, true),
                (opcode + 1, 1, 0, true, false),
            ] {
                let mut s = StartupState::new(0);
                s.p = 39;
                s.internal[reg] = old;
                step(&mut s, &mut Bus { bytes: vec![op] }).unwrap();
                assert_eq!(
                    (s.internal[reg], s.q, s.p, s.zero, s.carry),
                    (result, reg as u8, 39, zero, carry)
                );
            }
        }
    }
    #[test]
    fn call_return_and_counted_dispatch_keep_stack_order() {
        let mut bus = Bus {
            bytes: vec![0x78, 0, 3, 0x37],
        };
        let mut state = StartupState::new(0);
        state.r = 0x5c;
        step(&mut state, &mut bus).unwrap();
        assert_eq!((state.pc, state.r), (3, 0x5a));
        assert_eq!(&state.internal[0x5a..0x5c], &[3, 0]);
        step(&mut state, &mut bus).unwrap();
        assert_eq!((state.pc, state.r), (3, 0x5c));

        // Two cases, a default and a separately encoded return address.
        bus.bytes = vec![
            0x7a, 2, 0x12, 0x34, 0x69, 0x10, 0, 0x20, 0x11, 0, 0x30, 0, 0x40,
        ];
        for (a, target, zero) in [(0x10, 0x20, true), (0x11, 0x30, true), (0x12, 0x40, false)] {
            let mut s = StartupState::new(0);
            s.r = 0x5c;
            s.internal[2] = a;
            s.carry = true;
            step(&mut s, &mut bus).unwrap();
            assert_eq!(&s.internal[0x5a..0x5c], &[0x34, 0x12]);
            step(&mut s, &mut bus).unwrap();
            assert_eq!((s.pc, s.r, s.zero, s.carry), (target, 0x5a, zero, true));
        }
    }
}
