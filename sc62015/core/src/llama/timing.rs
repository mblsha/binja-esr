// PY_SOURCE: sc62015/pysc62015/instr/opcode_table.py
//! SC62015 instruction timing in the relative timing units used by the manual.
//!
//! A hardware-measured PC-E500 NOP is one unit.  The public instruction table
//! supplies the remaining documented relative counts; IR and RESET retain a
//! one-unit compatibility boundary because their complete timing is not
//! specified.  A fused PRE byte costs one additional unit.

use super::opcodes::{InstrKind, OpcodeEntry};

const FULL_I_COUNT: u64 = 0x1_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingProvenance {
    Documented,
    ProvisionalBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedInstructionTiming {
    resolved_opcode: u8,
    selector: Option<u8>,
    prefix_count: u8,
}

impl PreparedInstructionTiming {
    pub fn prepare<F>(
        first_opcode: u8,
        start_pc: u32,
        mut lookup: impl FnMut(u8) -> Option<&'static OpcodeEntry>,
        mut peek: F,
    ) -> Result<Self, &'static str>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let mut opcode = first_opcode;
        let mut pc = start_pc & 0x000f_ffff;
        let mut prefix_count = 0u8;
        loop {
            let entry = lookup(opcode).ok_or("instruction timing lookup failed")?;
            if entry.kind != InstrKind::Pre {
                break;
            }
            prefix_count = prefix_count.saturating_add(1);
            if prefix_count > 2 {
                return Err("more than two consecutive PRE prefixes are unverified");
            }
            pc = pc.wrapping_add(1) & 0x000f_ffff;
            opcode = peek(pc).ok_or("instruction timing requires stable opcode bytes")?;
        }

        let selector = selector_is_timing_relevant(opcode)
            .then(|| peek(pc.wrapping_add(1) & 0x000f_ffff))
            .flatten();
        if selector_is_timing_relevant(opcode) && selector.is_none() {
            return Err("instruction timing requires a stable selector byte");
        }
        Ok(Self {
            resolved_opcode: opcode,
            selector,
            prefix_count,
        })
    }

    pub fn timing_units(self, initial_i: u16, branch_taken: bool) -> u64 {
        let (base, per_i, provenance) =
            timing_formula(self.resolved_opcode, self.selector, branch_taken);
        let iterations = if initial_i == 0 {
            FULL_I_COUNT
        } else {
            u64::from(initial_i)
        };
        let _ = provenance;
        u64::from(self.prefix_count)
            .saturating_add(base)
            .saturating_add(per_i.saturating_mul(iterations))
    }

    pub fn provenance(self) -> TimingProvenance {
        timing_formula(self.resolved_opcode, self.selector, false).2
    }

    pub fn resolved_opcode(self) -> u8 {
        self.resolved_opcode
    }
}

fn selector_is_timing_relevant(opcode: u8) -> bool {
    matches!(opcode, 0x90..=0x96 | 0x98..=0x9e | 0xb0..=0xb6 | 0xb8..=0xbe | 0xe3 | 0xeb | 0xf0..=0xf3 | 0xf8..=0xfb)
}

fn effective_mode(selector: Option<u8>) -> u8 {
    selector.unwrap_or(0) & 0xf0
}

fn register_read_cycles(opcode: u8, selector: Option<u8>) -> u64 {
    let base = match opcode & 0x07 {
        0 => 4,
        1 => 5,
        2 | 3 => 5,
        4..=6 => 6,
        _ => 1,
    };
    match effective_mode(selector) {
        0x20 if (opcode & 0x0f) >= 4 => base + 1,
        0x30 if (opcode & 0x0f) >= 4 => base + 2,
        0x30 => base + 1,
        0x80 | 0xc0 => base + 2,
        _ => base,
    }
}

fn register_write_cycles(opcode: u8, selector: Option<u8>) -> u64 {
    let base = match opcode & 0x07 {
        0 | 1 => 4,
        2 | 3 => 5,
        4..=6 => 6,
        _ => 1,
    };
    match effective_mode(selector) {
        0x20 if (opcode & 0x0f) >= 4 => base + 1,
        0x30 if (opcode & 0x0f) >= 4 => base + 2,
        0x30 => base + 1,
        0x80 | 0xc0 => base + 2,
        _ => base,
    }
}

fn indirect_imem_cycles(opcode: u8, selector: Option<u8>) -> u64 {
    let base = match opcode & 0x07 {
        0 => 9,
        1 => 10,
        2 | 3 => 10,
        4..=6 => 11,
        _ => 1,
    };
    match effective_mode(selector) {
        0x80 | 0xc0 => base + 2,
        _ => base,
    }
}

fn indirect_imem_write_cycles(opcode: u8, selector: Option<u8>) -> u64 {
    let base = match opcode & 0x07 {
        0 | 1 => 9,
        2 | 3 => 10,
        4..=6 => 11,
        _ => 1,
    };
    match effective_mode(selector) {
        0x80 | 0xc0 => base + 2,
        _ => base,
    }
}

fn offset_imem_cycles(opcode: u8, selector: Option<u8>) -> u64 {
    let base = match opcode & 0x07 {
        0 => 11,
        1 => 12,
        2 => 13,
        3 => 10,
        _ => 1,
    };
    match effective_mode(selector) {
        0x80 | 0xc0 => base + 2,
        _ => base,
    }
}

/// Return `(base, per-I iteration, provenance)`.
fn timing_formula(
    opcode: u8,
    selector: Option<u8>,
    branch_taken: bool,
) -> (u64, u64, TimingProvenance) {
    use TimingProvenance::{Documented as D, ProvisionalBoundary as P};
    match opcode {
        0x00 => (1, 0, D),
        0x01 => (7, 0, D),
        0x02 => (4, 0, D),
        0x03 => (5, 0, D),
        0x04 => (6, 0, D),
        0x05 => (8, 0, D),
        0x06 => (4, 0, D),
        0x07 => (5, 0, D),
        0x08 => (2, 0, D),
        0x09 => (3, 0, D),
        0x0a..=0x0b => (3, 0, D),
        0x0c..=0x0f => (4, 0, D),
        0x10 => (6, 0, D),
        0x11 => (4, 0, D),
        0x12..=0x13 => (3, 0, D),
        0x14..=0x17 => (if branch_taken { 4 } else { 3 }, 0, D),
        0x18..=0x1f => (if branch_taken { 3 } else { 2 }, 0, D),
        0x28..=0x29 | 0x2e..=0x2f => (3, 0, D),
        0x2a..=0x2b => (4, 0, D),
        0x2c..=0x2d => (5, 0, D),
        0x38 => (2, 0, D),
        0x39..=0x3b => (3, 0, D),
        0x3c..=0x3d => (4, 0, D),
        0x3e..=0x3f => (2, 0, D),
        0x40 | 0x48 | 0x50 | 0x58 => (3, 0, D),
        0x41..=0x43 | 0x47 | 0x49..=0x4b | 0x51..=0x53 | 0x57 | 0x59..=0x5b => (4, 0, D),
        0x44 | 0x4c => (5, 0, D),
        0x45 | 0x4d => (7, 0, D),
        0x46 | 0x4e => (3, 0, D),
        0x4f => (3, 0, D),
        0x54 | 0x5c => (5, 2, D),
        0x55 | 0x5d => (4, 1, D),
        0x56 | 0x5e => (5, 2, D),
        0x5f => (2, 0, D),
        0x60 | 0x64 | 0x68 | 0x70 | 0x78 => (3, 0, D),
        0x61 | 0x63 | 0x65 | 0x67 | 0x69 | 0x6b | 0x6f | 0x71 | 0x73 | 0x77 | 0x79 | 0x7b
        | 0x7f => (4, 0, D),
        0x62 | 0x66 => (6, 0, D),
        0x6a | 0x72 | 0x7a => (7, 0, D),
        0x6c..=0x6d | 0x7c..=0x7d => (3, 0, D),
        0x6e | 0x76 | 0x7e => (6, 0, D),
        0x74..=0x75 => (1, 0, D),
        0x80 => (3, 0, D),
        0x81..=0x83 => (4, 0, D),
        0x84..=0x87 => (5, 0, D),
        0x88..=0x89 => (6, 0, D),
        0x8a..=0x8b => (7, 0, D),
        0x8c..=0x8f => (8, 0, D),
        0x90..=0x96 => (register_read_cycles(opcode, selector), 0, D),
        0x97 | 0x9f => (1, 0, D),
        0x98..=0x9e => (indirect_imem_cycles(opcode, selector), 0, D),
        0xa0..=0xa1 => (3, 0, D),
        0xa2..=0xa3 => (4, 0, D),
        0xa4..=0xa7 => (5, 0, D),
        0xa8..=0xa9 => (5, 0, D),
        0xaa..=0xab => (6, 0, D),
        0xac..=0xaf => (7, 0, D),
        0xb0..=0xb6 => (register_write_cycles(opcode, selector), 0, D),
        0xb7 => (6, 0, D),
        0xb8..=0xbe => (indirect_imem_write_cycles(opcode, selector), 0, D),
        0xc0 => (7, 0, D),
        0xc1 => (10, 0, D),
        0xc2 => (13, 0, D),
        0xc3 => (5, 3, D),
        0xc4 => (5, 2, D),
        0xc5 => (4, 1, D),
        0xc6 => (8, 0, D),
        0xc7 => (10, 0, D),
        0xc8 => (6, 0, D),
        0xc9 => (8, 0, D),
        0xca => (10, 0, D),
        0xcb | 0xcf => (5, 2, D),
        0xcc => (3, 0, D),
        0xcd => (4, 0, D),
        0xce => (1, 0, D),
        0xd0 => (7, 0, D),
        0xd1 => (8, 0, D),
        0xd2 => (9, 0, D),
        0xd3 => (6, 2, D),
        // The public table specifies the D0-D3 direction but does not state
        // D8-DB totals.  Keep the symmetric compatibility values visible as
        // provisional rather than silently presenting them as manual facts.
        0xd8 => (7, 0, P),
        0xd9 => (8, 0, P),
        0xda => (9, 0, P),
        0xdb => (6, 2, P),
        0xd4 => (5, 2, D),
        0xd5 => (4, 1, D),
        0xd6 => (7, 0, D),
        0xd7 => (9, 0, D),
        0xdc => (5, 0, D),
        0xdd => (3, 0, D),
        0xde..=0xdf => (1, 0, P),
        0xe0 => (6, 0, D),
        0xe1 => (7, 0, D),
        0xe2 => (8, 0, D),
        0xe3 => (7, 2, D),
        0xe4 | 0xe6 => (2, 0, D),
        0xe5 | 0xe7 => (3, 0, D),
        0xe8 => (6, 0, D),
        0xe9 => (7, 0, D),
        0xea => (8, 0, D),
        0xeb => (
            if effective_mode(selector) == 0x30 {
                7
            } else {
                5
            },
            2,
            D,
        ),
        0xec | 0xfc => (4, 1, D),
        0xed => (4, 0, D),
        0xee => (3, 0, D),
        0xef => (1, 1, D),
        0xf0..=0xf3 | 0xf8..=0xfb => {
            let base = offset_imem_cycles(opcode, selector);
            let per_i = u64::from((opcode & 0x07) == 3) * 2;
            (base, per_i, D)
        }
        0xf4 | 0xf6 => (2, 0, D),
        0xf5 | 0xf7 => (3, 0, D),
        0xfd => (2, 0, D),
        0xfe..=0xff => (1, 0, P),
        _ => (1, 0, P),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::dispatch;

    fn prepared(opcode: u8, following: &[u8]) -> PreparedInstructionTiming {
        PreparedInstructionTiming::prepare(opcode, 0, dispatch::lookup, |addr| {
            following.get(addr.saturating_sub(1) as usize).copied()
        })
        .expect("prepare timing")
    }

    #[test]
    fn hardware_measured_control_flow_slopes_match() {
        for (opcode, taken, expected) in [
            (0x00, false, 1),
            (0x02, true, 4),
            (0x14, true, 4),
            (0x15, false, 3),
            (0x12, true, 3),
            (0x18, true, 3),
            (0x1a, false, 2),
            (0x04, true, 6),
            (0x05, true, 8),
            (0x06, true, 4),
            (0x07, true, 5),
        ] {
            assert_eq!(prepared(opcode, &[]).timing_units(1, taken), expected);
        }
    }

    #[test]
    fn counted_forms_use_sixteen_bit_wrap_count() {
        assert_eq!(prepared(0xef, &[]).timing_units(3, false), 4);
        assert_eq!(prepared(0xef, &[]).timing_units(0, false), 0x1_0001);
        assert_eq!(prepared(0xc3, &[]).timing_units(3, false), 14);
    }

    #[test]
    fn pre_and_selector_modes_are_included() {
        assert_eq!(
            prepared(0x32, &[0xc8, 0x20, 0x21]).timing_units(1, false),
            7
        );
        assert_eq!(prepared(0x90, &[0x00]).timing_units(1, false), 4);
        assert_eq!(prepared(0x94, &[0x20]).timing_units(1, false), 7);
        assert_eq!(prepared(0x90, &[0x80]).timing_units(1, false), 6);
        assert_eq!(prepared(0x99, &[0x00]).timing_units(1, false), 10);
        assert_eq!(prepared(0xb9, &[0x00]).timing_units(1, false), 9);
        assert_eq!(prepared(0xf3, &[0x80]).timing_units(2, false), 16);
    }

    #[test]
    fn ir_and_reset_are_explicitly_provisional() {
        assert_eq!(
            prepared(0xfe, &[]).provenance(),
            TimingProvenance::ProvisionalBoundary
        );
        assert_eq!(
            prepared(0xff, &[]).provenance(),
            TimingProvenance::ProvisionalBoundary
        );
    }
}
