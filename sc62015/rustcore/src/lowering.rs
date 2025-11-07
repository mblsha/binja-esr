use crate::state::{Flag, Registers};

/// Flags produced by helper operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpFlags {
    pub carry: Option<u8>,
    pub zero: Option<u8>,
}

impl OpFlags {
    pub fn none() -> Self {
        Self {
            carry: None,
            zero: None,
        }
    }

    pub fn with(carry: Option<u8>, zero: Option<u8>) -> Self {
        Self { carry, zero }
    }

    /// Apply the flag updates to the provided register file.
    pub fn apply(self, registers: &mut Registers) {
        if let Some(value) = self.carry {
            registers.set_flag(Flag::Carry, value);
        }
        if let Some(value) = self.zero {
            registers.set_flag(Flag::Zero, value);
        }
    }
}

fn mask_for_width(width: u8) -> i128 {
    if width == 0 {
        return i128::MAX;
    }
    let bits = (width as i32) * 8;
    if bits >= 127 {
        i128::MAX
    } else {
        (1_i128 << bits) - 1
    }
}

fn normalize_width(width: u8) -> u8 {
    match width {
        0 => 8, // treat unspecified widths as 64-bit scratch
        other => other,
    }
}

pub fn add(width: u8, lhs: i64, rhs: i64) -> (i64, OpFlags) {
    let width = normalize_width(width);
    let mask = mask_for_width(width);
    let full = lhs as i128 + rhs as i128;
    let value = (full & mask) as i64;
    let carry = (full > mask) as u8;
    let zero = (value == 0) as u8;
    (value, OpFlags::with(Some(carry), Some(zero)))
}

pub fn sub(width: u8, lhs: i64, rhs: i64) -> (i64, OpFlags) {
    let width = normalize_width(width);
    let mask = mask_for_width(width);
    let full = lhs as i128 - rhs as i128;
    let value = (full & mask) as i64;
    let carry = (full < 0) as u8;
    let zero = (value == 0) as u8;
    (value, OpFlags::with(Some(carry), Some(zero)))
}

pub fn logical<F>(width: u8, lhs: i64, rhs: i64, op: F) -> (i64, OpFlags)
where
    F: FnOnce(i64, i64) -> i64,
{
    let width = normalize_width(width);
    let mask = mask_for_width(width);
    let value = (op(lhs, rhs) as i128 & mask) as i64;
    let zero = (value == 0) as u8;
    (value, OpFlags::with(Some(0), Some(zero)))
}

pub fn and(width: u8, lhs: i64, rhs: i64) -> (i64, OpFlags) {
    logical(width, lhs, rhs, |a, b| a & b)
}

pub fn or(width: u8, lhs: i64, rhs: i64) -> (i64, OpFlags) {
    logical(width, lhs, rhs, |a, b| a | b)
}

pub fn xor(width: u8, lhs: i64, rhs: i64) -> (i64, OpFlags) {
    logical(width, lhs, rhs, |a, b| a ^ b)
}

pub fn cmp_eq(lhs: i64, rhs: i64) -> i64 {
    (lhs == rhs) as i64
}

pub fn cmp_slt(width: u8, lhs: i64, rhs: i64) -> i64 {
    let width = normalize_width(width);
    let bits = (width as u32) * 8;
    if bits == 0 || bits >= 63 {
        return (lhs < rhs) as i64;
    }
    let sign_bit = 1_i64 << (bits - 1);
    let mask = (1_i64 << bits) - 1;
    let lhs_signed = ((lhs & mask) ^ sign_bit) - sign_bit;
    let rhs_signed = ((rhs & mask) ^ sign_bit) - sign_bit;
    (lhs_signed < rhs_signed) as i64
}

pub fn cmp_ugt(width: u8, lhs: i64, rhs: i64) -> i64 {
    let width = normalize_width(width);
    let mask = mask_for_width(width) as i64;
    (((lhs as i64) & mask) > ((rhs as i64) & mask)) as i64
}

pub fn shift_left(width: u8, value: i64, count: i64) -> (i64, OpFlags) {
    shift_impl(width, value, count, true)
}

pub fn shift_right(width: u8, value: i64, count: i64) -> (i64, OpFlags) {
    shift_impl(width, value, count, false)
}

fn shift_impl(width: u8, value: i64, count: i64, left: bool) -> (i64, OpFlags) {
    let width = normalize_width(width);
    let bits = (width as i64) * 8;
    if bits == 0 {
        return (value, OpFlags::none());
    }
    let mask = mask_for_width(width);
    if count == 0 {
        let val = (value as i128 & mask) as i64;
        return (val, OpFlags::with(Some(0), Some((val == 0) as u8)));
    }
    let mut carry = 0_u8;
    let result = if left {
        if count <= bits {
            carry = ((value >> (bits - count)) & 1) as u8;
        }
        ((value as i128) << count) & mask
    } else {
        if count > 0 && count <= bits {
            carry = ((value >> (count - 1)) & 1) as u8;
        }
        ((value as i128) >> count) & mask
    };
    let result = result as i64;
    (result, OpFlags::with(Some(carry), Some((result == 0) as u8)))
}

pub fn rotate(width: u8, value: i64, count: i64, left: bool) -> (i64, OpFlags) {
    let width = normalize_width(width);
    let bits = (width as i64) * 8;
    if bits == 0 {
        return (value, OpFlags::none());
    }
    let mask = mask_for_width(width);
    let mut count = count % bits;
    if count < 0 {
        count += bits;
    }
    if count == 0 {
        let val = (value as i128 & mask) as i64;
        let carry = if left {
            ((value >> (bits - 1)) & 1) as u8
        } else {
            (value & 1) as u8
        };
        return (val, OpFlags::with(Some(carry), Some((val == 0) as u8)));
    }
    let result = if left {
        (((value as i128) << count) | (((value as i128) & mask) >> (bits - count))) & mask
    } else {
        (((value as i128) >> count) | (((value as i128) & mask) << (bits - count))) & mask
    };
    let carry = if left {
        ((value >> (bits - count)) & 1) as u8
    } else {
        ((value >> (count - 1)) & 1) as u8
    };
    let result = result as i64;
    (result, OpFlags::with(Some(carry), Some((result == 0) as u8)))
}

pub fn rotate_through_carry(
    width: u8,
    value: i64,
    carry_in: i64,
    left: bool,
) -> (i64, OpFlags) {
    let width = normalize_width(width);
    let bits = (width as i64) * 8;
    if bits == 0 {
        return (value, OpFlags::none());
    }
    let mask = mask_for_width(width);
    let carry_in = carry_in & 1;
    let (result, new_carry) = if left {
        let carry_out = ((value >> (bits - 1)) & 1) as u8;
        let val = (((value as i128) << 1) | carry_in as i128) & mask;
        (val as i64, carry_out)
    } else {
        let carry_out = (value & 1) as u8;
        let val = (((value as i128) >> 1) | ((carry_in as i128) << (bits - 1))) & mask;
        (val as i64, carry_out)
    };
    (result, OpFlags::with(Some(new_carry), Some((result == 0) as u8)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sets_flags() {
        let (value, flags) = add(1, 0xFF, 1);
        assert_eq!(value, 0);
        assert_eq!(flags.carry, Some(1));
        assert_eq!(flags.zero, Some(1));
    }

    #[test]
    fn sub_borrow() {
        let (value, flags) = sub(1, 0, 1);
        assert_eq!(value, 0xFF);
        assert_eq!(flags.carry, Some(1));
    }

    #[test]
    fn rotate_left_wraps() {
        let (value, flags) = rotate(1, 0b1000_0001, 1, true);
        assert_eq!(value & 0xFF, 0b0000_0011);
        assert_eq!(flags.carry, Some(1));
    }
}
