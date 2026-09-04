//! IQ-7000 fixed LCD annunciator state.
// PY_SOURCE: pce500/display/lcd_visualization.py
// Python models the matrix only; IQ-7000 fixed segments are Rust-only.
//!
//! The 96x64 dot-matrix framebuffer does not contain the symbols printed on
//! the right edge of the physical glass.  The ROM keeps four workspace bytes
//! and mirrors them to four off-framebuffer LCD addresses.  `sub_41969` packs
//! exactly the thirteen bits named below into a two-byte display-status value.
//! Five assignments are directly tied to named ROM paths; the remaining
//! battery/card/beep/alarm/arrow names are an evidence-backed, explicitly
//! provisional mapping pending a one-hot real-hardware observation.

use crate::memory::MemoryImage;
use serde::Serialize;

pub const IQ7000_ANNUNCIATOR_STATE_ADDRS: [u32; 4] = [0x01_FDA3, 0x01_FDA4, 0x01_FDA5, 0x01_FDA6];
pub const IQ7000_ANNUNCIATOR_SHADOW_ADDRS: [u32; 4] = [0x00_6160, 0x00_6161, 0x00_61E0, 0x00_61E1];

pub const IQ7000_BATT: u8 = 0x80;
pub const IQ7000_CARD: u8 = 0x40;
pub const IQ7000_EDIT: u8 = 0x20;
pub const IQ7000_SHIFT: u8 = 0x10;
pub const IQ7000_CAPS: u8 = 0x08;
pub const IQ7000_SECRET_DATA: u8 = 0x04;
pub const IQ7000_KEY_BEEP: u8 = 0x02;
pub const IQ7000_MORE_UP: u8 = 0x01;

pub const IQ7000_SECRET_MODE: u8 = 0x04;
pub const IQ7000_ALARM: u8 = 0x02;
pub const IQ7000_MORE_DOWN: u8 = 0x01;

pub const IQ7000_MORE_LEFT: u8 = 0x80;
pub const IQ7000_MORE_RIGHT: u8 = 0x80;

pub const IQ7000_NAMED_MASKS: [u8; 4] = [0xFF, 0x07, 0x80, 0x80];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Iq7000Annunciators {
    /// Compatibility aliases for the first state/shadow byte.
    pub state_raw: u8,
    pub shadow_raw: u8,
    pub raw_union: u8,
    pub unmapped_state: u8,
    pub unmapped_shadow: u8,
    pub unmapped_union: u8,
    pub state_bytes: [u8; 4],
    pub shadow_bytes: [u8; 4],
    pub union_bytes: [u8; 4],
    pub unmapped_state_bytes: [u8; 4],
    pub unmapped_shadow_bytes: [u8; 4],
    pub desynchronized: bool,
    pub mapping_status: &'static str,
    pub batt: bool,
    pub card: bool,
    pub edit: bool,
    pub shift: bool,
    pub caps: bool,
    pub secret_data: bool,
    pub secret_mode: bool,
    pub key_beep: bool,
    pub alarm: bool,
    pub more_up: bool,
    pub more_down: bool,
    pub more_left: bool,
    pub more_right: bool,
}

impl Iq7000Annunciators {
    pub fn from_sources(state_bytes: [u8; 4], shadow_bytes: [u8; 4]) -> Self {
        let union_bytes = std::array::from_fn(|index| state_bytes[index] | shadow_bytes[index]);
        let unmapped_state_bytes =
            std::array::from_fn(|index| state_bytes[index] & !IQ7000_NAMED_MASKS[index]);
        let unmapped_shadow_bytes =
            std::array::from_fn(|index| shadow_bytes[index] & !IQ7000_NAMED_MASKS[index]);

        // The off-framebuffer LCD shadow is the physical display source.  The
        // workspace copy is retained for diagnostics instead of ORing stale
        // state into the rendered result.
        let byte0 = shadow_bytes[0];
        let byte1 = shadow_bytes[1];
        Self {
            state_raw: state_bytes[0],
            shadow_raw: shadow_bytes[0],
            raw_union: union_bytes[0],
            unmapped_state: unmapped_state_bytes[0],
            unmapped_shadow: unmapped_shadow_bytes[0],
            unmapped_union: union_bytes[0] & !IQ7000_NAMED_MASKS[0],
            state_bytes,
            shadow_bytes,
            union_bytes,
            unmapped_state_bytes,
            unmapped_shadow_bytes,
            desynchronized: state_bytes != shadow_bytes,
            mapping_status: "rom-derived-hypothesis-v1",
            batt: byte0 & IQ7000_BATT != 0,
            card: byte0 & IQ7000_CARD != 0,
            edit: byte0 & IQ7000_EDIT != 0,
            shift: byte0 & IQ7000_SHIFT != 0,
            caps: byte0 & IQ7000_CAPS != 0,
            secret_data: byte0 & IQ7000_SECRET_DATA != 0,
            secret_mode: byte1 & IQ7000_SECRET_MODE != 0,
            key_beep: byte0 & IQ7000_KEY_BEEP != 0,
            alarm: byte1 & IQ7000_ALARM != 0,
            more_up: byte0 & IQ7000_MORE_UP != 0,
            more_down: byte1 & IQ7000_MORE_DOWN != 0,
            more_left: shadow_bytes[2] & IQ7000_MORE_LEFT != 0,
            more_right: shadow_bytes[3] & IQ7000_MORE_RIGHT != 0,
        }
    }

    pub fn read(memory: &MemoryImage) -> Self {
        let state_bytes = std::array::from_fn(|index| {
            memory
                .load_silent(IQ7000_ANNUNCIATOR_STATE_ADDRS[index], 8)
                .unwrap_or(0) as u8
        });
        let shadow_bytes = std::array::from_fn(|index| {
            memory
                .load_silent(IQ7000_ANNUNCIATOR_SHADOW_ADDRS[index], 8)
                .unwrap_or(0) as u8
        });
        Self::from_sources(state_bytes, shadow_bytes)
    }

    pub const fn active_flags(&self) -> [bool; 13] {
        [
            self.batt,
            self.card,
            self.edit,
            self.shift,
            self.caps,
            self.secret_data,
            self.secret_mode,
            self.key_beep,
            self.alarm,
            self.more_up,
            self.more_down,
            self.more_left,
            self.more_right,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_thirteen_candidate_segments_from_lcd_shadows() {
        let annunciators = Iq7000Annunciators::from_sources([0; 4], [0xFF, 0x07, 0x80, 0x80]);
        assert!(annunciators.active_flags().into_iter().all(|active| active));
        assert_eq!(annunciators.unmapped_shadow_bytes, [0; 4]);
        assert!(annunciators.desynchronized);
    }

    #[test]
    fn every_bit_lights_only_its_named_segment() {
        let cases = [
            (0, 0x80),
            (0, 0x40),
            (0, 0x20),
            (0, 0x10),
            (0, 0x08),
            (0, 0x04),
            (1, 0x04),
            (0, 0x02),
            (1, 0x02),
            (0, 0x01),
            (1, 0x01),
            (2, 0x80),
            (3, 0x80),
        ];
        for (flag, (byte, bit)) in cases.into_iter().enumerate() {
            let mut shadows = [0; 4];
            shadows[byte] = bit;
            let ann = Iq7000Annunciators::from_sources([0; 4], shadows);
            let mut expected = [false; 13];
            expected[flag] = true;
            assert_eq!(ann.active_flags(), expected);
        }
    }

    #[test]
    fn reading_display_state_is_observational_and_tracks_cleared_bits() {
        let mut memory = MemoryImage::new();
        memory.store(0x6160, 8, 0x10).unwrap();
        memory.store(0x1FDA3, 8, 0x10).unwrap();
        let reads = memory.memory_read_count();
        assert!(Iq7000Annunciators::read(&memory).shift);
        assert_eq!(memory.memory_read_count(), reads);
        memory.store(0x6160, 8, 0).unwrap();
        let state = Iq7000Annunciators::read(&memory);
        assert!(!state.shift); // Do not resurrect the stale workspace bit.
        assert!(state.desynchronized);
        assert_eq!(memory.memory_read_count(), reads);
    }

    #[test]
    fn workspace_state_does_not_light_a_physical_segment_without_shadow_sync() {
        let annunciators = Iq7000Annunciators::from_sources([IQ7000_SHIFT, 0, 0, 0], [0; 4]);
        assert!(!annunciators.shift);
        assert_eq!(annunciators.raw_union, IQ7000_SHIFT);
        assert!(annunciators.desynchronized);
    }

    #[test]
    fn reports_bits_outside_the_thirteen_segment_candidate_set() {
        let annunciators = Iq7000Annunciators::from_sources([0; 4], [0, 0x80, 0x01, 0x01]);
        assert_eq!(annunciators.unmapped_shadow_bytes, [0, 0x80, 0x01, 0x01]);
        assert_eq!(annunciators.active_flags(), [false; 13]);
    }
}
