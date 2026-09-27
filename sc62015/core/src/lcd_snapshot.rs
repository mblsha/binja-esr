// PY_SOURCE: pce500/display/controller_wrapper.py:HD61202Controller
//! Typed LCD metadata and legacy JSON boundary conversion.
use crate::lcd::LcdKind;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChipSnapshot {
    pub on: bool,
    pub start_line: u8,
    pub page: u8,
    pub y_address: u8,
    pub instruction_count: u32,
    pub data_write_count: u32,
    pub data_read_count: u32,
    #[serde(default)]
    pub on_off_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum LcdSnapshotMetadata {
    #[serde(rename = "hd61202")]
    Hd61202 {
        chip_count: usize,
        pages: usize,
        width: usize,
        chips: [ChipSnapshot; 2],
        cs_both_count: u32,
        cs_left_count: u32,
        cs_right_count: u32,
    },
    #[serde(rename = "iq7000-vram")]
    Iq7000Vram {
        cols: usize,
        pages_per_buffer: usize,
        buffers: usize,
    },
    #[serde(rename = "unknown")]
    Unknown,
}

impl LcdSnapshotMetadata {
    pub fn kind(&self) -> LcdKind {
        match self {
            Self::Hd61202 { .. } => LcdKind::Hd61202,
            Self::Iq7000Vram { .. } => LcdKind::Iq7000Vram,
            Self::Unknown => LcdKind::Unknown,
        }
    }

    #[cfg(any(test, feature = "json-compat"))]
    pub fn from_legacy(value: &serde_json::Value, default: LcdKind) -> Result<Self, String> {
        let mut value = value.clone();
        let object = value
            .as_object_mut()
            .ok_or("LCD metadata must be an object")?;
        object
            .entry("kind")
            .or_insert_with(|| serde_json::json!(default));
        serde_json::from_value(value).map_err(|e| format!("LCD metadata: {e}"))
    }

    #[cfg(any(test, feature = "json-compat"))]
    pub fn to_legacy(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("LCD metadata contains only scalars and arrays")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcd::create_lcd;

    #[test]
    fn typed_and_legacy_roundtrip_all_controllers_atomically() {
        for kind in [LcdKind::Hd61202, LcdKind::Iq7000Vram, LcdKind::Unknown] {
            let mut original = create_lcd(kind);
            for address in 0x4000..0x4100 {
                original.write(address, address as u8);
            }
            let (metadata, payload) = original.snapshot_state();
            let mut restored = create_lcd(kind);
            restored.restore_state(&metadata, &payload).unwrap();
            assert_eq!(
                restored.snapshot_state(),
                (metadata.clone(), payload.clone())
            );
            restored
                .load_snapshot(&metadata.to_legacy(), &payload)
                .unwrap();
            assert_eq!(
                restored.snapshot_state(),
                (metadata.clone(), payload.clone())
            );
            let mut invalid = payload.clone();
            invalid.push(1);
            assert!(restored.restore_state(&metadata, &invalid).is_err());
            assert_eq!(restored.snapshot_state(), (metadata, payload));
        }
    }

    #[test]
    fn hd_selector_rejection_does_not_mutate_any_chip() {
        let mut lcd = create_lcd(LcdKind::Hd61202);
        let before = lcd.snapshot_state();
        let mut invalid = before.0.clone();
        if let LcdSnapshotMetadata::Hd61202 { chips, .. } = &mut invalid {
            chips[0].on = true;
            chips[1].page = 8;
        }
        assert!(lcd.restore_state(&invalid, &before.1).is_err());
        assert_eq!(lcd.snapshot_state(), before);
    }

    #[test]
    fn hd_legacy_metadata_preserves_on_off_count_and_accepts_old_snapshots() {
        let (metadata, _) = create_lcd(LcdKind::Hd61202).snapshot_state();
        let mut legacy = metadata.to_legacy();
        legacy["chips"][0]["on_off_count"] = serde_json::json!(3);
        let parsed = LcdSnapshotMetadata::from_legacy(&legacy, LcdKind::Hd61202).unwrap();
        let LcdSnapshotMetadata::Hd61202 { chips, .. } = parsed else {
            panic!("expected HD61202 metadata");
        };
        assert_eq!(chips[0].on_off_count, 3);

        legacy["chips"][0]
            .as_object_mut()
            .unwrap()
            .remove("on_off_count");
        let parsed = LcdSnapshotMetadata::from_legacy(&legacy, LcdKind::Hd61202).unwrap();
        let LcdSnapshotMetadata::Hd61202 { chips, .. } = parsed else {
            panic!("expected HD61202 metadata");
        };
        assert_eq!(chips[0].on_off_count, 0);

        legacy["chips"][0]["on_off_count"] = serde_json::json!(-1);
        assert!(LcdSnapshotMetadata::from_legacy(&legacy, LcdKind::Hd61202).is_err());
    }
}
