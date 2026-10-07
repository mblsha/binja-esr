// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/oz9600/input.py
//! Shared native/browser keycap -> physical matrix contacts, NOT translated
//! keyboard events. No FIFO writes, guessed IRQs, timing, or machine mutation.
//! Letter case follows the device's CAPS state; SHIFT means its printed legend.

use crate::DeviceModel;
include!(concat!(env!("OUT_DIR"), "/physical_keys.rs"));

pub fn matrix_key(model: DeviceModel, name: &str) -> Option<u8> {
    let keys = match model {
        DeviceModel::Iq7000 => IQ_7000_KEYS,
        DeviceModel::Oz9600 => OZ_9600_KEYS,
        DeviceModel::PcE500 | DeviceModel::PcE500Jp => PC_E500_KEYS,
    };
    keys.binary_search_by_key(&name, |(key, _)| *key)
        .ok()
        .map(|index| keys[index].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tables_match_every_shared_definition() {
        let maps: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u8>> =
            serde_json::from_str(include_str!("../data/physical_keys.json")).unwrap();
        for (name, model, table) in [
            ("pc-e500", DeviceModel::PcE500, PC_E500_KEYS),
            ("iq-7000", DeviceModel::Iq7000, IQ_7000_KEYS),
            ("oz-9600", DeviceModel::Oz9600, OZ_9600_KEYS),
        ] {
            assert_eq!(table.len(), maps[name].len());
            for (key, code) in &maps[name] {
                assert_eq!(matrix_key(model, key), Some(*code));
            }
        }
    }

    #[test]
    fn oz9600_arrows_follow_the_main_board_matrix() {
        // IQ9200 service manual PDF p.40: KO7 carries Left/Down, KO8 Up/Right.
        // The ROM's FBE02 table translates these to 1D/1F/1E/1C respectively.
        for (name, column, row) in [
            ("LEFT", 7, 0),
            ("DOWN", 7, 1),
            ("UP", 8, 0),
            ("RIGHT", 8, 1),
        ] {
            assert_eq!(
                matrix_key(DeviceModel::Oz9600, name),
                Some(column * 8 + row)
            );
        }
    }

    #[test]
    fn oz9600_calculator_contacts_follow_the_main_board_matrix() {
        // IQ9200 service manual PDF p.40 and the OZ-9600 front-panel photograph.
        // 2nd + M+ / M- also execute Move / Expand-Collapse in normal Outline.
        for (name, column, row) in [
            ("%", 1, 0),
            ("+", 2, 0),
            ("-", 3, 0),
            ("=", 4, 0),
            ("MENU", 5, 0),
            ("*", 2, 1),
            ("/", 3, 1),
            ("M+", 4, 1),
            ("M-", 5, 1),
        ] {
            assert_eq!(
                matrix_key(DeviceModel::Oz9600, name),
                Some(column * 8 + row),
                "{name}"
            );
        }
    }

    #[test]
    fn shared_key_maps_have_valid_unique_contacts_and_complete_basic_typing() {
        for model in [
            DeviceModel::PcE500,
            DeviceModel::Iq7000,
            DeviceModel::Oz9600,
        ] {
            let mut letters = std::collections::BTreeSet::new();
            for ch in 'A'..='Z' {
                let code = matrix_key(model, &ch.to_string()).expect("letter");
                assert!(code < 128);
                assert!(letters.insert(code));
            }
            for ch in '0'..='9' {
                let code = matrix_key(model, &ch.to_string()).expect("digit");
                assert!(code < 128);
                assert!(letters.insert(code));
            }
            for name in [
                "SPACE", "SHIFT", "CAPS", "ENTER", "BS", "DEL", "INS", "LEFT", "RIGHT", "UP",
                "DOWN",
            ] {
                assert!(matrix_key(model, name).unwrap() < 128);
            }
            assert_eq!(matrix_key(model, "NO SUCH KEY"), None);
        }
        assert_eq!(matrix_key(DeviceModel::Iq7000, "CAPS"), Some(0x24));
        assert_eq!(matrix_key(DeviceModel::Iq7000, "B"), Some(0x04));
        assert_eq!(matrix_key(DeviceModel::Iq7000, "PF1"), None);
        assert_eq!(matrix_key(DeviceModel::PcE500, "ENTER"), Some(0x27));
        assert_eq!(matrix_key(DeviceModel::Oz9600, "SPACE"), Some(4 * 8 + 6));
        assert_eq!(matrix_key(DeviceModel::Oz9600, "CALENDAR"), None);
    }
}
