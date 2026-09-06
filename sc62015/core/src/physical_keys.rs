// PY_SOURCE: pce500/run_pce500.py
//! Shared native/browser keycap -> physical matrix contacts, NOT translated
//! keyboard events. No FIFO writes, guessed IRQs, timing, or machine mutation.
//! Letter case follows the device's CAPS state; SHIFT means its printed legend.

use crate::DeviceModel;
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub fn matrix_key(model: DeviceModel, name: &str) -> Option<u8> {
    type Maps = BTreeMap<String, BTreeMap<String, u8>>;
    static MAPS: OnceLock<Maps> = OnceLock::new();
    let maps = MAPS.get_or_init(|| {
        serde_json::from_str(include_str!("../data/physical_keys.json"))
            .expect("checked-in physical key map must be valid")
    });
    let model = if model == DeviceModel::Iq7000 {
        "iq-7000"
    } else {
        "pc-e500"
    };
    maps.get(model)?.get(name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_key_maps_have_valid_unique_contacts_and_complete_basic_typing() {
        for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
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
    }
}
