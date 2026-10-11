// PY_SOURCE: pce500/pi3000/bundle.py
//! Bounded, identity-checked PI ROM bundle. It never imports captured live RAM.
use super::{BootstrapProfile, CoreRuntime, BANKS_SHA256, FIXED_SHA256};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
pub const MAGIC: &[u8; 8] = b"PIROM01\0";
pub const PAYLOAD_SIZE: usize = 0x420000;
pub fn from_rom_bundle(data: &[u8], profile: BootstrapProfile) -> Result<CoreRuntime, String> {
    if data.len() < 12 || &data[..8] != MAGIC {
        return Err("requires a PIROM01 bundle".into());
    }
    let length = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    if length == 0 || length > 0x40000 || data.len() != 12 + length + PAYLOAD_SIZE {
        return Err("PI bundle length mismatch".into());
    }
    let manifest: Value =
        serde_json::from_slice(&data[12..12 + length]).map_err(|e| e.to_string())?;
    if manifest["schema"] != "pi3000-rom-bank-set-2" {
        return Err("requires the verified 32-selector PI manifest".into());
    }
    let payload = &data[12 + length..];
    let fixed = &payload[..0x20000];
    let hash = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    if hash(fixed) != FIXED_SHA256
        || manifest["fixed_rom"]["sha256"] != FIXED_SHA256
        || hash(&payload[0x20000..]) != BANKS_SHA256
    {
        return Err("PI fixed/bank acquisition identity mismatch".into());
    }
    let entries = manifest["banks"]
        .as_array()
        .ok_or("missing PI bank inventory")?;
    if entries.len() != 32 {
        return Err("PI bank inventory length mismatch".into());
    }
    let mut banks = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let selector = entry["selector"].as_u64().ok_or("missing PI selector")?;
        let expected = if index < 16 { index } else { index + 224 };
        let offset = index * 0x20000;
        let bank = &payload[0x20000 + offset..0x40000 + offset];
        if selector != expected as u64
            || entry["logical_image_offset"] != offset
            || entry["bytes"] != 0x20000
            || entry["sha256"] != hash(bank)
        {
            return Err("PI bank selector/offset/hash mismatch".into());
        }
        banks.insert(selector as u8, bank.to_vec());
    }
    if banks.get(&255).map(Vec::as_slice) != Some(fixed) {
        return Err("PI FF/fixed ROM disagreement".into());
    }
    super::configure(fixed, banks, profile)
}
