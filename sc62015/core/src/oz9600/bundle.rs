// PY_SOURCE: pce500/oz9600/bundle.py
//! Hash-checked container for local native/WASM acquisition loading.
//! It transports original captures; it does not inject captured live RAM.
use crate::CoreRuntime;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"OZROM01\0";
const HEADER_SIZE: usize = 52;
const MAPPED_SIZE: usize = 0x100000;
const BANK_SIZE: usize = 0x20000;
const MAX_MANIFEST_SIZE: usize = 0x40000;

pub fn from_rom_bundle(image: &[u8]) -> Result<CoreRuntime, String> {
    if image.len() < HEADER_SIZE || &image[..8] != MAGIC {
        return Err("ROM bundle version/header mismatch".into());
    }
    let word =
        |offset: usize| u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap()) as usize;
    let mapped_size = word(8);
    let manifest_size = word(12);
    let bank_size = word(16);
    if mapped_size != MAPPED_SIZE
        || manifest_size == 0
        || manifest_size > MAX_MANIFEST_SIZE
        || bank_size > 16 * BANK_SIZE
        || bank_size % BANK_SIZE != 0
        || image.len() != HEADER_SIZE + mapped_size + manifest_size + bank_size
    {
        return Err("ROM bundle length mismatch".into());
    }
    let payload = &image[HEADER_SIZE..];
    if Sha256::digest(payload).as_slice() != &image[20..52] {
        return Err("ROM bundle payload checksum mismatch".into());
    }
    let mapped = &payload[..mapped_size];
    let manifest = &payload[mapped_size..mapped_size + manifest_size];
    let banks = &payload[mapped_size + manifest_size..];
    let mut offset = 0;
    let machine = super::from_verified_images(mapped, manifest, |_| {
        let bank = banks
            .get(offset..offset + BANK_SIZE)
            .ok_or("ROM bundle has too few bank payloads")?
            .to_vec();
        offset += BANK_SIZE;
        Ok(bank)
    })?;
    if offset != banks.len() {
        return Err("ROM bundle has unclaimed bank payloads".into());
    }
    Ok(machine)
}
