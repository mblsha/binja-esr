// PY_SOURCE: pce500/oz9600/retained.py
//! Restart image for one 256 KiB RAM backing and RTC registers.
//! The low workspace is a mapped view of the last RAM quarter, not a second
//! retained copy. CPU, timer, LCD, gate, GPIO and input state are not restored.
//! The top-page alias follows FE591 reservation arithmetic; mapper modes and
//! physical chip enables remain unqualified.
use super::{Hardware, SharedHardware, FIXED_ROM_HASH, WORKSPACE_RAM_OFFSET};
use crate::CoreRuntime;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"OZBAT02\0";
const LEGACY_MAGIC: &[u8; 8] = b"OZBAT01\0";
const LAYOUT_TAG: &[u8] = b"oz9600-top-page-ram-rtc-2";
const LEGACY_LAYOUT_TAG: &[u8] = b"oz9600-logical-ram-rtc-1";
const HEADER_SIZE: usize = 8 + 3 * 32;
pub const WORKSPACE_START: usize = 0x10000;
pub const WORKSPACE_SIZE: usize = 0x10000;
pub const MAIN_RAM_SIZE: usize = 0x40000;
const RTC_SIZE: usize = 32;
const PAYLOAD_SIZE: usize = MAIN_RAM_SIZE + RTC_SIZE;
const LEGACY_PAYLOAD_SIZE: usize = WORKSPACE_SIZE + PAYLOAD_SIZE;
pub const IMAGE_SIZE: usize = HEADER_SIZE + PAYLOAD_SIZE;

fn bank_identity(hardware: &Hardware, layout: &[u8]) -> [u8; 32] {
    let mut identity = Sha256::new();
    identity.update(layout);
    for (selector, bytes) in &hardware.banks {
        identity.update([*selector]);
        identity.update(bytes);
    }
    identity.finalize().into()
}

fn image_from_payload(runtime: &CoreRuntime, hardware: &Hardware, payload: &[u8]) -> Vec<u8> {
    let mut image = Vec::with_capacity(IMAGE_SIZE);
    image.extend_from_slice(MAGIC);
    image.extend_from_slice(&Sha256::digest(
        &runtime.memory.external_slice()[0xe0000..0x100000],
    ));
    image.extend_from_slice(&bank_identity(hardware, LAYOUT_TAG));
    image.extend_from_slice(&Sha256::digest(payload));
    image.extend_from_slice(payload);
    image
}

/// Read backing only. No mapped reads, acknowledgments or CPU steps occur.
pub fn retained_state(runtime: &CoreRuntime, hardware: &SharedHardware) -> Vec<u8> {
    let hw = hardware.borrow();
    let mut payload = Vec::with_capacity(PAYLOAD_SIZE);
    payload.extend_from_slice(&hw.ram);
    payload.extend_from_slice(&hw.rtc.retained_registers());
    image_from_payload(runtime, &hw, &payload)
}

fn validated_payload<'a>(
    runtime: &CoreRuntime,
    hardware: &Hardware,
    image: &'a [u8],
    magic: &[u8; 8],
    layout: &[u8],
    size: usize,
) -> Result<&'a [u8], String> {
    if image.len() != HEADER_SIZE + size || &image[..8] != magic {
        return Err("retained-state version/length mismatch".into());
    }
    let fixed = Sha256::digest(&runtime.memory.external_slice()[0xe0000..0x100000]);
    if format!("{fixed:x}") != FIXED_ROM_HASH {
        return Err("retained restore requires the verified fixed ROM".into());
    }
    let payload = &image[HEADER_SIZE..];
    if image[8..40] != fixed[..]
        || image[40..72] != bank_identity(hardware, layout)
        || image[72..104] != Sha256::digest(payload)[..]
    {
        return Err("retained-state ROM/bank identity or payload checksum mismatch".into());
    }
    Ok(payload)
}

/// Validate everything before copying the single retained backing. Restore is
/// allowed only before execution. Legacy independent copies require explicit
/// conversion; they are never silently overlaid onto an aliased machine.
pub fn restore_retained_state(
    runtime: &mut CoreRuntime,
    hardware: &SharedHardware,
    image: &[u8],
) -> Result<(), String> {
    if runtime.instruction_count() != 0 || runtime.cycle_count() != 0 {
        return Err("retained state must be restored before the first CPU boundary".into());
    }
    if image.starts_with(LEGACY_MAGIC) {
        return Err("OZBAT01 used independent workspace; convert explicitly to OZBAT02".into());
    }
    let mut hw = hardware.borrow_mut();
    let payload = validated_payload(runtime, &hw, image, MAGIC, LAYOUT_TAG, PAYLOAD_SIZE)?;
    hw.ram.copy_from_slice(&payload[..MAIN_RAM_SIZE]);
    hw.rtc.restore_retained_registers(
        payload[MAIN_RAM_SIZE..]
            .try_into()
            .expect("validated RTC length"),
    );
    hw.retained_loaded = true;
    Ok(())
}

fn merged_legacy_payload(payload: &[u8]) -> Result<Vec<u8>, String> {
    let workspace = &payload[..WORKSPACE_SIZE];
    let old_ram = &payload[WORKSPACE_SIZE..WORKSPACE_SIZE + MAIN_RAM_SIZE];
    let mut ram = old_ram.to_vec();
    if workspace.iter().any(|b| *b != 0) || old_ram.iter().any(|b| *b != 0) {
        let word =
            |a: usize| u32::from_le_bytes([workspace[a], workspace[a + 1], workspace[a + 2], 0]);
        let floor = word(0xff38);
        let start = word(0xfd00);
        let end = word(0xfd03);
        if !(0x10000..=0x20000).contains(&floor)
            || end != floor + 0xa0000
            || !(0x80000..end).contains(&start)
        {
            return Err("legacy image lacks consistent workspace/filesystem reservation".into());
        }
        let offset = floor as usize - WORKSPACE_START;
        // Below the live-workspace floor, main-RAM data wins only over an
        // unused zero byte or an identical workspace byte. Other conflicts
        // are ambiguous and must not be discarded by a converter.
        if workspace[..offset]
            .iter()
            .zip(&ram[WORKSPACE_RAM_OFFSET..WORKSPACE_RAM_OFFSET + offset])
            .any(|(low, high)| *low != 0 && low != high)
        {
            return Err("legacy workspace prefix conflicts with main RAM".into());
        }
        ram[WORKSPACE_RAM_OFFSET + offset..].copy_from_slice(&workspace[offset..]);
    }
    ram.extend_from_slice(&payload[WORKSPACE_SIZE + MAIN_RAM_SIZE..]);
    Ok(ram)
}

/// Offline conversion of a validated legacy image, with no runtime mutation.
/// Preserve main RAM below its stored filesystem end; preserve live workspace
/// above its stored floor. Reject inconsistent or ambiguous reservations.
pub fn convert_legacy_image(
    runtime: &CoreRuntime,
    hardware: &SharedHardware,
    image: &[u8],
) -> Result<Vec<u8>, String> {
    let hw = hardware.borrow();
    let payload = validated_payload(
        runtime,
        &hw,
        image,
        LEGACY_MAGIC,
        LEGACY_LAYOUT_TAG,
        LEGACY_PAYLOAD_SIZE,
    )?;
    let merged = merged_legacy_payload(payload)?;
    Ok(image_from_payload(runtime, &hw, &merged))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_payload() -> Vec<u8> {
        let mut p = vec![0; LEGACY_PAYLOAD_SIZE];
        for (a, v) in [(0xff38, 0x10300_u32), (0xfd00, 0x80100), (0xfd03, 0xb0300)] {
            p[a..a + 3].copy_from_slice(&v.to_le_bytes()[..3]);
        }
        p[WORKSPACE_SIZE + WORKSPACE_RAM_OFFSET + 0x100] = 0xfb;
        p[0x300] = 0x17;
        p[0xffff] = 0x52;
        p[LEGACY_PAYLOAD_SIZE - 1] = 0xa5;
        p
    }

    #[test]
    fn legacy_merge_preserves_both_reserved_regions_and_rtc_without_double_copy() {
        let p = legacy_payload();
        let merged = merged_legacy_payload(&p).unwrap();
        assert_eq!(merged.len(), PAYLOAD_SIZE);
        assert_eq!(merged[WORKSPACE_RAM_OFFSET + 0x100], 0xfb);
        assert_eq!(merged[WORKSPACE_RAM_OFFSET + 0x300], 0x17);
        assert_eq!(merged[MAIN_RAM_SIZE - 1], 0x52);
        assert_eq!(merged[PAYLOAD_SIZE - 1], 0xa5);
        assert_eq!(
            merged_legacy_payload(&vec![0; LEGACY_PAYLOAD_SIZE]).unwrap(),
            vec![0; PAYLOAD_SIZE]
        );
    }

    #[test]
    fn legacy_merge_rejects_inconsistent_reservation_and_nonzero_prefix_conflicts() {
        for a in [0xff38, 0xfd00, 0xfd03, 0x100] {
            let mut p = legacy_payload();
            p[a] ^= 1;
            if a == 0xfd00 {
                p[a..a + 3].fill(0);
            }
            assert!(merged_legacy_payload(&p).is_err(), "offset {a:x}");
        }
    }
}
