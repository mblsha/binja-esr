// PY_SOURCE: pce500/pvf1/retained.py
//! Restart backing for PV-F1 scratch RAM, main RAM and raw RTC registers.
//! CPU, mapper, LCD, auxiliary transport, gate, GPIO, input and clock phases stay
//! fresh. Optional PVRAM02 backing retains host-modeled registered pen templates.
//! This emulator format does not establish physical battery-backed regions.
use super::{Hardware, SharedHardware};
use crate::CoreRuntime;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"PVRAM01\0";
const CONTROLLER_MAGIC: &[u8; 8] = b"PVRAM02\0";
const LAYOUT_TAG: &[u8] = b"pvf1-scratch16k-main256k-rtc32-1";
const HEADER_SIZE: usize = 104;
pub const MAIN_RAM_SIZE: usize = 0x40000;
pub const SCRATCH_SIZE: usize = 0x4000;
const RTC_SIZE: usize = 32;
const PAYLOAD_SIZE: usize = MAIN_RAM_SIZE + SCRATCH_SIZE + RTC_SIZE;
pub const IMAGE_SIZE: usize = HEADER_SIZE + PAYLOAD_SIZE;

fn bank_identity(hardware: &Hardware) -> [u8; 32] {
    let mut identity = Sha256::new();
    identity.update(LAYOUT_TAG);
    for (selector, bytes) in &hardware.banks {
        identity.update([*selector]);
        identity.update(bytes);
    }
    identity.finalize().into()
}

/// Read backing only, without mapped reads, acknowledgments or CPU steps.
pub fn retained_state(runtime: &CoreRuntime, hardware: &SharedHardware) -> Vec<u8> {
    let hw = hardware.lock().unwrap();
    let mut payload = Vec::with_capacity(PAYLOAD_SIZE);
    payload.extend_from_slice(&hw.ram);
    payload.extend_from_slice(&hw.scratch);
    payload.extend_from_slice(&hw.rtc.retained_registers());
    let controller = hw.auxiliary.as_ref().and_then(|a| a.registered_state());
    if let Some(bytes) = &controller {
        payload.extend_from_slice(bytes);
    }
    let mut image = Vec::with_capacity(IMAGE_SIZE);
    image.extend_from_slice(if controller.is_some() {
        CONTROLLER_MAGIC
    } else {
        MAGIC
    });
    image.extend_from_slice(&Sha256::digest(
        &runtime.memory.external_slice()[0xe0000..0x100000],
    ));
    image.extend_from_slice(&bank_identity(&hw));
    image.extend_from_slice(&Sha256::digest(&payload));
    image.extend_from_slice(&payload);
    image
}

/// Validate all identities and lengths before copying backing. A fresh runtime
/// may restore a compatible image; a running runtime cannot replace its RAM.
pub fn restore_retained_state(
    runtime: &CoreRuntime,
    hardware: &SharedHardware,
    image: &[u8],
) -> Result<(), String> {
    if runtime.instruction_count() != 0 || runtime.elapsed_timing_units() != 0 {
        return Err("PV-F1 retained state must be restored before the first CPU boundary".into());
    }
    let with_controller = image.len() > IMAGE_SIZE && image.get(..8) == Some(CONTROLLER_MAGIC);
    if !with_controller && (image.len() != IMAGE_SIZE || image.get(..8) != Some(MAGIC)) {
        return Err("PV-F1 retained-state version/length mismatch".into());
    }
    let mut hw = hardware.lock().unwrap();
    let payload = &image[HEADER_SIZE..];
    let fixed = Sha256::digest(&runtime.memory.external_slice()[0xe0000..0x100000]);
    if image[8..40] != fixed[..]
        || image[40..72] != bank_identity(&hw)
        || image[72..104] != Sha256::digest(payload)[..]
    {
        return Err(
            "PV-F1 retained-state ROM/bank/layout identity or payload checksum mismatch".into(),
        );
    }
    // Parse and validate the optional controller backing before any RAM/RTC
    // mutation. Old PVRAM01 images retain their byte-identical layout.
    let glyphs = if with_controller {
        Some(super::auxiliary::Auxiliary::decode_registered_state(
            &payload[PAYLOAD_SIZE..],
        )?)
    } else {
        None
    };
    if glyphs.is_some() && hw.auxiliary.is_none() {
        return Err("PV-F1 controller backing requires a modeled auxiliary".into());
    }
    hw.ram.copy_from_slice(&payload[..MAIN_RAM_SIZE]);
    hw.scratch
        .copy_from_slice(&payload[MAIN_RAM_SIZE..MAIN_RAM_SIZE + SCRATCH_SIZE]);
    hw.rtc.restore_retained_registers(
        payload[MAIN_RAM_SIZE + SCRATCH_SIZE..PAYLOAD_SIZE]
            .try_into()
            .expect("validated RTC length"),
    );
    if let Some(glyphs) = glyphs {
        hw.auxiliary
            .as_mut()
            .unwrap()
            .restore_registered_state(glyphs);
    }
    Ok(())
}
