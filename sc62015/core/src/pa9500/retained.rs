// PY_SOURCE: pce500/pa9500/retained.py
//! Restart backing for the explicit PA bootstrap profiles. CPU, LCD, timers,
//! selector, scratch, input and card state are deliberately fresh/separate.
use super::{Hardware, SharedHardware, FIXED_ROM_HASH};
use crate::{llama::state::BlockTransferPolicy, CoreRuntime, DeviceModel};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"PABAT01\0";
const LAYOUT_TAG: &[u8] = b"pa9500-bootstrap-high-ram-80000-bffff-rtc-1";
const HEADER_SIZE: usize = 104;
pub const MAIN_RAM_SIZE: usize = 0x40000;
pub const IMAGE_SIZE: usize = HEADER_SIZE + MAIN_RAM_SIZE + 32;

fn board(runtime: &CoreRuntime) -> Result<SharedHardware, String> {
    if runtime.device_model() != DeviceModel::Pa9500
        || !runtime.pa9500_on_contact_edge_candidate
        || runtime.state.block_transfer_policy() != BlockTransferPolicy::CoupledPredecrement
    {
        return Err("PA retained backing requires an explicit bootstrap profile".into());
    }
    let hw = runtime
        .pa9500_compatibility()
        .map(|c| c.hardware.clone())
        .ok_or_else(|| "PA retained backing requires the shared coordinator".to_owned())?;
    if hw.borrow().irq_imr_only_candidate != runtime.pa9500_irq_imr_only_candidate {
        return Err("PA retained IRQ profile mismatch".into());
    }
    Ok(hw)
}

fn profile(hw: &Hardware) -> Result<u8, String> {
    if hw.ram.len() != MAIN_RAM_SIZE
        || hw.rtc.is_none()
        || hw.native_zero_data != Some(0xe0)
        || !hw.card_eil_presence_candidate
        || !hw.lcd_coordinate_restore_candidate
        || hw.scratch_candidate.as_ref().map(Vec::len) != Some(4096)
        || hw.card_lower_write_alias
    {
        return Err("PA retained backing layout/profile mismatch".into());
    }
    match (
        hw.observed_low_pair,
        hw.observed_low_triad,
        hw.observed_field_words,
        hw.observed_power_pair,
        hw.irq_imr_only_candidate,
    ) {
        (None, None, None, None, false) => Ok(1),
        (Some([0xe0, 0xe0]), Some([0xe0, 0xe0, 0xe0]), None, None, false) => Ok(2),
        (
            Some([0xe0, 0xe0]),
            Some([0xe0, 0xe0, 0xe0]),
            Some([0xe0, 0xe0, 0xe0, 0xe0]),
            None,
            false,
        ) => Ok(3),
        (
            Some([0xe0, 0xe0]),
            Some([0xe0, 0xe0, 0xe0]),
            Some([0xe0, 0xe0, 0xe0, 0xe0]),
            Some([0xe0, 0xe0]),
            true,
        ) => Ok(4),
        _ => Err("PA retained backing has a noncanonical bootstrap profile".into()),
    }
}

fn identity(hw: &Hardware) -> Result<[u8; 32], String> {
    let mut hash = Sha256::new();
    hash.update(LAYOUT_TAG);
    hash.update([profile(hw)?]);
    for (selector, bytes) in &hw.banks {
        hash.update([*selector]);
        hash.update(bytes);
    }
    Ok(hash.finalize().into())
}

fn fixed_identity(runtime: &CoreRuntime) -> Result<[u8; 32], String> {
    let hash = Sha256::digest(&runtime.memory.external_slice()[0xe0000..0x100000]);
    if format!("{hash:x}") != FIXED_ROM_HASH {
        return Err("PA retained backing requires the verified fixed ROM".into());
    }
    Ok(hash.into())
}

/// Read backing only; no mapped bus reads, CPU steps or acknowledgments.
pub fn retained_state(runtime: &CoreRuntime) -> Result<Vec<u8>, String> {
    let hardware = board(runtime)?;
    let hw = hardware.borrow();
    let fixed = fixed_identity(runtime)?;
    let identity = identity(&hw)?;
    let mut payload = hw.ram.clone();
    payload.extend_from_slice(&hw.rtc.as_ref().unwrap().retained_registers());
    let mut image = Vec::with_capacity(IMAGE_SIZE);
    image.extend_from_slice(MAGIC);
    image.extend_from_slice(&fixed);
    image.extend_from_slice(&identity);
    image.extend_from_slice(&Sha256::digest(&payload));
    image.extend_from_slice(&payload);
    Ok(image)
}

/// Restore only RAM/RTC into a newly constructed CPU. Validate model, ROM,
/// every acquired bank, profile, length and checksum before mutating backing.
pub fn restore_retained_state(runtime: &mut CoreRuntime, image: &[u8]) -> Result<(), String> {
    let hardware = board(runtime)?;
    let coordinator = runtime.pa9500_compatibility().unwrap();
    if runtime.instruction_count() != 0
        || runtime.elapsed_timing_units() != 0
        || coordinator.legacy_boundaries != 0
        || coordinator.in_legacy
        || hardware.borrow().cycle != 0
    {
        return Err("PA retained backing must be restored before the first CPU boundary".into());
    }
    if image.len() != IMAGE_SIZE || &image[..8] != MAGIC {
        return Err("PA retained backing version/length mismatch".into());
    }
    let mut hw = hardware.borrow();
    let payload = &image[HEADER_SIZE..];
    if image[8..40] != fixed_identity(runtime)?
        || image[40..72] != identity(&hw)?
        || image[72..104] != Sha256::digest(payload)[..]
    {
        return Err("PA retained backing ROM/bank/profile/checksum mismatch".into());
    }
    hw.ram.copy_from_slice(&payload[..MAIN_RAM_SIZE]);
    hw.rtc
        .as_mut()
        .unwrap()
        .restore_retained_registers(payload[MAIN_RAM_SIZE..].try_into().unwrap());
    Ok(())
}
