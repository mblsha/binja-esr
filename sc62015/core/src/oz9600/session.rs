// PY_SOURCE: pce500/oz9600/session.py
//! Full software session; RAM-only battery backups remain a separate format.
//! Restore resumes a saved CPU at a scheduler boundary, never repairs cold boot.
use super::{
    audio::AudioCapture, boundary::ExecutionState, lcd::LcdController, rtc::Rtc, tablet::Tablet,
};
use crate::{
    keyboard::{KeyboardMatrix, KeyboardSnapshot},
    llama::state::LlamaState,
    sio::SioStub,
    timer::TimerContext,
    CoreError, CoreRuntime, DeviceModel, Result, SnapshotMetadata,
};
use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
pub const MAGIC: &[u8; 8] = b"OZRUN01\0";
const HEADER: usize = 104;
const MAX_BODY: usize = 8 * 1024 * 1024;
const MAX_IMAGE: usize = 2 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardwareState {
    execution: ExecutionState,
    audio: AudioCapture,
    audio_ci_input: bool,
    selector: u8,
    retained_loaded: bool,
    gate: Vec<u8>,
    rtc: Rtc,
    lcd: LcdController,
    tablet: Tablet,
    gpio: Vec<u8>,
    ram: Vec<u8>,
    cycle: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Session {
    schema: u32,
    metadata: SnapshotMetadata,
    state: LlamaState,
    timer: TimerContext,
    keyboard: KeyboardSnapshot,
    sio: SioStub,
    external: Vec<u8>,
    internal: Vec<u8>,
    hardware: HardwareState,
    onk_level: bool,
    external_interrupt_level: bool,
    off_idle_timing_units: u64,
    memory_reads: u64,
    memory_writes: u64,
}
fn invalid(message: impl ToString) -> CoreError {
    CoreError::InvalidSnapshot(message.to_string())
}

impl CoreRuntime {
    pub fn oz9600_session_state(&self) -> Result<Vec<u8>> {
        if self.device_model() != DeviceModel::Oz9600 {
            return Err(invalid("OZ session requires OZ-9600"));
        }
        if self.poisoned.is_some()
            || self.host_read.is_some()
            || self.host_peek.is_some()
            || self.host_write.is_some()
        {
            return Err(invalid(
                "Session cannot represent a failed runtime or external host callbacks",
            ));
        }
        let hw = self
            .oz9600_hardware()
            .ok_or_else(|| invalid("OZ hardware unavailable"))?
            .borrow();
        if hw.card.is_some()
            || hw.fault.is_some()
            || hw.ram_access_watch.is_some()
            || hw.lcd_pixel_watch.is_some()
        {
            return Err(invalid("Session requires a healthy built-in organizer without card or diagnostic observers"));
        }
        let data = Session {
            schema: 1,
            metadata: self.metadata.clone(),
            state: self.state.clone(),
            timer: (*self.timer).clone(),
            keyboard: self
                .keyboard
                .as_ref()
                .ok_or_else(|| invalid("Keyboard unavailable"))?
                .snapshot_state(),
            sio: self
                .sio
                .as_ref()
                .ok_or_else(|| invalid("UART unavailable"))?
                .clone(),
            external: self.memory.external_slice()[..0xe0000].to_vec(),
            internal: self.memory.internal_slice().to_vec(),
            hardware: HardwareState {
                execution: hw.execution.clone(),
                audio: hw.audio.clone(),
                audio_ci_input: hw.audio_ci_input,
                selector: hw.selector,
                retained_loaded: hw.retained_loaded,
                gate: hw.gate.to_vec(),
                rtc: hw.rtc.clone(),
                lcd: hw.lcd.clone(),
                tablet: hw.tablet.clone(),
                gpio: hw.gpio.to_vec(),
                ram: hw.ram.clone(),
                cycle: hw.cycle,
            },
            onk_level: self.onk_level,
            external_interrupt_level: self.external_interrupt_level,
            off_idle_timing_units: self.off_idle_timing_units,
            memory_reads: self.memory.memory_read_count(),
            memory_writes: self.memory.memory_write_count(),
        };
        drop(hw);
        self.state.validate_session().map_err(invalid)?;
        let body =
            serde_json::to_vec(&serde_json::to_value(&data).map_err(invalid)?).map_err(invalid)?;
        if body.len() > MAX_BODY {
            return Err(invalid("Session exceeds decoded size limit"));
        }
        let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
        gzip.write_all(&body).map_err(invalid)?;
        let compressed = gzip.finish().map_err(invalid)?;
        let battery = self.oz9600_retained_state()?;
        let mut image = Vec::new();
        image.extend_from_slice(MAGIC);
        image.extend_from_slice(&battery[8..72]);
        image.extend_from_slice(&Sha256::digest(&compressed));
        image.extend_from_slice(&compressed);
        if image.len() > MAX_IMAGE {
            return Err(invalid("Session exceeds encoded size limit"));
        }
        Ok(image)
    }

    /// Only a fresh candidate receives the state. Frontends replace their live
    /// machine after this returns successfully; failed imports leave it intact.
    pub fn restore_oz9600_session(&mut self, image: &[u8]) -> Result<()> {
        if self.instruction_count() != 0
            || self.cycle_count() != 0
            || self.device_model() != DeviceModel::Oz9600
        {
            return Err(invalid("Session restore requires a fresh OZ candidate"));
        }
        if image.len() <= HEADER || image.len() > MAX_IMAGE || &image[..8] != MAGIC {
            return Err(invalid("OZ session version/length mismatch"));
        }
        let battery = self.oz9600_retained_state()?;
        if image[8..72] != battery[8..72] || image[72..104] != Sha256::digest(&image[104..])[..] {
            return Err(invalid("OZ session ROM/bank identity or checksum mismatch"));
        }
        let mut gzip = GzDecoder::new(&image[HEADER..]);
        let mut body = Vec::new();
        gzip.by_ref()
            .take(MAX_BODY as u64 + 1)
            .read_to_end(&mut body)
            .map_err(invalid)?;
        if body.len() > MAX_BODY || !gzip.into_inner().is_empty() {
            return Err(invalid("OZ session compression/length mismatch"));
        }
        let data: Session = serde_json::from_slice(&body).map_err(invalid)?;
        if data.schema != 1
            || data.metadata.device_model != Some(DeviceModel::Oz9600)
            || data.external.len() != 0xe0000
            || data.internal.len() != 256
            || data.hardware.ram.len() != 0x40000
            || data.hardware.gate.len() != 64
            || data.hardware.gpio.len() != 256
        {
            return Err(invalid("OZ session layout mismatch"));
        }
        if data.state.block_transfer_policy() != self.state.block_transfer_policy()
            || data.state.byte_arithmetic_source_policy()
                != self.state.byte_arithmetic_source_policy()
            || data.state.isr_software_write_policy() != self.state.isr_software_write_policy()
            || data.hardware.execution.session_policy()
                != self
                    .oz9600_hardware()
                    .expect("hardware")
                    .borrow()
                    .execution
                    .session_policy()
        {
            return Err(invalid(
                "Saved session uses a different OZ execution profile",
            ));
        }
        data.state.validate_session().map_err(invalid)?;
        data.timer
            .validate_session(data.metadata.cycle_count)
            .map_err(invalid)?;
        data.hardware.lcd.validate_session().map_err(invalid)?;
        data.hardware.tablet.validate_session().map_err(invalid)?;
        data.sio.validate_session().map_err(invalid)?;
        let mut keyboard = KeyboardMatrix::new();
        keyboard
            .load_snapshot_state(&data.keyboard)
            .map_err(invalid)?;
        if keyboard.snapshot_state() != data.keyboard {
            return Err(invalid("Session keyboard is not exactly representable"));
        }
        // Every fallible operation ends before the mutation boundary.
        let mut backing = self.memory.external_slice().to_vec();
        backing[..0xe0000].copy_from_slice(&data.external);
        self.memory
            .copy_external_from(&backing)
            .expect("validated memory size");
        self.memory.write_imem(&data.internal);
        self.memory.clear_dirty();
        self.memory
            .set_memory_counts(data.memory_reads, data.memory_writes);
        self.state = data.state;
        self.state.renew_session_stamp();
        *self.timer = data.timer;
        self.keyboard = Some(keyboard);
        self.sio = Some(data.sio);
        self.metadata = data.metadata;
        self.onk_level = data.onk_level;
        self.external_interrupt_level = data.external_interrupt_level;
        self.off_idle_timing_units = data.off_idle_timing_units;
        {
            let mut hw = self
                .oz9600_hardware()
                .expect("validated hardware")
                .borrow_mut();
            hw.execution = data.hardware.execution;
            hw.audio = data.hardware.audio;
            hw.audio_ci_input = data.hardware.audio_ci_input;
            hw.selector = data.hardware.selector;
            hw.retained_loaded = data.hardware.retained_loaded;
            hw.gate.copy_from_slice(&data.hardware.gate);
            hw.rtc = data.hardware.rtc;
            hw.lcd = data.hardware.lcd;
            hw.tablet = data.hardware.tablet;
            hw.gpio.copy_from_slice(&data.hardware.gpio);
            hw.ram = data.hardware.ram;
            hw.cycle = data.hardware.cycle;
            hw.execution.invalidate_bank_view();
        }
        let hardware = self.oz9600_hardware().expect("hardware").clone();
        super::boundary::sync_bank_view(&hardware, &mut self.memory);
        Ok(())
    }
}

// A restored session has no host contact owners. Frontends apply real release
// transitions once after replacement, retaining queued keys and latched IRQ/ADC.
impl CoreRuntime {
    pub fn release_oz9600_session_contacts(&mut self) -> Result<()> {
        if self.device_model() != DeviceModel::Oz9600 {
            return Err(invalid("OZ contacts require OZ-9600"));
        }
        let keys = self
            .keyboard
            .as_ref()
            .map(KeyboardMatrix::pressed_matrix_codes)
            .unwrap_or_default();
        for key in keys {
            self.set_physical_matrix_key(key, false);
        }
        if self.onk_level {
            self.release_on_key();
        }
        let (x, y, pressed) = {
            let hw = self.oz9600_hardware().expect("hardware").borrow();
            (hw.tablet.x, hw.tablet.y, hw.tablet.pressed)
        };
        if pressed {
            self.set_oz9600_tablet_contact(x, y, false)?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn runtime() -> CoreRuntime {
        let mut fixed = vec![0; 0x20000];
        fixed[0x1fffd..].copy_from_slice(&[0, 0, 14]);
        let mut rt =
            super::super::configure_hardware(&fixed, super::super::Hardware::default()).unwrap();
        rt.configure_oz9600_profile(super::super::ExecutionProfile::ProvisionalV1)
            .unwrap();
        rt
    }
    fn changed(image: &[u8], change: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut body = String::new();
        GzDecoder::new(&image[HEADER..])
            .read_to_string(&mut body)
            .unwrap();
        let mut data: serde_json::Value = serde_json::from_str(&body).unwrap();
        change(&mut data);
        let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
        gzip.write_all(&serde_json::to_vec(&data).unwrap()).unwrap();
        let body = gzip.finish().unwrap();
        let mut result = image[..72].to_vec();
        result.extend_from_slice(&Sha256::digest(&body));
        result.extend_from_slice(&body);
        result
    }
    #[test]
    fn checkpoint_restores_every_serialized_field_and_continues_exactly() {
        let mut a = runtime();
        a.step_scheduler_boundaries(10).unwrap();
        a.set_physical_matrix_key(0x10, true);
        a.press_on_key();
        a.set_oz9600_tablet_contact(411, 712, true).unwrap();
        {
            let mut hw = a.oz9600_hardware().unwrap().borrow_mut();
            hw.tablet.drive_control = 0xa8;
            hw.tablet.write_conversion_control(6).unwrap();
            hw.tablet.read_data().unwrap();
            hw.gpio[3] = 17;
            hw.ram[123] = 97;
        }
        let image = a.oz9600_session_state().unwrap();
        let mut b = runtime();
        b.restore_oz9600_session(&image).unwrap();
        assert_eq!(image, b.oz9600_session_state().unwrap());
        a.step_scheduler_boundaries(20).unwrap();
        b.step_scheduler_boundaries(20).unwrap();
        assert_eq!(
            a.oz9600_session_state().unwrap(),
            b.oz9600_session_state().unwrap()
        );
        b.release_oz9600_session_contacts().unwrap();
        assert!(b
            .keyboard
            .as_ref()
            .unwrap()
            .pressed_matrix_codes()
            .is_empty());
        let mut hw = b.oz9600_hardware().unwrap().borrow_mut();
        assert!(!hw.tablet.pressed);
        assert_eq!(hw.tablet.read_data().unwrap(), 3); // retained low half of 411
    }
    #[test]
    fn invalid_checkpoint_never_mutates_fresh_candidate() {
        let image = runtime().oz9600_session_state().unwrap();
        let mut bads = vec![vec![], image[..image.len() - 1].to_vec()];
        let mut corrupt = image.clone();
        corrupt[8] ^= 1;
        bads.push(corrupt);
        let mut corrupt = image.clone();
        corrupt[HEADER] ^= 1;
        bads.push(corrupt);
        bads.extend([
            changed(&image, |d| d["schema"] = serde_json::json!(2)),
            changed(&image, |d| d["external"] = serde_json::json!([])),
            changed(&image, |d| d["state"]["pc"] = serde_json::json!(0)),
            changed(&image, |d| {
                d["hardware"]["tablet"]["x"] = serde_json::json!(1024)
            }),
            changed(&image, |d| {
                d["hardware"]["lcd"]["pixels"] = serde_json::json!([])
            }),
            changed(&image, |d| d["unknown"] = serde_json::json!(true)),
        ]);
        let mut trailing = image.clone();
        trailing.extend_from_slice(b"tail");
        let digest = Sha256::digest(&trailing[HEADER..]);
        trailing[72..104].copy_from_slice(&digest);
        bads.push(trailing);
        for bad in bads {
            let mut candidate = runtime();
            let before = candidate.oz9600_session_state().unwrap();
            assert!(candidate.restore_oz9600_session(&bad).is_err());
            assert_eq!(before, candidate.oz9600_session_state().unwrap());
        }
    }
    #[test]
    fn profile_mismatch_and_live_restore_are_rejected() {
        let image = runtime().oz9600_session_state().unwrap();
        let mut other = runtime();
        other
            .configure_oz9600_profile(super::super::ExecutionProfile::Strict)
            .unwrap();
        assert!(other.restore_oz9600_session(&image).is_err());
        let mut live = runtime();
        live.step_scheduler_boundaries(1).unwrap();
        let before = live.oz9600_session_state().unwrap();
        assert!(live.restore_oz9600_session(&image).is_err());
        assert_eq!(before, live.oz9600_session_state().unwrap());
    }
}
