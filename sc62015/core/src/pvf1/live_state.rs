// PY_SOURCE: pce500/pvf1/hardware.py
// PY_SOURCE: pce500/emulator.py:save_snapshot
//! Exact PV-F1 continuation state. Assets/callbacks are rebuilt and validated
//! in a candidate runtime before replacing the active owner. Not PVRAM backing.
use super::{Hardware, HEIGHT, WIDTH};
use crate::{
    keyboard::KeyboardSnapshot, llama::state::LlamaState, sio::SioStub, CoreRuntime, DeviceModel,
    SnapshotMetadata, TimerContext,
};
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"PVLIVE01";
const LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    firmware: [String; 2],
    model: Option<String>,
    metadata: SnapshotMetadata,
    state: LlamaState,
    timer: TimerContext,
    keyboard: Option<KeyboardSnapshot>,
    sio: Option<SioStub>,
    hardware: Hardware,
    legacy: Legacy,
    boundary: Vec<u8>,
    external: Vec<u8>,
    internal: Vec<u8>,
    memory_counts: [u64; 2],
    off_idle_units: u64,
    onk_level: bool,
    external_interrupt_level: bool,
    flags: [bool; 4],
}
#[derive(serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Legacy {
    state: crate::pa9500::compat_probe::StartupState,
    active: bool,
    boundaries: u64,
    transitions: Vec<serde_json::Value>,
    trace: Vec<serde_json::Value>,
}

fn decode(bytes: &[u8]) -> Result<Document, String> {
    if bytes.len() < 40
        || bytes.len() as u64 > LIMIT
        || &bytes[..8] != MAGIC
        || Sha256::digest(&bytes[40..])[..] != bytes[8..40]
    {
        return Err("PV-F1 live-state format/checksum mismatch".into());
    }
    let mut decoder = ZlibDecoder::new(&bytes[40..]);
    let mut payload = Vec::new();
    (&mut decoder)
        .take(LIMIT + 1)
        .read_to_end(&mut payload)
        .map_err(|e| e.to_string())?;
    if payload.len() as u64 > LIMIT || decoder.total_in() as usize != bytes.len() - 40 {
        return Err("PV-F1 live-state extent/size mismatch".into());
    }
    let document: Document = serde_json::from_slice(&payload).map_err(|e| e.to_string())?;
    if document.version != 1
        || document.firmware != [super::FIXED_SHA256, super::BANKS_SHA256]
        || document.metadata.device_model != Some(DeviceModel::PvF1)
    {
        return Err("PV-F1 live-state identity/version mismatch".into());
    }
    Ok(document)
}

impl CoreRuntime {
    pub fn export_pvf1_live_state(&self) -> Result<Vec<u8>, String> {
        if self.device_model() != DeviceModel::PvF1
            || self.host_read.is_some()
            || self.host_peek.is_some()
            || self.host_write.is_some()
            || self.poisoned.is_some()
            || self.loop_detector.is_some()
            || self.lcd_bus_capture.is_some()
            || crate::PERFETTO_TRACER.is_installed()
        {
            return Err(
                "PV-F1 live state requires a healthy idle runtime without active diagnostics"
                    .into(),
            );
        }
        self.memory
            .validate_live_capture()
            .map_err(|e| e.to_string())?;
        let hardware = self
            .pvf1_hardware
            .as_ref()
            .ok_or("PV-F1 hardware missing")?
            .lock()
            .map_err(|e| e.to_string())?;
        let legacy = self
            .pvf1_compatibility
            .as_ref()
            .ok_or("PV-F1 legacy owner missing")?;
        let model = hardware
            .auxiliary
            .as_ref()
            .and_then(|aux| aux.live_model_identity());
        let legacy = Legacy {
            state: legacy.state.clone(),
            active: legacy.active,
            boundaries: legacy.boundaries,
            transitions: legacy.transitions.clone(),
            trace: legacy.trace.clone(),
        };
        let document = json!({
            "version":1,"firmware":[super::FIXED_SHA256,super::BANKS_SHA256],"model":model,
            "metadata":self.metadata,"state":self.state,"timer":self.timer,
            "keyboard":self.keyboard.as_ref().map(|k| k.snapshot_state()),
            "sio":self.sio,
            "hardware":*hardware,"legacy":legacy,
            "boundary":self.boundary_device.as_ref().ok_or("PV-F1 boundary missing")?.save_live_state().map_err(|e|e.to_string())?,
            "external":self.memory.external_slice(),"internal":self.memory.internal_slice(),
            "memory_counts":[self.memory.memory_read_count(),self.memory.memory_write_count()],
            "off_idle_units":self.off_idle_timing_units,"onk_level":self.onk_level,
            "external_interrupt_level":self.external_interrupt_level,
            "flags":[self.pvf1_irq_imr_only_candidate,self.pvf1_on_input_live_candidate,self.pvf1_on_contact_edge_candidate,self.pvf1_alarm_wake_candidate]
        });
        let payload = serde_json::to_vec(&document).map_err(|e| e.to_string())?;
        if payload.len() as u64 > LIMIT {
            return Err("PV-F1 live state exceeds size limit".into());
        }
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&payload).map_err(|e| e.to_string())?;
        let compressed = encoder.finish().map_err(|e| e.to_string())?;
        let mut result = MAGIC.to_vec();
        result.extend_from_slice(&Sha256::digest(&compressed));
        result.extend_from_slice(&compressed);
        Ok(result)
    }

    /// The matching model bytes are an immutable asset supplied by the host.
    /// A browser checkpoint includes these alongside this native state payload.
    pub fn restore_pvf1_live_state(
        &mut self,
        bundle: &[u8],
        model: Option<&[u8]>,
        bytes: &[u8],
    ) -> Result<(), String> {
        if self.device_model() != DeviceModel::PvF1 {
            return Err("requires PV-F1".into());
        }
        let mut document = decode(bytes)?;
        document.state.validate_live_state()?;
        document.timer.validate_live_state()?;
        if let Some(sio) = &document.sio {
            sio.validate_live_state()?;
        }
        if let Some(tablet) = &document.hardware.tablet {
            tablet.validate_live_state()?;
        }
        document.hardware.rtc_clock.validate_live_state()?;
        document.hardware.calendar_clock.validate_live_state()?;
        if let Some(aux) = &document.hardware.auxiliary {
            aux.validate_live_state()?;
        }
        let mut candidate =
            CoreRuntime::for_model(DeviceModel::PvF1, bundle).map_err(|e| e.to_string())?;
        candidate.configure_pvf1_recognizer(model)?;
        let owner = candidate.pvf1_hardware.as_ref().unwrap().clone();
        {
            let mut initial = owner.lock().map_err(|e| e.to_string())?;
            let identity = initial
                .auxiliary
                .as_ref()
                .and_then(|aux| aux.live_model_identity());
            if identity != document.model {
                return Err("PV-F1 live-state recognition model mismatch".into());
            }
            document.hardware.banks = std::mem::take(&mut initial.banks);
            if let (Some(dst), Some(src)) = (&mut document.hardware.auxiliary, &initial.auxiliary) {
                dst.restore_live_model(src);
            }
            if document
                .hardware
                .auxiliary
                .as_ref()
                .and_then(|aux| aux.live_model_identity())
                != document.model
            {
                return Err("PV-F1 live-state auxiliary/model mismatch".into());
            }
        }
        let h = &document.hardware;
        if h.ram.len() != 0x40000
            || h.scratch.len() != 0x4000
            || h.card_buffer.len() != 0x1000
            || !h.banks.contains_key(&h.selector)
            || h.fault.is_some()
            || (h.card.is_some() && h.ram_card.is_some())
        {
            return Err("PV-F1 live-state board invariants failed".into());
        }
        h.lcd.validate_live_geometry(WIDTH, HEIGHT)?;
        if let Some(card) = &h.card {
            crate::pa9500::card::Pa9c3Card::from_images(card.rom(), &card.sram)?;
        }
        if let Some(card) = &h.ram_card {
            crate::ram_media::RamMedia::from_image(card.image(), 0x20000, card.writable())?;
        }
        if document.internal.len() != candidate.memory.internal_slice().len()
            || document.external.len() != candidate.memory.external_len()
            || document.external[0xe0000..] != candidate.memory.external_slice()[0xe0000..]
        {
            return Err("PV-F1 live-state memory/firmware extents failed".into());
        }
        if let Some(card) = &document.hardware.card {
            candidate.configure_pvf1_pa9c3_candidate(card.rom(), &card.sram)?;
        }
        if let Some(card) = &document.hardware.ram_card {
            candidate.configure_pvf1_ram_card_candidate(card.image(), card.writable())?;
        }
        if let Some(mapped) = document.hardware.mapped {
            let bank = document
                .hardware
                .banks
                .get(&mapped)
                .ok_or("unacquired mapped bank")?;
            candidate.memory.remove_overlay("pvf1_bank");
            candidate.memory.add_rom_overlay(0xc0000, bank, "pvf1_bank");
        } else {
            return Err("PV-F1 live state missing mapped bank".into());
        }
        *owner.lock().map_err(|e| e.to_string())? = document.hardware;
        candidate
            .memory
            .copy_external_from(&document.external)
            .map_err(|e| e.to_string())?;
        candidate.memory.write_imem(&document.internal);
        candidate
            .memory
            .set_memory_counts(document.memory_counts[0], document.memory_counts[1]);
        candidate.state = document.state;
        candidate.metadata = document.metadata;
        *candidate.timer = document.timer;
        candidate.keyboard = match document.keyboard {
            Some(snapshot) => {
                let mut keyboard = crate::KeyboardMatrix::new();
                keyboard.load_snapshot_state(&snapshot)?;
                Some(keyboard)
            }
            None => return Err("PV-F1 live state missing keyboard".into()),
        };
        candidate.sio = document.sio;
        let legacy = candidate.pvf1_compatibility.as_mut().unwrap();
        legacy.state = document.legacy.state;
        legacy.active = document.legacy.active;
        legacy.boundaries = document.legacy.boundaries;
        legacy.transitions = document.legacy.transitions;
        legacy.trace = document.legacy.trace;
        candidate
            .boundary_device
            .as_mut()
            .unwrap()
            .restore_live_state(&document.boundary)
            .map_err(|e| e.to_string())?;
        candidate.off_idle_timing_units = document.off_idle_units;
        candidate.onk_level = document.onk_level;
        candidate.external_interrupt_level = document.external_interrupt_level;
        [
            candidate.pvf1_irq_imr_only_candidate,
            candidate.pvf1_on_input_live_candidate,
            candidate.pvf1_on_contact_edge_candidate,
            candidate.pvf1_alarm_wake_candidate,
        ] = document.flags;
        *self = candidate;
        Ok(())
    }
}
