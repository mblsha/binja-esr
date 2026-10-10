// PY_SOURCE: pce500/oz9600/profile.py
// PY_SOURCE: pce500/oz9600/input.py
// PY_SOURCE: pce500/oz9600/retained.py
// PY_SOURCE: pce500/oz9600/session.py
// PY_SOURCE: pce500/oz9600/audio.py
//! Browser model replacement is atomic; retained backing is not a CPU snapshot.
use super::*;
use js_sys::{Int16Array, Reflect};
use sc62015_core::oz9600::{input::PhysicalReplay, ExecutionProfile};

impl Sc62015Emulator {
    pub(crate) fn install_rom(
        &mut self,
        rom: &[u8],
        model: DeviceModel,
        profile: ExecutionProfile,
        retained: &[u8],
    ) -> Result<(), JsValue> {
        self.install_rom_checked(rom, model, profile, retained, None)
    }
    fn install_rom_checked(
        &mut self,
        rom: &[u8],
        model: DeviceModel,
        profile: ExecutionProfile,
        retained: &[u8],
        expected_battery: Option<&[u8]>,
    ) -> Result<(), JsValue> {
        self.require_no_active_call()?;
        if rom.is_empty() {
            return Err(JsValue::from_str("ROM not loaded"));
        }
        let error = |e: sc62015_core::CoreError| JsValue::from_str(&e.to_string());
        let mut candidate = CoreRuntime::for_model(model, rom).map_err(error)?;
        candidate.power_on_reset().map_err(error)?;
        if model == DeviceModel::Oz9600 {
            if retained.starts_with(sc62015_core::oz9600::session::MAGIC) {
                candidate.configure_oz9600_profile(profile).map_err(error)?;
                candidate.restore_oz9600_session(retained).map_err(error)?;
            } else {
                if !retained.is_empty() {
                    candidate
                        .restore_oz9600_retained_state(retained)
                        .map_err(error)?;
                }
                candidate.configure_oz9600_profile(profile).map_err(error)?;
            }
            if let Some(expected) = expected_battery {
                if candidate.oz9600_retained_state().map_err(error)?.as_slice() != expected {
                    return Err(JsValue::from_str(
                        "Saved session and RAM/RTC image disagree",
                    ));
                }
            }
            if retained.starts_with(sc62015_core::oz9600::session::MAGIC) {
                candidate.release_oz9600_session_contacts().map_err(error)?;
            }
            // Capture is a host preference. A successful replacement keeps it
            // enabled but starts a fresh sample timeline with no old backlog.
            candidate
                .set_oz9600_audio_enabled(
                    self.runtime
                        .oz9600_hardware()
                        .is_some_and(|hw| hw.borrow().audio.enabled()),
                )
                .map_err(error)?;
        } else if model == DeviceModel::Iq7000 {
            if let Some(seed) = self.iq7000_rtc_seed.as_deref() {
                candidate
                    .set_iq7000_clock_seed_yyyymmddhhmm(seed)
                    .map_err(error)?;
            }
        }
        // Commit only after bundle, reset, retained identity and profile validation.
        self.runtime = candidate;
        self.rom_image = rom.to_vec();
        self.model = model;
        self.text_decoder = model.text_decoder(rom);
        self.oz9600_profile = profile;
        if model != DeviceModel::Iq7000 {
            self.iq7000_rtc_seed = None;
        }
        self.last_lcd_source = None;
        self.pacer = Pacer::for_model(model, self.pacer.mode());
        Ok(())
    }
}

#[wasm_bindgen]
impl Sc62015Emulator {
    pub fn set_oz9600_audio_enabled(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.require_oz9600()?;
        self.runtime
            .set_oz9600_audio_enabled(enabled)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn oz9600_audio_status(&self) -> Result<JsValue, JsValue> {
        self.require_oz9600()?;
        self.runtime
            .oz9600_hardware()
            .expect("qualified OZ factory")
            .borrow()
            .audio
            .status()
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// An owned PCM block, separate from coalesced LCD frames. No guest writes.
    pub fn take_oz9600_audio(&mut self) -> Result<JsValue, JsValue> {
        self.require_oz9600()?;
        let chunk = self
            .runtime
            .take_oz9600_audio()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let metadata = serde_json::json!({"sample_rate":chunk.sample_rate,
            "first_sample":chunk.first_sample,"total_samples":chunk.total_samples,
            "dropped_samples":chunk.dropped_samples});
        let result = metadata
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Reflect::set(
            &result,
            &JsValue::from_str("samples"),
            &Int16Array::from(chunk.samples.as_slice()),
        )?;
        Ok(result)
    }

    /// Empty retained bytes explicitly request empty logical backing. No captured
    /// RAM or experimental execution settings are inferred from model selection.
    pub fn load_oz9600(
        &mut self,
        bundle: &[u8],
        retained: &[u8],
        profile: &str,
    ) -> Result<(), JsValue> {
        let profile = parse_profile(profile)?;
        self.install_rom(bundle, DeviceModel::Oz9600, profile, retained)
    }

    pub fn oz9600_profile(&self) -> Result<String, JsValue> {
        self.require_oz9600()?;
        Ok(match self.oz9600_profile {
            ExecutionProfile::Strict => "strict",
            ExecutionProfile::Experimental => "experimental",
            ExecutionProfile::ExperimentalIsrClearOnly => "experimental-isr-clear-only",
            ExecutionProfile::ExperimentalIsrMtiWritable => "experimental-isr-mti-writable",
            ExecutionProfile::ExperimentalOnEdge => "experimental-on-edge",
            ExecutionProfile::ExperimentalIrqImr => "experimental-irq-imr",
            ExecutionProfile::ExperimentalRtc => "experimental-rtc",
            ExecutionProfile::ExperimentalRtcIrqImr => "experimental-rtc-irq-imr",
            ExecutionProfile::ProvisionalV1 => "provisional-v1",
        }
        .into())
    }

    pub fn export_oz9600_session(&self) -> Result<Uint8Array, JsValue> {
        self.require_oz9600()?;
        self.require_no_active_call()?;
        let bytes = self
            .runtime
            .oz9600_session_state()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }
    pub fn load_oz9600_checkpoint(
        &mut self,
        bundle: &[u8],
        battery: &[u8],
        session: &[u8],
        profile: &str,
    ) -> Result<(), JsValue> {
        if session.is_empty() {
            return Err(JsValue::from_str("Saved session is empty"));
        }
        self.install_rom_checked(
            bundle,
            DeviceModel::Oz9600,
            parse_profile(profile)?,
            session,
            Some(battery),
        )
    }
    pub fn export_oz9600_retained(&self) -> Result<Uint8Array, JsValue> {
        self.require_oz9600()?;
        let bytes = self
            .runtime
            .oz9600_retained_state()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }

    /// Validates before replacing the running machine with a fresh CPU.
    pub fn restore_oz9600_retained(&mut self, retained: &[u8]) -> Result<(), JsValue> {
        self.require_oz9600()?;
        if retained.is_empty() {
            return Err(JsValue::from_str("Retained image is empty"));
        }
        self.install_rom(
            &self.rom_image.clone(),
            self.model,
            self.oz9600_profile,
            retained,
        )
    }

    pub fn set_oz9600_tablet_contact(
        &mut self,
        x: u16,
        y: u16,
        pressed: bool,
    ) -> Result<(), JsValue> {
        self.require_oz9600()?;
        self.runtime
            .set_oz9600_tablet_contact(x, y, pressed)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Whole-document validation precedes cooperative host replay. The host must
    /// use physical contacts and the shared boundary runner for the validated steps.
    pub fn validate_oz9600_replay(&self, document: &str) -> Result<(), JsValue> {
        self.require_oz9600()?;
        PhysicalReplay::parse(document.as_bytes())
            .map(|_| ())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Side-effect-free observations comparable with the native replay report.
    pub fn oz9600_state(&self) -> Result<JsValue, JsValue> {
        self.require_oz9600()?;
        let rt = &self.runtime;
        let hw = rt.oz9600_hardware().expect("qualified OZ factory").borrow();
        let mut state = serde_json::json!({
            "pc": rt.state.pc(), "instructions": rt.instruction_count(), "cycles": rt.cycle_count(),
            "cpu_halted": rt.state.is_halted(), "irq_total": rt.timer.irq_total, "selector": hw.selector,
            "registers": (["BA","I","X","Y","U","S","F"].iter().map(|n| (*n,rt.get_reg(n))).collect::<std::collections::BTreeMap<_,_>>()),
            "gate_registers": hw.gate.to_vec(), "rtc_registers": hw.rtc.registers(),
            "lcd_registers": hw.lcd.registers.to_vec(),
            "lcd_counters": [hw.lcd.data_writes,hw.lcd.data_reads,hw.lcd.block_operations],
            "lcd_windows": (0..16).map(|n|hw.lcd.window_descriptor(n)).collect::<Vec<_>>(),
            "tablet": [u64::from(hw.tablet.x),u64::from(hw.tablet.y),u64::from(hw.tablet.pressed),u64::from(hw.tablet.conversion_control),u64::from(hw.tablet.drive_control),hw.tablet.data_reads],
        });
        if let Some(progress) = hw.execution.rtc_progression_report() {
            state["rtc_progression"] = progress;
        }
        state
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Sc62015Emulator {
    fn require_oz9600(&self) -> Result<(), JsValue> {
        if self.model != DeviceModel::Oz9600 || self.runtime.oz9600_hardware().is_none() {
            return Err(JsValue::from_str("Requires a loaded OZ-9600 bundle"));
        }
        self.require_no_active_call()
    }
}
fn parse_profile(profile: &str) -> Result<ExecutionProfile, JsValue> {
    let profile = match profile {
        "strict" => ExecutionProfile::Strict,
        "experimental" => ExecutionProfile::Experimental,
        "experimental-isr-clear-only" => ExecutionProfile::ExperimentalIsrClearOnly,
        "experimental-isr-mti-writable" => ExecutionProfile::ExperimentalIsrMtiWritable,
        "experimental-on-edge" => ExecutionProfile::ExperimentalOnEdge,
        "experimental-irq-imr" => ExecutionProfile::ExperimentalIrqImr,
        "experimental-rtc" => ExecutionProfile::ExperimentalRtc,
        "experimental-rtc-irq-imr" => ExecutionProfile::ExperimentalRtcIrqImr,
        "provisional-v1" => ExecutionProfile::ProvisionalV1,
        _ => return Err(JsValue::from_str("Unknown OZ execution profile")),
    };
    Ok(profile)
}
