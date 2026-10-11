// PY_SOURCE: pce500/pa9500/bootstrap.py
// PY_SOURCE: pce500/pa9500/compat_runtime.py
//! Explicit PA firmware exploration profile; physical qualification is pending.
use super::*;
use sc62015_core::pa9500::{self, card::Pa9c3Card};

#[wasm_bindgen]
impl Sc62015Emulator {
    /// RAM/RTC restart backing only, bound to the verified PA ROM and profile.
    pub fn pa9500_retained_state(&self) -> Result<Vec<u8>, JsValue> {
        self.require_no_active_call()?;
        pa9500::retained::retained_state(&self.runtime).map_err(|e| JsValue::from_str(&e))
    }

    /// Explicit fresh-CPU loader; validate backing before replacing the machine.
    pub fn load_pa9500_with_retained(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
        retained: &[u8],
        uart: bool,
    ) -> Result<(), JsValue> {
        if retained.is_empty() {
            return Err(JsValue::from_str("PA retained backing is empty"));
        }
        self.install_pa9500_candidate_with_retained(
            bundle, card_rom, card_sram, profile, uart, retained,
        )
    }

    pub fn restore_pa9500_retained(&mut self, retained: &[u8]) -> Result<(), JsValue> {
        self.require_no_active_call()?;
        pa9500::retained::restore_retained_state(&mut self.runtime, retained)
            .map_err(|e| JsValue::from_str(&e))
    }
    /// Fresh organizer backing, optional verified PA-9C3 media. Replacement is
    /// atomic. No captured main RAM or CPU state is accepted by this loader.
    pub fn load_pa9500(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
    ) -> Result<(), JsValue> {
        self.install_pa9500_candidate(bundle, card_rom, card_sram, profile, false)
    }

    /// Explicit functional register-UART candidate. The ROM owns COM framing
    /// and flow control; this does not qualify physical baud/parity timing.
    pub fn load_pa9500_with_uart(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
    ) -> Result<(), JsValue> {
        self.install_pa9500_candidate(bundle, card_rom, card_sram, profile, true)
    }

    /// Drain completed external transmission, without advancing guest time.
    pub fn pa9500_take_serial(&mut self) -> Result<JsValue, JsValue> {
        self.require_no_active_call()?;
        if self.model != DeviceModel::Pa9500 || self.pa9500_bootstrap.is_none() {
            return Err(JsValue::from_str("PA bootstrap is not loaded"));
        }
        let bytes = self.sio_drain_tx_bytes().to_vec();
        let uart = self
            .runtime
            .sio
            .as_ref()
            .and_then(|s| s.uart())
            .map(|u| u.report());
        serde_json::json!({"bytes":bytes,"uart":uart})
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn pa9500_status(&self) -> Result<JsValue, JsValue> {
        pa9500::bootstrap_status(&self.runtime)
            .map_err(|e| JsValue::from_str(&e))?
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Logical card persistence only. Reset starts a fresh organizer/CPU while
    /// preserving these current card bytes; organizer/RTC retention is pending.
    pub fn pa9500_card_sram(&self) -> Result<Vec<u8>, JsValue> {
        let c = self
            .runtime
            .pa9500_compatibility()
            .ok_or_else(|| JsValue::from_str("PA bootstrap is not loaded"))?;
        Ok(c.hardware
            .borrow()
            .card
            .as_ref()
            .map_or_else(Vec::new, |card| card.sram.clone()))
    }
}

impl Sc62015Emulator {
    pub(crate) fn install_pa9500_candidate(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
        uart: bool,
    ) -> Result<(), JsValue> {
        self.install_pa9500_candidate_with_retained(bundle, card_rom, card_sram, profile, uart, &[])
    }

    fn install_pa9500_candidate_with_retained(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
        uart: bool,
        retained: &[u8],
    ) -> Result<(), JsValue> {
        self.require_no_active_call()?;
        if !matches!(
            profile,
            "bootstrap-v1" | "bootstrap-v2" | "bootstrap-v3" | "bootstrap-v4"
        ) {
            return Err(JsValue::from_str(
                "PA profile must be explicitly bootstrap-v1, bootstrap-v2, bootstrap-v3 or bootstrap-v4",
            ));
        }
        let card = match (card_rom.is_empty(), card_sram.is_empty()) {
            (true, true) => None,
            (false, false) => Some(
                Pa9c3Card::from_images(card_rom, card_sram).map_err(|e| JsValue::from_str(&e))?,
            ),
            _ => return Err(JsValue::from_str("PA card requires both ROM and SRAM")),
        };
        let (mut candidate, _) = match profile {
            "bootstrap-v4" => pa9500::from_bootstrap_bundle_with_power_resume(bundle, card),
            "bootstrap-v3" => pa9500::from_bootstrap_bundle_with_observed_field_reads(bundle, card),
            "bootstrap-v2" => pa9500::from_bootstrap_bundle_with_observed_low_reads(bundle, card),
            _ => pa9500::from_bootstrap_bundle(bundle, card),
        }
        .map_err(|e| JsValue::from_str(&e))?;
        if uart {
            pa9500::install_uart_candidate(&mut candidate).map_err(|e| JsValue::from_str(&e))?;
        }
        if !retained.is_empty() {
            pa9500::retained::restore_retained_state(&mut candidate, retained)
                .map_err(|e| JsValue::from_str(&e))?;
        }
        self.runtime = candidate;
        self.rom_image = bundle.to_vec();
        self.model = DeviceModel::Pa9500;
        self.text_decoder = None;
        self.iq7000_rtc_seed = None;
        self.oz9600_profile = Default::default();
        self.pa9500_bootstrap = Some(profile.to_owned());
        self.pa9500_card_rom = card_rom.to_vec();
        self.pa9500_uart_candidate = uart;
        self.last_lcd_source = None;
        self.pacer = Pacer::for_model(self.model, self.pacer.mode());
        Ok(())
    }
}
