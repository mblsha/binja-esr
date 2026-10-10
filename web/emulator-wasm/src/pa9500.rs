// PY_SOURCE: pce500/pa9500/bootstrap.py
// PY_SOURCE: pce500/pa9500/compat_runtime.py
//! Explicit PA firmware exploration profile; physical qualification is pending.
use super::*;
use sc62015_core::pa9500::{self, card::Pa9c3Card};

#[wasm_bindgen]
impl Sc62015Emulator {
    /// Fresh organizer backing, optional verified PA-9C3 media. Replacement is
    /// atomic. No captured main RAM or CPU state is accepted by this loader.
    pub fn load_pa9500(
        &mut self,
        bundle: &[u8],
        card_rom: &[u8],
        card_sram: &[u8],
        profile: &str,
    ) -> Result<(), JsValue> {
        self.require_no_active_call()?;
        if profile != "bootstrap-v1" {
            return Err(JsValue::from_str(
                "PA profile must be explicitly bootstrap-v1",
            ));
        }
        let card = match (card_rom.is_empty(), card_sram.is_empty()) {
            (true, true) => None,
            (false, false) => Some(
                Pa9c3Card::from_images(card_rom, card_sram).map_err(|e| JsValue::from_str(&e))?,
            ),
            _ => return Err(JsValue::from_str("PA card requires both ROM and SRAM")),
        };
        let (candidate, _) =
            pa9500::from_bootstrap_bundle(bundle, card).map_err(|e| JsValue::from_str(&e))?;
        self.runtime = candidate;
        self.rom_image = bundle.to_vec();
        self.model = DeviceModel::Pa9500;
        self.text_decoder = None;
        self.iq7000_rtc_seed = None;
        self.oz9600_profile = Default::default();
        self.pa9500_bootstrap = true;
        self.pa9500_card_rom = card_rom.to_vec();
        self.last_lcd_source = None;
        self.pacer = Pacer::for_model(self.model, self.pacer.mode());
        Ok(())
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
