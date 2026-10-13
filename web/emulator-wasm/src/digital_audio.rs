// PY_SOURCE: pce500/digital_audio.py
//! Model-routed digital audio, independent of LCD frame delivery.
use crate::*;
use js_sys::{Int16Array, Reflect};

#[wasm_bindgen]
impl Sc62015Emulator {
    pub fn set_audio_enabled(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.runtime
            .set_audio_enabled(enabled)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn audio_status(&self) -> Result<JsValue, JsValue> {
        self.runtime
            .audio_status()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn take_audio(&mut self) -> Result<JsValue, JsValue> {
        let chunk = self
            .runtime
            .take_audio()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = serde_json::json!({"sample_rate":chunk.sample_rate,
            "first_sample":chunk.first_sample,"total_samples":chunk.total_samples,
            "dropped_samples":chunk.dropped_samples})
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Reflect::set(
            &result,
            &JsValue::from_str("samples"),
            &Int16Array::from(chunk.samples.as_slice()),
        )?;
        Ok(result)
    }
}
