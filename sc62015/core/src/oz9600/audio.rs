// PY_SOURCE: pce500/oz9600/audio.py
//! OZ model routing for the shared digital CO capture component.
pub use crate::digital_audio::{
    AudioCapture, AudioChunk, QUEUE_CAPACITY, SAMPLE_RATE, TIMEBASE_HZ,
};
use crate::{CoreError, CoreRuntime, Result};

impl CoreRuntime {
    pub fn set_oz9600_audio_enabled(&mut self, enabled: bool) -> Result<()> {
        let hardware = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ audio requires OZ hardware".into()))?;
        hardware.borrow_mut().audio.set_enabled(enabled);
        Ok(())
    }

    pub fn take_oz9600_audio(&mut self) -> Result<AudioChunk> {
        let hardware = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ audio requires OZ hardware".into()))?;
        Ok(hardware.borrow_mut().audio.take())
    }
}
