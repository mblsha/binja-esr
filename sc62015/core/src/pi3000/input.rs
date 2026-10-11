// PY_SOURCE: pce500/pi3000/input.py
//! Raw contact replay through normal scheduler boundaries; no guest RAM events.
use crate::{CoreError, CoreRuntime, DeviceModel, Result};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contact {
    pub raw_x: u16,
    pub raw_y: u16,
    pub pressed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixContact {
    pub column: u8,
    pub row: u8,
    pub pressed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_entire_replay_before_running_any_step() {
        for data in [
            r#"{"steps":[{"boundaries":1},{"boundaries":1,"tablet":{"raw_x":1024,"raw_y":0,"pressed":true}}]}"#,
            r#"{"steps":[{"boundaries":1,"tablet":{"raw_x":2,"raw_y":3,"pressed":1}}]}"#,
            r#"{"steps":[{"boundaries":1,"native_ram_event":2}]}"#,
            r#"{"steps":[]}"#,
            r#"{"steps":[{"boundaries":1,"contact":{"column":11,"row":0,"pressed":true}}]}"#,
            r#"{"steps":[{"boundaries":1,"contact":{"column":0,"row":8,"pressed":true}}]}"#,
        ] {
            assert!(Replay::parse(data.as_bytes()).is_err());
        }
        let r=Replay::parse(br#"{"steps":[{"boundaries":2},{"boundaries":3,"tablet":{"raw_x":403,"raw_y":601,"pressed":true}}]}"#).unwrap();
        assert_eq!(r.total_boundaries(), 5);
        assert!(r
            .run_with_observer(&mut CoreRuntime::new(), |_, _| Ok(()))
            .is_err());
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub boundaries: u32,
    pub tablet: Option<Contact>,
    pub contact: Option<MatrixContact>,
    pub on_key: Option<bool>,
    pub label: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    steps: Vec<Step>,
}
impl Replay {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let replay: Self =
            serde_json::from_slice(bytes).map_err(|e| CoreError::Other(e.to_string()))?;
        if replay.steps.is_empty()
            || replay.steps.len() > 4096
            || replay.steps.iter().any(|s| {
                s.boundaries > 10_000_000
                    || s.tablet
                        .as_ref()
                        .is_some_and(|c| c.raw_x > 1023 || c.raw_y > 1023)
                    || s.contact
                        .as_ref()
                        .is_some_and(|c| c.column >= 11 || c.row >= 8)
            })
        {
            return Err(CoreError::Other(
                "PI contact replay/budget out of range".into(),
            ));
        }
        Ok(replay)
    }
    pub fn total_boundaries(&self) -> u64 {
        self.steps.iter().map(|s| u64::from(s.boundaries)).sum()
    }
    pub fn run_with_observer(
        &self,
        runtime: &mut CoreRuntime,
        mut observer: impl FnMut(usize, &CoreRuntime) -> Result<()>,
    ) -> Result<()> {
        if runtime.device_model() != DeviceModel::Pi3000 || runtime.pi3000_hardware().is_none() {
            return Err(CoreError::Other(
                "PI replay requires a PI board runtime".into(),
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            if let Some(c) = &step.tablet {
                runtime
                    .set_pi3000_raw_contact(c.raw_x, c.raw_y, c.pressed)
                    .map_err(CoreError::Other)?;
            }
            if let Some(on) = step.on_key {
                if on {
                    runtime.press_on_key();
                } else {
                    runtime.release_on_key();
                }
            }
            if let Some(c) = &step.contact {
                if !runtime.set_physical_matrix_key(c.column * 8 + c.row, c.pressed) {
                    return Err(CoreError::Other("PI matrix contact is unavailable".into()));
                }
            }
            runtime.step_scheduler_boundaries(step.boundaries as usize)?;
            observer(index, runtime)?;
        }
        Ok(())
    }
}
