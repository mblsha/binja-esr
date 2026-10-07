// PY_SOURCE: pce500/oz9600/input.py
//! Validated physical contacts with deterministic scheduler-boundary budgets.
use crate::{CoreError, CoreRuntime, DeviceModel, Result};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixContact {
    pub column: u8,
    pub row: u8,
    pub pressed: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabletContact {
    pub raw_x: u16,
    pub raw_y: u16,
    pub pressed: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalStep {
    pub boundaries: u32,
    pub contact: Option<MatrixContact>,
    pub tablet: Option<TabletContact>,
    /// Physical power/ON contact, separate from the keyboard matrix.
    pub on_key: Option<bool>,
    #[serde(default)]
    pub label: Option<String>,
}

pub struct PhysicalReplay {
    steps: Vec<PhysicalStep>,
    total: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalDocument {
    steps: Vec<PhysicalStep>,
}
impl PhysicalReplay {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let steps = serde_json::from_slice::<PhysicalDocument>(bytes)
            .map_err(|e| CoreError::Other(e.to_string()))?
            .steps;
        if steps.is_empty() || steps.len() > 4096 {
            return Err(CoreError::Other(
                "physical replay requires 1-4096 steps".into(),
            ));
        }
        let mut total = 0;
        for step in &steps {
            if step.boundaries > 10_000_000
                || step
                    .contact
                    .as_ref()
                    .is_some_and(|c| c.column >= 11 || c.row >= 8)
                || step
                    .tablet
                    .as_ref()
                    .is_some_and(|c| c.raw_x > 1023 || c.raw_y > 1023)
            {
                return Err(CoreError::Other(
                    "physical replay contact/budget out of range".into(),
                ));
            }
            total += u64::from(step.boundaries);
        }
        Ok(Self { steps, total })
    }
    pub fn total_boundaries(&self) -> u64 {
        self.total
    }
    pub fn run(&self, runtime: &mut CoreRuntime) -> Result<()> {
        self.run_with_observer(runtime, |_, _| Ok(()))
    }
    /// Observe completed steps through read-only state.
    pub fn run_with_observer(
        &self,
        runtime: &mut CoreRuntime,
        mut observer: impl FnMut(usize, &CoreRuntime) -> Result<()>,
    ) -> Result<()> {
        if runtime.device_model() != DeviceModel::Oz9600 || runtime.oz9600_hardware().is_none() {
            return Err(CoreError::Other(
                "OZ physical replay requires an OZ runtime".into(),
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            if let Some(pressed) = step.on_key {
                if pressed {
                    runtime.press_on_key();
                } else {
                    runtime.release_on_key();
                }
            }
            if let Some(c) = &step.contact {
                if !runtime.set_physical_matrix_key(c.column * 8 + c.row, c.pressed) {
                    return Err(CoreError::Other("physical keyboard unavailable".into()));
                }
            }
            if let Some(c) = &step.tablet {
                runtime.set_oz9600_tablet_contact(c.raw_x, c.raw_y, c.pressed)?;
            }
            runtime.step_scheduler_boundaries(step.boundaries as usize)?;
            observer(index, runtime)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_entire_input_before_any_mutation() {
        for input in [
            r#"{"steps":[{"boundaries":0},{"boundaries":1,"contact":{"column":11,"row":0,"pressed":true}}]}"#,
            r#"{"steps":[{"boundaries":1,"tablet":{"raw_x":1024,"raw_y":0,"pressed":true}}]}"#,
            r#"{"steps":[{"boundaries":1,"rtc_causes":{"a":4,"b":0}}]}"#,
            r#"{"steps":[{"boundaries":0,"on_key":1}]}"#,
            r#"{"steps":[{"boundaries":-1}]}"#,
            r#"{"steps":[]}"#,
        ] {
            assert!(PhysicalReplay::parse(input.as_bytes()).is_err());
        }
        let input=PhysicalReplay::parse(br#"{"steps":[{"boundaries":2,"contact":{"column":4,"row":6,"pressed":true}},{"boundaries":3}]}"#).unwrap();
        assert_eq!(input.total_boundaries(), 5);
        assert!(input.run(&mut CoreRuntime::new()).is_err());
    }
}
