// PY_SOURCE: pce500/oz9600/profile.py
//! Measured device instruction contracts and explicit timing/reset experiments.
use crate::{
    llama::state::{BlockTransferPolicy, ByteArithmeticSourcePolicy, IsrSoftwareWritePolicy},
    CoreError, CoreRuntime, Result,
};

#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProfile {
    #[default]
    Strict,
    Experimental,
    ExperimentalIsrClearOnly,
    ExperimentalIsrMtiWritable,
    /// Measured ISR replacement plus one ONKI latch assertion per physical press.
    /// A firmware-tested hypothesis, not an ESR-P silicon timing contract.
    ExperimentalOnEdge,
    /// ON-edge basis; IMR.IRM/source masks govern hardware IRQ acceptance even
    /// when ROM abandons a prior frame without RETI. ESR-P remains unqualified.
    ExperimentalIrqImr,
    /// Opt-in calendar/cyclic/deadline experiment at the nominal scheduler rate.
    ExperimentalRtc,
    /// Advancing RTC plus the explicit ON-edge and architectural IRQ policies.
    /// Neither composition nor its nominal clock rate qualifies physical wake.
    ExperimentalRtcIrqImr,
    /// Usable v1 composition: provisional initial BP=D0 for empty backing,
    /// advancing RTC, measured ISR replacement and provisional ON/IMR policies. Configuration is
    /// established before execution; no guest state is repaired during boot.
    ProvisionalV1,
    /// V1 basis plus opt-in EXI wake from OFF, inferred from ROM masks/wake code.
    /// Physical ESR-P OFF and gate-array signal behavior remain unqualified.
    ExperimentalExternalWake,
    /// V1 plus the 12 October software MTI/STI recognition contract.
    /// Electrical arrival and HALT/OFF recognition windows remain hypotheses.
    ExperimentalIrqRecognition,
}

impl CoreRuntime {
    /// Device-specific instruction contracts measured through ordinary OZ-707
    /// CALLs on 11 October 2026. These do not qualify reset or IRQ timing.
    pub(crate) fn configure_oz9600_cpu_instruction_contracts(&mut self) {
        self.state
            .set_block_transfer_policy(BlockTransferPolicy::CoupledPredecrement);
        self.state
            .set_byte_arithmetic_source_policy(ByteArithmeticSourcePolicy::LowByte);
        self.state
            .set_isr_software_write_policy(IsrSoftwareWritePolicy::Replace);
    }

    /// Configure before execution. A validated retained image must be restored
    /// first. ProvisionalV1 selects BP=D0 for all-zero SRAM, including a saved
    /// image exported before initialization. Experimental uses the load flag.
    /// ExperimentalIsrClearOnly instead selects the ISR hypothesis with BP=00.
    pub fn configure_oz9600_profile(&mut self, profile: ExecutionProfile) -> Result<()> {
        if self.instruction_count() != 0 || self.cycle_count() != 0 {
            return Err(CoreError::Other(
                "OZ profile must be set before the first CPU boundary".into(),
            ));
        }
        let hardware = self
            .oz9600_hardware
            .clone()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        let experimental = profile != ExecutionProfile::Strict;
        let mut hw = hardware.borrow_mut();
        self.configure_oz9600_cpu_instruction_contracts();
        // Retain the two named, historical ISR counterfactuals. Other profiles
        // use the measured replacement contract rather than the old MTI-only
        // assumption (OZ-9600 physical probes, 11 October 2026).
        self.state.set_isr_software_write_policy(match profile {
            ExecutionProfile::ExperimentalIsrClearOnly => IsrSoftwareWritePolicy::ClearOnly,
            ExecutionProfile::ExperimentalIsrMtiWritable => {
                IsrSoftwareWritePolicy::ClearOnlyExceptMti
            }
            _ => IsrSoftwareWritePolicy::Replace,
        });
        *self.timer = self
            .device_model()
            .timer_profile()
            .new_context(experimental);
        hw.execution.diagnostic_lcc7_halt_main_timer = experimental;
        hw.execution.diagnostic_on_irq_edge_only = matches!(
            profile,
            ExecutionProfile::ExperimentalOnEdge
                | ExecutionProfile::ExperimentalIrqImr
                | ExecutionProfile::ExperimentalRtcIrqImr
                | ExecutionProfile::ProvisionalV1
                | ExecutionProfile::ExperimentalExternalWake
                | ExecutionProfile::ExperimentalIrqRecognition
        );
        hw.execution.diagnostic_irq_imr_only = matches!(
            profile,
            ExecutionProfile::ExperimentalIrqImr
                | ExecutionProfile::ExperimentalRtcIrqImr
                | ExecutionProfile::ProvisionalV1
                | ExecutionProfile::ExperimentalExternalWake
                | ExecutionProfile::ExperimentalIrqRecognition
        );
        self.timer.recognition_delay = profile == ExecutionProfile::ExperimentalIrqRecognition;
        hw.execution.diagnostic_external_off_wake =
            profile == ExecutionProfile::ExperimentalExternalWake;
        let calendar = matches!(
            profile,
            ExecutionProfile::ExperimentalRtc
                | ExecutionProfile::ExperimentalRtcIrqImr
                | ExecutionProfile::ProvisionalV1
                | ExecutionProfile::ExperimentalExternalWake
                | ExecutionProfile::ExperimentalIrqRecognition
        );
        let second_period = self.device_model().timer_profile().timebase_hz;
        hw.execution.diagnostic_rtc_a2_period = if calendar {
            Some(second_period / 2)
        } else {
            experimental.then_some(1_000_000)
        };
        hw.execution.rtc_a2_phase_units = 0;
        hw.execution.rtc_a2_ticks = 0;
        hw.execution.experimental_rtc_second_period = calendar.then_some(second_period);
        hw.execution.rtc_second_phase_units = 0;
        hw.execution.rtc_elapsed_seconds = 0;
        hw.execution.rtc_held_units = 0;
        hw.execution.rtc_timing_units = 0;
        hw.execution.rtc_off_elapsed_units = 0;
        self.memory.write_internal_byte(
            0xec,
            if (matches!(
                profile,
                ExecutionProfile::ProvisionalV1
                    | ExecutionProfile::ExperimentalExternalWake
                    | ExecutionProfile::ExperimentalIrqRecognition
            ) && hw.ram.iter().all(|b| *b == 0))
                || (profile == ExecutionProfile::Experimental && !hw.retained_loaded)
            {
                0xd0
            } else {
                0
            },
        );
        Ok(())
    }

    pub fn oz9600_retained_state(&self) -> Result<Vec<u8>> {
        let hw = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        Ok(super::retained::retained_state(self, hw))
    }

    pub fn restore_oz9600_retained_state(&mut self, image: &[u8]) -> Result<()> {
        let hw = self
            .oz9600_hardware
            .clone()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        // Validate before modifying backing state or the diagnostic seed.
        super::retained::restore_retained_state(self, &hw, image).map_err(CoreError::Other)?;
        self.memory.write_internal_byte(0xec, 0);
        Ok(())
    }

    pub fn set_oz9600_tablet_contact(&mut self, x: u16, y: u16, pressed: bool) -> Result<()> {
        let hw = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        let mut hw = hw.borrow_mut();
        if hw
            .tablet
            .set_contact(x, y, pressed)
            .map_err(CoreError::Other)?
        {
            hw.gate[0x12] |= 2;
        }
        Ok(())
    }
}
