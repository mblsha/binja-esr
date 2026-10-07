// PY_SOURCE: pce500/runtime_device.py:BoundaryDevice
//! Peripheral work at each public scheduler boundary, including idle and IRQ
//! boundaries. A boundary may retire the selected IRQ handler instruction;
//! that internal continuation is not a second peripheral boundary.

use crate::{llama::state::LlamaState, memory::MemoryImage, timer::TimerContext, Result};

/// Restricted peripheral access. The owning runtime's timer allocation and
/// CPU executor cannot be replaced, and no recursive step API is exposed.
pub struct BoundaryContext<'a> {
    pub memory: &'a mut MemoryImage,
    pub state: &'a LlamaState,
    pub timer: &'a mut TimerContext,
    pub cycles: u64,
    pub instructions: u64,
    /// Independent scheduler time, including OFF idle; CPU cycles still freeze.
    pub elapsed_timing_units: u64,
}

pub trait BoundaryDevice: Send {
    /// Publish bank views and sample external IRQ before CPU/vector preflight.
    /// None leaves the existing external input unchanged; Some supplies its
    /// physical level through the runtime's ordinary IRQ setter.
    fn before_boundary(&mut self, context: &mut BoundaryContext<'_>) -> Result<Option<bool>>;

    /// Advance peripheral time after a successful boundary. Elapsed cycles
    /// may be zero (OFF) and are uncalibrated native scheduler timing units.
    /// A CPU fault skips this callback. An error here poisons the runtime:
    /// the preceding CPU boundary may already have committed side effects.
    fn after_boundary(&mut self, context: &mut BoundaryContext<'_>, elapsed: u64) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{llama::state::PowerState, CoreError, CoreRuntime};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Probe {
        log: Vec<(&'static str, u64, u64)>,
        timing: Vec<u64>,
        before_error_once: bool,
        after_error_once: bool,
        level: Option<bool>,
        banks: bool,
    }
    struct Device(Arc<Mutex<Probe>>);
    impl BoundaryDevice for Device {
        fn before_boundary(&mut self, c: &mut BoundaryContext<'_>) -> Result<Option<bool>> {
            let mut probe = self.0.lock().unwrap();
            probe.log.push(("before", c.instructions, c.cycles));
            if std::mem::take(&mut probe.before_error_once) {
                return Err(CoreError::Other("before fixture".into()));
            }
            if probe.banks {
                let value = 0x11 + c.instructions as u8;
                c.memory.remove_overlay("test-bank");
                c.memory.add_rom_overlay(
                    0xc0000,
                    &[0x08, value, 0x08, value, 0x08, value],
                    "test-bank",
                );
            }
            Ok(probe.level)
        }
        fn after_boundary(&mut self, c: &mut BoundaryContext<'_>, elapsed: u64) -> Result<()> {
            let mut probe = self.0.lock().unwrap();
            probe.log.push(("after", c.instructions, elapsed));
            probe.timing.push(c.elapsed_timing_units);
            if std::mem::take(&mut probe.after_error_once) {
                return Err(CoreError::Other("after fixture".into()));
            }
            Ok(())
        }
    }
    fn runtime(probe: &Arc<Mutex<Probe>>) -> CoreRuntime {
        let mut rt = CoreRuntime::new();
        rt.lcd = None;
        rt.keyboard = None;
        rt.pce500_peripherals = None;
        rt.sio = None;
        *rt.timer = TimerContext::new(false, 0, 0);
        rt.state.set_pc(0x10000);
        rt.set_reg("S", 0x3000);
        rt.set_reg("U", 0x3800);
        rt.install_boundary_device(Box::new(Device(Arc::clone(probe))))
            .unwrap();
        rt
    }
    #[test]
    fn each_public_boundary_ticks_device_in_running_halt_and_off_states() {
        for power in [PowerState::Running, PowerState::Halted, PowerState::Off] {
            let probe = Arc::new(Mutex::new(Probe::default()));
            let mut rt = runtime(&probe);
            rt.state.set_power_state(power);
            rt.step(0).unwrap();
            assert!(probe.lock().unwrap().log.is_empty());
            rt.step(5).unwrap();
            let p = probe.lock().unwrap();
            assert_eq!(p.log.len(), 10);
            assert_eq!(p.timing, [1, 2, 3, 4, 5]);
            for pair in p.log.as_chunks::<2>().0 {
                assert_eq!(pair[0].0, "before");
                assert_eq!(pair[1].0, "after");
                assert_eq!(pair[1].2, u64::from(power != PowerState::Off));
            }
            assert_eq!(
                rt.instruction_count(),
                if power == PowerState::Running { 5 } else { 0 }
            );
        }
    }
    #[test]
    fn batch_and_cooperative_slice_publish_bank_before_each_opcode() {
        for slice in [false, true] {
            let probe = Arc::new(Mutex::new(Probe {
                banks: true,
                ..Default::default()
            }));
            let mut rt = runtime(&probe);
            rt.state.set_pc(0xc0000);
            if slice {
                rt.run_slice(3, |_| false).unwrap();
            } else {
                rt.step(3).unwrap();
            }
            assert_eq!(rt.get_reg("A"), 0x13);
            assert_eq!(rt.get_reg("PC"), 0xc0006);
            assert_eq!(rt.instruction_count(), 3);
            assert_eq!(probe.lock().unwrap().log.len(), 6);
        }
    }
    #[test]
    fn irq_internal_handler_continuation_does_not_tick_device_twice() {
        let probe = Arc::new(Mutex::new(Probe {
            level: Some(true),
            ..Default::default()
        }));
        let mut rt = runtime(&probe);
        rt.load_rom(&[0, 0, 0x0f], crate::INTERRUPT_VECTOR_ADDR as usize);
        rt.load_rom(&[0x08, 0x59], 0xf0000); // handler's first instruction
        rt.memory
            .write_internal_byte(crate::memory::IMEM_IMR_OFFSET, 0xc0); // IRM | EXM
        rt.step(1).unwrap();
        assert_eq!(rt.timer.irq_total, 1);
        assert_eq!(rt.instruction_count(), 1);
        assert_eq!(rt.get_reg("PC"), 0xf0002);
        assert_eq!(rt.get_reg("A"), 0x59);
        assert_eq!(probe.lock().unwrap().log.len(), 2);
    }
    #[test]
    fn pre_boundary_error_keeps_device_and_consumes_no_cpu_boundary() {
        let probe = Arc::new(Mutex::new(Probe {
            before_error_once: true,
            ..Default::default()
        }));
        let mut rt = runtime(&probe);
        assert!(rt
            .step(2)
            .unwrap_err()
            .to_string()
            .contains("before fixture"));
        assert_eq!((rt.instruction_count(), rt.cycle_count()), (0, 0));
        rt.step(1).unwrap();
        assert_eq!(probe.lock().unwrap().log.len(), 3);
        assert_eq!(rt.instruction_count(), 1);
    }
    #[test]
    fn cpu_decode_fault_skips_after_callback_and_preserves_device() {
        let probe = Arc::new(Mutex::new(Probe::default()));
        let mut rt = runtime(&probe);
        rt.memory.write_external_slice(0x10000, &[0x20]); // quarantined TCL opcode
        assert!(rt.step(1).is_err());
        assert_eq!(rt.instruction_count(), 0);
        assert_eq!(probe.lock().unwrap().log.len(), 1);
        rt.memory.write_external_slice(0x10000, &[0]);
        rt.step(1).unwrap();
        assert_eq!(probe.lock().unwrap().log.len(), 3);
    }
    #[test]
    fn after_fault_poisons_committed_progress_and_retains_device_until_reset() {
        let probe = Arc::new(Mutex::new(Probe {
            after_error_once: true,
            ..Default::default()
        }));
        let mut rt = runtime(&probe);
        assert!(rt
            .step(2)
            .unwrap_err()
            .to_string()
            .contains("after fixture"));
        assert_eq!(rt.instruction_count(), 1);
        assert!(rt.step(1).unwrap_err().to_string().contains("poisoned"));
        assert_eq!(probe.lock().unwrap().log.len(), 2);
        rt.load_rom(&[0, 0, 1], crate::pce500::ROM_RESET_VECTOR_ADDR as usize);
        rt.power_on_reset().unwrap();
        rt.step(1).unwrap();
        assert_eq!(probe.lock().unwrap().log.len(), 4);
    }
    #[test]
    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    fn snapshot_v4_rejects_unrepresented_boundary_state_before_creating_file() {
        let probe = Arc::new(Mutex::new(Probe::default()));
        let rt = runtime(&probe);
        let path = std::env::temp_dir().join("core-boundary-device-reject.pcsnap");
        let _ = std::fs::remove_file(&path);
        assert!(rt
            .save_snapshot(&path)
            .unwrap_err()
            .to_string()
            .contains("boundary device state"));
        assert!(!path.exists());
        assert!(probe.lock().unwrap().log.is_empty());
    }
}
