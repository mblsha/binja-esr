// PY_SOURCE: pce500/pvf1/hardware.py
// PY_SOURCE: pce500/oz9600/lcd.py
use super::SharedHardware;
use crate::lcd::{LcdDisplayWrite, LcdHal, LcdStats, LcdWriteTrace};
use crate::lcd_frame::LcdFrame;
use crate::lcd_snapshot::LcdSnapshotMetadata;
use crate::LcdKind;

/// Observes the same controller written by the memory overlays. Address
/// routing remains there so PC/cycle diagnostics and read side effects retain
/// one owner. This view performs no bus operations and holds no pixel copy.
pub(super) struct LcdView(pub SharedHardware);

impl LcdHal for LcdView {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    // A shared typed LH1553F snapshot/model profile has not been introduced.
    fn kind(&self) -> LcdKind {
        LcdKind::Unknown
    }
    fn reset(&mut self) {
        self.0.lock().unwrap().lcd =
            super::LcdController::with_geometry(super::WIDTH, super::HEIGHT);
    }
    fn handles(&self, _address: u32) -> bool {
        false
    }
    fn fixed_windows(&self) -> Option<[(u32, u32); 2]> {
        Some([(1, 0), (1, 0)])
    }
    fn may_handle_span(&self, _start: u32, _end: u32) -> bool {
        false
    }
    fn read(&mut self, _address: u32) -> Option<u8> {
        None
    }
    fn write(&mut self, _address: u32, _value: u8) {
        panic!("PV-F1 LCD is routed by its bus overlay");
    }
    fn read_placeholder(&self, _address: u32) -> u32 {
        panic!("PV-F1 LCD is routed by its bus overlay");
    }
    fn begin_display_write_capture(&mut self) {
        panic!("use the PV-F1 LCD bus observation API");
    }
    fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite> {
        panic!("use the PV-F1 LCD bus observation API");
    }
    fn matrix_frame(&self) -> LcdFrame {
        self.0.lock().unwrap().lcd.matrix_frame()
    }
    fn display_buffer(&self) -> [[u8; 240]; 32] {
        panic!("use matrix_frame for the full PV-F1 LCD");
    }
    fn chip_display_buffer(&self, _chip_index: usize) -> [[u8; 64]; 64] {
        panic!("PV-F1 LCD has no HD61202 chips");
    }
    fn display_vram_bytes(&self) -> [[u8; 240]; 8] {
        panic!("use matrix_frame for the full PV-F1 LCD");
    }
    fn display_trace_buffer(&self) -> [[LcdWriteTrace; 240]; 8] {
        panic!("use the PV-F1 LCD bus observation API");
    }
    fn stats(&self) -> LcdStats {
        LcdStats {
            chip_on: [false; 2],
            instruction_counts: [0; 2],
            data_write_counts: [0; 2],
            cs_both_count: 0,
            cs_left_count: 0,
            cs_right_count: 0,
        }
    }
    fn snapshot_state(&self) -> (LcdSnapshotMetadata, Vec<u8>) {
        panic!("full PV-F1 LCD snapshots are not represented by v4");
    }
    fn restore_state(
        &mut self,
        _metadata: &LcdSnapshotMetadata,
        _payload: &[u8],
    ) -> std::result::Result<(), String> {
        Err("full PV-F1 LCD snapshots are not represented by v4".into())
    }
}
