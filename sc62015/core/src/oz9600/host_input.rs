// PY_SOURCE: pce500/oz9600/host_input.py
//! Read-only host assistance for the verified OZ-9600 ROM's tablet mailbox.
//! These are firmware queue records, not peripheral registers. Raw replay never
//! calls this helper; the guest continues to enqueue and acknowledge its events.
use super::WORKSPACE_RAM_OFFSET;
use crate::CoreRuntime;
impl CoreRuntime {
    pub fn oz9600_tablet_mailbox_busy(&self) -> Option<bool> {
        let hw = self.oz9600_hardware()?.borrow();
        let byte = |a: u32| hw.ram[WORKSPACE_RAM_OFFSET + (a - 0x10000) as usize];
        let pointer = u32::from(byte(0x1ff47))
            | (u32::from(byte(0x1ff48)) << 8)
            | (u32::from(byte(0x1ff49)) << 16);
        if !(0x10000..=0x1ffe4).contains(&pointer) {
            return Some(false);
        }
        // F0 DA326/DA37D/DA3E7 enqueue C8/C9/CA at these three record slots.
        // Waiting before the next down edge prevents DA341's occupied-slot
        // branch from discarding a later click. Nothing is written here.
        Some([0, 0x0b, 0x1a].iter().any(|o| {
            matches!(
                u16::from(byte(pointer + o)) | (u16::from(byte(pointer + o + 1)) << 8),
                0xc8..=0xca
            )
        }))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_observation_uses_valid_workspace_pointer_and_all_three_records() {
        let fixed = vec![0; 0x20000];
        let rt =
            super::super::configure_hardware(&fixed, super::super::Hardware::default()).unwrap();
        let hw = rt.oz9600_hardware().unwrap();
        assert_eq!(rt.oz9600_tablet_mailbox_busy(), Some(false));
        {
            let mut h = hw.borrow_mut();
            let p = WORKSPACE_RAM_OFFSET + 0xff47;
            h.ram[p..p + 3].copy_from_slice(&[0x49, 0xe5, 1]);
        }
        for (offset, code) in [(0, 0xc8), (0x0b, 0xc9), (0x1a, 0xca)] {
            {
                let mut h = hw.borrow_mut();
                h.ram[WORKSPACE_RAM_OFFSET + 0xe549 + offset] = code;
            }
            assert_eq!(rt.oz9600_tablet_mailbox_busy(), Some(true));
            {
                let mut h = hw.borrow_mut();
                h.ram[WORKSPACE_RAM_OFFSET + 0xe549 + offset] = 0;
            }
            assert_eq!(rt.oz9600_tablet_mailbox_busy(), Some(false));
        }
        {
            let mut h = hw.borrow_mut();
            let p = WORKSPACE_RAM_OFFSET + 0xff47;
            h.ram[p..p + 3].copy_from_slice(&[0xff, 0xff, 1]);
        }
        assert_eq!(rt.oz9600_tablet_mailbox_busy(), Some(false));
    }
}
