#![cfg(target_os = "macos")]

mod common;

use proptest::prelude::*;

proptest! {
    #[test]
    fn parity_proptest(seed in any::<u64>(), cases in 1usize..512) {
        let handle = common::load_parity();
        let rc = unsafe { (handle.run)(seed, cases) };
        prop_assert_eq!(rc, 0, "parity failed for seed={seed:#x} cases={cases}");
    }
}
