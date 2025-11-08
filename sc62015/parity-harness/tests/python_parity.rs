#![cfg(target_os = "macos")]

mod common;

#[test]
fn rust_llil_runtime_matches_python() {
    let handle = common::load_parity();
    let rc = unsafe { (handle.run)(0xC0FFEE, 1024) };
    assert_eq!(rc, 0, "Rust/Python parity invocation failed");
}
