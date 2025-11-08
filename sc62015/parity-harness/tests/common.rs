#![cfg(target_os = "macos")]

use libc::{RTLD_GLOBAL, RTLD_NOW};
use libloading::os::unix::{Library as UnixLibrary, Symbol as UnixSymbol};
use std::path::PathBuf;

pub struct ParityHandle {
    #[allow(dead_code)]
    lib: UnixLibrary,
    pub run: UnixSymbol<'static, unsafe extern "C" fn(u64, usize) -> i32>,
}

pub fn load_parity() -> ParityHandle {
    let python_dylib = std::env::var("PYTHON_DYLIB")
        .unwrap_or_else(|_| default_py_dylib().to_string_lossy().into_owned());
    unsafe {
        let _py = UnixLibrary::open(Some(python_dylib.as_str()), RTLD_NOW | RTLD_GLOBAL)
            .unwrap_or_else(|err| panic!("Failed to load libpython at {python_dylib}: {err}"));

        let parity_dylib = std::env::var("SC62015_PARITY_DYLIB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_parity_dylib());
        let parity = UnixLibrary::open(Some(parity_dylib.to_str().unwrap()), RTLD_NOW)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to load parity dylib {}: {err}",
                    parity_dylib.display()
                )
            });

        let run: UnixSymbol<unsafe extern "C" fn(u64, usize) -> i32> = parity
            .get(b"sc62015_run_parity\0")
            .expect("sc62015_run_parity symbol missing");
        ParityHandle { lib: parity, run }
    }
}

fn default_py_dylib() -> PathBuf {
    PathBuf::from("/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib")
}

fn default_parity_dylib() -> PathBuf {
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // parity-harness -> sc62015
    path.push("parity");
    path.push("target");
    path.push(&profile);
    path.push("libsc62015_parity.dylib");
    path
}
