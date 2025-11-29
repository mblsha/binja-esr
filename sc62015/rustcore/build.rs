// PY_SOURCE: sc62015/pysc62015/emulator.py:build_extension
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn python_lib_info() -> Option<(PathBuf, String)> {
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(python)
        .arg("-c")
        .arg("import sysconfig;print(sysconfig.get_config_var('LIBDIR'));print(sysconfig.get_config_var('LDLIBRARY'))")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let binding = String::from_utf8_lossy(&output.stdout).to_string();
    let mut lines = binding.lines();
    let libdir = lines.next()?.trim().to_string();
    let ldlib = lines.next()?.trim().to_string();
    Some((PathBuf::from(libdir), ldlib))
}

fn main() {
    // Ensure PyO3 passes the right flags for extension modules.
    pyo3_build_config::add_extension_module_link_args();

    let Some((lib_dir, ldlib)) = python_lib_info() else {
        return;
    };

    // Always prefer the Python-reported library directory.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:warning=python libdir={}, ldlib={}",
        lib_dir.display(),
        ldlib
    );

    let mut link_name = ldlib.trim_start_matches("lib").to_string();
    // Prefer linking against the reported shared library; if it is missing (e.g., only .so.1.0
    // is available), create a local shim symlink to satisfy -lpythonX.Y.
    let target = lib_dir.join(&ldlib);
    if target.exists() {
        if let Some(stripped) = link_name.strip_suffix(".so") {
            link_name = stripped.to_string();
        } else if let Some(stripped) = link_name.strip_suffix(".a") {
            link_name = stripped.to_string();
        }
        if !link_name.is_empty() {
            println!("cargo:rustc-link-lib=dylib={}", link_name);
        }
        return;
    }

    if ldlib.ends_with(".so") {
        let versioned = lib_dir.join(format!("{ldlib}.1.0"));
        if versioned.exists() {
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let shim = out_dir.join(&ldlib);
            let _ = fs::remove_file(&shim);
            #[cfg(unix)]
            {
                let _ = std::os::unix::fs::symlink(&versioned, &shim);
            }
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            if let Some(stripped) = link_name.strip_suffix(".so") {
                link_name = stripped.to_string();
            }
            if !link_name.is_empty() {
                println!("cargo:rustc-link-lib=dylib={}", link_name);
            }
        }
    }
}
