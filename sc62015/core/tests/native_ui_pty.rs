// PY_SOURCE: pce500/run_pce500.py
//! Actual terminal-process regression tests using a public synthetic ROM.
//! Every child is killed/reaped on failure; only test-owned pipes use nonblocking I/O.
#![cfg(all(unix, feature = "cli"))]

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

fn nonblocking(fd: i32, enabled: bool) {
    // SAFETY: fd is a live descriptor owned by this test.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert!(flags >= 0);
    let flags = if enabled {
        flags | libc::O_NONBLOCK
    } else {
        flags & !libc::O_NONBLOCK
    };
    assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFL, flags) }, 0);
}

fn termios(file: &File) -> libc::termios {
    let mut value = std::mem::MaybeUninit::uninit();
    // SAFETY: the successful call initializes value; file remains open.
    assert_eq!(
        unsafe { libc::tcgetattr(file.as_raw_fd(), value.as_mut_ptr()) },
        0,
        "tcgetattr: {}",
        io::Error::last_os_error()
    );
    unsafe { value.assume_init() }
}

fn full_pipe() -> (File, File) {
    let mut fds = [0; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let read = unsafe { File::from_raw_fd(fds[0]) };
    let mut write = unsafe { File::from_raw_fd(fds[1]) };
    for fd in fds {
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) },
            0
        );
    }
    nonblocking(write.as_raw_fd(), true);
    let started = Instant::now();
    loop {
        assert!(started.elapsed() < Duration::from_secs(1));
        match write.write(&[b'x'; 4096]) {
            Ok(count) => assert!(count > 0),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            Err(error) => panic!("fill pipe: {error}"),
        }
    }
    // The child deliberately receives a blocking, already-full pipe.
    nonblocking(write.as_raw_fd(), false);
    (read, write)
}

struct TerminalChild {
    child: Child,
    master: File,
    _slave: File,
    original: libc::termios,
}

impl TerminalChild {
    fn spawn(model: &str, stdout: Stdio, stderr: Stdio) -> Self {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../web/emulator-wasm/testdata/pf1_demo_rom_window.rom");
        Self::spawn_rom(model, fixture, stdout, stderr)
    }

    fn spawn_rom(model: &str, rom: PathBuf, stdout: Stdio, stderr: Stdio) -> Self {
        let (mut master, mut slave) = (0, 0);
        let mut size = libc::winsize {
            ws_row: 40,
            ws_col: 180,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: writable fd outputs, no name buffer, valid window size.
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        for fd in [master.as_raw_fd(), slave.as_raw_fd()] {
            assert_eq!(
                unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) },
                0
            );
        }
        let original = termios(&master);
        let mut command = Command::new(env!("CARGO_BIN_EXE_sc62015-lcd"));
        command
            .arg("--rom")
            .arg(rom)
            .args([
                "--model",
                model,
                "--mode",
                "turbo",
                "--force-tty",
                "--no-alt-screen",
                "--iq7000-rtc",
                "202609060000",
            ])
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(stdout)
            .stderr(stderr);
        // SAFETY: only async-signal-safe syscalls in the post-fork child;
        // stdin is the slave terminal installed by Command before this hook.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        Self {
            child,
            master,
            _slave: slave,
            original,
        }
    }

    fn wait_raw(&mut self) {
        let started = Instant::now();
        while termios(&self.master).c_lflag & libc::ICANON != 0 {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "child exited before raw input"
            );
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "raw-mode startup timed out"
            );
            sleep(Duration::from_millis(5));
        }
    }

    fn key(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).unwrap();
    }

    fn wait_exit(&mut self) -> ExitStatus {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "terminal shutdown timed out"
            );
            sleep(Duration::from_millis(5));
        }
    }

    fn assert_restored(&self) {
        let restored = termios(&self.master);
        assert_eq!(restored.c_lflag, self.original.c_lflag);
        assert_eq!(restored.c_iflag, self.original.c_iflag);
        assert_eq!(restored.c_oflag, self.original.c_oflag);
        assert_eq!(restored.c_cflag, self.original.c_cflag);
        assert_eq!(restored.c_cc, self.original.c_cc);
    }

    fn read_output(&mut self, text: &mut String) {
        let output = self.child.stdout.as_mut().unwrap();
        let mut bytes = [0; 8192];
        for _ in 0..128 {
            match output.read(&mut bytes) {
                Ok(0) => break,
                Ok(count) => text.push_str(&String::from_utf8_lossy(&bytes[..count])),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("read terminal: {error}"),
            }
        }
        assert!(
            text.len() < 2_000_000,
            "unbounded terminal output in smoke test"
        );
    }

    fn wait_status(&mut self, marker: &str) -> u64 {
        let started = Instant::now();
        let mut text = String::new();
        loop {
            self.read_output(&mut text);
            if let Some(value) = boundaries(&text, marker).last() {
                return *value;
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "child exited: {text}"
            );
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "missing {marker}: {text}"
            );
            sleep(Duration::from_millis(5));
        }
    }

    fn wait_text(&mut self, marker: &str) {
        let started = Instant::now();
        let mut text = String::new();
        loop {
            self.read_output(&mut text);
            if text.contains(marker) {
                return;
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "child exited: {text}"
            );
            assert!(
                started.elapsed() < Duration::from_secs(15),
                "missing {marker}: {}",
                text.chars().take(8000).collect::<String>()
            );
            sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for TerminalChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn boundaries(text: &str, marker: &str) -> Vec<u64> {
    text.split(marker)
        .skip(1)
        .filter_map(|part| {
            // Require the following space, so a partial read cannot be an ack.
            let number = part.split_once(' ')?.0;
            number.parse().ok()
        })
        .collect()
}

#[test]
fn terminal_pause_ack_freezes_progress_and_resume_advances_for_both_models() {
    for model in ["pc-e500", "iq-7000"] {
        let mut terminal = TerminalChild::spawn(model, Stdio::piped(), Stdio::null());
        nonblocking(terminal.child.stdout.as_ref().unwrap().as_raw_fd(), true);
        terminal.wait_raw();
        terminal.wait_status("RUNNING ack=0 boundaries=");
        terminal.key(&[0x10]); // Ctrl+P
        let paused = terminal.wait_status("PAUSED ack=1 boundaries=");
        let mut heartbeats = String::new();
        let started = Instant::now();
        while started.elapsed() < Duration::from_millis(180) {
            terminal.read_output(&mut heartbeats);
            sleep(Duration::from_millis(5));
        }
        let samples = boundaries(&heartbeats, "PAUSED ack=1 boundaries=");
        assert!(
            samples.len() >= 2,
            "missing paused heartbeats: {heartbeats}"
        );
        assert!(samples.iter().all(|value| *value == paused));
        terminal.key(&[0x10]);
        let resumed = terminal.wait_status("RUNNING ack=2 boundaries=");
        assert!(resumed > paused);
        terminal.key(&[0x03]); // Ctrl+C
        assert!(terminal.wait_exit().success());
        terminal.assert_restored();
    }
}

#[test]
fn full_stdout_and_stderr_cannot_block_quit_or_raw_mode_restoration() {
    for block_stderr in [false, true] {
        let (_stdout_read, stdout_write) = full_pipe();
        let (stderr_read, stderr) = if block_stderr {
            let (read, write) = full_pipe();
            (Some(read), Stdio::from(write))
        } else {
            (None, Stdio::null())
        };
        let mut terminal = TerminalChild::spawn("pc-e500", Stdio::from(stdout_write), stderr);
        terminal.wait_raw();
        terminal.key(&[0x10, 0x03]); // Pause then Quit, with no display consumer.
        let status = terminal.wait_exit();
        assert_eq!(
            status.code(),
            Some(2),
            "must report incomplete output, not pretend success"
        );
        terminal.assert_restored();
        drop(stderr_read);
    }
}

#[test]
#[ignore = "requires private IQ7000_ROM_PATH; explicit real-ROM acceptance"]
fn real_iq_rom_terminal_burst_entry_edit_store_and_reopen() {
    let rom = PathBuf::from(std::env::var_os("IQ7000_ROM_PATH").expect("IQ7000_ROM_PATH"));
    let mut terminal = TerminalChild::spawn_rom("iq-7000", rom, Stdio::piped(), Stdio::inherit());
    nonblocking(terminal.child.stdout.as_ref().unwrap().as_raw_fd(), true);
    terminal.wait_raw();
    terminal.wait_text("2026");
    // Calendar text appears before boot has finished installing its scanner.
    // Match the 500k-boundary ready baseline used by the deterministic ROM
    // browser/Function Runner proof, rather than racing its partial first draw.
    while terminal.wait_status("RUNNING ack=0 boundaries=") < 500_000 {}
    terminal.key(b"\x1bOS"); // F4 = MEMO
    terminal.wait_text("MEMO ?");
    // One terminal read can contain the whole burst. It must not become an
    // impossible simultaneous A+B+C+D+ENTER chord or a translated FIFO write.
    terminal.key(b"ABCD");
    terminal.wait_text("ABCD");
    terminal.key(b"\r\x1bOS");
    terminal.wait_text("MEMO ?");
    terminal.key(b"\x1b[6~"); // Search down: reopen saved record
    terminal.wait_text("ABCD");
    terminal.key(b"\x1b[20~aX"); // SHIFT+A = EDIT, overwrite A with X
    terminal.wait_text("XBCD");
    terminal.key(b"\r\x1bOS"); // Store, then return to an empty search prompt
    terminal.wait_text("MEMO ?");
    terminal.key(b"\x1b[6~");
    terminal.wait_text("XBCD");
    terminal.key(b"\x1bOS");
    terminal.wait_text("MEMO ?");
    // Comma uses the supported SHIFT legend. Unqualified ':' must not silently
    // insert a period; the native status reports it as unmapped.
    terminal.key(b"ABC,:D");
    terminal.wait_text("ABC,D");
    terminal.key(b"\x03");
    assert!(terminal.wait_exit().success());
    terminal.assert_restored();
}

#[test]
#[ignore = "requires private PCE500_ROM_PATH; explicit real-ROM acceptance"]
fn real_pc_rom_terminal_burst_calculator_expression() {
    let rom = PathBuf::from(std::env::var_os("PCE500_ROM_PATH").expect("PCE500_ROM_PATH"));
    let mut terminal = TerminalChild::spawn_rom("pc-e500", rom, Stdio::piped(), Stdio::inherit());
    nonblocking(terminal.child.stdout.as_ref().unwrap().as_raw_fd(), true);
    terminal.wait_raw();
    terminal.wait_text("S2(CARD):NEW CARD");
    terminal.wait_text("pc=0xF175F");
    terminal.key(b"\x1bOP");
    terminal.wait_text("S1(MAIN):NEW CARD");
    // This ROM draws S1 before its scanner has observed the first PF1's
    // release. F175F is the PC after HALT in the keyboard wait routine;
    // F1742 first checks that its remembered held-key slots are empty.
    // Wait for that observable firmware condition, not a host sleep or a
    // larger tap budget. The deterministic ROM test covers the early-key loss.
    terminal.wait_text("pc=0xF175F");
    terminal.key(b"\x1bOP");
    terminal.wait_text("MAIN MENU");
    terminal.key(b"\x1bOQ"); // PF2 = CAL
    terminal.wait_text("0.");
    terminal.key(b"2+2\r");
    terminal.wait_text("4.");
    terminal.key(b"\x03");
    assert!(terminal.wait_exit().success());
    terminal.assert_restored();
}
