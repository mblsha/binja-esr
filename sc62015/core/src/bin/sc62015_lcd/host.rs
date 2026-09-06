// PY_SOURCE: pce500/run_pce500.py
//! Terminal I/O is owned here, never by the machine scheduler.

use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableFocusChange, EnableFocusChange, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    },
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use sc62015_core::{
    device::{DeviceTextDecoder, DeviceTextFrame},
    native_ui::{ControlInbox, LatestOutput, OutputStats},
};
use std::{
    fs::File,
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub struct Frame {
    pub lcd: Option<DeviceTextFrame>,
    pub status: String,
    pub extra: Vec<String>,
    pub final_log: Option<String>,
}

// Duplicate handles without changing open-file flags. In particular do NOT
// set O_NONBLOCK on the parent's terminal or hold Rust's global stdout lock.
#[cfg(unix)]
fn output_file(stderr: bool) -> io::Result<File> {
    use std::os::fd::AsFd;
    Ok(if stderr {
        io::stderr().as_fd().try_clone_to_owned()?
    } else {
        io::stdout().as_fd().try_clone_to_owned()?
    }
    .into())
}
#[cfg(windows)]
fn output_file(stderr: bool) -> io::Result<File> {
    use std::os::windows::io::AsHandle;
    Ok(if stderr {
        io::stderr().as_handle().try_clone_to_owned()?
    } else {
        io::stdout().as_handle().try_clone_to_owned()?
    }
    .into())
}

struct Renderer {
    stdout: File,
    stderr: File,
    decoder: Option<DeviceTextDecoder>,
    rows: usize,
    cols: usize,
    tty: bool,
    alt: bool,
    initialized: bool,
    previous: Vec<String>,
    previous_extra: usize,
    previous_size: Option<(u16, u16)>,
    warnings: Vec<String>,
}

impl Renderer {
    fn render(&mut self, frame: Option<Frame>) -> Result<(), Box<dyn std::error::Error>> {
        let Some(frame) = frame else {
            if self.tty && self.initialized {
                crossterm::execute!(self.stdout, Show, DisableFocusChange)?;
                if self.alt {
                    crossterm::execute!(self.stdout, LeaveAlternateScreen)?;
                }
            }
            return Ok(());
        };
        if !self.initialized {
            for line in &self.warnings {
                writeln!(self.stderr, "{line}")?;
            }
            // Restoration must also be attempted after a partial escape write.
            self.initialized = true;
            if self.tty {
                if self.alt {
                    crossterm::execute!(self.stdout, EnterAlternateScreen)?;
                }
                crossterm::execute!(self.stdout, Hide, EnableFocusChange, Clear(ClearType::All))?;
            }
        }
        let lines = match (&self.decoder, &frame.lcd) {
            (Some(decoder), Some(lcd)) => decoder
                .decode_text_frame(lcd)
                .ok_or("LCD frame/decoder model mismatch")?,
            _ => Vec::new(),
        };
        let lines = super::normalize_lines(lines, self.rows, self.cols);
        let size = self.tty.then(|| crossterm::terminal::size().ok()).flatten();
        if self.previous != lines || size != self.previous_size || !self.tty {
            super::render_frame(
                &mut self.stdout,
                &lines,
                &frame.status,
                &frame.extra,
                self.tty,
            )?;
        } else {
            super::render_status_line(
                &mut self.stdout,
                &frame.status,
                self.rows as u16 + 1,
                self.tty,
            )?;
        }
        super::render_extra_lines(
            &mut self.stdout,
            &frame.extra,
            self.rows as u16 + 2,
            self.tty,
            &mut self.previous_extra,
        )?;
        self.previous = lines;
        self.previous_size = size;
        if let Some(log) = frame.final_log {
            writeln!(self.stderr, "{log}")?;
        }
        Ok(())
    }
}

pub struct NativeHost {
    pub controls: Arc<ControlInbox<KeyEvent>>,
    output: LatestOutput<Frame>,
    input_stop: Arc<AtomicBool>,
    input: Option<JoinHandle<()>>,
    raw: bool,
    finished: Option<bool>,
}

impl NativeHost {
    pub fn new(
        decoder: Option<DeviceTextDecoder>,
        geometry: (usize, usize),
        tty: bool,
        alt: bool,
        warnings: Vec<String>,
    ) -> io::Result<Self> {
        let mut renderer = Renderer {
            stdout: output_file(false)?,
            stderr: output_file(true)?,
            decoder,
            rows: geometry.0,
            cols: geometry.1,
            tty,
            alt,
            initialized: false,
            previous: Vec::new(),
            previous_extra: 0,
            previous_size: None,
            warnings,
        };
        let output = LatestOutput::spawn(move |frame| {
            renderer.render(frame).map_err(|error| error.to_string())
        })?;
        let controls = Arc::new(ControlInbox::default());
        let input_stop = Arc::new(AtomicBool::new(false));
        let mut host = Self {
            controls: controls.clone(),
            output,
            input_stop: input_stop.clone(),
            input: None,
            raw: false,
            finished: None,
        };
        if tty {
            crossterm::terminal::enable_raw_mode()?;
            host.raw = true;
            host.input = Some(thread::Builder::new().name("lcd-input".into()).spawn(
                move || {
                    let result = std::panic::catch_unwind(|| -> io::Result<()> {
                        while !input_stop.load(Ordering::Acquire) {
                            if !event::poll(Duration::from_millis(20))? {
                                continue;
                            }
                            match event::read()? {
                                Event::FocusLost => controls.release_all(),
                                Event::Key(key) => {
                                    let control = key.kind == KeyEventKind::Press
                                        && key.modifiers.contains(KeyModifiers::CONTROL);
                                    if control && matches!(key.code, KeyCode::Char('c' | 'C')) {
                                        controls.request_quit();
                                    } else if control
                                        && matches!(key.code, KeyCode::Char('p' | 'P'))
                                    {
                                        controls.toggle_pause();
                                    } else {
                                        controls.push(key);
                                    }
                                }
                                _ => {}
                            }
                        }
                        Ok(())
                    });
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => controls.fail(format!("terminal input failed: {error}")),
                        Err(_) => controls.fail("terminal input worker panicked"),
                    }
                },
            )?);
        }
        Ok(host)
    }
    pub fn publish(&self, frame: Frame) -> bool {
        self.output.publish(frame)
    }
    pub fn stats(&self) -> OutputStats {
        self.output.stats()
    }
    pub fn finish(&mut self) -> bool {
        if let Some(finished) = self.finished {
            return finished;
        }
        self.input_stop.store(true, Ordering::Release);
        let drained = self.output.finish(Duration::from_millis(250));
        let started = Instant::now();
        while self
            .input
            .as_ref()
            .is_some_and(|input| !input.is_finished())
            && started.elapsed() < Duration::from_millis(50)
        {
            thread::sleep(Duration::from_millis(1));
        }
        if self.input.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(input) = self.input.take() {
                let _ = input.join();
            }
        }
        let restored = !self.raw || crossterm::terminal::disable_raw_mode().is_ok();
        self.raw = false;
        let success =
            drained && restored && self.input.is_none() && self.output.stats().error.is_none();
        self.finished = Some(success);
        success
    }
}

impl Drop for NativeHost {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Even error reporting must not hang process termination behind a full pipe.
pub fn report_error_bounded(message: String) {
    let Ok(mut stderr) = output_file(true) else {
        return;
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = thread::Builder::new()
        .name("lcd-error".into())
        .spawn(move || {
            let _ = writeln!(stderr, "{message}");
            let _ = tx.send(());
        });
    let _ = rx.recv_timeout(Duration::from_millis(100));
}
