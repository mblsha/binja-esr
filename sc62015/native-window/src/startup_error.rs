// PY_SOURCE: pce500/oz9600/startup_recovery.py
//! Host-only startup failure. No machine, guest execution or automatic save.
use crate::{ui, Key, Result};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::PhysicalKey,
    platform::run_on_demand::EventLoopExtRunOnDemand,
    window::{Window, WindowId},
};

static NEXT_BACKUP: AtomicU64 = AtomicU64::new(0);

/// Copy the original bytes even when invalid. Never replace either file.
pub fn backup(path: &Path) -> Result<PathBuf> {
    let bytes = fs::read(path)?;
    let mut name = path
        .file_name()
        .ok_or("backup source needs a filename")?
        .to_os_string();
    name.push(format!(
        ".recovery-{}-{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id(),
        NEXT_BACKUP.fetch_add(1, Ordering::Relaxed)
    ));
    let target = path.with_file_name(name);
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&target)?;
    if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&target);
        return Err(error.into());
    }
    Ok(target)
}

fn lines(error: &str, path: Option<&Path>, status: &str) -> Vec<String> {
    let message = format!("{error}\n\nSaved file: {}\n\nNo replacement image is being saved.\nRestore a known-good backup at this path, then Retry.\nCopy backup preserves the original bytes separately.\n\n{status}",
        path.map_or_else(|| "(none)".to_owned(), |p| p.display().to_string()));
    message
        .lines()
        .flat_map(|line| {
            let chars: Vec<_> = line.chars().collect();
            if chars.is_empty() {
                vec![String::new()]
            } else {
                chars.chunks(62).map(|s| s.iter().collect()).collect()
            }
        })
        .collect()
}

fn action_at(point: Option<(usize, usize)>) -> Option<usize> {
    point.filter(|(_, y)| (470..505).contains(y)).map(|(x, _)| {
        if x < 110 {
            0
        } else if x < 270 {
            1
        } else {
            2
        }
    })
}

struct Failure<'a> {
    error: &'a str,
    path: Option<&'a Path>,
    status: String,
    page: usize,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    position: Option<(usize, usize)>,
    pressed: Option<usize>,
    retry: bool,
    fatal: Option<String>,
    scale: u8,
    started: Instant,
    quit_after_ms: Option<u64>,
}
impl Failure<'_> {
    fn choose(&mut self, action: usize, event_loop: &ActiveEventLoop) {
        match action {
            0 => {
                self.retry = true;
                event_loop.exit();
            }
            1 => {
                self.status = match self.path {
                    Some(path) => match backup(path) {
                        Ok(copy) => format!("Backup copied to:\n{}", copy.display()),
                        Err(error) => format!("Backup failed: {error}"),
                    },
                    None => "No saved-file path was supplied.".into(),
                };
                self.page = lines(self.error, self.path, &self.status)
                    .len()
                    .saturating_sub(26);
            }
            _ => event_loop.exit(),
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
    fn draw(&mut self) -> Result<()> {
        let mut pixels = vec![0xf2f0e4; ui::WIDTH * ui::HEIGHT];
        ui::text(&mut pixels, 12, 14, "OZ-9600 could not start", 0x992c25);
        ui::text(
            &mut pixels,
            12,
            34,
            "Page Up/Down: scroll details",
            0x263325,
        );
        for (index, line) in lines(self.error, self.path, &self.status)
            .iter()
            .skip(self.page)
            .take(26)
            .enumerate()
        {
            ui::text(&mut pixels, 12, 64 + index * 15, line, 0x263325);
        }
        ui::text(
            &mut pixels,
            12,
            486,
            "F5 Retry     F8 Copy backup      Esc Close",
            0x263325,
        );
        ui::text(
            &mut pixels,
            12,
            508,
            "Retry uses the same files and startup profile.",
            0x263325,
        );
        let size = self
            .window
            .as_ref()
            .ok_or("failure window unavailable")?
            .inner_size();
        if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            let surface = self.surface.as_mut().ok_or("failure surface unavailable")?;
            surface.resize(w, h)?;
            let mut buffer = surface.buffer_mut()?;
            for (i, pixel) in buffer.iter_mut().enumerate() {
                *pixel = pixels[(i / size.width as usize * ui::HEIGHT / size.height as usize)
                    * ui::WIDTH
                    + (i % size.width as usize * ui::WIDTH / size.width as usize)];
            }
            buffer.present()?;
        }
        Ok(())
    }
}
impl ApplicationHandler for Failure<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = (|| -> Result<()> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title(
                            "OZ-9600 | startup failed | F5 retry, F8 copy backup, Esc close",
                        )
                        .with_resizable(false)
                        .with_inner_size(LogicalSize::new(
                            ui::WIDTH as f64 * f64::from(self.scale),
                            ui::HEIGHT as f64 * f64::from(self.scale),
                        )),
                )?,
            );
            let context = softbuffer::Context::new(window.clone())?;
            self.surface = Some(softbuffer::Surface::new(&context, window.clone())?);
            self.window = Some(window);
            Ok(())
        })();
        if let Err(error) = result {
            self.fatal = Some(error.to_string());
            event_loop.exit();
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(Key::F5) => self.choose(0, event_loop),
                    PhysicalKey::Code(Key::F8) => self.choose(1, event_loop),
                    PhysicalKey::Code(Key::Escape) => self.choose(2, event_loop),
                    PhysicalKey::Code(Key::PageDown) => {
                        self.page = (self.page + 13).min(
                            lines(self.error, self.path, &self.status)
                                .len()
                                .saturating_sub(26),
                        )
                    }
                    PhysicalKey::Code(Key::PageUp) => self.page = self.page.saturating_sub(13),
                    _ => {}
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let size = self.window.as_ref().expect("event window").inner_size();
                self.position = ui::window_point(
                    position.x,
                    position.y,
                    size.width as usize,
                    size.height as usize,
                );
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                let action = action_at(self.position);
                if state == ElementState::Pressed {
                    self.pressed = action;
                } else if let Some(action) = self.pressed.take().filter(|a| Some(*a) == action) {
                    self.choose(action, event_loop);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw() {
                    self.fatal = Some(error.to_string());
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self
            .quit_after_ms
            .is_some_and(|ms| self.started.elapsed().as_millis() >= u128::from(ms))
        {
            event_loop.exit();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(100),
        ));
    }
}
pub fn show(
    event_loop: &mut EventLoop<()>,
    error: &str,
    path: Option<&Path>,
    scale: u8,
    quit_after_ms: Option<u64>,
) -> Result<bool> {
    let mut app = Failure {
        error,
        path,
        status: String::new(),
        page: 0,
        window: None,
        surface: None,
        position: None,
        pressed: None,
        retry: false,
        fatal: None,
        scale,
        started: Instant::now(),
        quit_after_ms,
    };
    event_loop.run_app_on_demand(&mut app)?;
    if let Some(error) = app.fatal {
        return Err(error.into());
    }
    Ok(app.retry)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn damaged_image_backup_preserves_both_original_and_prior_copy() {
        let dir = std::env::temp_dir().join(format!("oz-startup-copy-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ozbat");
        let invalid = b"invalid saved image\0\xff";
        fs::write(&path, invalid).unwrap();
        let first = backup(&path).unwrap();
        let second = backup(&path).unwrap();
        assert_ne!(first, second);
        for p in [&path, &first, &second] {
            assert_eq!(fs::read(p).unwrap(), invalid);
        }
        assert!(backup(&dir.join("missing")).is_err());
        assert_eq!(fs::read(&path).unwrap(), invalid);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn close_label_left_edge_does_not_copy_a_backup() {
        // The E of Esc is drawn at x=276 in the host recovery row.
        assert_eq!(action_at(Some((277, 490))), Some(2));
        assert_eq!(action_at(Some((130, 490))), Some(1));
        assert_eq!(action_at(Some((24, 490))), Some(0));
        assert_eq!(action_at(Some((277, 450))), None);
    }
    #[test]
    fn long_details_are_scrollable_and_error_remains_explicit() {
        let message = "checksum mismatch";
        let path = PathBuf::from("x".repeat(5000));
        let rows = lines(message, Some(&path), "Copy failed");
        assert!(rows.len() > 26);
        assert!(rows.iter().all(|line| line.chars().count() <= 62));
        assert_eq!(rows[0], message);
        assert_eq!(rows.last().unwrap(), "Copy failed");
    }
}
