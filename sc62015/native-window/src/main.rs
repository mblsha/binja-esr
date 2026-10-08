// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/oz9600/ui.py
//! Native host frontend for the ordinary public OZ machine factory.
mod audio;
mod retained_store;
mod ui;
use clap::Parser;
use retained_store::{atomic_write, RetainedStore};
use sc62015_core::{
    oz9600::{input::PhysicalReplay, ExecutionProfile},
    pacing::{ExecutionMode, Pacer},
    physical_keys::matrix_key,
    CoreRuntime, DeviceModel,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use ui::{Action, Contacts, Event, Target};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode as Key, PhysicalKey},
    window::{Window, WindowId},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(
    about = "Live OZ-9600 controller bitmap; strict defaults, explicit experimental profiles"
)]
struct Args {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, value_enum, default_value = "strict")]
    profile: ExecutionProfile,
    #[arg(long)]
    retained: Option<PathBuf>,
    #[arg(long)]
    retained_out: Option<PathBuf>,
    /// Load existing RAM/RTC and atomically save every five seconds and on exit.
    /// An invalid existing image rejects startup without overwriting it.
    #[arg(long, conflicts_with = "retained")]
    state: Option<PathBuf>,
    /// Verified OZ-707 ROM; explicit provisional logical cartridge view.
    #[arg(long, requires = "card_sram")]
    card_rom: Option<PathBuf>,
    #[arg(long, requires = "card_rom")]
    card_sram: Option<PathBuf>,
    #[arg(long, requires = "card_rom")]
    card_sram_out: Option<PathBuf>,
    /// Normal physical-input replay before opening the window.
    #[arg(long)]
    replay: Option<PathBuf>,
    #[arg(long)]
    replay_report: Option<PathBuf>,
    /// Save PBM, host-window PPM and read-only state report (F7 also captures).
    #[arg(long)]
    capture_prefix: Option<PathBuf>,
    /// Record accepted live physical contacts with exact boundary budgets.
    #[arg(long)]
    event_log: Option<PathBuf>,
    /// Read-only host input diagnostics, updated at most five times/second.
    #[arg(long)]
    host_status: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "interactive")]
    execution_mode: ExecutionMode,
    #[arg(long)]
    paused: bool,
    /// Enable nominal PCM playback; replay/headless execution stays silent.
    #[arg(long, conflicts_with = "headless")]
    sound: bool,
    /// Host click assistance; set zero for immediate contact release.
    #[arg(long,default_value_t=40_000,value_parser=clap::value_parser!(u32).range(0..=10_000_000))]
    minimum_contact_boundaries: u32,
    #[arg(long,default_value_t=2,value_parser=clap::value_parser!(u8).range(1..=2))]
    scale: u8,
    /// Observe the same factory/replay without creating an OS window.
    #[arg(long, requires = "replay")]
    headless: bool,
    #[arg(long)]
    quit_after_ms: Option<u64>,
}
fn machine(
    rom: &[u8],
    retained: Option<&[u8]>,
    profile: ExecutionProfile,
    card: Option<(&[u8], &[u8])>,
) -> Result<CoreRuntime> {
    let mut rt = CoreRuntime::for_model(DeviceModel::Oz9600, rom)?;
    if let Some(bytes) = retained {
        rt.restore_oz9600_retained_state(bytes)?;
    }
    rt.configure_oz9600_profile(profile)?;
    if let Some((rom, sram)) = card {
        rt.install_oz9600_oz707_card(rom, sram)?;
    }
    Ok(rt)
}
fn observation(rt: &CoreRuntime) -> Value {
    let frame = rt.lcd.as_deref().expect("factory LCD").matrix_frame();
    let hw = rt.oz9600_hardware().expect("factory OZ hardware").borrow();
    let mut state = json!({
        "pc":rt.state.pc(),"instructions":rt.instruction_count(),"cycles":rt.cycle_count(),
        "cpu_halted":rt.state.is_halted(),"irq_total":rt.timer.irq_total,"selector":hw.selector,
        "registers":(["BA","I","X","Y","U","S","F"].iter().map(|name|(*name,rt.get_reg(name))).collect::<BTreeMap<_,_>>()),
        "gate_registers":hw.gate.to_vec(),"rtc_registers":hw.rtc.registers(),
        "lcd_registers":hw.lcd.registers.to_vec(),
        "lcd_counters":[hw.lcd.data_writes,hw.lcd.data_reads,hw.lcd.block_operations],
        "lcd_windows":(0..16).map(|n|hw.lcd.window_descriptor(n)).collect::<Vec<_>>(),
        "tablet":[u64::from(hw.tablet.x),u64::from(hw.tablet.y),u64::from(hw.tablet.pressed),u64::from(hw.tablet.conversion_control),u64::from(hw.tablet.drive_control),hw.tablet.data_reads],
        "pbm_sha256":format!("{:x}",Sha256::digest(frame.pbm())),
    });
    if let Some(progress) = hw.execution.rtc_progression_report() {
        state["rtc_progression"] = progress;
    }
    state
}
fn sibling(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", prefix.display()))
}
fn capture(
    rt: &CoreRuntime,
    prefix: &Path,
    contacts: &Contacts,
    paused: bool,
    sound: bool,
) -> Result<()> {
    let frame = rt
        .lcd
        .as_deref()
        .ok_or("factory LCD missing")?
        .matrix_frame();
    let pixels = ui::render(
        &frame,
        &contacts.pressed_keys(),
        paused,
        contacts.pressed_on(),
        sound,
    )?;
    let mut ppm = format!("P6\n{} {}\n255\n", ui::WIDTH, ui::HEIGHT).into_bytes();
    for p in pixels {
        ppm.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    fs::write(sibling(prefix, ".pbm"), frame.pbm())?;
    fs::write(sibling(prefix, "-window.ppm"), ppm)?;
    fs::write(
        sibling(prefix, "-state.json"),
        serde_json::to_vec_pretty(&observation(rt))?,
    )?;
    Ok(())
}
fn host_key(key: Key) -> Option<u8> {
    let name = match key {
        Key::KeyA => "A",
        Key::KeyB => "B",
        Key::KeyC => "C",
        Key::KeyD => "D",
        Key::KeyE => "E",
        Key::KeyF => "F",
        Key::KeyG => "G",
        Key::KeyH => "H",
        Key::KeyI => "I",
        Key::KeyJ => "J",
        Key::KeyK => "K",
        Key::KeyL => "L",
        Key::KeyM => "M",
        Key::KeyN => "N",
        Key::KeyO => "O",
        Key::KeyP => "P",
        Key::KeyQ => "Q",
        Key::KeyR => "R",
        Key::KeyS => "S",
        Key::KeyT => "T",
        Key::KeyU => "U",
        Key::KeyV => "V",
        Key::KeyW => "W",
        Key::KeyX => "X",
        Key::KeyY => "Y",
        Key::KeyZ => "Z",
        Key::Digit0 | Key::Numpad0 => "0",
        Key::Digit1 | Key::Numpad1 => "1",
        Key::Digit2 | Key::Numpad2 => "2",
        Key::Digit3 | Key::Numpad3 => "3",
        Key::Digit4 | Key::Numpad4 => "4",
        Key::Digit5 | Key::Numpad5 => "5",
        Key::Digit6 | Key::Numpad6 => "6",
        Key::Digit7 | Key::Numpad7 => "7",
        Key::Digit8 | Key::Numpad8 => "8",
        Key::Digit9 | Key::Numpad9 => "9",
        Key::Space => "SPACE",
        Key::Enter | Key::NumpadEnter => "ENTER",
        Key::Backspace => "BS",
        Key::Delete => "DEL",
        Key::Insert => "INS",
        Key::Escape => "CANCEL",
        Key::CapsLock => "CAPS",
        Key::ShiftLeft | Key::ShiftRight => "SHIFT",
        Key::AltLeft | Key::AltRight => "2ND",
        Key::ArrowUp => "UP",
        Key::ArrowDown => "DOWN",
        Key::ArrowLeft => "LEFT",
        Key::ArrowRight => "RIGHT",
        Key::PageUp => "PREV",
        Key::PageDown => "NEXT",
        Key::Comma => ",",
        Key::Period | Key::NumpadDecimal => ".",
        Key::Minus | Key::NumpadSubtract => "-",
        Key::Equal => "=",
        Key::Slash | Key::NumpadDivide => "/",
        Key::NumpadAdd => "+",
        Key::NumpadMultiply => "*",
        Key::ContextMenu | Key::F3 => "MENU",
        Key::F1 => "NEW ENTRY",
        Key::F2 => "EDIT",
        Key::F4 => "2ND",
        Key::F6 => "SYMBOL",
        Key::Pause => "OFF",
        _ => return None,
    };
    matrix_key(DeviceModel::Oz9600, name)
}
#[derive(Default)]
struct HostKeys {
    down: BTreeSet<Key>,
}
impl HostKeys {
    fn events(&mut self, events: Vec<(Key, bool)>) -> BTreeSet<Key> {
        let mut presses = BTreeSet::new();
        for (key, pressed) in events {
            if pressed {
                if self.down.insert(key) {
                    presses.insert(key);
                }
            } else {
                self.down.remove(&key);
            }
        }
        presses
    }
}
fn host_on(keys: &HostKeys, pointer: Option<Target>) -> bool {
    keys.down.contains(&Key::F12) || pointer == Some(Target::On)
}
struct Recorder {
    steps: Vec<Value>,
    last: u64,
    enabled: bool,
}
impl Recorder {
    fn wait_to(&mut self, at: u64) {
        let mut remaining = at - self.last;
        while remaining > 0 {
            let count = remaining.min(10_000_000);
            self.steps.push(json!({"boundaries":count}));
            remaining -= count;
        }
        self.last = at;
    }
    fn contact(&mut self, event: &Event, at: u64) {
        if !self.enabled {
            return;
        }
        self.wait_to(at);
        self.steps.push(match event {
            Event::Matrix(code, pressed) => {
                json!({"boundaries":0,"contact":{"column":code/8,"row":code%8,"pressed":pressed}})
            }
            Event::Tablet(x, y, pressed) => {
                json!({"boundaries":0,"tablet":{"raw_x":x,"raw_y":y,"pressed":pressed}})
            }
            Event::On(pressed) => json!({"boundaries":0,"on_key":pressed}),
        });
    }
    fn save(&mut self, path: &Path, at: u64) -> Result<()> {
        self.wait_to(at);
        if self.steps.is_empty() {
            self.steps.push(json!({"boundaries":0}));
        }
        let bytes = serde_json::to_vec_pretty(&json!({"steps":self.steps}))?;
        PhysicalReplay::parse(&bytes)?;
        fs::write(path, bytes)?;
        Ok(())
    }
}
fn apply(rt: &mut CoreRuntime, events: Vec<Event>, record: &mut Recorder, at: u64) -> Result<()> {
    for event in events {
        match event {
            Event::Matrix(code, pressed) => {
                if !rt.set_physical_matrix_key(code, pressed) {
                    return Err("physical matrix unavailable".into());
                }
            }
            Event::Tablet(x, y, pressed) => rt.set_oz9600_tablet_contact(x, y, pressed)?,
            Event::On(true) => rt.press_on_key(),
            Event::On(false) => rt.release_on_key(),
        }
        record.contact(&event, at);
    }
    Ok(())
}
fn run(args: Args) -> Result<()> {
    let rom = fs::read(&args.rom)?;
    let (mut state_store, retained) = if let Some(path) = &args.state {
        let (store, saved) = RetainedStore::open(path)?;
        (Some(store), saved)
    } else {
        (None, args.retained.as_ref().map(fs::read).transpose()?)
    };
    let card = args
        .card_rom
        .as_ref()
        .map(|p| -> Result<(Vec<u8>, Vec<u8>)> {
            Ok((
                fs::read(p)?,
                fs::read(args.card_sram.as_ref().expect("paired card arguments"))?,
            ))
        })
        .transpose()?;
    // Parse the WHOLE replay before constructing/running the machine.
    let replay = args
        .replay
        .as_ref()
        .map(|p| -> Result<PhysicalReplay> { Ok(PhysicalReplay::parse(&fs::read(p)?)?) })
        .transpose()?;
    let mut rt = machine(
        &rom,
        retained.as_deref(),
        args.profile,
        card.as_ref().map(|(r, s)| (r.as_slice(), s.as_slice())),
    )?;
    eprintln!("profile={:?}; 336x240 full controller image; clocks, mapper/RAM aliases and physical crop remain provisional",args.profile);
    let mut boundaries = 0;
    if let Some(replay) = replay {
        let mut reports = Vec::new();
        replay.run_with_observer(&mut rt, |step, rt| {
            let mut value = observation(rt);
            value["step"] = json!(step);
            reports.push(value);
            Ok(())
        })?;
        boundaries = replay.total_boundaries();
        if let Some(path) = &args.replay_report {
            fs::write(path, serde_json::to_vec_pretty(&reports)?)?;
        }
    }
    let mut contacts = Contacts::new(u64::from(args.minimum_contact_boundaries));
    if !args.headless {
        let result = live(
            &args,
            &rom,
            &mut rt,
            &mut contacts,
            boundaries,
            state_store.as_mut(),
        );
        // Also preserve backing when the window exits with a host error.
        if let Some(store) = &mut state_store {
            store.save(&rt.oz9600_retained_state()?)?;
        }
        result?;
    } else if let Some(store) = &mut state_store {
        store.save(&rt.oz9600_retained_state()?)?;
    }
    if let Some(path) = &args.capture_prefix {
        capture(&rt, path, &contacts, true, false)?;
    }
    if let Some(path) = &args.retained_out {
        atomic_write(path, &rt.oz9600_retained_state()?)?;
    }
    if let Some(path) = &args.card_sram_out {
        fs::write(path, rt.oz9600_card_sram()?.ok_or("OZ card unavailable")?)?;
    }
    Ok(())
}

struct NativeApp<'a> {
    args: &'a Args,
    rom: &'a [u8],
    rt: &'a mut CoreRuntime,
    contacts: &'a mut Contacts,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    boundaries: u64,
    record: Recorder,
    host_keys: HostKeys,
    pointer: Option<Target>,
    position: Option<(usize, usize)>,
    lcd_drag: bool,
    mouse: bool,
    blocked_mouse: bool,
    active: bool,
    closing: bool,
    paused: bool,
    pacer: Pacer,
    pending: usize,
    started: Instant,
    next_frame: Instant,
    last_host_status: Instant,
    state_store: Option<&'a mut RetainedStore>,
    next_save: Instant,
    save_error: Option<String>,
    fault: Option<String>,
    fatal: Option<String>,
    status: String,
    sound: bool,
    audio_capture: bool,
    audio_output: Option<audio::Output>,
}
impl NativeApp<'_> {
    fn save_state(&mut self) {
        if let Some(store) = &mut self.state_store {
            match self
                .rt
                .oz9600_retained_state()
                .map_err(|e| e.to_string())
                .and_then(|bytes| store.save(&bytes).map_err(|e| e.to_string()))
            {
                Ok(true) => {
                    self.status = format!("Saved {}", store.path().display());
                    self.save_error = None;
                }
                Ok(false) => {
                    self.save_error = None;
                }
                Err(e) => self.save_error = Some(format!("Autosave failed: {e}")),
            }
        }
        self.next_save = Instant::now() + Duration::from_secs(5);
    }
    fn refresh_audio(&mut self) -> Result<()> {
        if let Some(error) = self.audio_output.as_ref().and_then(audio::Output::error) {
            self.status = format!("Sound unavailable: {error}; emulation continues");
            self.sound = false;
            self.audio_output = None;
        }
        let enabled = self.sound
            && self.active
            && !self.paused
            && self.pending == 0
            && self.fault.is_none()
            && self.args.execution_mode == ExecutionMode::Interactive;
        if enabled != self.audio_capture {
            self.rt.set_oz9600_audio_enabled(enabled)?;
            self.audio_capture = enabled;
        }
        if !enabled {
            if let Some(output) = &self.audio_output {
                output.clear();
            }
        }
        Ok(())
    }
    fn toggle_sound(&mut self) -> Result<()> {
        if self.sound {
            self.sound = false;
            self.audio_output = None;
            self.status = "Sound off".into();
        } else if self.args.execution_mode != ExecutionMode::Interactive {
            self.status = "Sound is available during interactive execution".into();
        } else {
            match audio::Output::open() {
                Ok(output) => {
                    self.audio_output = Some(output);
                    self.sound = true;
                    self.status = "Sound on; nominal digital PCM".into();
                }
                Err(error) => {
                    self.status = format!("Sound unavailable: {error}; emulation continues");
                }
            }
        }
        self.refresh_audio()
    }
    fn sync(&mut self) -> Result<()> {
        if self.fault.is_some() {
            self.pointer = None;
            self.lcd_drag = false;
            return apply(
                self.rt,
                self.contacts.cancel(),
                &mut self.record,
                self.boundaries,
            );
        }
        let mut keys = self
            .host_keys
            .down
            .iter()
            .copied()
            .filter_map(host_key)
            .collect::<BTreeSet<_>>();
        if let Some(Target::Matrix(code)) = self.pointer {
            keys.insert(code);
        }
        let tablet = if let Some(Target::Tablet(x, y)) = self.pointer {
            Some((x, y))
        } else {
            None
        };
        apply(
            self.rt,
            self.contacts.sync(
                &keys,
                tablet,
                self.boundaries,
                host_on(&self.host_keys, self.pointer),
            ),
            &mut self.record,
            self.boundaries,
        )
    }
    fn cancel(&mut self) -> Result<()> {
        self.host_keys.down.clear();
        self.pointer = None;
        self.lcd_drag = false;
        apply(
            self.rt,
            self.contacts.cancel(),
            &mut self.record,
            self.boundaries,
        )
    }
    fn prepare_close(&mut self) -> Result<()> {
        // Exit requests may leave a redraw queued. Seal this epoch before
        // saving its trace so later events cannot advance or re-enable sound.
        self.closing = true;
        self.paused = true;
        self.pending = 0;
        self.sound = false;
        self.audio_output = None;
        self.refresh_audio()?;
        self.cancel()?;
        if let Some(path) = &self.args.event_log {
            self.record.save(path, self.boundaries)?;
        }
        Ok(())
    }
    fn close(&mut self, event_loop: &ActiveEventLoop) {
        if self.closing {
            return;
        }
        if let Err(e) = self.prepare_close() {
            self.fatal = Some(e.to_string());
        }
        event_loop.exit();
    }
    fn action(&mut self, action: Action) -> Result<()> {
        if self.closing {
            return Ok(());
        }
        match action {
            Action::Sound => self.toggle_sound()?,
            Action::RunPause if self.fault.is_none() => {
                self.pending = 0;
                if self.args.execution_mode == ExecutionMode::Deterministic {
                    self.status = "Deterministic mode: use a STEP budget".into();
                } else {
                    self.paused = !self.paused;
                    self.pacer.rebase();
                    if self.paused {
                        self.save_state();
                    }
                }
            }
            Action::Step | Action::Wait if self.fault.is_none() => {
                self.paused = true;
                self.pacer.rebase();
                self.pending = self.pending.saturating_add(if action == Action::Step {
                    20_000
                } else {
                    1_000_000
                });
            }
            Action::Reset if self.args.event_log.is_some() => {
                self.status =
                    "Finish this recording before resetting; each replay has one reset epoch"
                        .into();
            }
            Action::Reset => {
                if let Some(output) = &self.audio_output {
                    output.clear();
                }
                self.audio_capture = false;
                self.cancel()?;
                self.save_state();
                let saved = self.rt.oz9600_retained_state()?;
                let card = self
                    .rt
                    .oz9600_hardware()
                    .unwrap()
                    .borrow()
                    .card
                    .as_ref()
                    .map(|c| (c.rom().to_vec(), c.sram().to_vec()));
                *self.rt = machine(
                    self.rom,
                    Some(&saved),
                    self.args.profile,
                    card.as_ref().map(|(r, s)| (r.as_slice(), s.as_slice())),
                )?;
                self.boundaries = 0;
                self.record = Recorder {
                    steps: Vec::new(),
                    last: 0,
                    enabled: false,
                };
                self.pending = 0;
                self.blocked_mouse = self.mouse;
                self.paused = true;
                self.fault = None;
                self.pacer.rebase();
                self.status = "Reset preserved logical retained memory; paused".into();
            }
            Action::Save => {
                self.save_state();
                let mut saved = Vec::new();
                if let Some(store) = &self.state_store {
                    saved.push(store.path().display().to_string());
                }
                if let Some(path) = &self.args.retained_out {
                    atomic_write(path, &self.rt.oz9600_retained_state()?)?;
                    saved.push(path.display().to_string());
                }
                if let Some(path) = &self.args.card_sram_out {
                    fs::write(
                        path,
                        self.rt.oz9600_card_sram()?.ok_or("OZ card unavailable")?,
                    )?;
                    saved.push(path.display().to_string());
                }
                self.status = if saved.is_empty() {
                    "Set --state, --retained-out or --card-sram-out to enable SAVE".into()
                } else {
                    format!("Saved {}", saved.join(", "))
                };
            }
            Action::Capture => {
                self.status = if let Some(path) = &self.args.capture_prefix {
                    capture(self.rt, path, self.contacts, self.paused, self.sound)?;
                    format!("Captured {}", path.display())
                } else {
                    "Set --capture-prefix to enable CAPTURE".into()
                };
            }
            _ => {}
        }
        self.refresh_audio()
    }
    fn tick(&mut self) -> Result<()> {
        if self.closing {
            return Ok(());
        }
        self.sync()?;
        self.refresh_audio()?;
        let deadline = Instant::now() + Duration::from_millis(4);
        let result = if self.pending > 0 && self.fault.is_none() {
            self.rt
                .run_slice(self.pending.min(4096), |_| Instant::now() >= deadline)
                .map(|slice| {
                    self.pending -= slice.progress.boundary_budget_used;
                    self.boundaries += slice.progress.boundary_budget_used as u64;
                })
        } else if !self.paused && self.fault.is_none() {
            self.rt
                .run_automatic_slice(
                    &mut self.pacer,
                    self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    4096,
                    |_| Instant::now() >= deadline,
                )
                .map(|result| {
                    if let Some(slice) = result.slice {
                        self.boundaries += slice.progress.boundary_budget_used as u64;
                    }
                })
        } else {
            Ok(())
        };
        if let Err(e) = result {
            let error = e.to_string();
            self.status = error.clone();
            self.fault = Some(error);
            self.paused = true;
            self.pending = 0;
            self.cancel()?;
        }
        if Instant::now() >= self.next_save {
            self.save_state();
        }
        self.refresh_audio()?;
        if self.audio_capture {
            let chunk = self.rt.take_oz9600_audio()?;
            if let Some(output) = &self.audio_output {
                output.push(&chunk);
            }
        }
        let window = self.window.as_ref().ok_or("native window unavailable")?;
        let mode = if self.pending > 0 {
            "stepping"
        } else if self.paused {
            "paused"
        } else {
            self.args.execution_mode.label()
        };
        window.set_title(&ui::window_title(
            &format!("{:?}", self.args.profile),
            mode,
            self.save_error.as_deref().unwrap_or(&self.status),
            self.fault.as_deref(),
        ));
        let frame = self
            .rt
            .lcd
            .as_deref()
            .ok_or("factory LCD missing")?
            .matrix_frame();
        let pixels = ui::render(
            &frame,
            &self.contacts.pressed_keys(),
            self.paused,
            self.contacts.pressed_on(),
            self.sound,
        )?;
        let size = window.inner_size();
        if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            let surface = self.surface.as_mut().ok_or("native surface unavailable")?;
            surface.resize(w, h)?;
            let mut buffer = surface.buffer_mut()?;
            for (i, pixel) in buffer.iter_mut().enumerate() {
                let x = i % size.width as usize;
                let y = i / size.width as usize;
                *pixel = pixels[(y * ui::HEIGHT / size.height as usize) * ui::WIDTH
                    + x * ui::WIDTH / size.width as usize];
            }
            buffer.present()?;
        }
        if let Some(path) = &self.args.host_status {
            if self.last_host_status.elapsed() >= Duration::from_millis(200) {
                fs::write(
                    path,
                    serde_json::to_vec_pretty(&json!({
                        "active":self.active,"mouse":self.mouse,"position":self.position,
                        "window_size":[size.width,size.height],"keys":format!("{:?}",self.host_keys.down),
                        "contacts":self.contacts.pressed_keys(),"pending_boundaries":self.pending,
                        "on_contact":self.contacts.pressed_on(),"cpu_off":self.rt.state.is_off(),
                        "boundaries":self.boundaries,"status":self.status,
                        "sound_requested":self.sound,"audio_capture":self.audio_capture,
                        "audio":self.audio_output.as_ref().map(audio::Output::status),
                    }))?,
                )?;
                self.last_host_status = Instant::now();
            }
        }
        Ok(())
    }
    fn focus_changed(&mut self, active: bool) -> Result<()> {
        self.active = active;
        if !active {
            self.cancel()?;
            self.blocked_mouse = self.mouse;
        }
        Ok(())
    }
    fn mouse_changed(&mut self, state: ElementState) -> Result<()> {
        self.mouse = state == ElementState::Pressed;
        if !self.mouse {
            self.blocked_mouse = false;
            self.pointer = None;
            self.lcd_drag = false;
            self.sync()?;
        } else if !self.active {
            // A press that starts in the background must not resume on focus.
            self.blocked_mouse = true;
        } else if !self.blocked_mouse {
            let target = self.position.and_then(|(x, y)| ui::hit(x, y));
            if let Some(Target::Control(a)) = target {
                self.pointer = None;
                self.lcd_drag = false;
                self.action(a)?;
            } else if self.fault.is_none() {
                self.pointer = target;
                self.lcd_drag = self.position.is_some_and(|(x, y)| ui::LCD.contains(x, y));
                self.sync()?;
            }
        }
        Ok(())
    }
    fn event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        match event {
            WindowEvent::CloseRequested => self.close(event_loop),
            WindowEvent::Focused(active) => {
                self.focus_changed(active)?;
                self.refresh_audio()?;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let size = self
                    .window
                    .as_ref()
                    .ok_or("native window unavailable")?
                    .inner_size();
                self.position = ui::window_point(
                    position.x,
                    position.y,
                    size.width as usize,
                    size.height as usize,
                );
                if self.lcd_drag {
                    if let Some((x, y)) = self.position {
                        let (x, y) = ui::lcd_tablet(
                            x.saturating_sub(ui::LCD.x),
                            y.saturating_sub(ui::LCD.y),
                        );
                        self.pointer = Some(Target::Tablet(x, y));
                        self.sync()?;
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.position = None;
                self.pointer = None;
                self.lcd_drag = false;
                self.sync()?;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_changed(state)?;
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } if self.active => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    let edges = self.host_keys.events(vec![(key, pressed)]);
                    if self.fault.is_none() {
                        self.sync()?;
                    }
                    if edges.contains(&key) {
                        let action = match key {
                            Key::F5 => Some(Action::Reset),
                            Key::F7 => Some(Action::Capture),
                            Key::F8 => Some(Action::Save),
                            Key::F9 => Some(Action::RunPause),
                            Key::F10 => Some(Action::Step),
                            Key::F11 => Some(Action::Wait),
                            _ => None,
                        };
                        if let Some(action) = action {
                            self.action(action)?;
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => self.tick()?,
            _ => {}
        }
        Ok(())
    }
}
impl ApplicationHandler for NativeApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let result = (|| -> Result<()> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("OZ-9600")
                        .with_resizable(false)
                        .with_inner_size(LogicalSize::new(
                            ui::WIDTH as f64 * f64::from(self.args.scale),
                            ui::HEIGHT as f64 * f64::from(self.args.scale),
                        ))
                        .with_min_inner_size(LogicalSize::new(ui::WIDTH as f64, ui::HEIGHT as f64)),
                )?,
            );
            let context = softbuffer::Context::new(window.clone())?;
            self.surface = Some(softbuffer::Surface::new(&context, window.clone())?);
            self.active = window.has_focus();
            self.window = Some(window);
            if self.args.sound {
                self.toggle_sound()?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.fatal = Some(e.to_string());
            event_loop.exit();
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.closing || self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        if let Err(e) = self.event(event_loop, event) {
            self.fatal = Some(e.to_string());
            self.close(event_loop);
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.closing {
            return;
        }
        let now = Instant::now();
        if self
            .args
            .quit_after_ms
            .is_some_and(|ms| self.started.elapsed().as_millis() >= u128::from(ms))
        {
            self.close(event_loop);
            return;
        }
        if now >= self.next_frame {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            self.next_frame = now + Duration::from_millis(4);
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}
fn live(
    args: &Args,
    rom: &[u8],
    rt: &mut CoreRuntime,
    contacts: &mut Contacts,
    boundaries: u64,
    state_store: Option<&mut RetainedStore>,
) -> Result<()> {
    let now = Instant::now();
    let mut app = NativeApp {
        args,
        rom,
        rt,
        contacts,
        window: None,
        surface: None,
        boundaries,
        record: Recorder {
            steps: Vec::new(),
            last: boundaries,
            enabled: args.event_log.is_some(),
        },
        host_keys: HostKeys::default(),
        pointer: None,
        position: None,
        lcd_drag: false,
        mouse: false,
        blocked_mouse: false,
        active: false,
        closing: false,
        paused: args.paused || args.execution_mode == ExecutionMode::Deterministic,
        pacer: Pacer::for_model(DeviceModel::Oz9600, args.execution_mode),
        pending: 0,
        started: now,
        next_frame: now,
        last_host_status: now,
        state_store,
        next_save: now + Duration::from_secs(5),
        save_error: None,
        fault: None,
        fatal: None,
        status: "F9 run/pause; F10 20K; F11 1M; F5 reset; F7 capture; F8 save".into(),
        sound: false,
        audio_capture: false,
        audio_output: None,
    };
    EventLoop::new()?.run_app(&mut app)?;
    if let Some(error) = app.fatal {
        return Err(error.into());
    }
    Ok(())
}
fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("oz9600-window: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component_runtime() -> CoreRuntime {
        let mut fixed = vec![0; 0x20000];
        fixed[0x1fffd..].copy_from_slice(&[0, 0, 14]);
        sc62015_core::oz9600::configure_hardware(&fixed, sc62015_core::oz9600::Hardware::default())
            .unwrap()
    }
    fn component_app<'a>(
        args: &'a Args,
        rt: &'a mut CoreRuntime,
        contacts: &'a mut Contacts,
    ) -> NativeApp<'a> {
        let now = Instant::now();
        NativeApp {
            args,
            rom: &[],
            rt,
            contacts,
            window: None,
            surface: None,
            boundaries: 0,
            record: Recorder {
                steps: Vec::new(),
                last: 0,
                enabled: true,
            },
            host_keys: HostKeys::default(),
            pointer: None,
            position: None,
            lcd_drag: false,
            mouse: false,
            blocked_mouse: false,
            active: true,
            paused: false,
            pacer: Pacer::for_model(DeviceModel::Oz9600, ExecutionMode::Interactive),
            pending: 0,
            started: now,
            next_frame: now,
            last_host_status: now,
            fault: None,
            fatal: None,
            status: String::new(),
        }
    }
    fn point_to(app: &mut NativeApp<'_>, target: Target) {
        let zone = ui::zones()
            .into_iter()
            .find(|z| z.target == target)
            .unwrap();
        app.position = Some((zone.rect.x + 1, zone.rect.y + 1));
    }
    #[test]
    fn focus_without_a_held_pointer_accepts_the_first_fresh_press() {
        let args = Args::try_parse_from(["window", "--rom", "unused.ozrom"]).unwrap();
        let mut rt = component_runtime();
        let mut contacts = Contacts::new(0);
        let mut app = component_app(&args, &mut rt, &mut contacts);
        let key = matrix_key(DeviceModel::Oz9600, "1").unwrap();
        point_to(&mut app, Target::Matrix(key));
        app.focus_changed(false).unwrap();
        app.focus_changed(true).unwrap(); // Keyboard activation, no pointer held.
        app.mouse_changed(ElementState::Pressed).unwrap();
        assert_eq!(app.contacts.pressed_keys(), BTreeSet::from([key]));
        app.mouse_changed(ElementState::Released).unwrap();
        assert!(app.contacts.pressed_keys().is_empty());
        assert_eq!(app.rt.instruction_count(), 0);
    }
    #[test]
    fn held_or_background_started_pointer_stays_cancelled_until_release() {
        for held in [false, true] {
            let args = Args::try_parse_from(["window", "--rom", "unused.ozrom"]).unwrap();
            let mut rt = component_runtime();
            let mut contacts = Contacts::new(40_000);
            let mut app = component_app(&args, &mut rt, &mut contacts);
            let key = matrix_key(DeviceModel::Oz9600, "1").unwrap();
            point_to(&mut app, Target::Matrix(key));
            if held {
                app.mouse_changed(ElementState::Pressed).unwrap();
            }
            app.focus_changed(false).unwrap();
            assert!(app.contacts.pressed_keys().is_empty());
            app.mouse_changed(ElementState::Pressed).unwrap(); // Background down.
            app.focus_changed(true).unwrap();
            app.mouse_changed(ElementState::Pressed).unwrap(); // Same held press.
            assert!(app.contacts.pressed_keys().is_empty());
            app.mouse_changed(ElementState::Released).unwrap();
            app.mouse_changed(ElementState::Pressed).unwrap();
            assert_eq!(app.contacts.pressed_keys(), BTreeSet::from([key]));
            app.focus_changed(false).unwrap();
            assert!(app.contacts.pressed_keys().is_empty()); // Cancel assistance immediately.
            assert_eq!(app.rt.instruction_count(), 0);
        }
    }
    #[test]
    fn fault_does_not_reapply_a_held_guest_key_on_pointer_release() {
        let args = Args::try_parse_from(["window", "--rom", "unused.ozrom"]).unwrap();
        let mut rt = component_runtime();
        let mut contacts = Contacts::new(0);
        let mut app = component_app(&args, &mut rt, &mut contacts);
        app.fault = Some("guest fault".into());
        app.host_keys.events(vec![(Key::Digit1, true)]);
        app.mouse_changed(ElementState::Released).unwrap();
        assert!(app.contacts.pressed_keys().is_empty());
        assert!(app.record.steps.is_empty());
        assert_eq!(app.rt.instruction_count(), 0);
    }
    #[test]
    fn fault_blocks_guest_contacts_but_keeps_visible_host_controls() {
        let args = Args::try_parse_from(["window", "--rom", "unused.ozrom"]).unwrap();
        let mut rt = component_runtime();
        let mut contacts = Contacts::new(0);
        let mut app = component_app(&args, &mut rt, &mut contacts);
        app.fault = Some("guest fault".into());
        app.host_keys.events(vec![(Key::Digit1, true)]);
        app.sync().unwrap(); // A mouse release/tick must not reapply keys held during a fault.
        assert!(app.contacts.pressed_keys().is_empty());
        point_to(
            &mut app,
            Target::Matrix(matrix_key(DeviceModel::Oz9600, "1").unwrap()),
        );
        app.mouse_changed(ElementState::Pressed).unwrap();
        app.mouse_changed(ElementState::Released).unwrap();
        assert!(app.contacts.pressed_keys().is_empty());
        point_to(&mut app, Target::Control(Action::Capture));
        app.mouse_changed(ElementState::Pressed).unwrap();
        assert_eq!(app.status, "Set --capture-prefix to enable CAPTURE");
        assert_eq!(app.fault.as_deref(), Some("guest fault"));
        point_to(&mut app, Target::Control(Action::Wait));
        app.mouse_changed(ElementState::Released).unwrap();
        app.mouse_changed(ElementState::Pressed).unwrap();
        assert_eq!(app.pending, 0); // Diagnostic stepping remains disabled during a fault.
        assert_eq!(app.rt.instruction_count(), 0);
    }
    #[test]
    fn queued_redraw_and_controls_cannot_run_after_trace_is_closed() {
        let args = Args::try_parse_from(["window", "--rom", "unused.ozrom"]).unwrap();
        let mut runtime = CoreRuntime::new();
        let mut contacts = Contacts::new(0);
        let now = Instant::now();
        let mut app = NativeApp {
            args: &args,
            rom: &[],
            rt: &mut runtime,
            contacts: &mut contacts,
            window: None,
            surface: None,
            boundaries: 0,
            record: Recorder {
                steps: Vec::new(),
                last: 0,
                enabled: true,
            },
            host_keys: HostKeys::default(),
            pointer: None,
            position: None,
            lcd_drag: false,
            mouse: false,
            blocked_mouse: false,
            active: true,
            closing: false,
            paused: false,
            pacer: Pacer::for_model(DeviceModel::Oz9600, ExecutionMode::Interactive),
            pending: 4096,
            started: now,
            next_frame: now,
            last_host_status: now,
            state_store: None,
            next_save: now,
            save_error: None,
            fault: None,
            fatal: None,
            status: String::new(),
            sound: false,
            audio_capture: false,
            audio_output: None,
        };
        app.prepare_close().unwrap();
        app.action(Action::Wait).unwrap();
        app.action(Action::Sound).unwrap();
        app.tick().unwrap(); // Would execute the queued budget before the fix.
        assert!(app.closing && app.paused);
        assert_eq!(app.pending, 0);
        assert_eq!(app.boundaries, 0);
        assert!(!app.sound && !app.audio_capture);
        assert_eq!(app.rt.instruction_count(), 0);
        assert_eq!(app.rt.cycle_count(), 0);
        assert!(app.record.steps.is_empty());
    }
    #[test]
    fn short_host_taps_keep_edges_and_two_modifiers_share_one_contact() {
        let mut keys = HostKeys::default();
        assert_eq!(
            keys.events(vec![(Key::KeyQ, true), (Key::KeyQ, false)]),
            BTreeSet::from([Key::KeyQ])
        );
        assert!(keys.down.is_empty());
        assert!(keys.events(vec![]).is_empty());
        assert_eq!(
            keys.events(vec![
                (Key::ShiftLeft, true),
                (Key::ShiftLeft, true),
                (Key::ShiftRight, true)
            ])
            .len(),
            2
        );
        assert!(keys.events(vec![(Key::ShiftLeft, false)]).is_empty());
        assert_eq!(
            keys.down
                .iter()
                .copied()
                .filter_map(host_key)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([5])
        );
    }
    #[test]
    fn physical_host_keys_share_contacts_and_on_is_separate_from_matrix() {
        assert_eq!(host_key(Key::ShiftLeft), host_key(Key::ShiftRight));
        assert_eq!(host_key(Key::AltLeft), host_key(Key::AltRight));
        assert_eq!(host_key(Key::KeyA), matrix_key(DeviceModel::Oz9600, "A"));
        assert_eq!(
            host_key(Key::F1),
            matrix_key(DeviceModel::Oz9600, "NEW ENTRY")
        );
        assert_eq!(host_key(Key::F5), None);
        assert_eq!(host_key(Key::F9), None);
        assert_eq!(host_key(Key::F10), None);
        assert_eq!(host_key(Key::F12), None);
        assert_eq!(host_key(Key::Pause), matrix_key(DeviceModel::Oz9600, "OFF"));
    }
    #[test]
    fn recording_preserves_exact_waits_and_contact_order_with_no_guest_steps() {
        let mut r = Recorder {
            steps: Vec::new(),
            last: 1_800_000,
            enabled: true,
        };
        r.contact(&Event::Matrix(3, true), 1_800_000);
        r.contact(&Event::Tablet(227, 68, true), 1_840_000);
        r.contact(&Event::Matrix(3, false), 1_840_000);
        r.wait_to(1_860_000);
        let replay =
            PhysicalReplay::parse(&serde_json::to_vec(&json!({"steps":r.steps})).unwrap()).unwrap();
        assert_eq!(replay.total_boundaries(), 60_000);
        assert_eq!(r.steps[1], json!({"boundaries":40_000}));
        assert_eq!(
            r.steps[3],
            json!({"boundaries":0,"contact":{"column":0,"row":3,"pressed":false}})
        );
    }
    #[test]
    fn model_selection_keeps_strict_default_and_rejects_invalid_host_options() {
        assert_eq!(
            Args::try_parse_from(["window", "--rom", "bundle.ozrom"])
                .unwrap()
                .profile,
            ExecutionProfile::Strict
        );
        assert!(Args::try_parse_from(["window", "--rom", "bundle.ozrom", "--headless"]).is_err());
        assert!(Args::try_parse_from([
            "window",
            "--rom",
            "bundle.ozrom",
            "--retained",
            "import.ozbat",
            "--state",
            "automatic.ozbat"
        ])
        .is_err());
        assert!(Args::try_parse_from(["window", "--rom", "bundle.ozrom", "--scale", "3"]).is_err());
        assert!(Args::try_parse_from([
            "window",
            "--rom",
            "bundle.ozrom",
            "--minimum-contact-boundaries",
            "10000001"
        ])
        .is_err());
    }
    #[test]
    fn recorded_on_edges_parse_without_inventing_a_matrix_contact() {
        let mut r = Recorder {
            steps: Vec::new(),
            last: 0,
            enabled: true,
        };
        r.contact(&Event::On(true), 0);
        r.contact(&Event::On(false), 40_000);
        r.wait_to(740_000);
        assert_eq!(r.steps[0], json!({"boundaries":0,"on_key":true}));
        assert_eq!(r.steps[2], json!({"boundaries":0,"on_key":false}));
        let replay =
            PhysicalReplay::parse(&serde_json::to_vec(&json!({"steps":r.steps})).unwrap()).unwrap();
        assert_eq!(replay.total_boundaries(), 740_000);
    }
    #[test]
    fn on_owners_and_focus_cancellation_reach_the_runtime_contact_api() {
        let mut keys = HostKeys::default();
        let mut contacts = Contacts::new(40_000);
        let mut runtime = CoreRuntime::new();
        let mut recorder = Recorder {
            steps: Vec::new(),
            last: 0,
            enabled: true,
        };
        keys.events(vec![(Key::F12, true)]);
        let events = contacts.sync(&BTreeSet::new(), None, 0, host_on(&keys, Some(Target::On)));
        apply(&mut runtime, events, &mut recorder, 0).unwrap();
        assert!(runtime.physical_on_key_pressed());
        keys.events(vec![(Key::F12, false)]);
        assert!(contacts
            .sync(
                &BTreeSet::new(),
                None,
                40_001,
                host_on(&keys, Some(Target::On))
            )
            .is_empty());
        // Focus loss releases the pointer owner before its assisted deadline.
        apply(&mut runtime, contacts.cancel(), &mut recorder, 40_001).unwrap();
        assert!(!runtime.physical_on_key_pressed());
        assert!(contacts.pressed_keys().is_empty());
        assert_eq!(runtime.instruction_count(), 0);
        assert_eq!(runtime.cycle_count(), 0);
        assert_eq!(recorder.steps[0], json!({"boundaries":0,"on_key":true}));
        assert_eq!(recorder.steps[2], json!({"boundaries":0,"on_key":false}));
    }
}
