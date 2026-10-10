// PY_SOURCE: pce500/oz9600/ui.py
//! Host artwork and input ownership. No ROM text, guest writes or interrupts.
use font8x8::{UnicodeFonts, BASIC_FONTS};
use sc62015_core::{lcd_frame::LcdFrame, physical_keys::matrix_key, DeviceModel};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const WIDTH: usize = 526;
pub const HEIGHT: usize = 530;

/// Keep fault state visible while allowing host Save/Capture acknowledgments.
pub fn window_title(profile: &str, mode: &str, status: &str, fault: Option<&str>) -> String {
    let mode = if fault.is_some() { "faulted" } else { mode };
    format!("OZ-9600 | {profile} | {mode} | {status}")
}
pub const LCD: Rect = Rect {
    x: 162,
    y: 36,
    w: 336,
    h: 240,
};
pub const PAPER: u32 = 0xd8dfbc;
pub const INK: u32 = 0x263325;

/// OS cursor and surface sizes both use physical pixels; invert presentation
/// scaling without assuming a platform backing-scale or screenshot size.
pub fn window_point(x: f64, y: f64, width: usize, height: usize) -> Option<(usize, usize)> {
    if width == 0 || height == 0 || !(x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64)
    {
        return None;
    }
    Some((
        (x * WIDTH as f64 / width as f64).floor() as usize,
        (y * HEIGHT as f64 / height as f64).floor() as usize,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}
impl Rect {
    pub fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x && y >= self.y && x - self.x < self.w && y - self.y < self.h
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    RunPause,
    Step,
    Wait,
    Reset,
    Save,
    Capture,
    Sound,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Matrix(u8),
    On,
    Tablet(u16, u16),
    Control(Action),
}
pub struct Zone {
    pub rect: Rect,
    pub label: &'static str,
    pub target: Target,
}

pub fn lcd_tablet(x: usize, y: usize) -> (u16, u16) {
    (
        ((4 * x.min(335) + 419) * 548 / 1008) as u16,
        ((4 * y.min(239) + 75) * 630 / 688) as u16,
    )
}
pub fn zones() -> Vec<Zone> {
    let mut output = vec![Zone {
        rect: Rect {
            x: 414,
            y: 7,
            w: 100,
            h: 22,
        },
        label: "SOUND OFF",
        target: Target::Control(Action::Sound),
    }];
    // Fixed printed panel, outside the controller image. Coordinates are the
    // ROM's default calibration, not measured digitizer voltages.
    for (index, (label, x, y)) in [
        ("CAL", 61, 141),
        ("SCHED", 122, 141),
        ("MENU", 182, 141),
        ("TO DO", 61, 288),
        ("ANN", 122, 288),
        ("CALC", 182, 288),
        ("TEL", 61, 434),
        ("USER", 122, 434),
        ("CLOCK", 182, 434),
        ("NOTE", 61, 581),
        ("OUTLN", 122, 581),
        ("SCRAP", 182, 581),
        ("FILER", 61, 727),
        ("CARD", 122, 727),
    ]
    .into_iter()
    .enumerate()
    {
        output.push(Zone {
            rect: Rect {
                x: 12 + index % 3 * 46,
                y: 36 + index / 3 * 40,
                w: 44,
                h: 38,
            },
            label,
            target: Target::Tablet(x, y),
        });
    }
    output.push(Zone {
        rect: Rect {
            x: 12,
            y: 236,
            w: 136,
            h: 40,
        },
        label: "SEARCH",
        target: Target::Tablet(122, 874),
    });
    for (col, (label, target)) in [
        ("ON", Target::On),
        (
            "OFF",
            Target::Matrix(matrix_key(DeviceModel::Oz9600, "OFF").expect("shared OFF key")),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        output.push(Zone {
            rect: Rect {
                x: 12 + col * 46,
                y: 278,
                w: 44,
                h: 18,
            },
            label,
            target,
        });
    }
    for (row, names) in [
        &["%", "*", "+", "-", "/", "=", "M+", "M-", "MENU", "EDIT"][..],
        &["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "BS"][..],
        &["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"][..],
        &["A", "S", "D", "F", "G", "H", "J", "K", "L", "ENTER"][..],
        &["SHIFT", "Z", "X", "C", "V", "B", "N", "M", ",", "."][..],
        &[
            "CAPS", "2ND", "WORD", "SYMBOL", "SPACE", "INS", "DEL", "CANCEL",
        ][..],
    ]
    .into_iter()
    .enumerate()
    {
        let w = 502 / names.len();
        for (col, &name) in names.iter().enumerate() {
            output.push(Zone {
                rect: Rect {
                    x: 12 + col * w,
                    y: 298 + row * 28,
                    w: w - 2,
                    h: 26,
                },
                label: name,
                target: Target::Matrix(
                    matrix_key(DeviceModel::Oz9600, name).expect("shared keycap"),
                ),
            });
        }
    }
    for (col, name) in ["NEW ENTRY", "LEFT", "DOWN", "UP", "RIGHT", "PREV", "NEXT"]
        .into_iter()
        .enumerate()
    {
        output.push(Zone {
            rect: Rect {
                x: 12 + col * 72,
                y: 466,
                w: 70,
                h: 26,
            },
            label: name,
            target: Target::Matrix(matrix_key(DeviceModel::Oz9600, name).expect("shared keycap")),
        });
    }
    for (col, (label, action)) in [
        ("RUN", Action::RunPause),
        ("STEP 20K", Action::Step),
        ("STEP 1M", Action::Wait),
        ("RESET", Action::Reset),
        ("SAVE", Action::Save),
        ("CAPTURE", Action::Capture),
    ]
    .into_iter()
    .enumerate()
    {
        output.push(Zone {
            rect: Rect {
                x: 12 + col * 84,
                y: 496,
                w: 82,
                h: 22,
            },
            label,
            target: Target::Control(action),
        });
    }
    output
}
pub fn hit(x: usize, y: usize) -> Option<Target> {
    if LCD.contains(x, y) {
        let (x, y) = lcd_tablet(x - LCD.x, y - LCD.y);
        return Some(Target::Tablet(x, y));
    }
    zones()
        .into_iter()
        .find(|z| z.rect.contains(x, y))
        .map(|z| z.target)
}
fn fill(buffer: &mut [u32], rect: Rect, color: u32) {
    for y in rect.y..(rect.y + rect.h).min(HEIGHT) {
        for x in rect.x..(rect.x + rect.w).min(WIDTH) {
            buffer[y * WIDTH + x] = color;
        }
    }
}
pub(crate) fn text(buffer: &mut [u32], x: usize, y: usize, label: &str, color: u32) {
    for (index, ch) in label.chars().enumerate() {
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (dy, bits) in glyph.into_iter().enumerate() {
                for dx in 0..8 {
                    if bits & (1 << dx) != 0 {
                        let px = x + index * 8 + dx;
                        let py = y + dy;
                        if px < WIDTH && py < HEIGHT {
                            buffer[py * WIDTH + px] = color;
                        }
                    }
                }
            }
        }
    }
}
pub fn render(
    frame: &LcdFrame,
    keys: &BTreeSet<u8>,
    paused: bool,
    on: bool,
    sound: bool,
) -> Result<Vec<u32>, &'static str> {
    if frame.geometry() != (LCD.w, LCD.h) {
        return Err("native window requires full 336x240 controller image");
    }
    let mut out = vec![0x333b40; WIDTH * HEIGHT];
    text(&mut out, 12, 10, "SHARP OZ-9600 / 256KB", 0xe0d9b2);
    fill(
        &mut out,
        Rect {
            x: 8,
            y: 32,
            w: 144,
            h: 248,
        },
        PAPER,
    );
    fill(
        &mut out,
        Rect {
            x: 158,
            y: 32,
            w: 344,
            h: 248,
        },
        0x17211b,
    );
    for zone in zones() {
        let color = match zone.target {
            Target::Tablet(..) => PAPER,
            Target::Matrix(code) if keys.contains(&code) => 0xb2c9a5,
            Target::Matrix(..) => 0xddd9c8,
            Target::On if on => 0xb2c9a5,
            Target::On => 0xddd9c8,
            Target::Control(Action::Sound) if sound => 0xb2c9a5,
            Target::Control(..) => 0x94acbb,
        };
        fill(&mut out, zone.rect, 0x4e574e);
        fill(
            &mut out,
            Rect {
                x: zone.rect.x + 1,
                y: zone.rect.y + 1,
                w: zone.rect.w - 2,
                h: zone.rect.h - 2,
            },
            color,
        );
        let label = if zone.target == Target::Control(Action::RunPause) && !paused {
            "PAUSE"
        } else if zone.target == Target::Control(Action::Sound) && sound {
            "SOUND ON"
        } else {
            zone.label
        };
        let lines = if label.len() * 8 > zone.rect.w {
            label.split_whitespace().collect::<Vec<_>>()
        } else {
            vec![label]
        };
        for (n, line) in lines.iter().enumerate() {
            text(
                &mut out,
                zone.rect.x + (zone.rect.w.saturating_sub(line.len() * 8)) / 2,
                zone.rect.y + (zone.rect.h - lines.len() * 8) / 2 + n * 8,
                line,
                INK,
            );
        }
    }
    // Copy every controller pixel LAST. Host text/artwork never touches it.
    for (i, &pixel) in frame.pixels().iter().enumerate() {
        out[(LCD.y + i / LCD.w) * WIDTH + LCD.x + i % LCD.w] = if pixel == 1 { INK } else { PAPER };
    }
    Ok(out)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Event {
    Matrix(u8, bool),
    On(bool),
    Tablet(u16, u16, bool),
}
#[derive(Default)]
pub struct Contacts {
    keys: BTreeMap<u8, u64>,
    tablet: Option<(u16, u16, u64)>,
    on_deadline: Option<u64>,
    pub minimum_hold: u64,
    tablet_grace_until: Option<u64>,
    tablet_host: Option<(u16, u16)>,
    tablet_queue: VecDeque<(u16, u16, bool)>,
    tablet_release: bool,
    tablet_gap_until: u64,
    pub tablet_queue_overflow: bool,
}
impl Contacts {
    pub fn new(minimum_hold: u64) -> Self {
        Self {
            minimum_hold,
            ..Self::default()
        }
    }
    pub fn pressed_keys(&self) -> BTreeSet<u8> {
        self.keys.keys().copied().collect()
    }
    /// Poll an assisted release without crossing its minimum contact deadline.
    pub fn limit_budget(&self, requested: usize, at: u64) -> usize {
        if self.tablet_gap_until > at {
            return requested.min((self.tablet_gap_until - at).min(usize::MAX as u64) as usize);
        }
        if self.minimum_hold != 0 && self.tablet_release {
            let remaining = self.tablet.map_or(0, |(_, _, end)| end.saturating_sub(at));
            if remaining != 0 {
                requested.min(remaining.min(usize::MAX as u64) as usize)
            } else {
                requested.min(256)
            }
        } else if !self.tablet_queue.is_empty() {
            requested.min(4096)
        } else {
            requested
        }
    }
    pub fn pressed_on(&self) -> bool {
        self.on_deadline.is_some()
    }
    /// Desired contacts are the union of all host owners (keyboard/pointer).
    /// Optional click assistance is counted in scheduler boundary budgets.
    pub fn sync(
        &mut self,
        desired: &BTreeSet<u8>,
        tablet: Option<(u16, u16)>,
        at: u64,
        on: bool,
    ) -> Vec<Event> {
        self.sync_tablet_ready(desired, tablet, at, on, true)
    }
    /// Assisted pen release also waits for a complete coordinate sample and idle panel drive, with
    /// a bounded 400k grace. Raw contacts and cancellation keep their contract.
    pub fn sync_tablet_ready(
        &mut self,
        desired: &BTreeSet<u8>,
        tablet: Option<(u16, u16)>,
        at: u64,
        on: bool,
        ready: bool,
    ) -> Vec<Event> {
        self.sync_tablet_state(desired, tablet, at, on, ready, false)
    }
    /// Read-only ROM mailbox state serializes complete host clicks. Raw mode
    /// bypasses mailbox, acquisition assistance and release-gap policies.
    pub fn sync_tablet_state(
        &mut self,
        desired: &BTreeSet<u8>,
        tablet: Option<(u16, u16)>,
        at: u64,
        on: bool,
        ready: bool,
        busy: bool,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        for &code in desired {
            if let std::collections::btree_map::Entry::Vacant(e) = self.keys.entry(code) {
                e.insert(at.saturating_add(self.minimum_hold));
                events.push(Event::Matrix(code, true));
            }
        }
        self.keys.retain(|&code, end| {
            if !desired.contains(&code) && at >= *end {
                events.push(Event::Matrix(code, false));
                false
            } else {
                true
            }
        });
        match (self.on_deadline, on) {
            (None, true) => {
                self.on_deadline = Some(at.saturating_add(self.minimum_hold));
                events.push(Event::On(true));
            }
            (Some(end), false) if at >= end => {
                self.on_deadline = None;
                events.push(Event::On(false));
            }
            _ => {}
        }
        match (self.tablet_host, tablet) {
            (None, Some((x, y))) => {
                // One native pointer; preserve a full tap even if its down/up
                // arrive before the preceding ROM event has been consumed.
                if self.tablet_queue.len() < 16 {
                    self.tablet_queue.push_back((x, y, true));
                } else {
                    self.tablet_queue_overflow = true;
                }
            }
            (Some((old_x, old_y)), Some((x, y))) if (x, y) != (old_x, old_y) => {
                if let Some(last) = self.tablet_queue.back_mut().filter(|q| q.2) {
                    last.0 = x;
                    last.1 = y;
                } else if let Some((_, _, end)) = self.tablet.filter(|_| !self.tablet_release) {
                    self.tablet = Some((x, y, end));
                    events.push(Event::Tablet(x, y, true));
                }
            }
            (Some(_), None) => {
                if let Some(last) = self.tablet_queue.back_mut().filter(|q| q.2) {
                    last.2 = false;
                } else if let Some((_, _, end)) = self.tablet {
                    self.tablet_release = true;
                    self.tablet_grace_until = Some(at.max(end).saturating_add(400_000));
                }
            }
            _ => {}
        }
        self.tablet_host = tablet;
        if let Some((x, y, end)) = self.tablet {
            if self.tablet_release
                && at >= end
                && (self.minimum_hold == 0
                    || ready
                    || at >= self.tablet_grace_until.unwrap_or(u64::MAX))
            {
                self.tablet = None;
                self.tablet_release = false;
                self.tablet_grace_until = None;
                self.tablet_gap_until =
                    at.saturating_add(if self.minimum_hold == 0 { 0 } else { 40_000 });
                events.push(Event::Tablet(x, y, false));
            }
        }
        if self.tablet.is_none() && at >= self.tablet_gap_until && (!busy || self.minimum_hold == 0)
        {
            if let Some((x, y, down)) = self.tablet_queue.pop_front() {
                let end = at.saturating_add(self.minimum_hold);
                self.tablet = Some((x, y, end));
                self.tablet_release = !down;
                self.tablet_grace_until = if down {
                    None
                } else {
                    Some(end.saturating_add(400_000))
                };
                events.push(Event::Tablet(x, y, true));
            }
        }
        events
    }
    /// Focus loss/close/fault override assistance; release every owned contact.
    pub fn cancel(&mut self) -> Vec<Event> {
        let mut events = self
            .keys
            .keys()
            .map(|&c| Event::Matrix(c, false))
            .collect::<Vec<_>>();
        self.keys.clear();
        if self.on_deadline.take().is_some() {
            events.push(Event::On(false));
        }
        self.tablet_grace_until = None;
        self.tablet_host = None;
        self.tablet_queue_overflow = false;
        self.tablet_queue.clear();
        self.tablet_release = false;
        self.tablet_gap_until = 0;
        if let Some((x, y, _)) = self.tablet.take() {
            events.push(Event::Tablet(x, y, false));
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_conversion_inverts_every_integer_scale_and_rejects_outside_samples() {
        for factor in [1, 2, 3, 4] {
            for (x, y) in [(0, 0), (125, 95), (WIDTH - 1, HEIGHT - 1)] {
                assert_eq!(
                    window_point(
                        (x * factor) as f64,
                        (y * factor) as f64,
                        WIDTH * factor,
                        HEIGHT * factor
                    ),
                    Some((x, y))
                );
            }
        }
        for (x, y, w, h) in [
            (f64::NAN, 1.0, WIDTH, HEIGHT),
            (-1.0, 1.0, WIDTH, HEIGHT),
            (WIDTH as f64, 1.0, WIDTH, HEIGHT),
            (1.0, 1.0, 0, HEIGHT),
        ] {
            assert_eq!(window_point(x, y, w, h), None);
        }
    }
    #[test]
    fn entire_frame_and_last_pixel_survive_host_artwork() {
        let pixels = (0..336 * 240)
            .map(|n| u8::from(n % 7 == 0 || n == 336 * 240 - 1))
            .collect::<Vec<_>>();
        let frame = LcdFrame::new(336, 240, pixels.clone()).unwrap();
        let out = render(&frame, &BTreeSet::new(), true, false, false).unwrap();
        for (i, p) in pixels.iter().enumerate() {
            assert_eq!(
                out[(LCD.y + i / 336) * WIDTH + LCD.x + i % 336],
                if *p == 1 { INK } else { PAPER }
            );
        }
        assert_eq!(frame.pixels(), pixels);
        assert_eq!(out[(LCD.y + 239) * WIDTH + LCD.x + 335], INK);
    }
    #[test]
    fn fixed_panel_and_controls_are_outside_lcd_and_hit_test_is_half_open() {
        for zone in zones() {
            for (x, y) in [
                (zone.rect.x, zone.rect.y),
                (zone.rect.x + zone.rect.w - 1, zone.rect.y + zone.rect.h - 1),
            ] {
                assert!(!LCD.contains(x, y));
                assert_eq!(hit(x, y), Some(zone.target));
            }
        }
        assert_eq!(hit(LCD.x, LCD.y), Some(Target::Tablet(227, 68)));
        assert_eq!(
            hit(LCD.x + 335, LCD.y + 239),
            Some(Target::Tablet(956, 944))
        );
        assert_eq!(hit(LCD.x + LCD.w, LCD.y), None);
        assert_eq!(hit(LCD.x, LCD.y + LCD.h), None);
    }
    #[test]
    fn overlapping_owners_and_fast_clicks_do_not_release_early_or_stick() {
        let mut contacts = Contacts {
            minimum_hold: 40_000,
            ..Default::default()
        };
        let union = BTreeSet::from([3]);
        assert_eq!(
            contacts.sync(&union, Some((61, 141)), 0, false),
            vec![Event::Matrix(3, true), Event::Tablet(61, 141, true)]
        );
        // One owner can release while the other keeps the same key down.
        assert!(contacts
            .sync(&union, None, 40_000, false)
            .iter()
            .all(|e| matches!(e, Event::Tablet(..))));
        assert_eq!(contacts.pressed_keys(), union);
        assert_eq!(
            contacts.sync(&BTreeSet::new(), None, 40_001, false),
            vec![Event::Matrix(3, false)]
        );
        assert_eq!(
            contacts.sync(&union, Some((227, 68)), 50_000, false).len(),
            1
        );
        assert!(contacts
            .sync(&BTreeSet::new(), None, 50_001, false)
            .is_empty());
        assert_eq!(
            contacts.sync(&BTreeSet::new(), None, 80_000, false),
            vec![Event::Tablet(227, 68, true)]
        );
        assert_eq!(
            contacts.cancel(),
            vec![Event::Matrix(3, false), Event::Tablet(227, 68, false)]
        );
        assert!(contacts.cancel().is_empty());
    }
    #[test]
    fn assisted_pen_waits_for_sample_with_bounded_timeout_and_immediate_cancel() {
        let mut c = Contacts::new(40_000);
        let empty = BTreeSet::new();
        assert_eq!(
            c.sync_tablet_ready(&empty, Some((61, 141)), 0, false, false),
            vec![Event::Tablet(61, 141, true)]
        );
        assert!(c
            .sync_tablet_ready(&empty, None, 40_000, false, false)
            .is_empty());
        assert_eq!(
            c.sync_tablet_ready(&empty, None, 60_000, false, true),
            vec![Event::Tablet(61, 141, false)]
        );
        c.sync_tablet_ready(&empty, Some((61, 141)), 100_000, false, false);
        assert!(c
            .sync_tablet_ready(&empty, None, 140_000, false, false)
            .is_empty());
        assert!(c
            .sync_tablet_ready(&empty, None, 539_999, false, false)
            .is_empty());
        assert_eq!(
            c.sync_tablet_ready(&empty, None, 540_000, false, false),
            vec![Event::Tablet(61, 141, false)]
        );
        c.sync_tablet_ready(&empty, Some((61, 141)), 600_000, false, false);
        assert_eq!(c.cancel(), vec![Event::Tablet(61, 141, false)]);
        let mut raw = Contacts::new(0);
        raw.sync_tablet_ready(&empty, Some((61, 141)), 0, false, false);
        assert_eq!(
            raw.sync_tablet_ready(&empty, None, 1, false, false),
            vec![Event::Tablet(61, 141, false)]
        );
    }

    #[test]
    fn pen_grace_begins_at_host_release_and_polls_without_splitting_acquisition() {
        let mut c = Contacts::new(40_000);
        let empty = BTreeSet::new();
        c.sync_tablet_ready(&empty, Some((61, 141)), 0, false, false);
        assert_eq!(c.limit_budget(4096, 0), 4096);
        c.sync_tablet_ready(&empty, Some((61, 141)), 1_000_000, false, false);
        assert!(c
            .sync_tablet_ready(&empty, None, 1_000_000, false, false)
            .is_empty());
        assert_eq!(c.limit_budget(4096, 1_000_000), 256);
        assert!(c
            .sync_tablet_ready(&empty, None, 1_399_999, false, false)
            .is_empty());
        assert_eq!(
            c.sync_tablet_ready(&empty, None, 1_400_000, false, false),
            vec![Event::Tablet(61, 141, false)]
        );
        assert_eq!(c.limit_budget(4096, 1_400_000), 4096);
    }
    #[test]
    fn raw_contact_edges_and_pen_movement_preserve_first_release_deadline() {
        let mut contacts = Contacts::default();
        assert_eq!(
            contacts.sync(&BTreeSet::new(), Some((227, 68)), 0, false),
            vec![Event::Tablet(227, 68, true)]
        );
        assert_eq!(
            contacts.sync(&BTreeSet::new(), Some((956, 944)), 1, false),
            vec![Event::Tablet(956, 944, true)]
        );
        assert_eq!(
            contacts.sync(&BTreeSet::new(), None, 2, false),
            vec![Event::Tablet(956, 944, false)]
        );
    }
    #[test]
    fn queued_pen_preserves_a_short_click_until_rom_mailbox_clears() {
        let mut c = Contacts::new(40_000);
        let keys = BTreeSet::new();
        assert!(c
            .sync_tablet_state(&keys, Some((627, 130)), 0, false, true, true)
            .is_empty());
        assert!(c
            .sync_tablet_state(&keys, None, 1, false, true, true)
            .is_empty());
        assert!(c
            .sync_tablet_state(&keys, None, 4096, false, true, true)
            .is_empty());
        assert_eq!(
            c.sync_tablet_state(&keys, None, 40960, false, true, false),
            vec![Event::Tablet(627, 130, true)]
        );
        assert!(c
            .sync_tablet_state(&keys, None, 80959, false, true, false)
            .is_empty());
        assert_eq!(
            c.sync_tablet_state(&keys, None, 80960, false, true, false),
            vec![Event::Tablet(627, 130, false)]
        );
        let mut raw = Contacts::new(0);
        assert_eq!(
            raw.sync_tablet_state(&keys, Some((627, 130)), 0, false, false, true),
            vec![Event::Tablet(627, 130, true)]
        );
        assert_eq!(
            raw.sync_tablet_state(&keys, None, 1, false, false, true),
            vec![Event::Tablet(627, 130, false)]
        );
    }
    #[test]
    fn pending_pen_cancellation_and_overflow_are_explicit() {
        let mut c = Contacts::new(40_000);
        let keys = BTreeSet::new();
        for n in 0..16 {
            c.sync_tablet_state(&keys, Some((627, 130)), n * 2, false, true, true);
            c.sync_tablet_state(&keys, None, n * 2 + 1, false, true, true);
        }
        assert_eq!(c.tablet_queue.len(), 16);
        assert!(!c.tablet_queue_overflow);
        c.sync_tablet_state(&keys, Some((627, 130)), 32, false, true, true);
        assert!(c.tablet_queue_overflow);
        assert_eq!(c.tablet_queue.len(), 16);
        assert!(c.cancel().is_empty());
        assert!(!c.tablet_queue_overflow);
        assert!(c
            .sync_tablet_state(&keys, None, 1000000, false, true, false)
            .is_empty());
    }
    #[test]
    fn on_contact_is_separate_assisted_owned_and_cancelled() {
        let mut contacts = Contacts::new(40_000);
        let empty = BTreeSet::new();
        assert_eq!(contacts.sync(&empty, None, 0, true), vec![Event::On(true)]);
        // A fast release is delayed until the original boundary deadline.
        assert!(contacts.sync(&empty, None, 1, false).is_empty());
        assert!(contacts.pressed_on());
        // Keeping one owner asserted does not reset that deadline.
        assert!(contacts.sync(&empty, None, 40_000, true).is_empty());
        assert_eq!(
            contacts.sync(&empty, None, 40_001, false),
            vec![Event::On(false)]
        );
        assert!(!contacts.pressed_on());
        assert_eq!(
            contacts.sync(&empty, None, 50_000, true),
            vec![Event::On(true)]
        );
        assert_eq!(contacts.cancel(), vec![Event::On(false)]);
        assert!(!contacts.pressed_on());
        assert!(contacts.cancel().is_empty());
        assert!(contacts.pressed_keys().is_empty());
        let mut raw = Contacts::default();
        assert_eq!(raw.sync(&empty, None, 0, true), vec![Event::On(true)]);
        assert_eq!(raw.sync(&empty, None, 1, false), vec![Event::On(false)]);
    }
    #[test]
    fn sound_control_changes_only_host_artwork_and_has_no_guest_contact() {
        let frame = LcdFrame::new(336, 240, vec![1; 336 * 240]).unwrap();
        let muted = render(&frame, &BTreeSet::new(), false, false, false).unwrap();
        let playing = render(&frame, &BTreeSet::new(), false, false, true).unwrap();
        assert_eq!(hit(420, 12), Some(Target::Control(Action::Sound)));
        assert_ne!(muted, playing);
        for y in LCD.y..LCD.y + LCD.h {
            let start = y * WIDTH + LCD.x;
            assert_eq!(muted[start..start + LCD.w], playing[start..start + LCD.w]);
        }
    }
    #[test]
    fn power_key_highlight_changes_only_host_artwork() {
        let frame = LcdFrame::new(336, 240, vec![1; 336 * 240]).unwrap();
        let before = render(&frame, &BTreeSet::new(), true, false, false).unwrap();
        let after = render(&frame, &BTreeSet::new(), true, true, false).unwrap();
        let on_zone = zones()
            .into_iter()
            .find(|z| z.target == Target::On)
            .unwrap();
        assert_eq!(hit(on_zone.rect.x, on_zone.rect.y), Some(Target::On));
        assert_eq!(
            zones()
                .into_iter()
                .find(|z| z.label == "OFF")
                .unwrap()
                .target,
            Target::Matrix(1)
        );
        assert_ne!(before, after);
        for (index, (a, b)) in before.iter().zip(after).enumerate() {
            if !on_zone.rect.contains(index % WIDTH, index / WIDTH) {
                assert_eq!(*a, b);
            }
        }
    }
    #[test]
    fn fault_keeps_successful_host_feedback_visible() {
        assert_eq!(
            window_title(
                "Strict",
                "paused",
                "Saved backup.ozbat",
                Some("guest fault")
            ),
            "OZ-9600 | Strict | faulted | Saved backup.ozbat"
        );
        assert_eq!(
            window_title("Strict", "paused", "guest fault", Some("guest fault")),
            "OZ-9600 | Strict | faulted | guest fault"
        );
        assert_eq!(
            window_title("Strict", "paused", "Captured guest", Some("")),
            "OZ-9600 | Strict | faulted | Captured guest"
        );
        assert_eq!(
            window_title("Strict", "paused", "Saved backup.ozbat", None),
            "OZ-9600 | Strict | paused | Saved backup.ozbat"
        );
    }
}
