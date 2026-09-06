# Rust interactive input correctness

Basic-input stage verified locally on 2026-09-06; unresolved cases are listed
below and are not reclassified as passing.

Native `sc62015-lcd` and the browser/WASM UI share
[`physical_keys.json`](../sc62015/core/data/physical_keys.json). Entries are
**physical matrix contacts**, not translated events, ASCII, FIFO entries, or
application selectors. Rust and TypeScript consume this same table.

## Controls

| Host control | PC-E500 | IQ-7000 |
| --- | --- | --- |
| A–Z, 0–9, Space | Device letter/digit/space keys | Device letter/digit/space keys |
| Arrows, Backspace, Delete, Insert | Cursor/edit keys | Cursor/edit keys |
| Enter | ENTER | ENTER (store) |
| F1–F5 | PF1–PF5 | Calendar, Schedule, TEL, MEMO, Calc |
| F6–F8 | BASIC, MENU, Clear | Card, World, Home |
| F9 / F10 | SHIFT / CAPS | SHIFT / CAPS |
| F11 | Unmapped | Return/newline |
| F12 | ON | ON |
| Page Up/Down | Unmapped | Search up/down |
| Escape | Clear | C·CE |

The browser additionally accepts the host Shift/Caps Lock keys, numeric keypad
operators, and model-specific punctuation keycaps. Its on-screen keyboard has
all basic letters/digits and the available independent operator keys.

This is **keycap input, not desktop text composition**. Device CAPS controls
case; host uppercase letters do not automatically imply guest uppercase, and
host Shift operates the device's shifted legends. For example, IQ SHIFT+A means
EDIT, not uppercase A. Browser users should use on-screen/numeric-keypad `+`
instead of expecting a desktop Shift+Equals chord to type it. Native terminal
characters retain supported assisted shifted-punctuation composition (IQ comma,
PC punctuation). IQ colon composition is intentionally unmapped: the proposed
SHIFT+period sequence produces a period in MEMO, including with separate taps.
Native unsupported characters report `unmapped:` in the status instead of
silently inserting a different character. Extended characters and IME/paste
composition are not fully covered.

## Native delivery

Normal native IQ input no longer injects translated events or modifies the
keyboard FIFO/masks. A terminal batch is drained **one assisted press at a time**,
with a release and gap before the next key, rather than pressing an entire burst
together. The backlog remains in the existing bounded 128-event queue; overflow
clears it, releases contacts and pauses. Pause freezes pending tap deadlines;
focus loss and quit invalidate them. Priority host controls remain independent.

Hold/release assistance still uses scheduler boundaries: IQ taps use a 40,000
boundary hold and gap; PC characters use a 10,000 hold and 20,000 gap, with longer
control-key holds and shifted-chord lead time. These are compatibility policies,
**not hardware timing measurements**. Native terminals do not reliably expose
physical key-up events. Browser raw down/up and virtual assistance retain their
existing independent source ownership and focus cleanup.

`--force-key-irq` remains an explicit diagnostic override, not normal input.
IQ rejects the PC-only `--auto-basic`, `--jump-basic`, and `--auto-type` shortcuts.
Headless CLI/Function Runner translated-event APIs remain diagnostic interfaces;
their existence is not evidence for the ordinary physical-input path.

## Acceptance and evidence limits

Actual-process PTY and compiled Chromium tests use private firmware through its
normal foreground UI. They do not inject FIFO events, write records directly,
jump into app routines, stub calls, or patch framebuffer contents:

- IQ: enter MEMO, type `ABCD`, store, return to the search prompt, reopen,
  SHIFT+A to edit, replace A with X, store and reopen `XBCD`.
- Native: type several characters in one terminal burst, including a shifted
  comma and unmapped-colon check. Browser: mix host letters and on-screen digits, use
  Backspace, and exercise the four cursor directions plus Insert/Delete in a
  multiline MEMO. Existing CAPS and owner/focus cleanup checks remain covered.
- PC-E500: initialize both card prompts, choose CAL, enter `2+2`, press ENTER,
  observe `4`. Native uses a single `2+2\r` burst; browser mixes host keys and the
  on-screen plus key. Browser captures are actual emulated LCD pixels; decoded
  text is only an assertion aid and can show `?` for unrecognized glyphs.

Basic input coverage is not exhaustive application, every-key, repeat/debounce,
or real-hardware qualification. In particular, **PC-E500 BASIC acceptance is
blocked**: both its direct BASIC key and main-menu PF1 path stop at the same
invalid-register-pair decoder guard. The guard is not relaxed here. CAL input
passing must not be described as BASIC passing. Investigate that CPU/decode
failure next with a minimal ROM-backed repro, separately from keyboard mapping.

## Reproduce

Final local checks: 571 Rust tests passed (six existing ignored tests and three
opt-in ROM tests excluded from that count); both new native private-ROM PTY
tests passed separately. Also passed: 26 WASM tests, 112 frontend tests, 36
bounded Chromium regressions, six selected real-ROM browser checks, Clippy,
Svelte/TypeScript and formatting. All five changed Rust files passed the
source-annotation check; this does not claim the pre-existing repository-wide
annotation failures have been fixed. The previously recorded 18 WASM
feature-gated dependency warnings remain. Browser proof advances deterministic
boundary budgets; this is not a live typing latency or hardware-timing benchmark.

Public tests cover table consistency, physical contacts without FIFO mutation,
bounded serialization/cleanup and browser ownership. Private-ROM checks remain
explicit opt-in, not an hour-long mandatory CI gate.

From the public root (set both ROM path variables to your local files):

```bash
cargo test --offline --manifest-path sc62015/core/Cargo.toml --all-targets --all-features
cargo test --offline --release --manifest-path sc62015/core/Cargo.toml --test native_ui_pty --all-features -- --ignored --nocapture
```

From `web/`, with `PCE500_ROM_PATH` and `IQ7000_ROM_PATH` exported:

```bash
npm run wasm:build
npm run test:ci
CI=1 PCE500_E2E_PORT=4197 PCE500_E2E_REAL_ROM=1 IQ7000_E2E_REAL_ROM=1 npm run e2e -- e2e/input_ownership.spec.ts --workers=1
```

Only one WASM build should write the generated package at a time. Chromium
proof PNGs are emitted beneath ignored `web/test-results/`. No ROM or ROM dump
belongs in the public repository.
