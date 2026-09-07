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
| F11 | CTRL (browser) | Return/newline |
| F12 | ON | ON |
| Page Up/Down | Unmapped | Search up/down |
| Escape | Clear | C·CE |

### Browser keyboard

Choose **⋯ → Keyboard focus** to enable physical input and move
focus to the device. Click **Resume** for live typing; paused key events never
secretly advance the CPU. Advanced contains the capture checkbox and raw mapping controls. Hover a
device key for host bindings; cyan outlines show host-held contacts, not proof
that the ROM has consumed them. Laptop function keys may require **Fn**.

- **Letters & symbols (buffered)** (default): follows `KeyboardEvent.key`, so letters match
  the host layout rather than QWERTY positions. Host Shift selects punctuation:
  `+ - * / = .` map to independent device keys, including laptop Shift+Equals
  and Shift+8. PC also supports `, ; ( )`. IQ comma requires **F9, release, K**:
  simultaneous raw SHIFT+K contacts produced `K` during live typing, so the
  browser deliberately does not compose a direct comma key. IQ Shift+Enter maps
  to Return/newline; unshifted Enter stores. Unsupported punctuation is reported instead of becoming a different
  unshifted character. Numeric keypad operators work in either mode; Num Lock
  off uses navigation semantics in Letters & symbols.
- **Device keycaps**: positional `KeyboardEvent.code` mapping, with both host
  Shift keys operating guest SHIFT. Shift+Equals means device SHIFT and equals,
  not plus. Use keypad/on-screen operators in this mode.

In both modes **F9 = device SHIFT**, **F10 or Caps Lock = device CAPS**, and
**F12 = ON**. This is **contact mapping, not desktop text composition**: device
CAPS controls case, and typing an uppercase host letter does not automatically
set guest uppercase. IQ SHIFT+A means EDIT, not uppercase A. Browser F11 maps
to PC CTRL without stealing host Ctrl/Cmd shortcuts; on IQ it remains newline.

Host Ctrl/Cmd/Alt shortcuts, text fields, and focused host scroll controls are
excluded. Key-up releases the original contact/chord even if Shift changes in
between. Separate owners prevent one host Shift from releasing the other.
Blur, hidden tab, editor focus, composition start, disabling capture,
changing mapping, and machine replacement clear held host contacts. Host
auto-repeat does not send repeated DOWNs; the ROM owns key repeat. Raw keycap
mode preserves immediate down/up, so extremely fast taps can miss firmware
scanning/debounce. Buffered mode now preserves fast typing as described below.
Paste/IME and automatic letter-case composition remain unsupported.

### Fast typing and optional catch-up

A zero-delay `AABBCCDDEE1122` burst originally produced only `C` in the IQ MEMO
editor: raw presses could be released before a ROM scan. Increasing the nominal
CPU speed alone cannot preserve a DOWN/UP pair delivered between two scans.

The default browser typing path now queues physical keys in DOWN order, including
repeated letters and overlapping host holds. It presents one contact at a time,
with a minimum 40,000 scheduler-boundary hold and a 40,000-boundary release gap.
These are compatibility budgets, not measured hardware timing. A longer host
hold stays held for the ROM's ordinary repeat behavior. No FIFO/IRQ/record writes
or translated input events are used. The same queue serves worker and fallback
frontends and advances only when Rust executes scheduler boundaries.

**Device pace** is the default and disables typing acceleration. The **Responsive**
preset (or Advanced's **Speed up while typing**) enables it. In interactive Run it executes
just the remaining buffered scan/gap work unthrottled, in slices bounded to four
milliseconds of host work. Every slice still yields to input and Stop. It then
rebases pacing so the user does not wait for an artificial time debt. A sustained
hold stops getting this boost after its minimum scan budget; arrow-key repeat
is not run at turbo speed. Turbo already runs unthrottled. Paused/explicit
deterministic execution does not get any automatic steps.

This acceleration advances **all emulated time, including RTC/peripherals**.
Disable it for normal interactive timing; buffering still works, but a burst
can take longer to drain. The visible counter counts active/pending keys, not
ROM-consumed characters. Capacity is 128. Overflow cancels pending typing,
pauses execution, and rejects the remainder of the burst until explicit cleanup;
it does not reset the machine. **Clear queued keys**, focus/lifecycle cleanup,
or model replacement cancel pending presses even after all host keys are UP.
ON bypasses and cancels the typing backlog rather than waiting behind it.

Native terminal characters retain supported assisted shifted-punctuation
composition (IQ comma, PC punctuation). IQ colon composition is intentionally
unmapped: the proposed SHIFT+period sequence produces a period in MEMO,
including with separate taps. Native unsupported characters report `unmapped:`
in the status instead of silently inserting a different character.

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
  on-screen controls. The browser follow-up uses host Shift+Equals for plus and
  Shift+8 for multiply, checking both `2+2=4` and `4*3=12`.
  Browser captures are actual emulated LCD pixels; decoded
  text is only an assertion aid and can show `?` for unrecognized glyphs.
- Browser live-run follow-up: boot IQ via a fixed initial budget, then use
  ordinary paced **Run**, DOM key down/up (150 ms hold, 100 ms release gap), and
  no per-character stepping. Enter `ABC+2,3`, newline, `DEF` in MEMO; store and
  reopen it through the normal ROM UI. Uses Shift+Equals, F9 then K for comma,
  and Shift+Enter for newline. This is a tested human-scale input cadence, not
  a guarantee that every typing speed or overlapping chord is accepted.

Basic input coverage is not exhaustive application, every-key, repeat/debounce,
or real-hardware qualification. In particular, **PC-E500 BASIC acceptance is
blocked**: both its direct BASIC key and main-menu PF1 path stop at the same
invalid-register-pair decoder guard. The guard is not relaxed here. CAL input
passing must not be described as BASIC passing. Investigate that CPU/decode
failure next with a minimal ROM-backed repro, separately from keyboard mapping.

An additional calculator-state issue is still unqualified: after the sequence
`2+2 ENTER`, Clear, `4*3 ENTER`, Clear, a slowly stepped `1` displayed `0.1`.
The same post-multiplication state made a fast `11+22 ENTER` produce `22.11`.
Slow raw and buffered `11`/`22` entry in a fresh calculator behaved normally.
This is not evidence of a dropped host key, and the input patch does not change
calculator state to hide it. Investigate Clear/decimal-entry semantics against
ROM and hardware separately; the rapid-input calculator acceptance starts fresh.

## Reproduce

Earlier basic-input stage checks: 571 Rust tests passed (six existing ignored
tests and three opt-in ROM tests excluded from that count); both new native private-ROM PTY
tests passed separately. Also passed: 26 WASM tests, 112 frontend tests, 36
bounded Chromium regressions, six selected real-ROM browser checks, Clippy,
Svelte/TypeScript and formatting. All five changed Rust files passed the
source-annotation check; this does not claim the pre-existing repository-wide
annotation failures have been fixed. The previously recorded 18 WASM
feature-gated dependency warnings remain. That stage's browser proof advances
deterministic boundary budgets; it is not a hardware-timing benchmark.

Browser keyboard UX follow-up (2026-09-06): 125 frontend tests, 43 bounded
Chromium regressions (eight opt-in skips), seven selected private-ROM Chromium
tests (including the live-run MEMO check above),
Svelte/TypeScript, formatting and production build passed. Rust/WASM core
sources were unchanged in this follow-up. The bounded public-browser suite
covers contact ownership, shortcuts, focus, mode changes, unsupported symbols,
IME exclusion, repeat suppression and pressed-key highlighting. Real-ROM tests
remain opt-in.

Fast-typing/catch-up follow-up (2026-09-06): 134 frontend tests, 46 bounded
Chromium regressions (11 opt-in skips), ten selected private-ROM checks,
Svelte/TypeScript, formatting and build passed. Zero-delay IQ input preserves
`AABBCCDDEE1122` with acceleration both disabled and enabled; fresh PC CAL
preserves repeated digits in `11+22 ENTER = 33`. In two local runs with decoded
LCD text visible, the IQ burst reached the display in approximately 4.9 seconds
without catch-up and 2.9 seconds with it. These are sample end-to-end timings,
not a universal latency guarantee or hardware benchmark. No Rust core, ROM
keyboard FIFO or guest application state was patched for this fix.

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
