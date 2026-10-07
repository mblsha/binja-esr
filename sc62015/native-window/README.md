# OZ-9600 native bitmap window

This host frontend uses `CoreRuntime::for_model(DeviceModel::Oz9600, bundle)`.
The LCD is the complete 336 × 240 controller observation. Fixed left-panel
artwork and physical keycaps sit outside it. The bezel is an approximation of
the user's unit; the physical LCD crop and annunciator wiring remain unresolved.

From the public repository root:

```sh
cargo build --release --manifest-path sc62015/native-window/Cargo.toml
sc62015/native-window/target/release/oz9600-window \
  --rom /path/to/verified.ozrom \
  --profile provisional-v1 \
  --retained-out /tmp/oz-retained.bin \
  --capture-prefix /tmp/oz-screen
```

Add `--retained /path/to/saved.ozbat` to recover saved records. For empty memory,
touch initialization YES, wait for Welcome, and use ADJUST to set the clock.
The default profile is **strict**; `provisional-v1` must be selected explicitly.
It selects initial CPU BP=D0 for all-zero SRAM and BP=00 for populated saved
SRAM before execution, without guest-memory repairs or injected callbacks.
It retains the shared
core's provisional arithmetic, interrupt and clock assumptions. UI integration
does not qualify hardware reset, timing, SRAM aliases, serial or card mapping.

Keyboard letters/digits, arrows, Space, Enter, Backspace, Delete and modifiers
drive physical matrix contacts through the shared key table. Escape is Cancel;
Alt is 2nd; Page Up/Down are Prev/Next. F1 is New Entry, F2 Edit, F3 Menu and
F6 Symbol. Numpad operators map to their physical calculator contacts. Pointer
presses on the printed panel and LCD drive raw tablet samples using the ROM's
default calibration. Those samples have not been measured on a physical unit.

Clock is a momentary popup. Hold the printed Clock target to keep the ROM
view visible; release closes it after input processing. Exact physical release
timing is unqualified.

ON and OFF keycaps appear below the fixed panel. F12 holds the separate CPU
ON contact; Pause holds the matrix OFF contact. The ON key uses the same shared
API as physical replays and the browser. It does not expose a guessed RTC B0
cause or qualify the RTC's electrical ON-output route. Manual ON resumes CPU
execution under the experimental RTC profile; previous-application restoration,
automatic alarm wake and alarm dismissal remain unresolved.

F9 runs/pauses, F10 steps 20,000 boundaries, F11 steps 1,000,000, F5 resets
while preserving logical retained memory, F7 captures, and F8 saves. Host
controls are also visible below the keyboard. Faults stop execution until reset.

`--execution-mode interactive` uses nominal shared pacing; `turbo` runs bounded
slices without throttling; `deterministic` requires explicit step budgets.
`--paused` starts paused. Very short contacts receive 40,000 scheduler
boundaries of host assistance by default; `--minimum-contact-boundaries 0`
selects immediate release. Focus loss cancels all contacts immediately.
Ordered OS key/mouse events preserve short taps and modifier ownership.

`--replay` applies validated physical inputs before opening the window.
`--headless --replay PATH` uses the same factory without an OS window.
`--replay-report` exports the ordinary CPU/peripheral observations. Captures
include unscaled `.pbm`, host `-window.ppm` and read-only `-state.json`.
`--event-log` records accepted live contacts and exact boundary budgets at
close; concatenate it after the startup replay to reproduce the session.
Reset is disabled during recording because each trace has one reset epoch.
`--host-status PATH` enables read-only host-input diagnostics.

The frontend uses [winit's ordered window events](https://docs.rs/winit/0.30.13/winit/event/enum.WindowEvent.html)
and [softbuffer's window surface](https://docs.rs/softbuffer/0.4.8/softbuffer/struct.Surface.html).
Cursor and surface dimensions use physical pixels, so rendering and tablet
conversion share the same scaling on a Retina display. Python layout/contact
references live in `pce500/oz9600/ui.py`.

On macOS, create a local app bundle after building:

```sh
uv run python sc62015/native-window/package_macos.py
open -n 'sc62015/native-window/target/release/OZ-9600 Emulator.app' --args \
  --rom /absolute/path/to/verified.ozrom \
  --profile provisional-v1 --retained-out /tmp/oz-retained.bin
```

The app requires access to the window server. Builds, ROMs and captures remain
local. macOS was exercised directly; other desktop platforms are not yet
qualified by a live-window run.

An optional OZ-707 BASIC card uses separate ROM and SRAM files:

```sh
sc62015/native-window/target/release/oz9600-window \
  --rom /path/to/verified.ozrom \
  --profile experimental-isr-mti-writable \
  --retained /path/to/rom-generated-retained.bin \
  --card-rom /path/to/verified-oz707.bin \
  --card-sram /path/to/oz707-sram.bin \
  --card-sram-out /tmp/oz707-sram.bin
```

Both card inputs are required together. The shared factory checks the exact
128 KiB ROM identity and 32 KiB SRAM size before execution. Card selects the
application through normal ROM input; the guest draws and handles its keypad.
The host sends raw tablet coordinates rather than translating keypad cells.
F5 preserves both current main memory and card SRAM; F8 saves either configured
output. Card SRAM remains separate from the main retained-state format.

This logical slot uses an explicit provisional SSR bit-1 presence signal,
repeated ROM views and an upper writable SRAM view with a lower read-only
mirror. Normal-input replay checks SIN, inverse SIN, RUN/PRO, storing
`10 PRINT 2+3`, running it and recovering it after reset. Physical slot
wiring, mapper aliases and the remaining card functions are unqualified.
