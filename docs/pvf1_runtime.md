# PV-F1 runtime checkpoint

The shared Rust factory accepts `DeviceModel::PvF1` when built with the `pvf1`
feature. It loads a verified `PVROM01\0` bundle and runs the acquired firmware
with fresh RAM. The bounded scheduler coordinates native ESR-L execution and
the PV-F1 legacy gateway. The WASM loader and browser use this same runtime.
Initialization, calendar rendering and a schedule-view tablet interaction have
been checked in native Rust, WASM and the browser. This remains an experimental
model; physical calibration and broader application behavior are incomplete.

## Components and model boundaries

`pvf1::Hardware` owns the PV-F1 bank selector, gate registers, RAM, auxiliary
port gating and GPIO feedback. It reuses the shared LCD controller, RTC and
tablet ADC with explicit PV-F1 geometry and pressure parameters. It also
implements the PV-F1 packed-pixel expansion port at `0401E–0401F`:
one written byte returns sixteen little-endian bits with each pixel doubled.
The acquired generated-font routine uses it to produce a 16×32 glyph from 8×16
source data; native Rust and Python getter/render consumers check that result.
The mapped-port operation is checked in WASM, separately from native IOCS proof.
The shared ESR-H CPU uses a separate PV-F1 bus and gateway coordinator. Native and legacy
instructions retain separate counters; legacy instructions do not manufacture
native cycles. `LcdView` observes the board's existing pixel owner.

PI-3000, PA-9500/DB-Z, OZ-9600 and IQ-7000 retain their own factories, board
routing and model profiles. The `pvf1` feature depends on `pi3000` to reuse its
auxiliary transport engine and bounded legacy CPU, without adopting PI mapper
or mode-register addresses. The Python PV-F1 modules mirror the board contracts
and provide an independent legacy replay bus.

## Bundle and bootstrap profiles

The bundle consists of an eight-byte magic, a little-endian JSON length, the
acquisition manifest, fixed 128 KiB firmware and all 32 logical 128 KiB bank
views. The loader checks exact lengths, schema, sorted selectors, offsets,
individual hashes, the complete logical-bank-set hash and the fixed bank-FF
view. The factory validates a replacement before altering the running machine.
Equal logical banks remain separate views with acquisition provenance.

The factory's `Firmware` profile enables explicit candidates: BP starts at D0,
LCD save/restore uses coupled predecrement, byte arithmetic truncates wide sources
with the explicit `LowByte` candidate, auxiliary control replies are gated
by PV-F1 firmware bytes, absent expansion reads FF, and tablet pressure is 1023.
Timers start at the first native HALT. The main timer then advances during HALT
only when LCC bit 7 is set, excluding OFF. These choices reproduce the bounded
firmware path; they are not measured oscillator, peripheral or CPU contracts.
Firmware also selects an auxiliary capture clock candidate of 4096 native
cycles. With 03FC capture active and control 06 bit7 set, a sample-ready latch
raises GA source 3 (08); 05 exposes 80 until 07=7F acknowledges it. Native
FF575/FE DB508 then sample the shared ADC and build guest pen events. Control 06
reads back through 04 as established by the acquired diagnostic. No host
character or recognition/query reply is fabricated. The cadence and edge
wiring remain hardware hypotheses; the runtime API can disable or vary the
candidate. ControlCandidate retains a disabled sample clock.

Direct `pvf1::configure` callers can select `Strict`, `AbsentAuxiliary` or
`ControlCandidate` instead. A loaded PV-F1 reset requires constructing a fresh
runtime so board state cannot survive accidentally.

## Browser use

Serve a locally generated bundle using `PVF1_ROM_PATH`, then select `pv-f1` or
open `/?model=pv-f1`. The browser starts paused. Resume, send a raw tablet contact
at X=362/Y=585 to accept the fresh-memory initialization dialog, then use
X=940/Y=64 to open the schedule view. The raw controls accept ten-bit samples;
LCD clicks read live IOCS calibration coefficients and sample each pixel center.
The main 336×240 viewport accepts press, movement and release; the remaining
52 controller-storage rows are excluded. Native conversion fixtures and normal
WASM/browser clicks open New Entry and its description editor. Physical
calibration, auxiliary capture/recognition and typing inside that editor remain
unqualified. The native auxiliary sample-ready route now opens Board Switch
and Typewriter; normal calibrated contacts produce a full-width Latin TEST
payload and the firmware's saved-data notification. Native Rust and WASM
agree on eleven foreground frames and both observed RAM copies. Native ROM
checks also reopen the entry after a retained-state restart. ON and matrix Focus are
the currently verified virtual keys. There is no inferred alphabetic mapping.

The browser's automatic pacing adapter accepts a zero-progress `host_yield`
when Rust's four-millisecond host deadline expires before the first batch.
Input holds advance only by executed scheduler boundaries. Invalid budgets,
missing results and false zero-progress budget completion still fail closed.

## Evidence limits

The acquisition and private proof repository retain the source-bound native,
WASM and browser receipts. All 39 LCD IOCS slots have a static inventory and
all 31 non-error commands have successful focused native cases. These cases
do not qualify every context, raster mode, font, codec or error branch.
Remaining work includes broader applications, auxiliary-mode input, RTC clock
setting/alarms and PV-F1 card/serial routing. No physical LCD or timing qualification
is claimed, and this checkpoint does not establish a fully working emulator.

## Native saved-entry wait and RTC A2 candidate

Firmware now enables a PV-specific nominal half-second RTC A2 divider. Fixed
F1BEB loads BFE15=F210A; that native callback increments BFE32. IOCS0049 with
CL14 waits on this counter, independently of the CPU MTI/STI handlers. Without
A2, the normal saved-data notification remains visible. The shared RTC latch
and native acknowledgement own delivery; no guest counters, masks or queues
are patched. The divider consumes completed timing units in running, HALT and
OFF states. Strict stays disabled; ControlCandidate enables it only through
configure_pvf1_rtc_half_second_candidate. Period zero is rejected atomically.

The unmodified ROM now stores Typewriter text, leaves the notification for its
day view, and reconstructs the editor text through Modify. Native and WASM
consumers agree on the reopened frame and full-width Latin CP932 bytes. A
slower divider trial reaches the same stored text with a different cursor
phase. This is software qualification of a nominal timing candidate; physical
alarm/power wiring, HOLD behavior and recognition remain unqualified. Calendar
progression and retained restart have subsequent native Rust checks. The RTC
component and Python logical reference are separate
from the firmware-specific mapper and auxiliary protocol.
## Retained restart backing

`CoreRuntime::pvf1_retained_state()` exports the board's main RAM, scratch RAM
and raw RTC registers. `restore_pvf1_retained_state(&image)` validates exact
length, fixed-ROM identity, ordered bank/layout identity and payload checksum
before copying backing into a fresh board, prior to its first scheduler
boundary. CPU, LCD, mapper, input, GPIO/gate, auxiliary and clock phases are
fresh. The emulator-only `PVRAM01` format does not identify physical
battery-backed regions. Native ROM checks save and reopen Typewriter TEST
after a fresh restart through normal calendar/Modify contacts.

The standalone `pce500` CLI accepts `--pvf1-retained PATH` and
`--pvf1-retained-out PATH` with `--model pv-f1 --rom verified.pvrom` and the
`pvf1` build feature. It preserves PV-F1 factory ownership of reset, card
mapping and first-HALT timer initialization. Normal UI clock setting,
alarm wake and handwriting recognition remain incomplete.

## Advancing calendar candidate

Firmware now advances the RTC calendar at a nominal 1,024,000 scheduler units
per second, including HALT/OFF idle. The separate A2 source stays at 512,000.
Strict is disabled; ControlCandidate opts in through
`configure_pvf1_calendar_candidate(Some(period))`, or disables with `None`.
Zero rejects without changing phase. The model freezes calendar phase during
HOLD and skips invalid/uninitialized dates without synthesizing a date or
replaying skipped time. Reports expose phase, seconds and held/skipped units.

Native PV-F1 FC getters establish the shared packed year/month/day/hour/minute/
second encoding. Public IOCS device 0A reads/writes the five mapped RTC bytes
with its own HOLD protocol. A retained unmodified-ROM boot crosses midnight,
and the native Typewriter save/restart/reopen regression still passes with
progression enabled. Cyclic/deadline matching, HOLD freeze and the oscillator
rate remain candidates; normal UI clock-setting and alarm/ON behavior need
further foreground checks. This is not hardware timing qualification.

## Native OFF/ON resume

The Firmware profile admits hardware IRQs using the guest IMR master and source
masks even when the diagnostic interrupt monitor remains active after a native
context change. Strict retains monitor gating; an explicit pre-execution
`configure_pvf1_irq_imr_candidate` switch supports bounded comparisons. This
change is specific to PV-F1 and does not alter the related model policies.

A normal physical matrix contact at column0/row7 reaches the acquired system41
power routine and its OFF instruction. CPU execution stops while RTC time
continues. Releasing that contact and briefly pressing ON resumes the native
calendar, RTC callbacks and saved-entry Modify operation. The private Rust
suite includes this flow. The matrix contact's physical
key label and electrical timing remain unverified. No guest flags, queues, counters or pixels are
patched to produce this result, and alarm wake is not yet qualified.

Firmware also samples SSR.ONK from the live external ON contact. Native
power execution can retain an old ON bit in SSR backing; an OR-only read
would then report held ON after release, leaving the native abort flag set.
The live-input candidate replaces only SSR.ONK on architectural reads and
passive CPU peeks. It preserves other SSR bits and leaves backing and latched
ISR.ONKI acknowledgement to the guest. Strict keeps the original policy;
ControlCandidate can opt in through the pre-execution
`configure_pvf1_on_input_candidate` API.

The native suite now checks scalar and word input reads and OFF/ON holds of
50,000 and150,000 boundaries. A separate1,000,000-boundary hold also resumes
and reopens stored TEST. Firmware consumes its own abort notification after
release; no host clearing is used. All26 native cases pass. Electrical timing,
debounce and alarm wake remain unqualified.
