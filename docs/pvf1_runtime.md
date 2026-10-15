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

The logical PV board shares `10000–1FFFF` with the live `B0000–BFFFF` RAM
page. PV's own sequential acquisition matches 65,487 of those 65,536 bytes;
the runtime shares backing and does not replay acquired RAM. Non-device spans
`00000–03FFF`, `05000–07FFF`, `09000–0EFFF` and `0F200–0FFFF` read `E0`
and have no writable backing. Known peripheral routes take precedence, and
unknown peripheral ports remain strict. This prevents a stale invented size
word at `00402` from shortening a longer native editor extraction. Physical
write aliases and every mapper state remain unmeasured.

Retained images keep their original 16 KiB low-backing field for format
compatibility; those bytes no longer affect mapped low-bus reads.

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

Firmware also enables ordered LCD window clipping. PV-F1's native CityChange
callback draws its list in window2, then Calendar redraws through window3.
Earlier non-bypass descriptors protect the foreground list from that redraw.
The same rule preserves active input boards, search controls and the Mini clock
that were previously erased by parent drawing. A direct Numeric-board contact
edits the native time field; these controls are active ROM contexts. Native city
selection and fresh retained restart are checked, and LCD reset preserves this
profile setting. Other PV-F1 profiles retain selected-window clipping. This is
software evidence from the acquired firmware; the silicon truth table remains
unverified.

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
checks also reopen the entry after a retained-state restart. The panel exposes
ON and the ten contacts in the ROM's two-column keyboard table. The former
"Focus" contact1 is C/CE: raw0C translates to001A and dismisses the native
Daily Alarm notification. Menu6 opens Edit; C/CE closes it. Contacts9/13 move
to the previous/next calendar month; Function5 followed by C/CE opens date
entry. The panel uses Escape for C/CE, F7 for Menu, F9 for Function, Page Up/Down
for previous/next and Pause for OFF. These are functional aliases; physical
case geometry/legends remain unverified. There is no inferred alphabetic mapping.

The functional application panel sends normal raw pen contacts through the
ROM's two-column region table beyond the main LCD. It exposes Calendar,
Schedule, Telephone, Business cards, Action List, Memo, World Clock, Drawing,
Calculator, Optical communication, Add-in Software, Card, Action Planner and Search. Native
Rust checks open these routes; a focused calculator consumer displays 5 after
normal 2 + 3 = input. Optical communication exits with C/CE. Opening an application
does not qualify its complete behavior or a physical exchange. Native drawing
creation, Modify and fresh retained reopen now preserve the checked pixels.
Card delivers raw980,770 (logical40FF), matching the printed physical control.
Without media it leaves the checked application unchanged. With the qualified
PA-9C3 program image, native keyboard End and the Schedule contact return to
saved TEST; Card reopens PA-9C3 and another End/Schedule returns to that same
day. This qualifies the program-card route; formatted S1/S2 memory-card volume
operations remain unverified.

Native consumers can configure the verified PA-9C3 before execution with
`configure_pvf1_pa9c3_candidate(rom, sram)`. The PV-F1 acquisition confirms both
128 KiB ROM views match that card image. The adapter separates MMIO,
external-media and main-RAM views, then mounts immutable ROM and passive card
SRAM. The prior broad callback shadowed executable card ROM and failed scheduler
preflight at native entry 59F94. MMIO remains quarantined from instruction fetch.
Image, duplicate, profile and running-state guards are checked.

SSR.CI (bit1) presence and SRAM aliases remain explicit wiring/mapping candidates.
Normal PV-F1 initialization executes the card ROM and reaches its foreground
input screen. Native AUTO arithmetic, serial SAVE/LOAD and pen PRO line editing
have fresh-SRAM restart checks. Full card UI/mode coverage and a complete
emulated bank-dumper session remain open. The SRAM alias, presence and input
buffer are functional candidates, without physical mapping verification.

Firmware now installs the shared register-only UART with the PV nominal
1,024,000-unit clock and divisor two. Native serial 43 maps format code 60
to UCR 70; this gives nominal 9600/8E1, matching the acquired stream settings.
The clock/divider remains a functional timing candidate. ControlCandidate
can opt in with `configure_pvf1_uart_candidate`; Strict remains unchanged.
There are no PC IOCS replacements, automatic peer replies or guest receive-data
writes. Native serial 41 sends ABC as scheduled completed frames. Public FFFD8
consumes incoming Z and parity/framing variants once and returns their status.
Those IOCS checks use an explicitly initialized workspace. Native foreground
open/RX IRQ/ring consumers and a Port15 peer transfer are also checked:189 bytes
reach a receiver, which reopens TEST after a fresh RAM/RTC-backed restart.
Other formats/rates and broader peer protocols remain open.

The browser exposes the shared Serial panel for PV-F1. It admits bytes only
while the native UART is enabled, keeps raw output separate from LCD delivery,
and resets capture credit on a new machine generation. A changed-host smoke
transfers the same 189-byte native Port15 payload through `SerialDelivery` and
`queueSerialInput`, then reopens TEST with a fresh receiver. The normal firmware
controls UART setup, IRQ processing and storage; the host supplies wire bytes.

The PV-F1 card controls select the verified 128 KiB PA-9C3 ROM and 32 KiB SRAM for
the next load. Import card and restart preserves organizer RAM/RTC and installs
selected card assets into a fresh paused machine. Export current card SRAM
saves the firmware-mutated card separately from organizer backups. Session reset
and organizer backup restore preserve the currently mounted ROM/SRAM; selecting
new media and importing it explicitly replaces that media. Restart without card
keeps organizer backing. Plain ROM replacement still loads the selected assets.

`load_pvf1(bundle, retained, model, card_rom, card_sram)` validates all assets in
a candidate before replacing the current runtime. Empty retained/model/card
arguments select fresh main backing, no recognizer and no card respectively.
The changed-binding smoke rejects malformed/incorrect ROM, SRAM, retained state
and model without replacement, checks native AUTO BASIC5, and checks reset,
restore and removal preservation. This reuses native Rust behavioral evidence;
it does not qualify physical card mapping or a hardware transfer.

The browser's automatic pacing adapter accepts a zero-progress `host_yield`
when Rust's four-millisecond host deadline expires before the first batch.
Input holds advance only by executed scheduler boundaries. Invalid budgets,
missing results and false zero-progress budget completion still fail closed.

## Evidence limits

The acquisition and private proof repository retain the source-bound native,
WASM and browser receipts. All 39 LCD IOCS slots have a static inventory and
all 31 non-error commands have successful focused native cases. These cases
do not qualify every context, raster mode, font, codec or error branch.
Native menu contacts now open clock/daily-alarm settings. Numeric11:23AM entry
commits through native RTC IOCS; an enabled daily alarm wakes native OFF,
displays the ROM notification and rearms the next day. Firmware infers an
ON/power-wake level only from enabled RTC alarm causesA0/A1, excluding periodic
sources. Strict preserves its prior policy; ControlCandidate has a guarded
pre-execution `configure_pvf1_alarm_wake_candidate` comparison. Physical wiring
and timing are unverified. C/CE notification dismissal and normal Menu return
have focused native Rust checks and a UI-map-to-WASM smoke.
Remaining work includes broader applications, auxiliary-mode input, remaining
clock/date/city paths and PV-F1 card/serial routing. No physical LCD or timing qualification
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


## Explicit handwriting controller replay

The Rust runtime exposes `supply_pvf1_controller_result_fixture(index, records)`
for an explicit type-1 controller response. Indices 0..3 and up to ten opaque
three-byte records follow the native FE `DD4DF`/`DD511` consumers. This API
supplies peripheral data; the ROM owns interrupt delivery, query, copying and
rendering. It installs no recognizer and does not interpret pen samples.

Result-ready is separate from sample-ready. Native GA source 6 reaches fixed
`FF463`; `F1A94` acknowledges it with `08A08=BF`. The candidate query then sends
`FC,02,index`, waits for transport completion and acknowledges with `08A03=BF`.
Indexed response bytes survive the earlier interrupt acknowledgment. Reset
cancels pending fixture state. PI command admission remains unchanged.

Standalone Rust checks cover zero, one and ten records through unmodified
`DD511`, plus a normal Handwriting selection and native sample upload followed
by a supplied `82,73,00` response. The native IRQ/query path displays `Ｔ` in the
editor. This proves response transport with a fixture, not handwriting recognition.
The default has no result provider; result type 2, multi-result sequencing,
physical timing and stroke classification remain open.


## Portable handwriting inference

`configure_pvf1_recognizer(Some(model_bytes))` attaches a replaceable
Zinnia-compatible model before boot or with the native capture editor closed.
The pure Rust shared `handwriting` module
loads version-1 sparse linear models and extracts ordered-stroke features.
It also accepts an optional hybrid model containing an unchanged version-1
Japanese engine and ordered-stroke shape templates for Latin recognition.
PV-F1 owns the native markers, four character slots, canvas geometry, CP932
conversion and result transport. Reset preserves the model and clears gesture
state. The Python mirror implements the same component boundaries.

Native uploads use 8000/index to select a character slot, FFFF/FFFF
to end a pen stroke, and 00FC to finish the idle character. In the qualified
Handwriting case, native03FE supplies a 138-by-222 canvas. Normal pen movement
for two-stroke hiragana `い` produces126 uploads, one classification and one
native result IRQ/query. The classifier ranks `い` first and the ROM renders it
without a supplied result record. The Japanese model has6420 classes and is
loaded externally; its license/source and model identity belong to that data
package. It is not embedded in the CPU or an implementation of Sharp's original
recognizer. The default unconfigured controller still has no provider.

Native recognition, adoption, save and fresh-boot reopen now have regressions.
Four successive native boxes query indices0..3 and retain/reopen `い井い井`
in order. The native foreground finishes each character's recognition before
the next begins. Rapid overlapping trials can upload incomplete strokes and
produce different candidates; they are not qualified accuracy results.
The initial persistence driver selected an older TEST record; selecting the
second day-list row at (100,98) normally reopens the saved handwritten `い`.
Stored TEST remains intact; storage code needed no repair.

The native runner accepts `--pvf1-handwriting-model PATH`. The browser's
Handwriting model input installs a supported `.model` asset while paused, keeps
it for subsequent ROM reloads in the session and rejects replacement during
active capture. Host model replacement preserves native canvas geometry.
For a configured local launch, set `PVF1_HANDWRITING_MODEL_PATH` alongside
`PVF1_ROM_PATH`. Initial PV-F1 auto-load stages that model with the ROM and
validates it through the existing Rust loader before the first instruction.
The selected asset/name is cached only after installation succeeds; a manually
selected model takes precedence on later ROM reloads. An unconfigured server
leaves the normal model picker available. No firmware or model is embedded in
the public source tree.
The browser also exports `.pvram` RAM/scratch/RTC backups and restores compatible
backing with the selected model into a fresh paused runtime. ROM/bank identity,
payload checksum and model validation complete before the old machine changes.
The PV host queue preserves ordered deferred motion while paused or behind
another stroke. Distinct points replay at20,000 scheduler boundaries; tap holds
and50,000-boundary stroke gaps remain. The4096-point bound cancels overflow,
and clear/discard cancels held/pending movement. PI/OZ use their existing default
policy. One changed-host integration submits both い strokes before execution,
delivers all55 distinct coordinate/release updates and renders い through the
normal native controller. With the old queue and the same core binary, the
incomplete path instead renders 八. This is assisted input, not measured timing.
Broader handwriting accuracy and physical type-2 triggers remain unmeasured.
Normal application/card flow coverage is described below. Routine
behavioral checks use standalone Rust; WASM checks only the changed bindings.

The hybrid engine centers each character, preserves its aspect ratio and
resamples each ordered stroke into 32 arc-length points. It compares only
templates with the same stroke count. Latin recognition is selected only when
the nearest Latin shape's squared distance is less than the configured fraction
of the nearest Japanese shape's distance. Ambiguous shapes and stroke counts
without Latin templates use the original Japanese classifier, including its
unchanged candidate scores. Latin candidates are ranked by negative shape
distance and deduplicated by label. These distances are not probabilities and
are not compared with the original classifier's scores.

One explicit host model uses 4,960 Latin training samples from 40 writers and
6,410 Japanese routing templates, with a distance ratio of 0.75. It supplies
6,472 output classes. Standalone Rust evaluation recognizes both captured T
inputs and preserves the two Japanese い results. All 225 reserved Japanese
samples retain their original first choice; these samples may have been in the
original Japanese model's training set. The separate 20 Latin test writers give
1,747/2,480 first-choice and 2,353/2,480 top-ten hits. These are measured limits,
not a claim of comprehensive handwriting accuracy or Sharp engine equivalence.

The native ROM regression captures four slots for TいTい, adopts and saves the
text, exports retained backing, then boots a fresh runtime and reopens it through
Modify. Both the record and editor contain ASCII54, CP93282A2, ASCII54, CP93282A2;
the original TEST record is intact. No supplied result or guest memory patch
provides the text. The existing model remains optional and host supplied.

Hybrid binary format: `PVHW02\0\0`, followed by little-endian base byte length,
template count, samples-per-stroke (32), and float32 distance ratio. The complete
version-1 base follows unchanged. Each template contains a NUL-terminated UTF-8
label in 16 bytes, a little-endian stroke count, then 32 float32 X/Y pairs per
stroke. The parser rejects nested hybrid bases, invalid configurations, nonfinite
or out-of-range points, truncation and trailing bytes. Japanese routing templates
do not introduce new output classes. Legacy version-1 model handling is retained.

## Action Planner keyword matching

Native related-word queries now match the full-width TEST schedule and TASK
Action List records. The query's Input contact executes the search; the result
view's top-right contact opens Report creation. Report cancellation returns to
the intact result list. A positive report trial creates one Action Board page.

The failing multi-character path exposed a shared CPU alias error. At bankFC
C4B37, INC IL/DEC IL inspect a pattern word's low byte before SUB BA,I. Those
instructions must preserve IH; routing them through the IL value-load setter
changed8264 to0064 and rejected the second character. The scoped Rust and LLIL
correction preserves IH, low-byte wrap and carry, and derives zero from IL.
IL value-load behavior retains its existing high-byte clearing. Tiny Rust
opcode/word-compare regressions and normal unmodified-ROM query consumers cover
the correction. Normal report/edit/navigation flows are also covered below.


## Normal built-in application flows

The acquired PV-F1 firmware has a closed software inventory of 304 grouped
normal-flow rows: 294 verified and 10 justified inapplicable on the unit without
installed add-ins. This includes ordinary create/edit/delete/search/navigation,
settings, text and graphics clipboard, Planner reports, card startup, backup,
connection recovery and fresh retained restart. A disabled Move menu entry was
observed in the reviewed built-in contexts; no universal absence claim is made.

Qualification uses one standalone Rust core. The 93-case integration milestone
was followed by focused repairs and reruns of its 19 failures, with unchanged
passing evidence reused. Current web artifacts are rebuilt from the same Rust
source; loader bindings are checked without a second CPU-flow campaign. These
software results do not establish physical bus timing, accessory interoperability
or comprehensive handwriting accuracy.
