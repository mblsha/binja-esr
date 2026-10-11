# PI-3000 native runtime

The `pi3000` Cargo feature adds a PI-3000 board adapter to the shared
CoreRuntime. It reuses the native CPU, matrix LCD controller, RTC reference
and bounded ESR-H interpreter. PI ROM mapping, controller geometry and input
ports come from the acquired PI firmware. This is an implementation in
progress. Native and browser raw input complete calibration, pass contrast
setup, initialize fresh memory, enter the organizer calendar and select dates.
The selected numeral blinks through ordinary ROM-controlled context updates;
the previous selected numeral reappears at its original date cell. Native and
browser idle samples agree on both complete framebuffer phases. Raw matrix
input opens the date dialog and context menu, changes months and reaches OFF;
ON resumes the calendar after its normal redraw. Complete organizer navigation,
keyboard/RTC/power qualification and cards remain open.

The calendar's New Entry control also opens the Schedule editor. FD:CECA9
executes `SUB A,I` (`4E03`) directly while constructing its controls. The
normal Firmware profile uses the existing ESR-P `LowByte` arithmetic-source
hypothesis: wider sources are truncated before byte arithmetic and flags.
Strict decoding remains available and rejects this form before commit.
This reuses the shared CPU policy without changing other device defaults;
the silicon contract remains unqualified. Native fresh-entry output has an
exact full-frame regression. The browser checks the invariant Schedule header
and retains the entire body, whose cursor/input phases still need recovery.
Schedule content input, validation and stored-record workflows remain open.

The normal factory accepts only a `PIROM01` bundle of the verified fixed ROM
and 32 complete selector views (00..0F and F0..FF). It does not accept a mapped
1 MiB dump as startup RAM. The private acquisition tool packs the bundle;
ROM data is not distributed with this source tree.

```sh
cargo run --manifest-path sc62015/core/Cargo.toml --features pi3000 \
  --bin pce500 -- --model pi-3000 --rom /path/to/pi-3000.pirom --steps 20
```

The normal model factory starts fresh scratch/high RAM with the `Firmware`
startup candidate. It combines initial BP=D0, MTI-writable/otherwise-clear-only ISR writes, coupled
predecrement transfers, nominal organizer timer periods, the LCC.bit7/HALT
main-clock candidate, the periodic RTC A2 candidate, integrated ADC and Gate1
press edges, plus the capture-derived E0 read candidate for 05000..07FFF. These settings are
supported by ROM consumers; physical reset state and timing remain unqualified.
Unsupported bus operations stop and poison the runtime.
F06CA sets ISR.MTI (bit0) after its masked keyboard HALT. F13A9/F1BC5 dispatch
that request, and F1BEB calls the ordinary FBB42/FBD9A matrix scanner. FBE41
translates contacts through the ROM table at FBAC6; FC151/FC113 produce press/
release records. The previous clear-only setting suppressed this software
request. The shared `ClearOnlyExceptMti` CPU policy now permits it without
allowing guest software to create other IRQ sources. This is a ROM-derived
operational policy; the silicon write truth table remains unqualified. The
private CLI retains `--isr-policy clear-only` for the previous controls.
The sequential physical acquisition returned E0 at all 12288 bytes in that
unmapped window. The read-only candidate supplies that value without creating
RAM or permitting writes. Firmware initialization makes ignored cleanup calls
with invalid filename pointers; allowing the resolver to return its ordinary
error21 lets startup proceed. E0 and FF controls produce identical complete
calendar states. Electrical bus-hold behavior in other execution contexts is
not qualified. `Strict` and the explicit configuration API retain the trap
control; `configure_pi3000_unmapped_low_read_candidate(value)` selects the
candidate before execution, and `pi3000_status` reports its value.
`configure_pi3000_rtc_calendar_candidate(period)` separately opts into packed
calendar advancement before execution. It uses independent scheduler time in
running, HALT and OFF states, freezes calendar phase under CONTROL.HOLD, and
never seeds a date. The separate `Hardware.rtc_event_candidate` policy generates
A3/A4/A5/A6 for
second/minute/hour/day boundaries and A0/A1 for ordinary alarm deadlines in
08010/08018 with sixth-byte 07. PI's FC:C1938/C196A getter/setter fixtures
confirm the shared packed fields; C1A09 programs ordinary A0 deadlines and
raises A0 immediately for a due request. The second alarm's matching, other
wildcard fields, weekday names and physical timing remain unqualified. Invalid
dates stop rather than being repaired. Calendar-only behavior remains available
with the event policy disabled, which is the default in every profile. This
option is disabled in every bootstrap profile, preserving the current startup
consumer. `pi3000_status` reports its period, phase, elapsed seconds and held units.
An unknown instruction mapping fails preflight; an unknown data access stops
after its committed instruction. Generic card overlays are not applied.
Restart constructs a new board; v4 snapshots do not represent its devices.

`BootstrapProfile::Strict` retains the native-only/unknown-auxiliary control
through the explicit bundle API; periodic timers stay disabled in that profile.
`Compatibility` opts into the firmware-derived gate/page handoff.
`AuxiliaryAbsent` additionally supplies status02 and retains auxiliary requests
without completion or response bytes. The firmware can take its own timeout
path; this does not implement the auxiliary device.

`AuxiliaryControl` is a separate stateful candidate for status control headers
00FE,01FE,30FB,02FE,00F9..02F9,03FE,04FE. FE:DE211's 00FE initialization control and
DE202/DE207/DE20C's F9 controls, requested when entering Schedule content,
share DE219/DE21E's
single-byte acknowledgment path with 01FE. No command-specific response data
is read by those wrappers. DDFDB's 04FE parameter request writes three bytes;
DE006's 03FE geometry request writes eight. Both read only response byte0,
accept status01 and acknowledge BF. Their physical configuration effects are
unqualified. The model requires the exact fresh request-byte mask (two, three
or eight bytes), enters
busy, completes after 32 native timing units, exposes status01 and requires BF
acknowledgment before another request. Unknown commands and query bytes stay
unsupported. Its idle auxiliary event ports return zero; physical protocol,
device identity, timing and recognition behavior remain unqualified.

Ordinary Schedule content-label input opens the ROM's character-entry panel;
a recovered left tablet-strip contact opens Phonebook 1. Further contact
with BFF20.bit7 set requests 03FC through FD6EA. This command completes and
acknowledges BF without reading response bytes; 05FE instead reads status01.
The prior analysis annotation had these two branches reversed.

The experimental Firmware profile now models a capture session between
03FC/05FE and a sample-clock candidate enabled by 08A06.bit7. Gate3 reaches
FD859/FE:DBBD8. Its contact debounce needs a second stable sample before
coordinate collection; one enable pulse is insufficient. The provisional
cadence is 4096 native timing units and is inactive outside the capture
session. AuxiliaryControl and Strict keep the candidate disabled. The private
CLI retains explicit clock-disabled and period controls. Physical cadence,
wiring, parameter effects and auxiliary recognition data remain unqualified.

Normal raw contacts now select the numeric keypad, type1 and save Schedule
content. The ROM's day view shows the stored record; switching to Phonebook
and back, then using Edit, reopens it. Actual browser LCD contacts and the
Calendar/Schedule and Phonebook application controls reproduce the workflow
and its visible panel. No guest queues, coefficients or storage are injected.
The saved record also survives OFF/ON. Wake returns to Calendar; selecting
January 1, opening its weekly view, choosing the Saturday day row and then
selecting the record enables Edit. Native and actual browser contacts reopen
the same content with identical visible pixels. Complete application coverage,
handwriting recognition remain open.

The PI browser now provides Save backup and Restore backup. A backup contains
the separate 256 KiB main RAM, 16 KiB scratch RAM and 32 raw RTC registers.
`PIRAM01` binds the fixed ROM, sorted bank inventory, PI backing layout and
payload SHA-256. Restore validates every identity and length before copying
backing, and the core API accepts it only before the first CPU boundary,
including elapsed HALT time. This layout describes emulator persistence;
physical battery-backed regions remain unqualified.

`CoreRuntime::pi3000_retained_state()` reads backing without mapped reads,
acknowledgments or execution. `restore_pi3000_retained_state(image)` restores
it into a fresh runtime. CPU, LCD, mapper, auxiliary device, gate, GPIO, input
ownership and clock phases stay fresh. The browser validates a candidate
before replacing the running machine; corrupt imports leave its state intact.
Worker and no-Worker backup controls use the same WASM/core implementation.

The actual-ROM restart consumer completes touch calibration, returns to
Calendar without the erase-data prompt, selects the saved day and record,
and reopens content `1` through Edit. Native and browser visible pixels match
the original saved-record panel. The private native CLI exposes
`--retained-out <path>` and `--retained-in <path>`; the Python reference
restores and re-exports a native-produced image byte-identically. Complete
application, clock/alarm, card and serial coverage remain open.

The browser also exposes the verified Clock contact at raw `(240,572)`.
The ROM keeps Clock open after release and shows its Tokyo/New York views,
time-setting dialog and daily-alarm dialog. Enabling the default daily alarm
through normal touch input programs an ordinary next-day RTC deadline.
Entering both hour digits `0`, `1` saves 01:00; entering `0`, `1`, `0`, `1`
in the daily-alarm dialog saves 01:01. Native and actual browser input agree
on the packed RTC bytes and complete visible frames. The earlier single-digit
input leaves hour-field status bit7 set and takes the ROM's incorrect-time path.
With the separately enabled calendar/cyclic/alarm candidate, normal input and
scheduler advancement deliver the daily-alarm popup and reschedule its deadline
for the next day. Calendar-only advancement does not deliver it. A no-input
counterfactual reaches the same fragmented Clock repaint after the next minute
as the earlier contact sequence. That repaint does not establish an alarm-clear
contact; timed refresh/modal ownership, acknowledgement and audible output
remain under recovery.
The browser's Clear control and Escape use column1,row6 (scanner byte1C,
virtual contact14). They close the incorrect-time popup and clear its hour
field through the ROM's matrix scanner, preserving RTC. Native and browser
input agree on the complete visible result. Column0,row0 instead closes that
smaller popup while retaining the entered digit.
Clock redraws can require several million scheduler boundaries after closing
a dialog. These are emulator contracts, not physical timing qualification.

The historical strict ISR replacement control triggers an early IRQ before
reset has installed the handler workspace. An explicitly selected clear-only
ISR experiment reaches a four-cross calibration frame after two million
scheduler boundaries, including 29 legacy instructions. Initial CPU and ISR
semantics remain unqualified. The normal model explicitly uses the combined
`Firmware` candidate described above.

`configure_pi3000_tablet_route` selects one of the two ADC branches in
FE:DBE41. `Integrated` routes control/data to 08B80/08BC0; `External` routes
to 08800/08C00. `set_pi3000_raw_contact` supplies ten-bit samples and contact
state. It never populates firmware event queues. Strict/absence profiles supply
only samples; the auxiliary-control candidate routes press edges through the
selected gate source and the normal EX interrupt. The ADC route cannot change once selected. Both original ROM branches
return independently chosen raw coordinates with balanced stacks.

ADC completion-on-first-read, held split-byte samples, binary contact pressure,
LCD sampling sync and physical route selection remain provisional. The
controller frame is 240×336 native storage; the physical panel specification
is 239×168 and scan orientation is still being recovered.

`configure_pi3000_tablet_interrupt_source` selects `Gate3` (the unchanged
default) or `Gate1` while contact is released and no ADC reads have occurred.
Both remain physical
wiring candidates. Gate3 reaches FD859, whose interrupt sampler requires
BFF20.bit7. Gate1 reaches FD6B0, posts a foreground contact flag and reaches
FE:DBC74 -> DBE41 without that mode bit. A reset/press/release replay with
nominal timers increases ADC reads from 2 to 9 to 10 under Gate1; Gate3 stays
at two. No guest event records or mode flags are seeded.

`pi3000::input::Replay` validates raw tablet/ON contacts and boundary budgets
before execution. Startup allocates windows 10 and 11. With independent block
transfers, F6DF9 overwrites the caller's saved window 11 with FF, and FAFEB
rejects that selector before glyph rendering. The existing shared CPU
`CoupledPredecrement` policy preserves the window and renders the Japanese
calibration instructions. This policy remains an explicit firmware experiment.

`configure_pi3000_halt_main_timer_candidate(true)` is a separate clock
experiment: while HALTed with LCC.bit7 set, the main timer keeps its deadline
and advances through the ordinary timer/interrupt path. It never advances the
main timer in OFF. Strict retains the shared CPU's frozen main clock; `Firmware` enables the
candidate.
The F01C..F01E readbacks retain the F00C..F00E output latch values used by
F8DC4..F8DD2's save and F9106..F9111's restore sequence; electrical behavior
and mapping meanings remain unqualified.

Starting BP at zero lets a startup scratch write at F93CB wrap into SSR and
set the ON input bit. With explicit initial BP=D0, clear-only ISR writes,
coupled transfers, nominal clocks, the HALT clock candidate and Gate1, repeated
raw contacts rearm and advance the calibration screen. The preserved four-point
consumer reaches 455 ADC reads without a bus fault or the earlier OFF stop.
Physical reset CPU state and scan orientation remain
open. Model selection does not enable these
controls independently; `Firmware` selects the tested combination.

The recovered calibration requires ten held samples per target. The initial
quarter-pixel acceptance bounds are X=[64,192]/[620,1020] and
Y=[64,192]/[340,740]. Raw inputs (340,240), (700,240), (340,700), (700,700)
with sufficient holds complete all four targets and produce the ROM's
calibration-complete message. A shorter third hold collected only nine samples.
The completion message is followed by a timed wait, not a confirmation-key
requirement. F3:CC527 passes four ticks to IOCS14 command49; F7E43 waits for
BFE32 to advance. The current firmware profile completes organizer entry;
physical key legends and complete application input remain open.

`configure_pi3000_rtc_tick_candidate(period)` explicitly configures the RTC
A2 periodic source before execution. `Firmware` uses the related board's
nominal 512,000 timing-unit half-second period; `Strict` leaves it disabled.
Each native/HALT/OFF boundary advances its independent phase. Ticks coalesce
in CAUSE_A.A2 (bit2); masks do not prevent latching. The RTC interrupt reasserts
gate bit0 until the ROM clears the cause and then the gate latch. This matches
F15C4/F1655 -> BFE15 -> F1B83 -> F1860/F14C5. F1B83 increments BFE32 through
ordinary instructions. Reads and preflight peeks do not advance time.
Calendar progression, alarm deadlines, retention and the physical crystal
remain separate recovery gaps. No calendar or firmware counter is seeded.

With this source, the original ROM automatically leaves calibration and draws
the contrast setup. A raw touch on Finish opens the fresh-memory initialization
confirmation. Selecting Yes in the empty emulated machine runs 00FE and clears
RAM. The strict low-bus control stops at F11D0 with X=05844. The current E0
candidate lets that ignored cleanup call return error21 and reaches the calendar.

Raw replay also accepts `contact` with `column`, `row` and `pressed`, validated
for the shared 11-column/eight-row matrix before any input runs. This supplies
electrical contacts only; no physical PI key legends or C/CE mapping are asserted.
The browser accepts the bundle through PI-3000 model selection or
`PI3000_ROM_PATH`. It presents a 239×168 crop/transposition of native storage;
IOCS extents and legible Japanese support this orientation, while the physical
scan mapping remains unqualified. LCD clicks map to a virtual digitizer whose
four endpoints pass the recovered calibration. A short tap stays pressed for
200,000 scheduler boundaries; blur/cancellation releases immediately.

Worker and no-Worker browser paths both reproduce the native ten-sample
calibration result from fresh RAM. They do not seed firmware queues or
coefficients. The browser exposes functional DATE, MENU, PREV, NEXT and OFF
controls alongside ON. Their labels describe observed calendar actions on
matrix contacts 1,6,9,13,7; they do not reconstruct the physical keycap layout.
F1/F7, PageUp/PageDown and Pause provide the same raw contacts. The periodic RTC advances past the completion
message without keyboard input. Native
`lcd_capture` and the Function Runner retain the full controller geometry;
the presentation transform applies only to UI frames.

A bounded OFF/ON consumer returns to the F06BA idle loop and the calendar's
initial-date visible pixels after redraw. Wake returns the focus to day1,
even after selecting day16. This is an in-session power
transition, not persistent snapshot or full retention qualification. A long
Cancel contact released between independent X/Y conversions can produce a
mixed final coordinate under the provisional ADC model; shorter or upper-button
contacts close the modal. Complete ADC sampling/release timing remains open.
