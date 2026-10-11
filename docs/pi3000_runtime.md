# PI-3000 native runtime

The `pi3000` Cargo feature adds a PI-3000 board adapter to the shared
CoreRuntime. It reuses the native CPU, matrix LCD controller, RTC reference
and bounded ESR-H interpreter. PI ROM mapping, controller geometry and input
ports come from the acquired PI firmware. This is an implementation in
progress. Native and browser raw input complete calibration, pass contrast
setup and reach the fresh-memory initialization prompt. Organizer navigation,
complete keyboard/RTC/power behavior and cards remain open.

The normal factory accepts only a `PIROM01` bundle of the verified fixed ROM
and 32 complete selector views (00..0F and F0..FF). It does not accept a mapped
1 MiB dump as startup RAM. The private acquisition tool packs the bundle;
ROM data is not distributed with this source tree.

```sh
cargo run --manifest-path sc62015/core/Cargo.toml --features pi3000 \
  --bin pce500 -- --model pi-3000 --rom /path/to/pi-3000.pirom --steps 20
```

The normal model factory starts fresh scratch/high RAM with the `Firmware`
startup candidate. It combines initial BP=D0, clear-only ISR writes, coupled
predecrement transfers, nominal organizer timer periods, the LCC.bit7/HALT
main-clock candidate, the periodic RTC A2 candidate, integrated ADC and Gate1
press edges. These settings are
supported by ROM consumers; physical reset state and timing remain unqualified.
Unsupported bus operations stop and poison the runtime.
An unknown instruction mapping fails preflight; an unknown data access stops
after its committed instruction. Generic card overlays are not applied.
Restart constructs a new board; v4 snapshots do not represent its devices.

`BootstrapProfile::Strict` retains the native-only/unknown-auxiliary control
through the explicit bundle API; periodic timers stay disabled in that profile.
`Compatibility` opts into the firmware-derived gate/page handoff.
`AuxiliaryAbsent` additionally supplies status02 and retains auxiliary requests
without completion or response bytes. The firmware can take its own timeout
path; this does not implement the auxiliary device.

`AuxiliaryControl` is a separate stateful candidate for startup control headers
00FE,01FE,30FB,02FE. FE:DE211's 00FE initialization control shares DE219's
single-byte acknowledgment path with 01FE. The model requires exactly two
freshly written request bytes, enters
busy, completes after 32 native timing units, exposes status01 and requires BF
acknowledgment before another request. Unknown commands and query bytes stay
unsupported. Its idle auxiliary event ports return zero; physical protocol,
device identity, timing and recognition behavior remain unqualified.

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
Physical reset CPU state, scan orientation and organizer GUI support remain
open. Model selection does not enable these
controls independently; `Firmware` selects the tested combination.

The recovered calibration requires ten held samples per target. The initial
quarter-pixel acceptance bounds are X=[64,192]/[620,1020] and
Y=[64,192]/[340,740]. Raw inputs (340,240), (700,240), (340,700), (700,700)
with sufficient holds complete all four targets and produce the ROM's
calibration-complete message. A shorter third hold collected only nine samples.
The completion message is followed by a timed wait, not a confirmation-key
requirement. F3:CC527 passes four ticks to IOCS14 command49; F7E43 waits for
BFE32 to advance. Ordinary key mapping and entry into the organizer remain open.

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
RAM. The next preserved failure is a filename lookup at F11D0 with X=05844;
the recovered bus rejects that address. It is not organizer-entry proof.

Raw replay also accepts `contact` with `column`, `row` and `pressed`, validated
for the shared 11-column/eight-row matrix before any input runs. This supplies
electrical contacts only; no PI key-name table or C/CE mapping is asserted.
The browser accepts the bundle through PI-3000 model selection or
`PI3000_ROM_PATH`. It presents a 239×168 crop/transposition of native storage;
IOCS extents and legible Japanese support this orientation, while the physical
scan mapping remains unqualified. LCD clicks map to a virtual digitizer whose
four endpoints pass the recovered calibration. A short tap stays pressed for
200,000 scheduler boundaries; blur/cancellation releases immediately.

Worker and no-Worker browser paths both reproduce the native ten-sample
calibration result from fresh RAM. They do not seed firmware queues or
coefficients. The browser exposes only ON as a named key until the PI button
mapping is recovered. The periodic RTC now advances past the completion
message without keyboard input. Native
`lcd_capture` and the Function Runner retain the full controller geometry;
the presentation transform applies only to UI frames.
