# PI-3000 native runtime

The `pi3000` Cargo feature adds a PI-3000 board adapter to the shared
CoreRuntime. It reuses the native CPU, matrix LCD controller, RTC reference
and bounded ESR-H interpreter. PI ROM mapping, controller geometry and input
ports come from the acquired PI firmware. This is an implementation in
progress; foreground calibration, ordinary input, RTC/power, cards and browser
integration remain open.

The normal factory accepts only a `PIROM01` bundle of the verified fixed ROM
and 32 complete selector views (00..0F and F0..FF). It does not accept a mapped
1 MiB dump as startup RAM. The private acquisition tool packs the bundle;
ROM data is not distributed with this source tree.

```sh
cargo run --manifest-path sc62015/core/Cargo.toml --features pi3000 \
  --bin pce500 -- --model pi-3000 --rom /path/to/pi-3000.pirom --steps 20
```

The factory starts fresh scratch/high RAM and leaves periodic timers disabled
pending calibration. Unsupported bus operations stop and poison the runtime.
An unknown instruction mapping fails preflight; an unknown data access stops
after its committed instruction. Generic card overlays are not applied.
Restart constructs a new board; v4 snapshots do not represent its devices.

`BootstrapProfile::Strict` is the default native-only/unknown-auxiliary control.
`Compatibility` opts into the firmware-derived gate/page handoff.
`AuxiliaryAbsent` additionally supplies status02 and retains auxiliary requests
without completion or response bytes. The firmware can take its own timeout
path; this does not implement the auxiliary device.

`AuxiliaryControl` is a separate stateful candidate for startup control headers
01FE,30FB,02FE. It requires exactly two freshly written request bytes, enters
busy, completes after 32 native timing units, exposes status01 and requires BF
acknowledgment before another request. Unknown commands and query bytes stay
unsupported. Its idle auxiliary event ports return zero; physical protocol,
device identity, timing and recognition behavior remain unqualified.

The default ISR replacement policy currently triggers an early IRQ before
reset has installed the handler workspace. An explicitly selected clear-only
ISR experiment reaches a four-cross calibration frame after two million
scheduler boundaries, including 29 legacy instructions. Initial CPU and ISR
semantics remain unqualified; model selection does not infer that experiment.

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
main timer in OFF. The default retains the shared CPU's frozen main clock.
The F01C..F01E readbacks retain the F00C..F00E output latch values used by
F8DC4..F8DD2's save and F9106..F9111's restore sequence; electrical behavior
and mapping meanings remain unqualified.

Starting BP at zero lets a startup scratch write at F93CB wrap into SSR and
set the ON input bit. With explicit initial BP=D0, clear-only ISR writes,
coupled transfers, nominal clocks, the HALT clock candidate and Gate1, repeated
raw contacts rearm and advance the calibration screen. The preserved four-point
consumer reaches 455 ADC reads without a bus fault or the earlier OFF stop.
Calibration completion, physical reset CPU state, scan orientation and
organizer GUI support remain open. Model selection does not enable these
experiments automatically.
