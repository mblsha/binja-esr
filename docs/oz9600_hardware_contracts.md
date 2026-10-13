# Measured OZ-9600 contracts for related emulators

Start here when reusing the OZ-9600 research for PV-F1 or another ESR-family
model. Measurements below used a physical OZ-9600 CPU through an uploaded
OZ-707 card helper. They are evidence for that silicon and configuration;
CPU-family resemblance alone does not qualify another model.

| Contract | Evidence and implementation | Reuse limit |
| --- | --- | --- |
| ISR stores replace requested pending bits | 81 hardware trials; core ISR store policy | Hardware-generated arrivals and reset state remain separate |
| Arithmetic with a wider register source uses its low byte | 12 arithmetic fixtures; measured core operand policy | Untested instructions and flags need their own controls |
| External predecrement block transfers use coupled descending pointers | Six distinct-marker transfer fixtures; `coupled_predecrement` policy | Other overlap/direction combinations remain bounded by existing tests |
| Internal direct/BP-relative selectors and wrap | 16 paired MV/PRE cases and five wrap cases on hardware; current model already matches | Does not establish true power-on BP/AMC/internal-RAM values |
| Software interrupt recognition | Following instruction executes before entry, including tested masking/clearing and nested STI/MTI | `experimental-irq-recognition` stays opt-in; asynchronous/electrical arrival and broader RETI cases remain open |
| LCD Boolean raster operations and clipping | [Measured raster table and window controls](oz9600_lcd_raster.md); shared Rust/Python controller | Controller branch is OZ-9600 LH1553F; other displays require verification |
| Native RTC and manual OFF/ON | Actual RTC callbacks, BP-minus-one preservation and one manual OFF/ON return | Does not prove automatic alarm wake or earliest wake cause |
| LCD completed block count | Native two-row copy returns count zero; fill then copy without reload does no further transfer; explicit reloads work in both directions | Completed-operation model; intermediate decrement and electrical busy timing remain open |
| SRAM windows | Paired BASIC/native4400 reads and prior distinct markers establish that the low/top alias is view-dependent | Unconditional alias in the provisional model does not cover BASIC3507FF0002; coupled mapper/GPIO bits, other pages and capacity remain open |

The implementation preserves related models' geometry, PI input hold interval
and PA runners. It does not add PV-F1 hardware defaults from unqualified
OZ-specific experiments. [CPU recognition](oz9600_irq_recognition.md),
[host tap assistance](oz9600_host_taps.md) and
[session persistence](oz9600_sessions.md) document their separate scopes.

The companion private research repository preserves exact native sources,
counted serial acquisitions, checksums, restoration receipts and negative
controls. Its current entry point is
`docs/oz-9600/analysis/hardware_investigation.md`. Particularly useful receipts:

- `reference/cpu_physical_20261011/proof.json`: arithmetic and block fixtures.
- `reference/cpu_irq_arrival_20261011/`: interrupt arrival and nesting controls.
- `reference/cpu_internal_addressing_20261013/physical/physical-proof.json`:
  all 21 addressing cases and full scratch restoration.
- `reference/rtc_bp_minus_one_20261013/physical/physical-proof.json`: two quiet
  RTC entries at BP B2, native callback/cache preservation and restoration.
- `reference/lcd_block_count_20261013/README.md`: consumed-count explanation,
  native/count/increment hardware controls and unchanged-code integration replays.
  The original ineffective copies followed a fill without reloading the count.
- `reference/memory_native_view_20261014/physical/physical-proof.json`: identical
  targets in BASIC3507FF0002 and native3007FF0100. B2000 is F6 in BASIC and
  equals12000 in the native view; ROM/card controls and exact recovery pass.
  After restoration, BASIC35FCFF0002 also returns F6 at BFB10 while1FB10
  contains the live cache. This does not isolate the coupled control bits or
  justify a universal alias.
- `reference/power_owner_20261013/physical-manual/physical-proof.json`: native
  OFF followed by one manual ON, responsive BASIC and verified restoration.

Paths in this list are relative to `docs/oz-9600/`. The Schedule cache race,
physical short-tap converter timing, automatic alarm wake, complete RAM mapping
and UART electrical timing remain open. Passing emulator replays and legible
instrumented screenshots do not close those hardware contracts.
