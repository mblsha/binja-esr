# Explicit PA-9500 firmware bootstrap

The `pa9500` feature provides an explicit `pa9500::from_bootstrap_bundle`
constructor for the verified ROM bundle and optional PA-9C3 media. It combines
the firmware exploration candidates and creates fresh organizer backing.
The strict model factory keeps its original behavior.

Native `CoreRuntime` steps, cooperative slices and automatic pacing use the
borrowed compatibility coordinator. Gate bit1 plus native JPF into low space
enters the bounded ESR-H interpreter; legacy WRIT clears the gate and the
7B far transfer returns to native execution. C/Z and low IRAM are shared.
Physical trigger/timing and the complete legacy instruction set are open;
legacy boundaries have their own counter and no fabricated timing units.

WASM exposes `load_pa9500(bundle, card_rom, card_sram, "bootstrap-v1")`,
`pa9500_status()` and `pa9500_card_sram()`. Loading is atomic. LCD capture
observes the actual board controller. Reset creates a fresh CPU/organizer
while keeping logical card SRAM. Organizer/RTC/OFF retention is pending.
PA debugger function-call capture is unavailable; use normal matrix input.

The Function Runner CLI accepts the PA model, explicit profile and paired
card files. `e.pa9500.status()` and `e.pa9500.cardSram()` observe state;
`e.step`, `e.keyboard` and `e.onKey` keep their existing interfaces.
The graphical browser loader is not yet qualified for this profile.

`bootstrap-v2` adds exactly five native read-only bytes from the verified
1 MiB acquisition: addresses 1..3 and C/D returned E0. The original v1
constructor remains available; other unknown reads and all writes at those
addresses retain their traps. The v2 constructor is
`from_bootstrap_bundle_with_observed_low_reads`. Physical decode under other
modes remains unqualified. The WASM loader accepts an explicit version
and preserves the selected version across fresh reset with retained card SRAM.
Normal Phonebook, Memo and Business Cards entry/Calendar return have source-
bound native/WASM proofs. Their text entry/conversion and persistence remain
open; v2 is an explicit board candidate, not complete application support.

`bootstrap-v3` adds only native read-only addresses5..8, also E0 in the
acquisition. Its constructor is `from_bootstrap_bundle_with_observed_field_reads`.
The Phonebook name editor's immutable NAPR template has a null optional-format
pointer; native code reads words at pointer+5 and pointer+7. A strict component
reproduces that access and preserves strict neighboring reads, every low write
and legacy data reads. Acquired-byte and zero controls produce identical pixels
through a normal kana/edit/return replay, while temporary format metadata differs.
The public candidate selects the acquired bytes and retains all v1/v2 behavior.
Its selected profile and logical card SRAM survive fresh WASM reset.

Native/WASM normal inputs now have a source-bound v3 consumer: M then A produces
the native narrow kana glyph, Backspace clears it, E replaces it, Enter executes
without a runtime fault, and Calendar returns. Immutable ROM font bits verify
the two kana glyphs independently; the empty cell has an explicit underline
cursor layout fixture. This does not qualify complete text conversion/modifiers,
search semantics, registration, organizer retention or physical low-address
decode. The native EC font codes differ from standard Shift JIS dialog glyphs.

Normal Clock popup/save/restore uses the shared controller's signed pixel
coordinates. A stream beginning outside backing can cross zero within one
byte; each off-screen bit is blank or discarded individually. The negative
origin used by firmware picture save/restore must preserve the remaining bits.
This contract is mirrored in Rust and Python and covers both orientations;
physical-controller timing and clipping behavior remain provisional.

The private companion repository contains source-bound native/WASM reset and
normal BASIC animation replays, independent pixel checks and proof images.
These qualify firmware behavior under an explicit candidate; they do not
establish physical hardware equivalence or complete all-application support.
