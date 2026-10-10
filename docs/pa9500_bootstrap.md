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
