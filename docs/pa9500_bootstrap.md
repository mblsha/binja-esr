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
The interactive browser bench now routes PA through the same explicit constructor, with normal ON release, paired optional card import and retained-card reset. Source-bound DOM consumers cover native and no-Worker paths; see the interactive bench instructions below.

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


## Interactive browser bench

Open Advanced, select PA-9500 / DB-Z and Bootstrap v3, then load a verified
`.parom` bundle. Optional PA-9C3 input requires its verified128 KiB ROM and a
32 KiB SRAM image; Import card and restart creates fresh organizer backing.
Click the LCD to enable host typing, then Resume or use explicit Step controls.
The displayed controls use ROM-derived contacts and a functional layout;
physical case geometry, key legends and timing remain provisional.

F1–F7 select Calendar, Phonebook, Memo, Business Cards, Handwritten Memo,
Calculator and Clock. F8 selects the native kana input behavior. F9 taps the
native modifier; release it, then native N for Special Functions or K for
SECRET. F10 operates the ROM input-mode key; F11 selects BASIC Run/Pro.
F12 operates normal ON/BREAK. Unsupported host characters are reported.

The PA configuration selects native kana or Latin layout / BASIC letter bindings.
These follow different ROM conversion tables. Selecting bindings changes
ordinary host contacts; the organizer's own Mode key continues to determine
its text-entry mode. The accepted BASIC replay uses normal Mode transitions,
types10 PRINT1, edits the constant to2 with Backspace, lists and runs it.

Reset starts a fresh CPU/organizer and keeps current logical card SRAM. Export
card SRAM downloads the current image for a later paired import. Export PA
observations includes actual controller pixels, exposed machine state and card
bytes. Browser proofs match42 native checkpoints, independently check22 glyph/
bitmap regions, and recover the guest-created10 PRINT2 through normal firmware
LIST/RUN after reset. Every captured UI canvas matches its controller bytes.
Organizer/RTC/OFF retention and complete text/navigation/modifier coverage
remain open. Default PA bootstrap still leaves UART disabled.

The private companion preserves exact input closures, compiled WASM/Vite
artifacts, normal DOM replays, raw frames and inspected screenshots. Earlier
proof capsules retain their historical source versions. All171 web unit tests
and five existing PC/IQ browser regressions pass for this browser integration.


## Serial and ON/BREAK consumers

F12 now has a normal foreground GOTO-loop consumer: ON/BREAK displays
`Break in 10`, Enter dismisses it, and PRO/LIST reconstructs the guest's cached
line. The private proof compares native, Function Runner and browser states
and independently checks every main-area pixel at the running/break/list points.

For PA serial, select **Enable experimental serial on next load** before
Import card and restart or loading the bundle. BASIC owns `OPEN COM`, `PRINT`
and `CLOSE`. The existing Serial panel captures output and Save raw bytes
exports it. Fresh Reset preserves the loaded transport choice and logical
card bytes, starts new organizer/CPU backing and clears host capture.

The explicit WASM API is `load_pa9500_with_uart(bundle, rom, sram, profile)`;
ordinary `load_pa9500` leaves UART unavailable and clears a previous choice.
Invalid replacements preserve the running model, media and transport. Other
model loads clear PA configuration. Function Runner accepts `--pa9500-uart`
with an explicit PA profile; `e.pa9500.takeSerial()` drains completed output
without advancing guest time. No firmware workspace or software ring is patched.

The authored serial TEST requests9600, even parity,8 data bits and1 stop bit.
Its271-byte payload equals the independent physical TEST capture; CLOSE adds
EOF1A. Native, Function Runner and browser captures agree. The enabled browser
captures272 bytes and repeats them after fresh reset; the disabled control
emits nothing even though firmware reaches DONE. The byte device uses nominal
clock/divisor assumptions and does not establish physical parity waveforms.
Receive/flow-control behavior remains a separate qualification target.

This checkpoint has62 source-bound comparisons, seven independent complete
main-display fixtures,14 loader/reset guards,33 actual Node/WASM tests,
18 acquired-ROM component tests,171 web tests and five existing PC/IQ browser
regressions. The companion repository retains raw PNG/PBM, actual UI images,
compiled artifacts and exact replay/source hashes. Native cargo tests alone
run no wasm-bindgen tests and are not counted as the WASM regression gate.


## Cursor and editing bindings

The acquired matrix/base-word tables and card cursor-output code establish
Left/contact8, Right14, Up0 and Down15. Escape/contact5 clears input; the virtual
key is labelled CLEAR. Host Delete and Insert queue a normal modifier tap and
release, then Left or Right respectively. The worker uses the existing contact
queue and rejects stale generations or unavailable models; lifecycle cleanup
uses the same input ownership rules. No firmware FIFO/RAM injection is used.

The private proof's native Phonebook sequence changes AB to ACB through
insertion, deletes B, backspaces C, clears A and returns to Calendar. The BASIC
sequence changes10 PRINT13 to10 PRINT2 using cursor movement, Delete and
Backspace, stores it and prints2. Native, Function Runner and browser replay
results are compared, with independent acquired font pixels and strict program
round trips. The40,000-boundary queue timing is an explicit host policy, not a
physical timing measurement. Full input/search/registration and organizer/RTC/
OFF retention remain separate work.
