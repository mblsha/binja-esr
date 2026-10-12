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
tablet ADC with explicit PV-F1 geometry and pressure parameters. The shared
ESR-H CPU uses a separate PV-F1 bus and gateway coordinator. Native and legacy
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
LCD save/restore uses coupled predecrement, auxiliary control replies are gated
by PV-F1 firmware bytes, absent expansion reads FF, and tablet pressure is 1023.
Timers start at the first native HALT. The main timer then advances during HALT
only when LCC bit 7 is set, excluding OFF. These choices reproduce the bounded
firmware path; they are not measured oscillator, peripheral or CPU contracts.
Direct `pvf1::configure` callers can select `Strict`, `AbsentAuxiliary` or
`ControlCandidate` instead. A loaded PV-F1 reset requires constructing a fresh
runtime so board state cannot survive accidentally.

## Browser use

Serve a locally generated bundle using `PVF1_ROM_PATH`, then select `pv-f1` or
open `/?model=pv-f1`. The browser starts paused. Resume, send a raw tablet contact
at X=362/Y=585 to accept the fresh-memory initialization dialog, then use
X=940/Y=64 to open the schedule view. The raw controls accept ten-bit samples;
screen-coordinate calibration is still unresolved. ON and matrix Focus are
the currently verified virtual keys. There is no inferred alphabetic mapping.

The browser's automatic pacing adapter accepts a zero-progress `host_yield`
when Rust's four-millisecond host deadline expires before the first batch.
Input holds advance only by executed scheduler boundaries. Invalid budgets,
missing results and false zero-progress budget completion still fail closed.

## Evidence limits

The acquisition and private proof repository retain the source-bound native,
WASM and browser receipts. Thirty-nine LCD IOCS slots have a static inventory;
six non-error commands still lack a successful focused native case.
Remaining work includes broader applications, input calibration, RTC, power,
retention and PV-F1 card/serial routing. No physical LCD or timing qualification
is claimed, and this checkpoint does not establish a fully working emulator.
