# Rust emulator performance

This note records how the SC62015 Rust core (`sc62015/core`) was made roughly
an order of magnitude faster without changing emulated behaviour, how that was
verified, and how to measure it again.

## Results

Baseline is `147e829` built with the default release profile. Both workloads are
CPU-bound, use fixed inputs (and a fixed IQ-7000 RTC seed), and print identical
output on both builds.

| Host | Workload | Baseline | Current | Speedup |
| --- | --- | --- | --- | --- |
| Linux x86-64 (i5-9600KF), wall, 3 interleaved pairs | PC-E500 | 17.46 s | 1.71 s | 10.2× |
| Linux x86-64 (i5-9600KF), wall, 3 interleaved pairs | IQ-7000 | 19.10 s | 1.68 s | 11.4× |
| macOS (Apple M2), wall | PC-E500 | 5.09 s | 0.83 s | 6.1× |
| macOS (Apple M2), wall | IQ-7000 | 5.47 s | 0.84 s | 6.5× |

Host instructions per emulated instruction fell about fivefold (Apple M1:
~4,990 → ~970). The larger Linux wall-clock gain also includes removing futex
system calls that the old per-instruction tracer locking made on Linux.

Workloads (`pce500` CLI, ROMs not distributed):

- PC-E500: `--model pc-e500 --steps 80000000 --key-seq 'wait-op:200000;on:40000;wait-op:3000000;text:10 FOR I=1 TO 20000:NEXT I;enter;wait-op:300000;text:RUN;enter;wait-op:60000000'`
  (about 17.6M executed instructions; the ROM polls at its card prompt).
- IQ-7000: `--model iq-7000 --iq7000-rtc 199201101050 --steps 20000000 --key-seq 'wait-op:600000;memo:40000;wait-op:400000;text:lorem ipsum dolor sit amet consectetur;wait-op:10000000'`
  (about 19.5M executed instructions of MEMO typing).

## Equivalence checking

Every change was checked against the baseline binary before it was kept:

- Seven runs across both models (the two workloads, PC-E500 card initialisation
  to MAIN MENU, IQ-7000 TEL/CALENDAR/SCHEDULE/CALC key traffic, boots, and a
  `--disable-timers` run), each compared byte-for-byte on stdout, the LCD JSON
  capture and PNG, and a debug probe that dumps external RAM `0x00000-0x3FFFF`,
  all of IMEM, registers, interrupt and timer state.
- Two Perfetto-traced runs (one per model), compared after normalising each
  trace (track UUIDs replaced by names, interned strings resolved, annotations
  sorted) so every event, timestamp and annotation must match.
- The core, PyO3 and WASM test suites, and the WASM IQ-7000 MEMO benchmark,
  whose `retired` instruction and `timingUnits` counts stay identical.

## What changed

Tracing and build:

- `PerfettoHandle` keeps an `installed` flag, so untraced execution never
  enters the reentrant handle (previously several mutex operations and a
  condvar notify per instruction). `RuntimeBus` reports the per-call tracing
  state on every target.
- Release builds use fat LTO with one codegen unit (core, PyO3, WASM).

Instruction preflight and decode:

- The silent preflight of each instruction is memoized per PC for plain
  upper-ROM bytes (`preflight_cache.rs`); a hit requires the live bytes to equal
  the validated ones, so no invalidation is needed. The executor no longer
  repeats the scheduler's validation.
- Each cache entry carries a decode memo: operands that depend only on the
  instruction bytes and IMEM base registers are replayed with the same
  BP/PX/PY reads and the same read accounting.
- Instruction timing is evaluated once per preparation.

Scheduler boundary:

- A quiet steady-state prelude handles a running core that will not take an
  IRQ, has no input level to re-latch, and executes memoized ROM; it performs
  exactly the side effects of the general prelude in that case.
- IRQ-level helpers, SIO and RTC ticks, and the timer next-event query have
  inlined no-op/early-out paths with the general logic out of line.
- Device and host hooks are gathered once per `step` call.

Memory and devices:

- `RuntimeBus` sends plain internal and external accesses straight to
  `MemoryImage`; device routing stays out of line. `LcdHal::fixed_windows`
  publishes the built-in controllers' address maps.
- `MemoryImage` keeps the upper-ROM plainness of its mapping as a maintained
  flag and skips overlay walks outside the overlay span; trace emission in
  IMEM accessors moved to cold helpers.
- The keyboard scans and computes KIL only over non-idle keys.

Executor:

- `RegName` is two bytes (its unused `Unknown` payload was removed), shrinking
  every decoded operand; the deferred trace is boxed.
- Per-thread execution context lives in one thread-local, its pointer is cached
  per call, and the call-stack snapshot is republished only when a stamp shows
  the stack changed.

## Measuring

- Correctness first: compare a candidate against the baseline with the
  equivalence runs above.
- Counters: `instructions:u` and `cycles:u` under `perf stat` on Linux, or
  `/usr/bin/time -l` on macOS; divide by the executed-instruction count the CLI
  prints.
- Wall time: interleave baseline and candidate runs on an otherwise idle host.
- Attribution: `perf record` on an LTO build with
  `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`, then map samples to source
  lines with `addr2line -i` (most hot code is inlined into
  `CoreRuntime::step_scheduler_boundaries`).
- A ROM copy whose reset target is patched to a NOP/`JR` loop isolates the fixed
  per-boundary cost (about 550 host instructions per NOP after these changes,
  2,670 before).
