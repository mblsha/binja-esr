- sc62015/core/src/keyboard.rs:412-504 scans/enqueues events on `press_matrix_code`,
  `release_matrix_code`, and every KIL/KOL/KOH read/write. This immediately sets KEYI and fills the
  FIFO even if the ROM never strobes columns or timers fire. Python `KeyboardMatrix`
  (`pce500/keyboard_matrix.py:215-238, 241-257, 229-238`) only toggles state, relies on
  timer-driven `scan_tick`, and `read_kil` just recomputes the latch. Result: Rust delivers
  keyboard interrupts and repeats much earlier than Python and will log extra KEYI/Perfetto events.
- sc62015/core/src/llama/eval.rs:2054-2073 pushes IR context little-endian (`push_stack(...,
  false)`) before jumping to the vector. Python IR spec/LLIL
  (`sc62015/pysc62015/instr/instructions.py:1123-1140`) pushes PC high-byte first, then F and IMR.
  Stack layout/RETI restore order likely diverges, affecting IRQ return PCs and parity traces.
- sc62015/core/src/llama/eval.rs:335-375 Perfetto tracing reads IMR/ISR via `bus.load`, which in
  the LLAMA runner calls into `StandaloneBus::load` and triggers keyboard side effects (KIL read
  asserts KEYI, extra logging). Python tracing (`pce500/emulator.py:543-570` fast WAIT path and
  `tools/llama_parity_runner.py:48-100`) snapshots registers without touching device state, so
  enabling tracing changes runtime behavior only on the Rust side.
- sc62015/core/src/bin/pce500.rs:427-437,492-539 `press_key`/`release_key` call the eager-enqueue
  paths above and `irq_pending` only treats KEYI as pending when IMR_KEY is set (with a master-bit
  shortcut). Python keeps a `_key_irq_latched` flag and reasserts KEYI even when IMR is masked
  (`pce500/emulator.py:672-706,2366-2395`), so masked keyboard interrupts can be delivered/cleared
  on different schedules.
- Perfetto/IRQ visibility differs: the runner emits `IrqPerfetto` tracks
  (`sc62015/core/src/bin/pce500.rs:420-489`) with custom event names, while Python’s new tracer
  uses `TimerFired`/`KEYI_*` instants (`pce500/emulator.py:2318-2368`). These track/name mismatches
  mean IRQ-oriented traces aren’t directly comparable even when instruction/memory tracks align.
