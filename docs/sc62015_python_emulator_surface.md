# SC62015 Python Emulator Surface & Test Coverage

This note captures the Python-side contract that the upcoming Rust core must mirror. It focuses on the CPU/LLIL execution surface, memory expectations, configuration knobs, and how the existing pytest suite exercises those behaviours.

## Module Inventory
- `sc62015.pysc62015.emulator`
  - `RegisterName (Enum)`: canonical names for architected registers. Includes 8/16/24-bit general registers, `PC`, `FC/FZ` flag bits, and 14 temporaries (`TEMP0`–`TEMP13`). The enum values are relied upon by downstream consumers (e.g. `pce500`) and tests.
  - `REGISTER_SIZE (Dict[RegisterName, int])`: register widths in bytes; the Rust core must apply the same masking semantics.
  - `Registers`: register file abstraction with `get`, `set`, `get_by_name`, `set_by_name`, `get_flag`, and `set_flag`. Handles PC masking to 20 bits, flag aliases, and composite register updates (e.g. writes to `A` update `BA`). Also tracks `call_sub_level` for tracing call depth.
  - `CALL_STACK_EFFECTS`: map of opcode → call-depth delta. Used during execution to keep `call_sub_level` in sync when calls/returns fire.
  - `InstructionEvalInfo`: lightweight dataclass bundling `InstructionInfo` and lifted `Instruction` after execution.
  - `Emulator`: orchestrates instruction fetch/decode (`decode_instruction`), execution (`execute_instruction` / `_execute_instruction_impl`), LLIL evaluation (`evaluate`), and system reset handling (`power_on_reset`). The constructor accepts a `Memory` object and `reset_on_init` flag. Execution special-cases opcode `0xEF` (WAIT) to avoid long LLIL loops.
  - Module-level constants: `NUM_TEMP_REGISTERS`, `USE_CACHED_DECODER` (set when `CachedFetchDecoder` import succeeds), and `FLAG_TO_REGISTER`.
- `sc62015.pysc62015.stepper`
  - `CPURegistersSnapshot`: serializable snapshot of architected registers + temps + `call_sub_level`. Provides `from_registers`, `apply_to`, `to_dict`, and `diff` helpers.
  - `_SnapshotMemory`: `binja_test_mocks.eval_llil.Memory` adapter that records writes while serving reads from a provided mapping. Exposes `writes` and `snapshot`.
  - `MemoryWrite`: dataclass describing a captured memory mutation.
  - `CPUStepResult`: named container for a single-step outcome (registers, diffs, writes, resulting memory image, instruction metadata).
  - `CPUStepper`: stateless helper that takes a snapshot + sparse memory image, executes one instruction through `Emulator`, and returns a `CPUStepResult`. Used heavily by higher-level device tests.
- `sc62015.pysc62015.constants`
  - Address space and PC metadata: `ADDRESS_SPACE_SIZE`, `INTERNAL_MEMORY_START`, `INTERNAL_MEMORY_LENGTH`, `PC_MASK`.
  - Flag enumerations: `IMRFlag`, `ISRFlag` (both `IntFlag`), providing bit masks for interrupt control/status registers.
- `sc62015.pysc62015.cached_decoder`
  - `CachedFetchDecoder`: drop-in replacement for `binja_test_mocks.coding.Decoder` with an LRU byte cache (`_CACHE_LIMIT = 32`). Provides `peek`, `unsigned_byte`, `advance`, `get_cache_stats`, and `clear_cache`. The emulator falls back to the uncached `FetchDecoder` when importing fails.
- `sc62015.pysc62015.intrinsics`
  - Intrinsic evaluators: `eval_intrinsic_tcl`, `eval_intrinsic_halt`, `eval_intrinsic_off`, `eval_intrinsic_reset`. All accept the LLIL node, register interface, memory, state, and flag callbacks and perform the documented side effects.
  - `_enter_low_power_state`: shared helper for HALT/OFF side effects.
  - `register_sc62015_intrinsics`: registers the above handlers with `binja_test_mocks.eval_llil.register_intrinsic()`; called by `Emulator.__init__`.
- `sc62015.pysc62015.instr` package
  - Entry points: `decode(decoder, address, OPCODES)` and `encode(...)`.
  - `Instruction` hierarchy plus operand helpers (`Operand`, `Reg`, `Imm8`, `Reg3`, `AddressingMode`, etc.) that wrap Binary Ninja’s LLIL.
  - `opcode_table.OPCODES`: canonical decode table for the architecture (consumed by both emulator and tests).
  - `opcodes.IMEMRegisters`: register identifiers for internal memory-mapped peripherals; shared with the emulator, assembler, and downstream device model.
- `sc62015.pysc62015.sc_asm`
  - `Assembler`: Lark-based assembler used by tests to emit machine code; exposes `.assemble(text)` returning binary segments.
- `sc62015.pysc62015.__init__`
  - Re-exports the new `CPU` facade alongside the legacy `Emulator`, `RegisterName`, and `Registers` helpers for convenience.
- `sc62015.pysc62015.cpu`
  - `CPU`: facade that selects between the Python `Emulator` and the optional Rust backend (when available). Accepts the same constructor arguments as `Emulator` and forwards attribute access to the underlying implementation. The selection honours an explicit `backend` argument or the `SC62015_CPU_BACKEND` environment variable.
  - Helper functions: `available_backends()` and `select_backend()` to inspect or override backend resolution.

### External Types the Rust Core Must Respect
- `binja_test_mocks.eval_llil.Memory`: wraps provided `read_mem` / `write_mem` callables. Exposes `read_byte`, `write_byte`, `read_bytes`, `write_bytes`.
- `binja_test_mocks.eval_llil.State`: simple dataclass with `halted: bool`.
- `binja_test_mocks.eval_llil.ResultFlags`: `TypedDict` with optional `C` and `Z` fields. Returned by LLIL evaluation helpers.

## CPU Execution Lifecycle
1. **Construction**: `Emulator(memory, reset_on_init=True)` stores a fresh `Registers` instance, keeps the supplied `Memory`, initialises `State()`, records `call_sub_level`, and registers intrinsic evaluators.
2. **Reset path**: `power_on_reset()` delegates to `intrinsics.eval_intrinsic_reset`, ensuring PC, IMEM registers, and flags are set per hardware spec. External users often pass `reset_on_init=False` and call `power_on_reset` manually when they need fine-grained control.
3. **Fetch/decode**: `decode_instruction(address)` builds a `FetchDecoder` (cached variant when available) that reads bytes via `Memory.read_byte`, then calls `instr.decode`. Any Rust replacement must honour `ADDRESS_SPACE_SIZE` bounds and PC masking.
4. **Execution**: `execute_instruction(address)` updates tracing metadata, optionally wraps the LLIL evaluation in a Perfetto slice when the memory stub exposes `_perf_tracer`. It then defers to `_execute_instruction_impl`.
5. **LLIL evaluation**: `_execute_instruction_impl` sets the architected PC, decodes the instruction, bumps `call_sub_level` per opcode, fast-paths `WAIT`, lifts to LLIL, and runs the resulting graph via `evaluate_llil`, feeding register and memory accessors. It updates the PC using the decoded `InstructionInfo.length`.
6. **Post-step state**: The emulator returns `InstructionEvalInfo` containing the Binary Ninja metadata. Register and memory mutations have already been applied through the `Registers` and `Memory` abstractions.

The Rust backend must mimic each stage so that existing consumers (assemblers, snapshot steppers, and the PC-E500 device model) continue to work unchanged.

## Memory & Peripheral Expectations
- `Memory` callbacks are provided by callers; the emulator assumes `read_byte`/`write_byte` raise on out-of-range access. Internal memory accesses use `INTERNAL_MEMORY_START` offsets. Tests rely on this to mirror IMEM behaviour.
- The emulator does not own peripheral emulation; instead the backing memory object is expected to intercept writes/reads for IO behaviour. Several pce500 components hang attributes (e.g. `_perf_tracer`) or override methods on the memory instance.
- `Registers.call_sub_level` is used downstream for profiling (`pce500/tests/test_tracing_call_stack.py`). The Rust implementation must expose the same attribute on the register file or an equivalent property reachable from Python.

## Configuration & Environment Hooks
- `USE_CACHED_DECODER` toggles automatically depending on whether `cached_decoder` imports successfully. Parity requires the Rust backend to expose a similar knob or transparently outperform the Python cache.
- Tests set `FORCE_BINJA_MOCK=1` so imports from `binaryninja` resolve to `binja_test_mocks`. The Rust tooling should continue to honour this environment variable, especially when generating LLIL metadata.
- `reset_on_init`: honoured by both `Emulator` and `CPUStepper`; the Rust constructor must take—and default—this flag identically.

## pytest Coverage Matrix
The following files are under `sc62015/pysc62015/tests` and define the behavioural envelope the Rust backend must satisfy:

| Feature area | Expectations | Tests |
| --- | --- | --- |
| Register file semantics | Read/write masking, composite register updates, flag aliases, PC masking | `test_emulator.py::test_registers`, `test_emulator.py::test_pc_mask` |
| Instruction decode + execution semantics | Broad sweep of load/store, arithmetic, logic, branch, stack, interrupt, and vector operations. Each `InstructionTestCase` validates decode string, register effects, memory writes, and PC advance | `test_emulator.py::test_instruction_execution` parametrized suite (covers ~80 instruction forms); specialist cases such as PUSH/POP, call/return, stack behaviour |
| WAIT fast path | Opcode `0xEF` clears `I` and advances PC without full LLIL walk | `test_stepper.py::test_stepper_wait_clears_i_register` (no direct emulator unit test today) |
| Call stack tracking | `call_sub_level` increments/decrements via CALL/RET opcodes and feeds tracing | Covered indirectly by `pce500/tests/test_tracing_call_stack.py`; no dedicated unit test, so parity should rely on opcode map in `emulator.CALL_STACK_EFFECTS` |
| HALT/OFF/RESET intrinsics | IMEM register side effects, halt state toggles, PC jump to reset vector | `test_halt_off_reset.py` (`test_halt_instruction`, `test_off_instruction`, `test_reset_instruction`, `test_power_on_reset`) |
| Snapshot stepper contract | Serialisation/diff of register snapshots, capturing memory writes, instruction metadata passthrough | `test_stepper.py` cases for NOP, WAIT, and SC flag-setting |
| Instruction metadata + LLIL lifting | Instruction naming, operand rendering, LLIL jump targets, mode decoding | `test_instr.py` (covers `JP`, addressing modes, operand renderers, HALT/OFF/TCL intrinsics etc.) |
| Assembler integration | Round-tripping assembly text to opcode bytes, internal register naming, full-program assembly | `test_asm.py`, `test_asm_e2e.py`, `test_asm_imem_register_names.py`, `test_opcode_assembly.py` |
| Architecture hookup | Binary Ninja `Architecture` shim returns LLIL for simple instructions | `test_arch_failure.py` (skipped when real BN present) |

**Downstream integration tests** (outside this package) rely on these APIs as well:
- `pce500/emulator.py` and associated tests instantiate `Emulator` and `CPUStepper` for the handheld device model.
- Web UI tests import `RegisterName`, `IMRFlag`, and IMEM register constants to drive state displays.

## Open Questions / Follow-ups
- Documented downstream usages indicate that `Registers` and `RegisterName` must remain importable at their current module paths. The Rust bindings should mirror those names to avoid churn.
- `call_sub_level` has no dedicated unit test; confirm whether more coverage is needed when implementing the Rust equivalent.
- Validate that `binja_test_mocks` provides all LLIL constructs needed for full opcode coverage before relying on auto-generated metadata from Binary Ninja.

Keeping this document current as the Rust implementation lands will help ensure both cores stay in feature lockstep.
