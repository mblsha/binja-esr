# SC62015 Python Emulator Surface & Test Coverage

The Python LLIL evaluator and optional Rust LLAMA core share this public CPU
surface. This note captures the boundary both implementations must preserve.
It focuses on CPU/LLIL execution, memory expectations, failure atomicity,
configuration knobs, and how pytest exercises those behaviours.

## Module Inventory
- `sc62015.pysc62015.emulator`
  - `RegisterName (Enum)`: canonical names for architected registers. Includes 8-bit and 16-bit registers plus the 20-bit `X/Y/U/S/PC` class, whose Binary Ninja and stack containers are three bytes but whose architectural mask is `0xFFFFF`; it also includes `FC/FZ`, the byte-wide `F` stack image (only `C | Z<<1`, values `0..3`, are modeled; raw upper-bit images fail closed), and 16 temporaries (`TEMP0`–`TEMP15`). The enum values are relied upon by downstream consumers (e.g. `pce500`) and tests.
  - `REGISTER_SIZE (Dict[RegisterName, int])`: Binary Ninja/storage widths in whole bytes, not a claim that every bit in a three-byte container is architecturally retained; the Rust core must apply the same register-specific masks.
  - `Registers`: register file abstraction with `get`, `set`, `get_by_name`, `set_by_name`, `get_flag`, and `set_flag`. Handles `X/Y/U/S/PC` masking to 20 bits, flag aliases, and composite register updates (e.g. writes to `A` update `BA`). Also tracks `call_sub_level` for tracing call depth.
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
  - Intrinsic evaluators: `eval_intrinsic_halt`, `eval_intrinsic_off`, and `eval_intrinsic_reset` accept the LLIL node, register interface, memory, state, and flag callbacks and perform the current model side effects. `eval_intrinsic_tcl` deliberately raises because the timer-phase behavior is not implemented or hardware-traced.
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
  - Facade exports: `CPU`, `CPUBackendName`, `available_backends`,
    `select_backend`, `Emulator`/`PythonEmulator`, `RegisterName`, and
    `Registers`.

### External Types the Rust Core Must Respect
- `binja_test_mocks.eval_llil.Memory`: wraps provided `read_mem` / `write_mem` callables. Exposes `read_byte`, `write_byte`, `read_bytes`, `write_bytes`.
- `binja_test_mocks.eval_llil.State`: simple dataclass with `halted: bool`.
  The SC62015 intrinsic layer additionally tags it with `power_state` =
  `running`, `halted`, or `off` so OFF is not silently given HALT wake behavior.
- `binja_test_mocks.eval_llil.ResultFlags`: `TypedDict` with optional `C` and `Z` fields. Returned by LLIL evaluation helpers.

## CPU Execution Lifecycle
1. **Construction**: `Emulator(memory, reset_on_init=True)` stores a fresh `Registers` instance, keeps the supplied `Memory`, initialises `State()`, records `call_sub_level`, and registers intrinsic evaluators.
2. **Reset path**: `power_on_reset()` resolves the three-byte reset vector at `0xFFFFD` (the interrupt vector is separately stored at `0xFFFFA`) only when the vector and decoded target bytes are explicitly immutable. It binds the one matching architectural vector fetch, decoded target length, and memory provenance into an owner-held one-shot operation before delegating any SFR/state change to `intrinsics.eval_intrinsic_reset`. The whole-machine wrapper prepares that operation before clearing RAM or LCD state. External users often pass `reset_on_init=False` and call `power_on_reset` manually when they need fine-grained control. The remaining RESET register side effects are model contracts pending dedicated hardware tracing.
3. **Fetch/decode**: `decode_instruction(address)` builds a `FetchDecoder` (cached variant when available) that reads bytes via `Memory.read_byte`, then calls `instr.decode`. Any LLAMA/native replacement must honour `ADDRESS_SPACE_SIZE` bounds and PC masking.
4. **Preflight**: device wrappers prepare the final instruction before advancing timers, peripherals, or fallible observers. Reserved/noncanonical encodings, `TCL`, and every `I=0` counted instruction are rejected there and again inside the selected backend. Every opcode byte inspected by scheduling and every vector/target byte selected for same-pass transfer must be explicitly callback-free; the normal opcode/vector fetch is captured with its PC, fused length, and memory provenance in an owner-held one-shot operation. Execution must consume that exact operation. RESET additionally requires immutable vector and target bytes. HALT/OFF wake is an idle-step boundary and does not fetch the dormant PC or IRQ vector; masked pending status, an active handler, and a timer that can only arm a next-pass IRQ likewise do not inspect the vector. An already-unmasked IRQ may replace the saved PC on the following scheduling pass, where its vector is proved before delivery. A mismatch, remap, duplicate use, or missing proof fails before any frame, SFR, trace, or transfer-related wrapper metadata changes.
5. **Execution**: `execute_instruction(address)` snapshots CPU-owned state, updates tracing metadata, optionally wraps evaluation in a Perfetto slice when the memory stub exposes `_perf_tracer`, and then defers to `_execute_instruction_impl`. A failure restores CPU-owned state. If a host callback may already have committed an external effect, the CPU is poisoned until a complete RESET reconciles it.
6. **LLIL evaluation**: `_execute_instruction_impl` wraps the architected PC to 20 bits, decodes and validates the instruction, updates `call_sub_level`, and lifts to LLIL. Nonzero `WAIT` uses a required `memory.wait_cycles` hook, clears `I` only after the callback succeeds, and preserves flags. Ordinary instructions pre-advance PC by the decoded length before evaluating their graph; control-flow LLIL may replace it.
7. **Post-step state**: The emulator returns `InstructionEvalInfo` containing Binary Ninja metadata only after all synchronous work succeeds. Native deferred host writes are flushed before success is reported.

The Python emulator must stay in lockstep with the Rust core so existing consumers (assemblers, snapshot steppers, and the PC-E500 device model) continue to work unchanged.

## Memory & Peripheral Expectations
- `Memory` callbacks are provided by callers; the emulator assumes `read_byte`/`write_byte` raise on out-of-range access. Internal memory accesses use `INTERNAL_MEMORY_START` offsets. Tests rely on this to mirror IMEM behaviour.
- The emulator does not own peripheral emulation; instead the backing memory object is expected to intercept writes/reads for IO behaviour. Several pce500 components hang attributes (e.g. `_perf_tracer`) or override methods on the memory instance.
- `Registers.call_sub_level` is used downstream for profiling (`pce500/tests/test_tracing_call_stack.py`). The Rust implementation must expose the same attribute on the register file or an equivalent property reachable from Python.

The higher-level PCE snapshot wrapper uses version 4 and rejects legacy v3
outright because v3 cannot attest that its source omitted no host state. Version
4 serializes exact present/absent memory-card mode, capacity, writability, and
payload. Save and load still reject pending serial RX/TX bytes, cassette tape
blocks/cursor state, custom handler-backed or writable memory overlays, static
read-only overlays that overlap another mapping or extend outside the flattened
image, and a custom IMEM access callback because those contracts are not
encoded. Disjoint, fully flattened static ROM overlays remain representable.
This is an implementation-integrity guarantee, not a claim that the remaining
modeled device state matches silicon.

## Configuration & Environment Hooks
- `USE_CACHED_DECODER` toggles automatically depending on whether `cached_decoder` imports successfully. Parity requires the native backend to expose a similar knob or transparently outperform the Python cache.
- Tests set `FORCE_BINJA_MOCK=1` so imports from `binaryninja` resolve to `binja_test_mocks`. The native tooling should continue to honour this environment variable, especially when generating LLIL metadata.
- `reset_on_init`: honoured by both `Emulator` and `CPUStepper`; the native constructor must take—and default—this flag identically.

## pytest Coverage Matrix
The following files are under `sc62015/pysc62015/` and define the behavioural envelope the Rust backend must satisfy:

| Feature area | Expectations | Tests |
| --- | --- | --- |
| Register file semantics | Read/write masking, composite register updates, flag aliases, PC masking | `test_emulator.py::test_registers`, `test_emulator.py::test_pc_mask` |
| Instruction decode + execution semantics | Broad sweep of load/store, arithmetic, logic, branch, stack, interrupt, and vector operations. Each `InstructionTestCase` validates decode string, register effects, memory writes, and PC advance | `test_emulator.py::test_instruction_execution` parametrized suite (covers ~80 instruction forms); specialist cases such as PUSH/POP, call/return, stack behaviour |
| WAIT fast path | For `I>0`, opcode `0xEF` accounts exactly `I` model cycles, clears `I` after a successful timing callback, preserves flags, and advances PC; `I=0` or a missing/failing hook is atomic failure | Direct and lifted cases in `test_emulator.py`, bridge failures in `test_llama_callback_errors.py`, parity in `test_llama_parity_misc.py`, plus `test_stepper.py` |
| Call stack tracking | `call_sub_level` increments/decrements via CALL/RET opcodes and feeds tracing | Covered indirectly by `pce500/tests/test_tracing_call_stack.py`; no dedicated unit test, so parity should rely on opcode map in `emulator.CALL_STACK_EFFECTS` |
| HALT/OFF/RESET intrinsics | IMEM register side effects, halt state toggles, validated single-fetch PC jump to reset vector, and atomic rejection of invalid/volatile vectors | `test_halt_off_reset.py` (`test_halt_instruction`, `test_off_instruction`, `test_reset_instruction`, `test_power_on_reset`, and vector failure cases) |
| Snapshot stepper contract | Serialisation/diff of register snapshots, capturing memory writes, instruction metadata passthrough | `test_stepper.py` cases for NOP, WAIT, and SC flag-setting |
| Instruction metadata + LLIL lifting | Instruction naming, operand rendering, LLIL jump targets, mode decoding | `test_instr.py` (covers `JP`, addressing modes, operand renderers, HALT/OFF/TCL intrinsics etc.) |
| Assembler integration | Round-tripping assembly text to opcode bytes, internal register naming, full-program assembly | `test_asm.py`, `test_asm_e2e.py`, `test_asm_imem_register_names.py`, `test_opcode_assembly.py` |
| Architecture hookup | Binary Ninja `Architecture` shim returns canonical metadata/LLIL and rejects invalid decoder results | `sc62015/test_arch.py` |

**Downstream integration tests** (outside this package) rely on these APIs as well:
- `pce500/emulator.py` and associated tests instantiate `Emulator` and `CPUStepper` for the handheld device model.

## Open Questions / Follow-ups
- Documented downstream usages indicate that `Registers` and `RegisterName` must remain importable at their current module paths. The Rust bindings should mirror those names to avoid churn.
- `call_sub_level` is covered by `pce500/tests/test_tracing_call_stack.py` and
  snapshot/runtime restoration tests; new call-like opcodes must update both
  backends and those integrity checks together.
- Validate that `binja_test_mocks` provides all LLIL constructs needed for full opcode coverage before relying on auto-generated metadata from Binary Ninja.

Keeping this document current as the Rust implementation lands will help ensure both cores stay in feature lockstep.
