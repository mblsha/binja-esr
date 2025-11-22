# Project LLAMA Parity Plan

Goal: verify the LLIL-less Rust core against the existing Python emulator before wiring it into rustcore. Use the Python implementation as an oracle for register/memory/flag effects.

## Strategy
- **Opcode sweep:** iterate the typed opcode table (`llama::opcodes::OPCODES`), decode a synthetic instruction stream, and run one step on both cores.
- **Length hints:** reuse `operand_len_bytes` to synthesize minimal-length streams for each operand mix; fix or override heuristics when dealing with mode placeholders (post-inc/pre-dec EMEM, EMemImemOffset).
- **Bus hooks:** LLAMA’s bus exposes `resolve_emem` to map EMEM bases; align Python side to use the same address resolution for fairness.
- **Feature gate:** put the parity harness under the `llama-tests` feature (see `sc62015/core/src/llama/parity.rs` stub) so normal builds stay unaffected.
- **Oracle:** call into the Python emulator (via PyO3 or JSON round-trips) to execute a single instruction on a scratch memory image; capture registers/flags and memory deltas.
- **Subject:** run the same opcode through LLAMA evaluator with the same initial state/memory and compare end state (registers, flags, memory writes).
- **Edge cases:** include PRE variants, IMEM/EMEM offset modes, intrinsics (HALT/OFF/RESET/TCL), and CALL/RET/IR control flow for PC updates.

## Harness shape (Rust side)
- Build a thin FFI bridge or JSON channel to invoke the Python emulator for a single step, given opcode bytes and initial register/memory snapshots.
- For each opcode entry:
  - Synthesize operands (constants or simple addresses) based on operand kinds to avoid undefined behaviour (e.g., safe IMEM addresses).
  - Seed state/memory with deterministic values (non-zero registers, flags set/cleared).
  - Execute Python oracle, capture end state and memory writes.
  - Execute LLAMA evaluator, capture end state and memory writes.
  - Diff registers (respect aliasing/masking) and memory deltas.
- Surface failures with opcode/operand context for debugging.

## Python helper (suggested)
- Expose a helper in the Python emulator that accepts opcode bytes + initial state/memory and returns a JSON blob with registers/flags/memory writes. This avoids invoking full pytest.
- Use `binja_test_mocks` memory stubs as the backing memory for consistency.

## Test selection
- Start with a representative subset (e.g., arithmetic/logic, IMEM/EMEM moves, intrinsics) then scale to all opcodes.
- Skip PRE explosion initially; add PRE-aware cases once base parity passes.
- Include WAIT/HALT/OFF/RESET and IRQ entry/exit (IR) to exercise control flow.

## Acceptance criteria
- Registers/flags match after masking/aliasing (BA<->A/B, I<->IL/IH, PC mask, FC/FZ).
- Memory writes match (address + value, IMEM vs EMEM classification).
- Control flow agrees (PC advance vs branch/call/ret behaviour).
