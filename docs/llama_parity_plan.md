# LLAMA Parity Guardrails

- Purpose: keep the LLIL-free Rust CPU core (“LLAMA”) behaviour-aligned with the Python emulator.
- Primary checks:
  - Opcode sweep parity: `FORCE_BINJA_MOCK=1 uv run python tools/llama_parity_sweep.py` (uses Python as oracle; emits Perfetto traces for deeper diffing).
  - Python test suite with LLAMA backend: `PYTHONPATH=. FORCE_BINJA_MOCK=1 uv run pytest --cpu-backend=llama sc62015/pysc62015`.
  - Perfetto smoke traces in CI: nightly/dispatch workflow compares NOP, CALL, EMEM MV, DSBL between LLAMA and Python using `scripts/compare_perfetto_traces.py`.
- Perfetto traces: LLAMA emits streaming retrobus-perfetto binary traces (instruction + memory threads, instruction-index timestamps). Python parity runner emits the same format; `scripts/compare_perfetto_traces.py` compares two traces.
- Scope: avoid SCIL/LLIL dependencies; prefer enums/typed operands over stringly identifiers.
- Maintenance notes:
  - If opcode table changes, regenerate Rust dispatch by hand (no JSON indirection).
  - Keep stack/init semantics in sync (S/U start in internal RAM; CALL/IR push order matches Python).
  - Temp registers are ignored in parity sweep unless explicitly under test; flags and observable memory writes must match.
