# LLAMA/Python Parity Fix Plan

Source: `tools/llama_parity_sweep.py` (latest run, 16 memory-only mismatches). No register diffs observed.

## Mismatch Inventory (by opcode/encoding)
- `0x47`: writes seen in Python only (`47abcd`, `470007`, `470707`) — LLAMA missing external writes at `0x1002BB`, `0x100000`, `0x100007`.
- `0xD3`: writes seen in LLAMA only (`d300000000`, `d307000000`) — LLAMA writing `0x100000`/`0x100007`, Python not.
- `0xDB`: writes seen in LLAMA only (`db00000000`, `db00000007`) — LLAMA writing `0x0`, Python not.
- `0xDE`, `0xDF`: writes seen in Python only — internal trace bytes at `0x1002E8`, `0x1002EF` missing in LLAMA.
- `0xEB`: writes seen in LLAMA only (`eb0400`, `eb0407`) — LLAMA writing `0x0`, Python not.
- `0xF3`: writes seen in LLAMA only (`f300abcd`, `f300cdab`) — LLAMA writing `0x1002BB`, `0x1003D5`, Python not.
- `0xFB`: writes seen in LLAMA only (`fb00abcd`) — LLAMA writing `0x0`, Python not.
- `0xFE`: write seen in Python only — internal `0x10000D`=1 missing in LLAMA.
- `0xFF`: write seen in Python only — internal `0x1002E8`=24 missing in LLAMA.

## Fix Passes (proposed order)
1) Align LLAMA internal-state writes for opcodes with Python-only effects (`0xDE/DF`, `0xFE`, `0xFF`).
2) Align LLAMA external memory side effects for data-move opcodes (`0x47`, `0xF3`); verify addressing modes/widths.
3) Remove unintended LLAMA writes for opcodes where Python is silent (`0xD3`, `0xDB`, `0xEB`, `0xFB`); confirm pointer arithmetic and guarding of read-only/IO ranges.
4) Re-run full parity sweep (no `--allow-mismatches`) and iterate until zero mismatches.
