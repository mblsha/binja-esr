# Binary Ninja ESR-* plugin

The ESR plugin provides an SC62015 (aka ESR-L) architecture for Binary Ninja.

Currently it only works as a crude disassembler, with the goal to lift all the
instructions and create memory mapping for Sharp PC-E500 and Sharp Organizers.

## Acknowledgements

Overall structure of instruction logic based on
[binja-avnera](https://github.com/whitequark/binja-avnera) plugin by
@whitequark.

## License

Apache License 2.0.

## Development

Install dependencies using uv and run the checks:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv

	# Install all dependencies and create virtual environment
	uv sync --extra dev --extra pce500

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright sc62015/pysc62015

	# Run tests with coverage
	FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml
	```

The CI workflow uploads coverage results to Codecov on each commit.

## CLI emulator (terminal LCD)

The Rust LLAMA CLI is the primary emulator core. Run it with a terminal-rendered LCD view:

```bash
cargo run --manifest-path sc62015/core/Cargo.toml --bin sc62015-lcd -- --model pc-e500
```

Notes:
- Use `--model iq-7000` to switch ROM/profile.
- IQ-7000 date/time screens seed the CLOCK workspace from the host clock by default; use
  `--iq7000-rtc YYYYMMDDHHMM` for deterministic captures or `--iq7000-rtc off` for raw ROM
  behavior.
- Use `--refresh-steps 20000` to control redraw cadence.
- Use `--input-steps 1000` to poll for key presses more frequently.
- Use `--no-alt-screen` for tmux capture panes.
- Use `--force-tty` when running detached.
- Use `--pf-numbers` to map digits 1–5 to PF1–PF5 (disables typing those digits).
- Use `--bnida PATH` to show function names in the status line (defaults to `rom-analysis/.../bnida.json` if present).
- Use `--force-key-irq` if the ROM stays halted at the boot menu (forces KEY interrupts on key press).
- Use `--card auto|present|absent` to control memory card slot state. `auto`
  selects a blank writable PC-E500 card and an absent IQ-7000 card.
- Keys: Ctrl+1..5 or F1..F5 → PF1..PF5, Enter → `=`, Backspace → `BS`, Ctrl+C exits.

The headless runner (`--bin pce500`) also supports reusable IQ-7000 probe captures:

```bash
cargo run --manifest-path sc62015/core/Cargo.toml --bin pce500 -- \
  --model iq-7000 \
  --runtime core \
  --key-seq "memo,text:PASSPORT NO.\\nM6711888\\nEXPIRES 12/25/90,memo-enter,memo,search-down" \
  --capture-png /tmp/iq7000-a0.png \
  --capture-json /tmp/iq7000-a0.json \
  --debug-probe-json /tmp/iq7000-a0.debug.json \
  --debug-probe-range storage@0x1fd00:0x40
```

Ordinary headless execution uses the shared `CoreRuntime`, matching the WASM,
terminal, and IQ-7000 PC-Link frontends. PC stop/trace, final LCD provenance,
structured memory probes, and exact bounded LCD-write logging now run on that
same scheduler. Historical trace-replay boot overlays, raw external-bus
transaction logging, and snapshots still require an explicit legacy runtime;
unsupported combinations fail instead of silently changing schedulers.

Runtime `cycle_count` advances in the SC62015 relative timing units documented
by the instruction table: a NOP is one unit, conditional/counting forms use
their selected path and initial `I`, and each fused PRE adds one unit. Forms
without a complete published total (the D8-DB transfer direction, HALT/OFF
entry, IR, and RESET) retain explicitly provisional compatibility values.
These units are not calibrated oscillator cycles: PC-E500 absolute timer
cadence and IQ-7000 timing still require machine-level qualification. CLI
timing diagnostics label this basis explicitly.

`CoreRuntime::step_scheduler_boundaries(n)` is the unambiguous execution API.
The shorter `step(n)` remains a compatibility alias. A running boundary usually
retires one instruction, while HALT/OFF can consume a boundary without retiring
one; neither argument is a timing-unit or wall-clock duration. Use
`instruction_count()` for retired work and `cycle_count()` for relative timing.

`SCR.MTS` selects the 4 ms or 16 ms main-timer compatibility period and
`SCR.STS` selects the approximately 0.5 s or 2 s sub-timer period. Changing a
selector currently starts a fresh period; exact divider phase at an `SCR`
write remains a hardware-timing question.

The Rust SIO bridge also advances from these retired-instruction timing units.
RX-ready and TX-complete status therefore no longer depend on when a TCP/WASM
host polls the bridge. Its delay constants remain a functional compatibility
model, not a measured baud-rate model. Historical replacements for three
PC-E500 serial ROM entries are disabled by default and available only through
an explicit diagnostic opt-in; normal model runs execute the ROM and never
manufacture a serial peer response.

Keyboard ingress is likewise explicit. Physical matrix contacts affect KIL
and can assert `ISR.KEYI` only while their column is selected. Translated host
events (for example digitizer samples) enter the host event FIFO without
changing KIL or manufacturing a silicon interrupt. The legacy combined
immediate-matrix/FIFO injection remains available only as a diagnostic helper.

The Python `CPU(..., backend="llama")` facade and `pce500/run_pce500.py` remain
CPU differential/parity tools. Their machine scheduler and peripheral callbacks
are Python-owned, so they are not another Rust machine runtime and should not be
used to benchmark or qualify the shared emulator. `backend_stats()` reports
`execution_scope=cpu-only` and `scheduler_owner=python-caller` to make this
boundary visible to tooling.

`digitizer:0xNN` and `event:0xNN` still inject exact translated input bytes. For
IQ-7000 app/editor work, the runner also accepts named event keys such as
`calendar`, `memo`, `tel`, `home`, `world`, `shift`, `caps`, `caps-off`,
`search-up`, `search-down`, `newline`, and `memo-enter`. `caps-off` injects the
CAPS key once; the IQ-7000 ROM starts with CAPS enabled, so include it before
lowercase/mixed-case text entry on a freshly booted image. `text:...` expands
printable characters through the generated per-model input map; use `\\n` inside
text to emit the MEMO newline key. IQ-7000 PNG/JSON captures draw only the
confirmed SHIFT/CAPS LCD annunciators. JSON diagnostics preserve the two
sources separately as `state_raw` (`0x1FDA3`) and `shadow_raw` (`0x006160`),
name their OR explicitly as `raw_union`, and report unmapped bits numerically
per source. In particular, raw bit `0x80` has no assigned icon or physical
meaning.

In the live terminal LCD (`sc62015-lcd --model iq-7000`), `F6` injects the
IQ-7000 SHIFT event, `F7` injects CAPS, and `F8` injects the FUNCTION event;
Caps Lock is also accepted when the terminal reports it. The status line shows
`lcd=SHIFT:...,CAPS:...`; unknown bits are appended numerically as
`UNMAPPED_STATE:0xNN` and/or `UNMAPPED_SHADOW:0xNN` without drawing an icon.

`CoreRuntime::set_external_interrupt_level` is currently a neutral API/test
hook. Its level-sensitive EXI re-latch policy is an explicit emulator model
contract, not a measured device fact. Neither command-line frontend exposes a
host switch for that input yet.

See [`docs/sc62015_runtime_evidence.md`](docs/sc62015_runtime_evidence.md) for
the concise boundary between real-device-derived instruction behavior,
ROM-grounded interrupt dispatch, provisional machine timing/peripherals, and
implementation-only safeguards.

For repeatable captures, put the same settings in a scenario JSON file:

```json
{
  "model": "iq-7000",
  "steps": 3000000,
  "key_seq": [
    "memo",
    "text:PASSPORT NO.\\nM6711888\\nEXPIRES 12/25/90",
    "memo-enter",
    "memo",
    "search-down"
  ],
  "capture_png": "memo.png",
  "capture_json": "memo.json",
  "debug_probe_json": "memo.debug.json",
  "debug_probe_range": ["storage@0x1fd00:0x40"]
}
```

Run it with `--scenario path/to/scenario.json`. Relative capture/debug paths are
resolved relative to the scenario file.

The web/WASM Function Runner has the same deterministic IQ-7000 RTC seeding for
screen probes:

```bash
cd web
npm run fnr:cli -- --model iq-7000 --iq7000-rtc 202604261330 --eval '
await e.step(100000);
await e.keys.app.tap("calendar");
await e.wait.screenChange();
await e.wait.lcdStable();
const calendar = await e.lcd.assertCalendarMonth({ year: 2026, month: 4, day: 26 });
const proof = await e.proof.metadata({ label: "calendar-apr-2026", assertions: { calendar } });
return { lines: await e.lcd.text(), proof };
' --proof-yaml calendar-proof.yaml
```

For new Function Runner scripts, prefer the explicit key namespaces:
`e.keys.app.tap("calendar")` for app selectors, `e.keys.event.tap(0x18)` for
ROM-visible translated events, and `e.keys.phys.tap(0xNN)` for raw physical
matrix/scanner codes. `e.lcd.assertCalendarMonth(...)` validates the compact
calendar day-number pixels directly, while `e.lcd.text()` now decodes those
same compact monthly-calendar day rows for readable CI output. `--proof-yaml`
writes concise YAML metadata with the RTC seed, ROM path, assertions, LCD text,
and pixel signature.
