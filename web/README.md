# PC‑E500 Web Emulator (LLAMA/WASM)

SvelteKit + TypeScript UI that runs the Rust LLAMA SC62015 core compiled to WebAssembly via `wasm-pack`.

## Prerequisites
- Node.js `20.19+` (CI pins `20.19.0`)
- Rust stable + `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- `wasm-pack` (`cargo install wasm-pack` or your package manager)

## Develop
```sh
cd public-src/web
npm ci
npm run dev
```

`npm run dev` runs a Rust/WASM rebuild watcher alongside the Vite dev server. Edit Rust under `web/emulator-wasm/` or `sc62015/core/` and the WASM package will be rebuilt automatically.

`npm run dev` listens on all interfaces (`0.0.0.0`) so you can connect via LAN/public IPs. To restrict it back to localhost, run `npm run dev -- --host 127.0.0.1`.

If you access the dev server via a hostname (e.g. `vibe-qemu2.local`) and see a “host is not allowed” error, set `VITE_ALLOWED_HOSTS` (comma-separated) or set it to `true` to allow any Host header:
`VITE_ALLOWED_HOSTS=vibe-qemu2.local npm run dev`

Open the dev server, then use the ROM file picker.

## Build
```sh
cd public-src/web
npm run build
npm run preview
```

## Tests
```sh
cd public-src/web
npm run check
npm run test
npm run wasm:test
```

## IQ-7000 MEMO typing benchmark

From `public-src/web`, with your own ROM available:

```sh
npm run bench:iq-memo -- --rom /path/to/iq-7000.bin --out /tmp/iq-memo-bench
```

The default workload boots a fresh machine, enters MEMO, switches CAPS off,
types the full 445-character lowercase Lorem Ipsum paragraph, then stores and
reopens it through normal keys. It uses the shared physical-key paste planner,
40,000-boundary holds/gaps and cooperative Rust slices, without FIFO injection,
application stubs or direct RAM writes. It runs headlessly and unthrottled, with
`setImmediate` yields rather than real-time pacing or per-key host sleeps.
RTC is fixed at `199201101050`; this is a throughput benchmark, not calibrated
hardware timing or a benchmark of browser rendering/input-event dispatch.

`--rom` defaults to `IQ7000_ROM_PATH`, then `public-src/data/iq-7000.bin`.
Use `--repeats 3` for independent cold-machine runs (later runs have a warmed
V8 engine), or `--text 'lorem ipsum dolor sit amet.'` for a short smoke test.
Custom text must use lowercase ASCII letters/digits, spaces, commas and periods;
unsupported characters are rejected rather than dropped. See `--help`.

After a known-current release WASM build, skip rebuilding with:

```sh
npm run bench:iq-memo:built -- --repeats 3 --out /tmp/iq-memo-bench
```

### Profiling inside WASM

Build a separate optimized artifact with Rust function names retained:

```sh
npm run wasm:build:profile
npm run bench:iq-memo:built -- --wasm-dir node_modules/.cache/iq-memo-wasm-profile \
  --cpu-profile --out /tmp/iq-memo-cpu
```

This records only the typing phase in `run-N.cpuprofile`, viewable in Chrome
DevTools' JavaScript Profiler. WASM stack frames carry Rust function names;
inlined operations remain attributed to their enclosing optimized function.
The profiling build does not replace the normal web UI WASM artifact. Its
`wasm-opt -g` setting is important: `--profiling` alone can still lose names in
the final optimizer. Compare runtime counters and artifact hashes before
comparing results; use separate unprofiled runs for throughput measurement.

For guest behavior, the WASM core already integrates **retrobus-perfetto**.
Capture a bounded prefix of typing using its existing start/stop API:

```sh
npm run bench:iq-memo:built -- --trace-boundaries 2000 --out /tmp/iq-memo-guest
```

This writes `run-N.perfetto-trace` with SC62015 instruction/function slices,
memory activity and IRQ events. Its timestamps are **guest instruction-index
ticks, not host CPU nanoseconds**. It explains what the ROM executes, not which
Rust function consumes wall time. The full typing/store/reopen verification
still runs afterward. The capture is limited to 1–20,000 scheduler boundaries
to avoid an enormous all-instruction trace; serialization and tracing overhead
are included in this diagnostic run's typing time. CPU sampling and guest
tracing cannot be enabled together, to avoid profiling trace-generation costs
as normal execution. Both require `--out`; distinct directories preserve runs.

### Experimental adaptive MEMO input

```sh
npm run bench:iq-memo -- --adaptive-memo --out /tmp/iq-memo-adaptive
```

This opt-in, MEMO-specific experiment first types up to 16 seed characters with
normal timing and discovers a newly appearing, unique NUL-terminated editor
copy. It then holds each character contact until the expected complete prefix
and terminator appear in emulated memory, polling every 256 boundaries with a
40,000-boundary timeout. All reads are observational: there is no FIFO, RAM,
register or interrupt injection. Modifier contacts and menu navigation retain
conservative timing. The release gap defaults to 8,000 boundaries; timers and
interrupts continue operating normally.

`--adaptive-gap 1..40000` and `--input-phase 0..10000` support qualification
experiments. Shorter gaps are not necessarily safe: 4,000 failed the full Lorem
Ipsum workload. This is not a general app/scanner acknowledgement and does not
change browser input defaults. It requires a unique discovered buffer, checks
for unexpected extra text, and fails rather than silently retrying a potentially
accepted contact. `run-N-failure.json` records failed trials when `--out` is set.
Exact full-text/store/reopen verification remains mandatory. Adaptive results
execute fewer guest instructions, so compare them separately from fixed-pacing
CPU benchmarks. The report records per-contact hold budgets and input strategy.

Qualification includes the 445-character paragraph at several scanner-phase
offsets, repeated letters, punctuation, and a 512-character MEMO. An oversized
856-character MEMO failed with both adaptive and fixed pacing; adaptive input
must not be treated as bypassing the application's storage limits. Other apps,
editor modes, and arbitrary existing device contents remain unqualified.

### Comparing optimized WASM builds

With two separately built artifact directories and a private ROM:

```sh
node node_modules/vite-node/vite-node.mjs --script scripts/compare_wasm_builds.ts \
  /tmp/base-wasm /tmp/candidate-wasm /path/to/iq-7000.bin iq-7000 /tmp/wasm-comparison
```

Use `pc-e500` and its ROM for the other model. The smoke compares register state,
instruction/timing counts, power, contacts, complete internal/external memory
hashes, LCD pixels/text and IQ RTC state after boot and physical key transitions.
It is a differential regression check, not an independent silicon oracle or an
exhaustive app sweep.

Every run prints boot, MEMO-entry, typing and store/reopen times, wall totals,
character throughput, retired-instruction and scheduler-boundary counts. The
build, WASM compilation and artifact output are outside the measured phases.
The first run can include V8 execution warm-up. `--out` additionally writes
`benchmark.json` (including ROM/WASM hashes and all verification observations)
and `run-N-lcd.json` (actual full-glass grayscale capture). Output files with the
same names are replaced; use a separate directory for comparisons.

Success requires the complete exact text to be absent initially, present in
emulated memory after typing, and copied to a new location after store/reopen;
the reopened ROM display must show its beginning and all contacts must be
released. RAM is inspected read-only, not injected or treated as a fully parsed
record-format proof. Failure exits nonzero; runs have a 120-second host timeout.
The full default workload measured about 32 seconds locally before CPU-side
optimizations; this is roughly 14 characters/s through the conservative input
policy, not a claim that shorter contact timings are safe.

## Function runner stubs

Live display delivery keeps one frame in transit and coalesces pending refresh
requests into a lazy capture of the newest state. A slow UI cannot accumulate
a queue of old screenshots. Pause/input acknowledgements do not wait for display
credit. Explicit Function Runner LCD artifacts are not coalesced or dropped.

The browser Function Runner uses a disposable script worker, separate from the
worker that owns the Rust/WASM machine. Stop terminates runaway user JavaScript
(including stub/probe callbacks), then waits for cooperative machine execution
and debugger cleanup. It does not reset or roll back guest state. Browser
scripts own raw matrix/ON contacts only for their invocation: those contacts
are released on success, error or cancellation without releasing another host
input owner's contact. Keep a held-key experiment within one invocation.

Serve the app over HTTPS or localhost with these headers on **both HTML and
worker JavaScript**, including immutable assets served by a proxy/CDN:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Vite development/preview and SvelteKit responses are configured here. Production
static-asset hosting must preserve the headers too. Plain HTTP on a remote LAN
hostname is not a secure context: normal emulation may work but Function Runner
will refuse scripts. There is no UI-thread script fallback. See
[MDN's worker-isolation requirements](https://developer.mozilla.org/en-US/docs/Web/API/WorkerGlobalScope/crossOriginIsolated).

`e.reg()` and stub memory readers remain synchronous, but always query the live
machine owner. Only the script worker waits on a shared reply mailbox; emulated
RAM itself is not shared or duplicated. `e.calls`/`events`/`prints` and `last()`
return owned data copies, not mutable references to the owner's artifact store.
RPC data is JSON-shaped (BigInts become decimal strings); functions travel only
through the explicit callback APIs. Nonfinite numbers are rejected. A reply is
limited to 16 MiB, synchronous RPC waits to 30 seconds, and a browser script to
270 seconds of host time. These are host safety limits, not emulated timing.
The Node `fnr:cli` retains its offline execution model; browser script isolation
is not a security sandbox or a claim that the CLI preempts arbitrary JavaScript.

The Function Runner (UI + `fnr:cli`) can intercept execution at a specific PC and apply patches.

```js
e.stub(0x00F1234, 'demo_stub', (mem, regs, flags) => ({
  mem_writes: { 0x2000: mem.read8(0x2000) ^ 0xff },
  regs: { A: 0x42 },
  flags: { Z: 0, C: 1 },
  ret: { kind: 'ret' }, // or retf/jump/stay
}));
await e.call(0x00F1234, undefined, { maxInstructions: 5_000 });
```

## Notes
- WASM package output is generated into `src/lib/wasm/pce500_wasm` (ignored by git).
  - One-off rebuild: `npm run wasm:build`
  - Dev rebuild + watch: `npm run wasm:build:dev` and `npm run wasm:watch` (included in `npm run dev`)
- ROM loading mirrors the native runner: a 256 KiB image (or the top 256 KiB
  of a full capture) maps to `0xC0000..0xFFFFF`, while an exact 128 KiB base
  ROM maps to `0xE0000..0xFFFFF`; `power_on_reset` uses the vector at
  `0xFFFFD` in either layout.
- Initial model-specific control mappings live in `src/lib/keymap.ts` (full
  character-key mapping is still being qualified). PC-E500 uses F1–F5 for PF
  keys; IQ-7000 uses F1–F8 for app keys. F12 is ON for both models.
- Virtual controls default to a minimum 40,000 **scheduler-boundary** hold from
  DOWN; uncheck assistance for immediate raw releases. No machine work runs
  solely to complete a tap while paused. Focus loss, hidden documents and
  cancellation force releases. Keyboard/mouse/script owners cannot release
  each other's contacts. An input acknowledgement is not ROM-consumption proof.
- Use `e.keys.phys` for physical-contact experiments. Legacy `keys.event` and
  `keyboard.injectEvent` still perturb debounce/FIFO state and are diagnostic
  compatibility helpers, not normal-UI or hardware input evidence.

## Function Runner CLI

The following commands use the repository root as their starting directory.

- **JS function runner (WASM):** Run an async JS snippet against the same Rust core compiled to WASM:
  - Install deps once: `cd web && npm install`
  - Run a script (auto-builds wasm): `cd web && npm run fnr:cli -- --model pc-e500 path/to/script.js` (or `--eval "<js>"`, or `--stdin`)
  - Script API: `e` is the same `EvalApi` used by the web Function Runner; output JSON is compatible with `FunctionRunnerOutput`.
  - IQ-7000 PC-Link scripts can use `await e.pclinkSerial.serve("127.0.0.1:7700", { clients: 1 })` after the ROM reaches `LINK READY`; this serves a TCP bridge backed by the emulated SC62015 COM/SIO registers, not a RAM injection path.
  - Stubs: `e.stub(0x00F1234, 'demo', (mem, regs, flags) => ({ mem_writes: { 0x2000: 0x41 }, regs: { A: 1 }, flags: { Z: 0, C: 1 }, ret: { kind: 'ret' } }))` intercepts a PC and returns a patch; `mem.read8/read16/read24` are read-only and writes flow through `mem_writes` (array or `{addr: value}` map).
    Return kinds: `ret`, `retf`, `jump`, `stay`.
  - Note: this is separate from the native Rust CLI (no TS wrapper for the Rust CLI); the web UI uses `web/src/lib/wasm/sc62015_wasm.ts` to keep `Pce500Emulator` as an alias for `Sc62015Emulator`.
