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
