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
- ROM loading mirrors the native runner: the last `0x40000` bytes are mapped to `0xC0000..0xFFFFF` and `power_on_reset` uses the reset vector near `0xFFFFD`.
- Keyboard mapping lives in `src/lib/keymap.ts` (currently `F1/F2` + arrow keys).
