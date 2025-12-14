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

## Notes
- WASM package output is generated into `src/lib/wasm/pce500_wasm` (ignored by git). Rebuild it with `npm run wasm:build`.
- ROM loading mirrors the native runner: the last `0x40000` bytes are mapped to `0xC0000..0xFFFFF` and `power_on_reset` uses the reset vector near `0xFFFFD`.
- Keyboard mapping lives in `src/lib/keymap.ts` (currently `F1/F2` + arrow keys).
