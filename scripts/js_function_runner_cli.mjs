#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

function usage() {
	return `
Usage:
  node scripts/js_function_runner_cli.mjs [options] <script.js>
  node scripts/js_function_runner_cli.mjs [options] --eval "<js>"
  cat script.js | node scripts/js_function_runner_cli.mjs [options] --stdin

Options:
  --model <iq-7000|pc-e500>   ROM preset (default: Rust runtime default)
  --rom <path>               Explicit ROM path (overrides --model)
  --eval <js>                Inline script (async JS)
  --stdin                    Read script from stdin
  --help                     Show this help

Prereqs:
  - Install web deps + build wasm:
      cd web && npm install && npm run wasm:build

Script context:
  - Uses the same Function Runner API as the web UI (EvalApi from web/src/lib/debug/sc62015_eval_api.ts).
`;
}

function die(msg) {
	console.error(msg);
	process.exit(2);
}

const repoRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));
const webDir = resolve(repoRoot, 'web');
const viteNodePath = resolve(webDir, 'node_modules', 'vite-node', 'vite-node.mjs');
const entryPath = resolve(webDir, 'scripts', 'js_function_runner_cli.ts');

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
	console.log(usage().trimEnd());
	process.exit(0);
}

if (!existsSync(viteNodePath)) {
	die(`error: missing ${viteNodePath}\nRun: cd web && npm install`);
}
if (!existsSync(entryPath)) {
	die(`error: missing ${entryPath}`);
}

const child = spawn(process.execPath, [viteNodePath, entryPath, ...argv], {
	cwd: webDir,
	stdio: 'inherit',
});
child.on('exit', (code) => process.exit(code ?? 1));
child.on('error', (err) => die(String(err)));

