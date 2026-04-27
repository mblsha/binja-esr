import { spawn } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import process from 'node:process';

async function readStdin() {
	const chunks = [];
	for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
	return Buffer.concat(chunks).toString('utf8');
}

function run(command, args, options = {}) {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, {
			stdio: options.stdio ?? 'inherit',
			env: process.env,
			shell: false,
		});
		if (options.stdin !== undefined && child.stdin) {
			child.stdin.end(options.stdin);
		}
		child.on('error', reject);
		child.on('exit', (code, signal) => {
			if (signal) {
				reject(new Error(`${command} exited via ${signal}`));
				return;
			}
			resolve(code ?? 1);
		});
	});
}

const args = process.argv.slice(2);
const stdinSource = args.includes('--stdin') ? await readStdin() : undefined;
let stdinScriptDir = null;
let stdinScriptPath = null;
let runnerArgs = args;
if (stdinSource !== undefined) {
	stdinScriptDir = await mkdtemp(join(tmpdir(), 'fnr-cli-'));
	stdinScriptPath = join(stdinScriptDir, 'stdin.js');
	await writeFile(stdinScriptPath, stdinSource, 'utf8');
	runnerArgs = args.filter((arg) => arg !== '--stdin');
	runnerArgs.push(stdinScriptPath);
}

const buildCode = await run('npm', ['run', 'wasm:build'], { stdio: ['ignore', 'inherit', 'inherit'] });
if (buildCode !== 0) process.exit(buildCode);

let runCode = 1;
try {
	runCode = await run('vite-node', ['--script', 'scripts/js_function_runner_cli.ts', ...runnerArgs], {
		stdio: ['ignore', 'inherit', 'inherit'],
	});
} finally {
	if (stdinScriptDir) await rm(stdinScriptDir, { recursive: true, force: true });
}
process.exit(runCode);
