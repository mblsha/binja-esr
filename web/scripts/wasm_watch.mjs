import chokidar from 'chokidar';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const webRoot = process.cwd();
const relevantBasenames = new Set(['Cargo.toml', 'Cargo.lock', 'build.rs']);

function isRelevantChange(filePath) {
	if (filePath.endsWith('.rs')) return true;
	return relevantBasenames.has(path.basename(filePath));
}

function log(message) {
	console.log(`[wasm-watch] ${message}`);
}

async function runBuild() {
	return await new Promise((resolve) => {
		const child = spawn(npmCommand, ['run', 'wasm:build:dev'], {
			cwd: webRoot,
			stdio: 'inherit'
		});
		child.on('exit', (code) => resolve(code ?? 0));
	});
}

let isBuilding = false;
let needsRebuild = false;
let debounceTimer = null;

async function buildOnce() {
	isBuilding = true;
	needsRebuild = false;

	log('building wasm...');
	const code = await runBuild();
	if (code === 0) {
		log('build finished');
	} else {
		log(`build failed (exit ${code})`);
	}

	isBuilding = false;
	if (needsRebuild) {
		log('changes detected during build; rebuilding...');
		void buildOnce();
	}
}

function requestBuild(reason) {
	if (debounceTimer) clearTimeout(debounceTimer);
	debounceTimer = setTimeout(() => {
		debounceTimer = null;
		if (isBuilding) {
			needsRebuild = true;
			return;
		}
		void buildOnce();
	}, 100);
	log(reason);
}

const emulatorWasmRoot = path.resolve(webRoot, 'emulator-wasm');
const coreRoot = path.resolve(webRoot, '..', 'sc62015', 'core');

const watcher = chokidar.watch(
	[
		path.join(emulatorWasmRoot, 'src'),
		path.join(emulatorWasmRoot, 'Cargo.toml'),
		path.join(emulatorWasmRoot, 'Cargo.lock'),
		path.join(emulatorWasmRoot, 'build.rs'),
		path.join(coreRoot, 'src'),
		path.join(coreRoot, 'Cargo.toml'),
		path.join(coreRoot, 'Cargo.lock')
	],
	{
		ignoreInitial: true,
		awaitWriteFinish: {
			stabilityThreshold: 150,
			pollInterval: 50
		}
	}
);

watcher.on('ready', () => {
	log('watching Rust sources for WASM rebuilds (Ctrl+C to stop)');
});

watcher.on('all', (event, filePath) => {
	if (!filePath || !isRelevantChange(filePath)) return;
	requestBuild(`${event} ${path.relative(webRoot, filePath)}`);
});

async function shutdown(exitCode) {
	log('stopping');
	try {
		await watcher.close();
	} finally {
		process.exit(exitCode);
	}
}

process.on('SIGINT', () => {
	void shutdown(0);
});

process.on('SIGTERM', () => {
	void shutdown(0);
});

