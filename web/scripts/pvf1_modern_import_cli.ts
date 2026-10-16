/** Thin offline consumer of the browser's exact preview/conversion implementation. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { previewModernImport, type ImportOptions } from '../src/lib/emulator/pvf1_modern_import';
const [specPath, output] = process.argv.slice(2);
if (!specPath || !output) throw new Error('Requires spec.json and output directory');
const spec = JSON.parse(await readFile(specPath, 'utf8')) as { source: string; options: ImportOptions };
const p = previewModernImport(new Uint8Array(await readFile(spec.source)), basename(spec.source), spec.options);
await mkdir(output, { recursive: true });
await writeFile(join(output, 'source-' + basename(spec.source)), p.source);
await writeFile(
	join(output, 'preview.json'),
	JSON.stringify({ ...p, source: undefined, records: p.records ? { ...p.records, bytes: undefined } : null }, null, 2) +
		'\n',
);
if (p.records) await writeFile(join(output, 'native.csv'), p.records.bytes);
else process.exitCode = 1;
