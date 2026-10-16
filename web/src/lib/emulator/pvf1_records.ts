/** PV-F1 native Port15 record CSV. Original CP932 bytes are always retained. */
export type NativeRecords = {
	bytes: Uint8Array;
	columns: string[];
	titles: string[];
	rows: string[][];
	family: 'contacts' | 'business' | 'schedule' | 'memo' | 'tasks' | 'unknown';
	warnings: string[];
};
const LIMIT = 4 * 1024 * 1024;
// Base64, JSON escaping and pretty-printed short/empty cells expand the wire.
// Keep the original 4 MiB wire limit after decoding this bounded envelope.
export const RECORD_JSON_LIMIT = 16 * LIMIT + 65536;
const decoder = new TextDecoder('shift_jis');
const readable = (bytes: Uint8Array) => decoder.decode(bytes).replace(/\x1f/g, '\n');
export function parseNativeRecords(input: Uint8Array): NativeRecords {
	const bytes = new Uint8Array(input);
	if (bytes.length < 2 || bytes.length > LIMIT || bytes.at(-1) !== 0x1a)
		throw new Error('Native transfer is incomplete: final EOF is required');
	if (bytes.subarray(0, -1).includes(0x1a)) throw new Error('Embedded EOF in native transfer');
	const lines: Uint8Array[][] = [];
	let i = 0;
	while (i < bytes.length - 1) {
		const row: Uint8Array[] = [];
		while (true) {
			const field: number[] = [];
			if (bytes[i] === 34) {
				i++;
				let closed = false;
				while (i < bytes.length - 1) {
					const c = bytes[i++];
					if (c === 34) {
						if (bytes[i] === 34) {
							field.push(34);
							i++;
						} else {
							closed = true;
							break;
						}
					} else field.push(c);
				}
				if (!closed) throw new Error('Unclosed native CSV field');
			} else {
				while (i < bytes.length - 1 && ![44, 13, 10].includes(bytes[i])) {
					if (bytes[i] === 34) throw new Error('Quote inside unquoted native field');
					field.push(bytes[i++]);
				}
			}
			row.push(Uint8Array.from(field));
			if (row.length > 1024) throw new Error('Too many native record columns');
			if (bytes[i] === 44) {
				i++;
				continue;
			}
			if (bytes[i] !== 13 || bytes[i + 1] !== 10) throw new Error('Native record row requires CRLF');
			i += 2;
			break;
		}
		lines.push(row);
		if (lines.length > 100_000) throw new Error('Too many native record rows');
	}
	if (i !== bytes.length - 1 || lines.length < 2) throw new Error('Invalid native record extents');
	const text = lines.map((row) => row.map(readable)),
		columns = text[0],
		titles = text[1];
	if (!['i', 'I'].includes(columns[0]) || titles[0] !== 'T' || columns.length !== titles.length)
		throw new Error('Missing native field/title headers');
	if (text.slice(2).some((row) => row[0] === 'D' && row.length !== columns.length))
		throw new Error('Native data row does not match its header');
	const warnings: string[] = [];
	if (text.flat().some((s) => s.includes('\ufffd')))
		warnings.push('Some bytes have no text mapping; the native file and lossless JSON preserve them.');
	if (text.slice(2).some((row) => row[0] !== 'D'))
		warnings.push('Unknown row types are preserved in the native file and JSON.');
	const family = columns.includes('TIM1')
		? 'schedule'
		: columns.includes('ETDY')
			? 'tasks'
			: columns.includes('OFCE')
				? 'business'
				: columns.includes('NAME')
					? 'contacts'
					: columns.includes('CLSC') && columns.includes('MEM1')
						? 'memo'
						: 'unknown';
	return { bytes, columns, titles, rows: text.slice(2), family, warnings };
}
const hash = async (bytes: Uint8Array) =>
	Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes))), (b) =>
		b.toString(16).padStart(2, '0'),
	).join('');
function base64(bytes: Uint8Array) {
	let text = '';
	for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
	return btoa(text);
}
export async function recordsJson(records: NativeRecords): Promise<string> {
	return (
		JSON.stringify(
			{
				schema: 'pvf1-native-records-1',
				nativeBase64: base64(records.bytes),
				nativeSha256: await hash(records.bytes),
				readable: { family: records.family, columns: records.columns, titles: records.titles, rows: records.rows },
			},
			null,
			2,
		) + '\n'
	);
}
export async function importRecordsJson(text: string): Promise<NativeRecords> {
	if (text.length > RECORD_JSON_LIMIT) throw new Error('Record JSON exceeds size limit');
	const value = JSON.parse(text);
	if (
		value.schema !== 'pvf1-native-records-1' ||
		typeof value.nativeBase64 !== 'string' ||
		value.nativeBase64.length > Math.ceil(LIMIT / 3) * 4 ||
		!/^[A-Za-z0-9+/]*={0,2}$/.test(value.nativeBase64)
	)
		throw new Error('Invalid lossless record JSON');
	const bytes = Uint8Array.from(atob(value.nativeBase64), (c) => c.charCodeAt(0));
	if ((await hash(bytes)) !== value.nativeSha256) throw new Error('Record JSON checksum mismatch');
	const records = parseNativeRecords(bytes);
	const expected = { family: records.family, columns: records.columns, titles: records.titles, rows: records.rows };
	if (
		!value.readable ||
		Object.keys(expected).some(
			(key) => JSON.stringify(value.readable[key]) !== JSON.stringify(expected[key as keyof typeof expected]),
		)
	)
		throw new Error('Readable fields changed; import an encoded native CSV to edit records');
	return records;
}
const escape = (s: string) =>
	s
		.replace(/\\/g, '\\\\')
		.replace(/\r\n?|\n/g, '\\n')
		.replace(/;/g, '\\;')
		.replace(/,/g, '\\,');
/** RFC5545/6350: fold at 75 octets without splitting UTF-8 characters. */
function folded(lines: string[]): string {
	const utf8 = new TextEncoder();
	return (
		lines
			.map((line) => {
				let result = '',
					n = 0;
				for (const char of line) {
					const size = utf8.encode(char).length;
					if (n + size > 75) {
						result += '\r\n ';
						n = 1;
					}
					result += char;
					n += size;
				}
				return result;
			})
			.join('\r\n') + '\r\n'
	);
}
function fields(records: NativeRecords, row: string[]) {
	return (key: string) => row[records.columns.indexOf(key)] ?? '';
}
function extras(records: NativeRecords, row: string[]) {
	return records.columns
		.slice(1)
		.map((key, i) => `X-PVF1-${i + 1}-${key.toUpperCase().replace(/[^A-Z0-9-]/g, '-')}:${escape(row[i + 1] ?? '')}`);
}
export async function recordsVcard(records: NativeRecords): Promise<string> {
	if (!['contacts', 'business'].includes(records.family)) throw new Error('This transfer is not a contact book');
	const identity = await hash(records.bytes),
		lines: string[] = [];
	records.rows.forEach((row, index) => {
		if (row[0] !== 'D') return;
		const f = fields(records, row);
		lines.push(
			'BEGIN:VCARD',
			'VERSION:4.0',
			`UID:https://pv-f1.invalid/records/${identity}/${index}`,
			`FN:${escape(f('NAME') || f('OFCE'))}`,
		);
		if (f('TEL1')) lines.push(`TEL;VALUE=text:${escape(f('TEL1'))}`);
		if (f('FAX1')) lines.push(`TEL;TYPE=fax;VALUE=text:${escape(f('FAX1'))}`);
		if (f('OFCE') || f('SCTN')) lines.push(`ORG:${escape(f('OFCE'))};${escape(f('SCTN'))}`);
		if (f('PSTN')) lines.push(`TITLE:${escape(f('PSTN'))}`);
		if (f('ADR1') || f('ZIPC')) lines.push(`ADR:;;${escape(f('ADR1'))};;;${escape(f('ZIPC'))};`);
		if (f('RMRK')) lines.push(`NOTE:${escape(f('RMRK'))}`);
		lines.push(...extras(records, row), 'END:VCARD');
	});
	return folded(lines);
}
function date(value: string): { parameter: string; value: string } | null {
	const text = value.trim();
	if (!text) return null;
	if (!/^\d{8}(\d{6})?$/.test(text)) throw new Error('Native calendar date cannot be represented without guessing');
	const year = Number(text.slice(0, 4)),
		month = Number(text.slice(4, 6)),
		day = Number(text.slice(6, 8)),
		check = new Date(Date.UTC(year, month - 1, day));
	if (year < 100 || check.getUTCFullYear() !== year || check.getUTCMonth() !== month - 1 || check.getUTCDate() !== day)
		throw new Error('Invalid native calendar date');
	if (text.length === 8) return { parameter: ';VALUE=DATE', value: text };
	if (Number(text.slice(8, 10)) > 23 || Number(text.slice(10, 12)) > 59 || Number(text.slice(12, 14)) > 59)
		throw new Error('Invalid native calendar time');
	return { parameter: '', value: text.slice(0, 8) + 'T' + text.slice(8) };
}
export async function recordsCalendar(
	records: NativeRecords,
	stamp = new Date().toISOString().replace(/[-:]|\.\d{3}/g, ''),
): Promise<string> {
	if (records.family !== 'schedule') throw new Error('This transfer is not Schedule calendar data');
	if (!/^\d{8}T\d{6}Z$/.test(stamp)) throw new Error('Invalid export timestamp');
	const identity = await hash(records.bytes),
		lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//PV-F1 Emulator//Record export//EN'];
	records.rows.forEach((row, index) => {
		if (row[0] !== 'D') return;
		const f = fields(records, row);
		const start = date(f('TIM1'));
		if (!start) throw new Error('Calendar row has no start date');
		lines.push(
			'BEGIN:VEVENT',
			`UID:${identity}-${index}@pv-f1.invalid`,
			`DTSTAMP:${stamp}`,
			`DTSTART${start.parameter}:${start.value}`,
			`SUMMARY:${escape(f('MEM1'))}`,
		);
		const end = date(f('TIM2'));
		if (end) {
			if (end.parameter !== start.parameter || end.value <= start.value)
				throw new Error('Calendar end must follow its start with matching precision');
			lines.push(`DTEND${end.parameter}:${end.value}`);
		}
		lines.push(...extras(records, row), 'END:VEVENT');
	});
	lines.push('END:VCALENDAR');
	return folded(lines);
}
export function recordsMemoText(records: NativeRecords): string {
	if (records.family !== 'memo') throw new Error('This transfer is not Memo data');
	return records.rows
		.filter((row) => row[0] === 'D')
		.map((row, i) => {
			const f = fields(records, row);
			return `Memo ${i + 1}\nClassification: ${f('CLAS')}\n\n${f('MEM1')}\n`;
		})
		.join('\n');
}
