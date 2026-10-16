import { describe, expect, it } from 'vitest';
import {
	parseNativeRecords,
	recordsJson,
	importRecordsJson,
	recordsVcard,
	recordsCalendar,
	recordsMemoText,
} from './pvf1_records';
const wire = (s: string) => Uint8Array.from([...s].map((c) => c.charCodeAt(0)));
describe('native PV-F1 records', () => {
	it('reopens its own expanded lossless JSON for a large opaque field', async () => {
		const start = wire('"i","FUTURE"\r\n"T","Unknown"\r\n"D","');
		const end = wire('"\r\n\x1a');
		const bytes = new Uint8Array(start.length + 2 * 1024 * 1024 + end.length);
		bytes.set(start);
		bytes.set(end, bytes.length - end.length);
		const record = parseNativeRecords(bytes),
			text = await recordsJson(record);
		expect(text.length).toBeGreaterThan(12 * 1024 * 1024);
		expect((await importRecordsJson(text)).bytes).toEqual(bytes);
	});
	it('preserves original unknown bytes/columns, quoted fields, empty cells and memo paragraph separators', async () => {
		const bytes = wire(
			'"i","SCRT","CLSC","CLAS","MEM1","FUTURE"\r\n"T","Secret","Code","Class","Memo","Future"\r\n"D"," ",,"Group","A\x1fB, ""quoted""","\x82"\r\n\x1a',
		);
		const record = parseNativeRecords(bytes);
		expect(record.family).toBe('memo');
		expect(record.rows[0][4]).toBe('A\nB, "quoted"');
		expect(record.warnings).toHaveLength(1);
		expect((await importRecordsJson(await recordsJson(record))).bytes).toEqual(bytes);
		expect(recordsMemoText(record)).toContain('A\nB, "quoted"');
		const json = JSON.parse(await recordsJson(record));
		json.readable.rows[0][4] = 'edited';
		await expect(importRecordsJson(JSON.stringify(json))).rejects.toThrow('Readable fields changed');
		json.nativeSha256 = '0'.repeat(64);
		await expect(importRecordsJson(JSON.stringify(json))).rejects.toThrow('checksum');
	});
	it('exports UTF-8 contacts and floating/all-day calendar entries with escaped values and octet-safe folding', async () => {
		const contact = parseNativeRecords(
			wire(
				'"i","NAME","TEL1","ADR1"\r\n"T","Name","Phone","Address"\r\n"D","' +
					'\x82\xa0'.repeat(40) +
					'","123","A;B"\r\n\x1a',
			),
		);
		const vcard = await recordsVcard(contact);
		expect(vcard).toContain('VERSION:4.0');
		expect(vcard).toContain('TEL;VALUE=text:123');
		expect(vcard).toContain('A\\;B');
		for (const line of vcard.split('\r\n')) expect(new TextEncoder().encode(line).length).toBeLessThanOrEqual(75);
		const calendar = parseNativeRecords(
			wire(
				'"i","TIM1","TIM2","MEM1"\r\n"T","Start","End","Body"\r\n"D","20240229      ",,"Leap day"\r\n"D","20240301123000","20240301133000","A,B"\r\n\x1a',
			),
		);
		const ics = await recordsCalendar(calendar, '20261016T000000Z');
		expect(ics).toContain('DTSTART;VALUE=DATE:20240229');
		expect(ics).toContain('DTSTART:20240301T123000');
		expect(ics).toContain('DTEND:20240301T133000');
		expect(ics).toContain('SUMMARY:A\\,B');
		expect(ics).not.toContain('TZID');
		const minute = parseNativeRecords(
			wire('"i","TIM1","MEM1"\r\n"T","Start","Body"\r\n"D","199301010033  ","Native12:33AM"\r\n\x1a'),
		);
		expect(await recordsCalendar(minute, '20261016T000000Z')).toContain('DTSTART:19930101T003300');
		const bad = parseNativeRecords(wire('"i","TIM1","MEM1"\r\n"T","Date","Memo"\r\n"D","20230229","invalid"\r\n\x1a'));
		await expect(recordsCalendar(bad)).rejects.toThrow('Invalid native calendar date');
	});
	it('rejects incomplete, malformed and early-EOF transfers rather than silently truncating data', () => {
		for (const s of [
			'"i","MEM1"\r\n"T","Body"\r\n',
			'"i","MEM1"\n"T","Body"\n\x1a',
			'"i","MEM1"\r\n"T","Body"\r\n"D","open\r\n\x1a',
			'"i","MEM1"\r\n"T","Body"\r\n"D","A\x1aB"\r\n\x1a',
			'"i","MEM1"\r\n"T","Body"\r\n"D","A","extra"\r\n\x1a',
		])
			expect(() => parseNativeRecords(wire(s))).toThrow();
	});
	it.runIf(!!process.env.PVF1_RECORD_FIXTURE_DIR)(
		'reads all six original native record exports and round trips every byte',
		async () => {
			const { readFile } = await import('node:fs/promises'),
				{ join } = await import('node:path');
			for (const [name, family, count] of [
				['schedule', 'schedule', 5],
				['schedule-minute', 'schedule', 1],
				['telephone1', 'contacts', 2],
				['telephone2', 'contacts', 2],
				['business', 'business', 3],
				['memo', 'memo', 2],
				['actionlist', 'tasks', 2],
			] as const) {
				const original = new Uint8Array(await readFile(join(process.env.PVF1_RECORD_FIXTURE_DIR!, name + '.csv'))),
					record = parseNativeRecords(original);
				expect(record.family).toBe(family);
				expect(record.rows).toHaveLength(count);
				expect((await importRecordsJson(await recordsJson(record))).bytes).toEqual(original);
				if (family === 'contacts' || family === 'business') expect(await recordsVcard(record)).toContain('BEGIN:VCARD');
				if (family === 'schedule') expect(await recordsCalendar(record, '20261016T000000Z')).toContain('BEGIN:VEVENT');
				if (name === 'schedule-minute')
					expect(await recordsCalendar(record, '20261016T000000Z')).toContain('DTSTART:19930101T003300');
				if (family === 'memo') expect(recordsMemoText(record)).toContain('ＡＴＥＳＴ');
			}
		},
	);
});
