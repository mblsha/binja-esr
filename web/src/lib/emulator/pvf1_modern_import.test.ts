import { describe, it, expect } from 'vitest';
import { encodeCp932 } from './cp932';
import { recordsVcard } from './pvf1_records';
import { nativeEditorCapacities } from './pvf1_record_limits';
import { previewModernImport, type ImportOptions } from './pvf1_modern_import';
const defaults: ImportOptions = {
	target: 'memo',
	memoCode: '１',
	memoClassification: '',
	duplicates: 'skip-identical',
	deviceTimeZone: '',
};
const bytes = (s: string) => new TextEncoder().encode(s);
const preview = (s: string, file = 'memo.txt', options: Partial<ImportOptions> = {}) =>
	previewModernImport(bytes(s), file, { ...defaults, ...options });
const card = (body: string) => `BEGIN:VCARD\r\nVERSION:4.0\r\n${body}\r\nEND:VCARD\r\n`;
const event = (body: string) =>
	`BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n${body}\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n`;
describe('reviewable modern PV-F1 imports', () => {
	it('matches capacities independently observed in native Telephone and Business Modify calls', () => {
		const row = { NAME: 'ＴＥＳＴ', NAPR: 'ＴＥＳＴ', TEL1: '１２３', OFCE: 'ＴＥＳＴ', OFPR: 'ＴＥＳＴ' };
		expect(nativeEditorCapacities(row, 'telephone1')).toEqual({
			NAME: 993,
			NAPR: 17,
			TEL1: 992,
			ZIPC: 986,
			ADR1: 986,
			RMRK: 986,
		});
		expect(nativeEditorCapacities(row, 'business')).toEqual({
			NAME: 966,
			NAPR: 17,
			TEL1: 966,
			FAX1: 966,
			OFCE: 966,
			OFPR: 17,
			SCTN: 963,
			PSTN: 963,
			ZIPC: 963,
			ADR1: 963,
			RMRK: 963,
		});
		expect(
			nativeEditorCapacities({ NAME: 'ＩＭＰＯＲＴ', TEL1: '１２３', RMRK: '日本語\nSecond line' }, 'telephone1').RMRK,
		).toBe(990);
		const p = preview(card('FN:' + 'A'.repeat(993)), 'long.vcf', { target: 'telephone1' });
		expect(p.records).toBeNull();
		expect(p.issues.some((x) => x.field === 'NAME' && x.message.includes('992-byte'))).toBe(true);
		expect(
			preview(card('FN:Name\r\nN;SORT-AS="ＴＥＳＴ":Name;;;;'), 'sort.vcf', { target: 'telephone1' }).rows[0].NAPR,
		).toBe('ＴＥＳＴ');
	});
	it('encodes Japanese, half-width Kana and exact punctuation without substitutions', () => {
		const original = '名前 ｶﾅ あいう \\ " ①';
		expect(new TextDecoder('shift_jis', { fatal: true }).decode(encodeCp932(original))).toBe(original);
		expect(() => encodeCp932('😀')).toThrow('U+1F600');
		expect(() => encodeCp932('¥')).toThrow('U+A5');
	});
	it('keeps source bytes, converts only line breaks, and refuses unsafe Memo editor lengths', () => {
		const s = bytes('あいう\r\nsecond'),
			opts = { ...defaults };
		const p = previewModernImport(s, 'one.txt', opts);
		expect(p.records?.rows[0]).toEqual(['D', ' ', '１', '', 'あいう\nsecond']);
		expect(p.records?.bytes).toContain(0x1f);
		s[0] = 0;
		opts.memoCode = '９';
		expect(p.source[0]).not.toBe(0);
		expect(p.options.memoCode).toBe('１');
		expect(preview('a'.repeat(997)).records).not.toBeNull();
		expect(
			preview('a'.repeat(998)).issues.some((x) => x.severity === 'error' && x.message.includes('998 CP932 bytes')),
		).toBe(true);
		expect(preview('あ'.repeat(499)).records).toBeNull();
		expect(preview('x\x1ay').records).toBeNull();
		expect(preview('x', 'one.txt', { memoCode: '' }).records).toBeNull();
	});
	it('unfolds, unescapes and maps vCard fields, and explicitly lists omissions', async () => {
		const p = preview(
			card(
				'FN:名前 "literal\r\nTEL;TYPE=work;PREF=1;VALUE=uri:tel:+61-123\r\nTEL;TYPE=fax;VALUE=text:456\r\nORG:会社;営\r\n 業\r\nADR:;;Street\\; one;City;;123;JP\r\nNOTE:First\\nSecond\\, third\r\nEMAIL:user@example.com',
			),
			'contact.vcf',
			{ target: 'business' },
		);
		expect(p.records).not.toBeNull();
		expect(p.rows[0]).toMatchObject({
			NAME: '名前 "literal',
			TEL1: '+61-123',
			FAX1: '456',
			OFCE: '会社',
			SCTN: '営業',
			ZIPC: '123',
			ADR1: 'Street; one\nCity\nJP',
			RMRK: 'First\nSecond, third',
		});
		expect(p.issues).toContainEqual(expect.objectContaining({ field: 'EMAIL', severity: 'warning' }));
		expect(p.issues).toContainEqual(expect.objectContaining({ field: 'TEL;TYPE', severity: 'warning' }));
		expect(p.issues).toContainEqual(expect.objectContaining({ field: 'TEL;PREF', severity: 'warning' }));
		expect(p.records?.rows[0][p.records.columns.indexOf('RMRK')]).toBe('MEM1:First\nSecond, third\x1e\x0c');
		const exported = await recordsVcard(p.records!);
		expect(exported).toContain('NOTE:First\\nSecond\\, third');
		expect(exported).not.toMatch(/[\x1e\x0c]/);
		expect(exported).toContain('RMRK-UTF8-BASE64:');
	});
	it('reports exact duplicate handling within the source file without claiming guest deduplication', () => {
		const source = card('FN:TEST') + card('FN:TEST');
		const p = preview(source, 'book.vcf', { target: 'telephone1' });
		expect(p.skippedDuplicates).toBe(1);
		expect(p.records?.rows).toHaveLength(1);
		expect(preview(source, 'book.vcf', { target: 'telephone2', duplicates: 'keep' }).records?.rows).toHaveLength(2);
	});
	it('keeps a fax-only Business contact in FAX1 and accepts standard parameter-list quoting', () => {
		const p = preview(card('FN:Fax\r\nORG:Company\r\nTEL;TYPE="work","fax";VALUE=text:１２３'), 'fax.vcf', {
			target: 'business',
		});
		expect(p.rows[0].FAX1).toBe('１２３');
		expect(p.rows[0].TEL1).toBeUndefined();
		expect(preview(card('FN:Only a person'), 'person.vcf', { target: 'business' }).records).toBeNull();
		expect(
			preview(card('FN:Fax\r\nTEL;TYPE=fax:123'), 'fax.vcf', { target: 'telephone1' }).issues.some((x) =>
				x.message.includes('no separate fax'),
			),
		).toBe(true);
		expect(preview('tab\there').records).toBeNull();
	});
	it('does not silently accept lossy encodings, repeated scalars, malformed escapes or unbalanced records', () => {
		for (const body of ['FN;ENCODING=QUOTED-PRINTABLE:One', 'FN:A\r\nFN:B', 'FN:A\\q', 'FN:A\\', 'FN:😀'])
			expect(preview(card(body), 'bad.vcf', { target: 'telephone1' }).records, body).toBeNull();
		expect(() => preview(card('FN:A').replace('END:VCARD', 'END:VEVENT'), 'bad.vcf', { target: 'telephone1' })).toThrow(
			'Unbalanced',
		);
		expect(() => previewModernImport(Uint8Array.from([0xff]), 'bad.txt', defaults)).toThrow();
	});
	it('keeps floating and all-day events local and explicitly converts fixed times to a chosen device zone', () => {
		const p = preview(event('DTSTART:20261016T003300\r\nDTEND:20261016T013300\r\nSUMMARY:TEST'), 'one.ics', {
			target: 'schedule',
		});
		expect(p.rows[0]).toMatchObject({ TIM1: '20261016003300', TIM2: '20261016013300', MEM1: 'TEST', ALON: ' ' });
		expect(
			preview(event('DTSTART;VALUE=DATE:20261016\r\nSUMMARY:Day'), 'day.ics', { target: 'schedule' }).rows[0].TIM1,
		).toBe('20261016      ');
		const fixed = event('DTSTART:20261015T140000Z\r\nSUMMARY:Midnight');
		expect(preview(fixed, 'utc.ics', { target: 'schedule' }).records).toBeNull();
		expect(preview(fixed, 'utc.ics', { target: 'schedule', deviceTimeZone: 'Australia/Sydney' }).rows[0].TIM1).toBe(
			'20261016010000',
		);
		expect(
			preview(event('DTSTART;TZID=Asia/Tokyo:20261016T090000\r\nSUMMARY:Meeting'), 'zone.ics', {
				target: 'schedule',
				deviceTimeZone: 'Australia/Sydney',
			}).rows[0].TIM1,
		).toBe('20261016110000');
	});
	it('refuses ambiguous DST times, recurrence, invalid dates and nested alarms', () => {
		for (const body of [
			'DTSTART;VALUE=DATE:20261016\r\nDTEND;VALUE=DATE:20261018\r\nSUMMARY:Range',
			'DTSTART:20260230T010000\r\nSUMMARY:Bad',
			'DTSTART;TZID=America/New_York:20261101T013000\r\nSUMMARY:Ambiguous',
			'DTSTART:20261016T010000\r\nSUMMARY:Repeat\r\nRRULE:FREQ=DAILY',
			'DTSTART:20261016T010000\r\nSUMMARY:Alarm\r\nBEGIN:VALARM\r\nTRIGGER:-PT5M\r\nEND:VALARM',
		])
			expect(preview(event(body), 'bad.ics', { target: 'schedule', deviceTimeZone: 'UTC' }).records, body).toBeNull();
	});
	it('handles case-insensitive calendar tokens and rejects oversized property inventories', () => {
		const p = preview(
			event('dtstart;value=date:20261016\r\nsummary:Day')
				.replaceAll('VCALENDAR', 'vcalendar')
				.replaceAll('VEVENT', 'vevent'),
			'one.ics',
			{ target: 'schedule' },
		);
		expect(p.rows[0].TIM1).toBe('20261016      ');
		expect(p.records).not.toBeNull();
		expect(() =>
			preview(card('FN:Test\r\n' + 'X-UNKNOWN:x\r\n'.repeat(513)), 'too-many.vcf', { target: 'telephone1' }),
		).toThrow('Too many properties');
	});
});
