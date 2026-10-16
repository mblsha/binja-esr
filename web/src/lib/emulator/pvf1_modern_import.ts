/** Modern text -> reviewable native records. The ROM owns the actual Receive/storage transaction. */
import { encodeCp932 } from './cp932';
import { parseNativeRecords, type NativeRecords } from './pvf1_records';
import { nativeEditorCapacities } from './pvf1_record_limits';

export type ImportTarget = 'telephone1' | 'telephone2' | 'business' | 'schedule' | 'memo';
export type ImportOptions = {
	target: ImportTarget;
	duplicates: 'keep' | 'skip-identical';
	memoCode: string;
	memoClassification: string;
	deviceTimeZone: string;
};
export type ImportIssue = { row: number; field: string; severity: 'error' | 'warning'; message: string };
export type ImportPreview = {
	source: Uint8Array;
	sourceName: string;
	options: ImportOptions;
	rows: Record<string, string>[];
	issues: ImportIssue[];
	skippedDuplicates: number;
	records: NativeRecords | null;
};
type Property = { name: string; params: Map<string, string>; value: string };
const SCHEMAS = {
	telephone1: ['SCRT', 'NAME', 'NAPR', 'TEL1', 'ZIPC', 'ADR1', 'BRTH', 'RMRK', 'CHK1', 'CHK2', 'CHK3', 'CHK4'],
	telephone2: ['SCRT', 'NAME', 'NAPR', 'TEL1', 'ZIPC', 'ADR1', 'BRTH', 'RMRK', 'CHK1', 'CHK2', 'CHK3', 'CHK4'],
	business: [
		'SCRT',
		'NAME',
		'NAPR',
		'TEL1',
		'FAX1',
		'OFCE',
		'OFPR',
		'SCTN',
		'PSTN',
		'ZIPC',
		'ADR1',
		'BRTH',
		'RMRK',
		'CHK1',
		'CHK2',
		'CHK3',
		'CHK4',
		'ASCT',
	],
	schedule: ['SCRT', 'TIM1', 'TIM2', 'ALRM', 'ALON', 'MEM1'],
	memo: ['SCRT', 'CLSC', 'CLAS', 'MEM1'],
} satisfies Record<ImportTarget, string[]>;
const LABELS: Record<string, string> = {
	SCRT: 'シークレット',
	NAME: '個人名',
	NAPR: '個人名読み',
	TEL1: '電話番号',
	ZIPC: '郵便番号',
	ADR1: '住所',
	BRTH: '生年月日',
	RMRK: '備考',
	CHK1: '暑中見舞い',
	CHK2: '年賀状',
	CHK3: 'お中元',
	CHK4: 'お歳暮',
	FAX1: 'ＦＡＸ番号',
	OFCE: '会社名',
	OFPR: '会社名読み',
	SCTN: '所属',
	PSTN: '役職',
	ASCT: '交際録',
	TIM1: '開始時刻',
	TIM2: '終了時刻',
	ALRM: 'アラーム時刻',
	ALON: 'アラーム設定',
	MEM1: '内容',
	CLSC: '分類コード',
	CLAS: '分類名',
};
function splitOutside(value: string, delimiter: string, parameterQuotes = true): string[] {
	const result: string[] = [];
	let start = 0,
		quoted = false;
	for (let i = 0; i < value.length; i++) {
		if (value[i] === '\\') {
			i++;
			continue;
		}
		if (parameterQuotes && value[i] === '"') quoted = !quoted;
		if (!quoted && value[i] === delimiter) {
			result.push(value.slice(start, i));
			start = i + 1;
		}
	}
	if (quoted) throw new Error('Unclosed quoted content-line parameter');
	result.push(value.slice(start));
	return result;
}
function unescape(value: string): string {
	let trailing = 0;
	for (let i = value.length - 1; value[i] === '\\'; i--) trailing++;
	if (trailing % 2) throw new Error('Incomplete text escape');
	return value.replace(/\\(.)/gs, (_, char: string) => {
		if (char === 'n' || char === 'N') return '\n';
		if ('\\,;'.includes(char)) return char;
		throw new Error(`Unsupported text escape: \\${char}`);
	});
}
function properties(text: string): Property[] {
	const unfolded = text.replace(/\r\n[ \t]/g, '').replace(/\n[ \t]/g, '');
	if (unfolded.length > 4 * 1024 * 1024) throw new Error('Modern import exceeds 4 MiB');
	const lines = unfolded.split(/\r\n|\n/).filter(Boolean);
	if (lines.length > 100_000) throw new Error('Too many modern content lines');
	return lines.map((line) => {
		if (line.includes('\r')) throw new Error('Invalid bare carriage return');
		let at = -1,
			quoted = false;
		for (let i = 0; i < line.length; i++) {
			if (line[i] === '"') quoted = !quoted;
			if (!quoted && line[i] === ':') {
				at = i;
				break;
			}
		}
		if (at < 1) throw new Error('Content line has no value separator');
		const heading = line.slice(0, at);
		const [raw, ...parameters] = splitOutside(heading, ';');
		const name = raw.split('.').at(-1)!.toUpperCase(),
			params = new Map<string, string>();
		if (!/^[A-Z0-9-]+$/.test(name)) throw new Error('Invalid content property name');
		for (const parameter of parameters) {
			const at = parameter.indexOf('=');
			if (at < 1) throw new Error('Only named content-line parameters are supported');
			const key = parameter.slice(0, at).toUpperCase(),
				value = splitOutside(parameter.slice(at + 1), ',')
					.map((v) => v.replace(/^"(.*)"$/, '$1'))
					.join(',');
			if (params.has(key)) throw new Error('Repeated content-line parameter');
			params.set(key, value);
		}
		const value = line.slice(at + 1);
		return { name, params, value: ['BEGIN', 'END'].includes(name) ? value.toUpperCase() : value };
	});
}
function components(lines: Property[], kind: 'VCARD' | 'VEVENT'): Property[][] {
	const stack: string[] = [],
		rows: Property[][] = [];
	let current: Property[] | null = null;
	for (const p of lines) {
		if (p.name === 'BEGIN') {
			const value = p.value.toUpperCase();
			if (value === kind) {
				if (current) throw new Error('Nested record component');
				current = [];
				rows.push(current);
			} else if (current) current.push(p); // Nested alarm is explicitly rejected below.
			stack.push(value);
		} else if (p.name === 'END') {
			const value = p.value.toUpperCase();
			if (stack.pop() !== value) throw new Error('Unbalanced record component');
			if (value === kind) current = null;
			else if (current) current.push(p);
		} else if (current) current.push(p);
		if (current && current.length > 512) throw new Error('Too many properties in one record');
	}
	if (stack.length || current || !rows.length) throw new Error(`Requires complete ${kind} records`);
	if (rows.length > 10_000) throw new Error('Too many modern records');
	return rows;
}
const one = (ps: Property[], name: string) => {
	const values = ps.filter((p) => p.name === name);
	if (values.length > 1) throw new Error(`Repeated ${name} is not supported`);
	return values[0];
};
const textValue = (ps: Property[], name: string) => unescape(one(ps, name)?.value ?? '');
function parts(epoch: number, zone: string): string {
	const values = new Intl.DateTimeFormat('en-US-u-ca-gregory-nu-latn', {
		timeZone: zone,
		year: 'numeric',
		month: '2-digit',
		day: '2-digit',
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit',
		hourCycle: 'h23',
	}).formatToParts(epoch);
	return ['year', 'month', 'day', 'hour', 'minute', 'second']
		.map((key) => values.find((v) => v.type === key)!.value.padStart(key === 'year' ? 4 : 2, '0'))
		.join('');
}
function checkedDate(value: string): number {
	if (!/^\d{8}(T\d{6}Z?)?$/.test(value)) throw new Error('Requires a complete date or date-time');
	const plain = value.replace('T', '').replace(/Z$/, ''),
		y = Number(plain.slice(0, 4)),
		m = Number(plain.slice(4, 6)),
		d = Number(plain.slice(6, 8));
	if (y < 100) throw new Error('Years below 0100 are unsupported');
	const epoch = Date.UTC(
		y,
		m - 1,
		d,
		Number(plain.slice(8, 10)),
		Number(plain.slice(10, 12)),
		Number(plain.slice(12, 14)),
	);
	if (parts(epoch, 'UTC') !== plain.padEnd(14, '0')) throw new Error('Invalid calendar date/time');
	return epoch;
}
function calendarDate(p: Property, zone: string): string {
	p = { ...p, value: p.value.toUpperCase() };
	const epoch = checkedDate(p.value),
		allDay = !p.value.includes('T'),
		sourceZone = p.params.get('TZID');
	if (allDay) {
		if (sourceZone || (p.params.has('VALUE') && p.params.get('VALUE')!.toUpperCase() !== 'DATE'))
			throw new Error('Invalid all-day date parameters');
		return p.value.padEnd(14, ' ');
	}
	if (p.params.has('VALUE') && p.params.get('VALUE')!.toUpperCase() !== 'DATE-TIME')
		throw new Error('Unsupported calendar value type');
	if (!sourceZone && !p.value.endsWith('Z')) return p.value.replace('T', '');
	if (!zone) throw new Error('Choose the organizer time zone to convert fixed-time events');
	if (!sourceZone) return parts(epoch, zone);
	if (p.value.endsWith('Z')) throw new Error('UTC date-time cannot also specify TZID');
	const local = p.value.replace('T', ''),
		candidates = new Set<number>();
	// Find both possible offsets around a DST transition; refuse gaps and ambiguous wall times.
	for (const delta of [-36, 0, 36]) {
		const sample = epoch + delta * 3600_000,
			offset = checkedDate(parts(sample, sourceZone).replace(/^(\d{8})(\d{6})$/, '$1T$2')) - sample;
		const candidate = epoch - offset;
		if (parts(candidate, sourceZone) === local) candidates.add(candidate);
	}
	if (candidates.size !== 1) throw new Error('Ambiguous or nonexistent zoned time; provide an explicit UTC time');
	return parts([...candidates][0], zone);
}
export function previewModernImport(source: Uint8Array, sourceName: string, options: ImportOptions): ImportPreview {
	if (!Object.hasOwn(SCHEMAS, options.target) || !['keep', 'skip-identical'].includes(options.duplicates))
		throw new Error('Invalid import target or duplicate policy');
	if (!source.length || source.length > 4 * 1024 * 1024) throw new Error('Modern import requires 1 byte to 4 MiB');
	const text = new TextDecoder('utf-8', { fatal: true }).decode(source),
		issues: ImportIssue[] = [],
		rows: Record<string, string>[] = [];
	const result: ImportPreview = {
		source: source.slice(),
		sourceName,
		options: { ...options },
		rows,
		issues,
		skippedDuplicates: 0,
		records: null,
	};
	const issue = (row: number, field: string, severity: 'error' | 'warning', message: string) =>
		issues.push({ row, field, severity, message });
	const ext = sourceName.split('.').at(-1)?.toLowerCase(),
		seen = new Set<string>();
	if (ext === 'txt') {
		if (options.target !== 'memo') throw new Error('Text files import into Memo');
		rows.push({
			SCRT: ' ',
			CLSC: options.memoCode,
			CLAS: options.memoClassification,
			MEM1: text.replace(/\r\n?|\n/g, '\n'),
		});
	} else if (ext === 'vcf' || ext === 'vcard') {
		if (!['telephone1', 'telephone2', 'business'].includes(options.target))
			throw new Error('Choose a contact book for vCard');
		for (const [index, ps] of components(properties(text), 'VCARD').entries()) {
			const row: Record<string, string> = { SCRT: ' ' };
			rows.push(row);
			try {
				if (!['3.0', '4.0'].includes(one(ps, 'VERSION')?.value ?? ''))
					throw new Error('Only UTF-8 vCard 3.0/4.0 is supported');
				for (const p of ps) {
					if (
						p.params.has('ENCODING') ||
						(p.params.has('CHARSET') && p.params.get('CHARSET')!.toUpperCase() !== 'UTF-8')
					)
						throw new Error('Encoded/non-UTF-8 vCard properties are unsupported');
				}
				row.NAME = textValue(ps, 'FN');
				if (!row.NAME) throw new Error('Contact requires a nonempty FN');
				const sort = one(ps, 'N')?.params.get('SORT-AS');
				if (sort) row.NAPR = splitOutside(sort, ',', false).map(unescape).join(' ');
				else
					issue(
						index + 1,
						'NAPR',
						'warning',
						'No phonetic reading supplied; find this contact under the organizer’s Other tab or set its reading there',
					);
				const phones = ps.filter((p) => p.name === 'TEL'),
					isFax = (p: Property) => /(^|,)fax(,|$)/i.test(p.params.get('TYPE') ?? ''),
					fax = phones.find(isFax);
				const phone = phones.find((p) => !isFax(p)) ?? (options.target === 'business' ? undefined : fax);
				const phoneText = (p: Property) =>
					p.value.toLowerCase().startsWith('tel:') ? decodeURIComponent(p.value.slice(4)) : unescape(p.value);
				if (phone) row.TEL1 = phoneText(phone);
				if (fax && options.target === 'business') row.FAX1 = phoneText(fax);
				if (phone && isFax(phone))
					issue(
						index + 1,
						'TEL',
						'warning',
						'Fax stored as the telephone number; this personal book has no separate fax field',
					);
				for (const p of phones)
					if (p !== phone && !(p === fax && options.target === 'business'))
						issue(index + 1, 'TEL', 'warning', 'Additional phone/fax omitted; retained in source file');
				const adr = one(ps, 'ADR');
				if (adr) {
					const values = splitOutside(adr.value, ';', false).map(unescape);
					if (values.length !== 7) throw new Error('ADR requires seven components');
					row.ZIPC = values[5];
					row.ADR1 = [values[0], values[1], values[2], values[3], values[4], values[6]].filter(Boolean).join('\n');
				}
				row.RMRK = textValue(ps, 'NOTE');
				if (options.target === 'business') {
					const org = one(ps, 'ORG');
					if (org) {
						const values = splitOutside(org.value, ';', false).map(unescape);
						row.OFCE = values[0];
						row.SCTN = values.slice(1).join(' / ');
						const reading = org.params.get('SORT-AS');
						if (reading) row.OFPR = splitOutside(reading, ',', false).map(unescape).join(' ');
					}
					row.PSTN = textValue(ps, 'TITLE');
					if (!row.OFCE)
						throw new Error(
							'Business Receive requires an organization name (ORG); choose a personal telephone book for individual-only contacts',
						);
					if (!row.OFPR)
						issue(
							index + 1,
							'OFPR',
							'warning',
							'No organization sort reading supplied; set it in the organizer if needed',
						);
				}
				const handled = new Set([
					'VERSION',
					'FN',
					'TEL',
					'ADR',
					'NOTE',
					...(options.target === 'business' ? ['ORG', 'TITLE'] : []),
				]);
				for (const p of ps) {
					if (!handled.has(p.name))
						issue(index + 1, p.name, 'warning', 'Property omitted from native record; original source retained');
					else {
						const mapped = p.name === 'TEL' ? ['TYPE', 'VALUE'] : p.name === 'ORG' ? ['SORT-AS'] : [];
						for (const key of p.params.keys())
							if (key !== 'CHARSET' && !mapped.includes(key))
								issue(
									index + 1,
									`${p.name};${key}`,
									'warning',
									'Parameter omitted from native field; original source retained',
								);
						if (p.name === 'TEL' && (p.params.get('TYPE') ?? '').split(',').some((v) => v && v.toLowerCase() !== 'fax'))
							issue(
								index + 1,
								'TEL;TYPE',
								'warning',
								'Only fax classification maps to a separate Business field; other phone type labels are omitted',
							);
					}
				}
			} catch (e) {
				issue(index + 1, 'contact', 'error', String(e));
			}
		}
	} else if (ext === 'ics' || ext === 'ical') {
		if (options.target !== 'schedule') throw new Error('Calendar files import into Schedule');
		const lines = properties(text);
		if (
			lines[0]?.name !== 'BEGIN' ||
			lines[0].value.toUpperCase() !== 'VCALENDAR' ||
			lines.at(-1)?.value.toUpperCase() !== 'VCALENDAR'
		)
			throw new Error('Requires a complete VCALENDAR');
		if (!lines.some((p) => p.name === 'VERSION' && p.value === '2.0')) throw new Error('Requires iCalendar 2.0');
		if (lines.some((p) => p.name === 'BEGIN' && ['VTIMEZONE', 'VTODO', 'VJOURNAL', 'VFREEBUSY'].includes(p.value)))
			issue(
				0,
				'calendar',
				'error',
				'Custom zone rules and non-event components are unsupported; provide standalone events',
			);
		for (const [index, ps] of components(lines, 'VEVENT').entries()) {
			const row: Record<string, string> = { SCRT: ' ', ALON: ' ' };
			rows.push(row);
			try {
				if (ps.some((p) => ['RRULE', 'RDATE', 'EXDATE', 'RECURRENCE-ID', 'DURATION', 'BEGIN', 'END'].includes(p.name)))
					throw new Error('Recurrence, duration and embedded alarms require explicit standalone events');
				const start = one(ps, 'DTSTART');
				if (!start) throw new Error('Event requires DTSTART');
				row.TIM1 = calendarDate(start, options.deviceTimeZone);
				const end = one(ps, 'DTEND');
				if (end) {
					if (!start.value.toUpperCase().includes('T'))
						throw new Error(
							'All-day DTEND ranges require a verified native end-date contract; import standalone days instead',
						);
					row.TIM2 = calendarDate(end, options.deviceTimeZone);
					if (row.TIM2.trim().length !== row.TIM1.trim().length || row.TIM2 <= row.TIM1)
						throw new Error('Event end must follow its start with matching date/time types');
				}
				row.MEM1 = [textValue(ps, 'SUMMARY'), textValue(ps, 'DESCRIPTION'), textValue(ps, 'LOCATION')]
					.filter(Boolean)
					.join('\n');
				if (!row.MEM1) throw new Error('Event requires some summary, description or location text');
				for (const p of ps)
					if (!['DTSTART', 'DTEND', 'SUMMARY', 'DESCRIPTION', 'LOCATION'].includes(p.name))
						issue(index + 1, p.name, 'warning', 'Property omitted from native event; original source retained');
				if (start.value.toUpperCase().endsWith('Z') || start.params.has('TZID'))
					issue(
						index + 1,
						'DTSTART',
						'warning',
						`Fixed time converted to organizer wall time in ${options.deviceTimeZone}; native record has no time-zone identity`,
					);
			} catch (e) {
				issue(index + 1, 'event', 'error', String(e));
			}
		}
	} else throw new Error('Choose a .vcf, .ics or UTF-8 .txt file');

	const columns = SCHEMAS[options.target],
		accepted: Record<string, string>[] = [];
	for (const [index, row] of rows.entries()) {
		if (options.target === 'memo') {
			if (!/^[１-９]$/.test(row.CLSC))
				issue(index + 1, 'CLSC', 'error', 'This importer supports full-width classification codes １–９');
			issue(
				index + 1,
				'CLSC',
				'warning',
				`Uses organizer classification ${row.CLSC}; existing classification membership is controlled by native Receive`,
			);
		}
		let capacities: Record<string, number> = {};
		try {
			capacities = nativeEditorCapacities(row, options.target);
		} catch {
			/* Individual encoding errors are reported below. */
		}
		for (const [field, value] of Object.entries(row)) {
			if (/[\x00-\x09\x0b-\x1f\x7f-\x9f]/.test(value)) {
				issue(index + 1, field, 'error', 'Embedded control characters are not importable');
				continue;
			}
			try {
				const n = encodeCp932(value.replace(/\r\n?|\n/g, '\x1f')).length;
				const limit = capacities[field] === undefined ? null : capacities[field] - 1;
				if (limit !== null && n > limit)
					issue(
						index + 1,
						field,
						'error',
						`${n} CP932 bytes exceeds the native editor's ${limit}-byte payload capacity; no truncation performed`,
					);
			} catch (e) {
				issue(index + 1, field, 'error', String(e));
			}
		}
		const key = JSON.stringify(columns.map((k) => row[k] ?? ''));
		if (seen.has(key)) {
			issue(
				index + 1,
				'duplicate',
				'warning',
				options.duplicates === 'skip-identical'
					? 'Identical record within this file skipped'
					: 'Identical record within this file retained',
			);
			if (options.duplicates === 'skip-identical') {
				result.skippedDuplicates++;
				continue;
			}
		}
		seen.add(key);
		accepted.push(row);
	}
	if (!accepted.length) issue(0, 'records', 'error', 'No records selected for import');
	if (!issues.some((p) => p.severity === 'error')) {
		const csv = (cells: string[]) =>
			cells.map((c) => '"' + c.replace(/\r\n?|\n/g, '\x1f').replace(/"/g, '""') + '"').join(',') + '\r\n';
		const wire =
			csv(['i', ...columns]) +
			csv(['T', ...columns.map((k) => LABELS[k])]) +
			accepted
				.map((r) => csv(['D', ...columns.map((k) => (k === 'RMRK' && r[k] ? `MEM1:${r[k]}\x1e\x0c` : (r[k] ?? '')))]))
				.join('');
		const encoded = encodeCp932(wire),
			bytes = new Uint8Array(encoded.length + 1);
		bytes.set(encoded);
		bytes[encoded.length] = 0x1a;
		result.records = parseNativeRecords(bytes);
	}
	return result;
}
