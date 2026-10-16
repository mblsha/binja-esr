/** Native Modify capacities for the modern-import subset. Includes each field's NUL.
 * Acquired ROM: F9 D1F53..D23D6, F2 D8277..D8AAE, FA D2E96/D2F32;
 * normal field reads independently checked in the record-import workflow campaign.
 * These describe the native editor, not physical storage capacity or wire limits.
 */
import { encodeCp932 } from './cp932';
export function nativeEditorCapacities(row: Record<string, string>, target: string): Record<string, number> {
	if (target === 'memo') return { CLSC: 3, CLAS: 21, MEM1: 998 };
	if (target === 'schedule') return { MEM1: 1025 };
	const business = target === 'business',
		max = business ? 966 : 993;
	const stages: [string, number, number, number][] = business
		? [
				['NAME', 24, max, 1],
				['NAPR', 23, 17, 3],
				['TEL1', 20, max, 1],
				['FAX1', 19, max, 1],
				['OFCE', 18, max, 1],
				['OFPR', 17, 17, 3],
				['SCTN', 14, max, 1],
				['PSTN', 13, max, 1],
				['ZIPC', 12, max, 1],
				['ADR1', 11, max, 1],
			]
		: [
				['NAME', 17, max, 1],
				['NAPR', 16, 17, 3],
				['TEL1', 13, max, 1],
				['ZIPC', 12, max, 1],
				['ADR1', 11, max, 1],
			];
	let used = 0;
	const capacities: Record<string, number> = {};
	for (const [key, reserve, bound, overhead] of stages) {
		capacities[key] = Math.max(0, Math.min(bound, 1025 - used - reserve));
		used += encodeCp932((row[key] ?? '').replace(/\r\n?|\n/g, '\r')).length + overhead;
	}
	// Modern imports leave birthday unset. The native birthday display occupies
	// nine characters plus NUL even in that empty case (F9 D2350..D23C1).
	used += 10;
	capacities.RMRK = Math.max(0, Math.min(max, 1025 - used));
	return capacities;
}
