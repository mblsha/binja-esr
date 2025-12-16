import { describe, expect, it } from 'vitest';

import { runUserJs } from './run_user_js';

describe('runUserJs', () => {
	it('executes user code with Eval API and shadows globals', async () => {
		const prints: unknown[] = [];
		const api = {
			print: (...items: unknown[]) => {
				for (const item of items) prints.push(item);
			}
		};
		const Reg = { A: 'A' };
		const Flag = { Z: 'Z' };
		const IOCS = { LCD_PUTC: 0x0d, DISPLAY_PUTCHAR_XY: 0x0041 };

		const result = await runUserJs(
			`
e.print("hello");
return { hasGlobalThis: typeof globalThis !== "undefined", hasWindow: typeof window !== "undefined" };
			`,
			api as any,
			Reg,
			Flag,
			IOCS
		);

		expect(prints).toEqual(['hello']);
		expect(result).toEqual({ hasGlobalThis: false, hasWindow: false });
		expect(Object.isFrozen(Reg)).toBe(true);
		expect(Object.isFrozen(Flag)).toBe(true);
		expect(Object.isFrozen(IOCS)).toBe(true);
	});
});
