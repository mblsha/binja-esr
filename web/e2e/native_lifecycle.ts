import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import type { CDPSession } from '@playwright/test';

/** Playwright pins every page to forced focus in its primary CDP session.
 * A second CDP session cannot undo that override. This test-only adapter turns
 * off that one command at its source; Chromium generates the lifecycle events.
 * No page visibility property or DOM event is replaced. Pinned internal API
 * use is confined to this file and restored after the lifecycle test file.
 */
export function nativeLifecycle() {
	const require = createRequire(import.meta.url);
	const { CRSession } = require(
		join(dirname(require.resolve('playwright-core/package.json')), 'lib/server/chromium/crConnection.js'),
	);
	const original = CRSession.prototype.send;
	let allowForcedFocus = false;
	CRSession.prototype.send = function (method: string, params: any) {
		return original.call(
			this,
			method,
			method === 'Emulation.setFocusEmulationEnabled' && !allowForcedFocus ? { enabled: false } : params,
		);
	};
	return {
		restore() {
			CRSession.prototype.send = original;
		},
		async foreground(cdp: CDPSession) {
			allowForcedFocus = true;
			try {
				await cdp.send('Emulation.setFocusEmulationEnabled', { enabled: true });
			} finally {
				allowForcedFocus = false;
			}
		},
		async freeze(cdp: CDPSession) {
			await cdp.send('Emulation.setFocusEmulationEnabled', { enabled: false });
			await cdp.send('Page.setWebLifecycleState', { state: 'frozen' });
		},
	};
}
