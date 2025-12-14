import type { CallHandle } from '$lib/debug/sc62015_eval_api';

export type FunctionRunnerOutput = {
	events: any[];
	calls: CallHandle[];
	prints: any[];
	resultJson: string | null;
	error: string | null;
};

