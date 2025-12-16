import type { CallHandle, EvalEvent, PrintEntry } from '$lib/debug/sc62015_eval_api';

export type FunctionRunnerOutput = {
	events: EvalEvent[];
	calls: CallHandle[];
	prints: PrintEntry[];
	resultJson: string | null;
	error: string | null;
};
