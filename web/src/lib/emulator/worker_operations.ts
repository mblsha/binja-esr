/** One mutating job owns the machine across await points. Stop is priority control. */
export class WorkerOperations {
	private active: { controller: AbortController; done: Promise<void> } | null = null;
	private stopping = 0;

	get busy(): boolean {
		return this.active !== null || this.stopping !== 0;
	}

	async run<T>(action: (signal: AbortSignal) => Promise<T>): Promise<T> {
		if (this.busy) throw new Error('Emulator is busy; pause and wait for acknowledgement first');
		const controller = new AbortController();
		let finish!: () => void;
		const operation = {
			controller,
			done: new Promise<void>((resolve) => {
				finish = resolve;
			}),
		};
		this.active = operation;
		try {
			return await action(controller.signal);
		} finally {
			this.active = null;
			finish();
		}
	}

	async stop(): Promise<void> {
		this.stopping++;
		try {
			const operation = this.active;
			operation?.controller.abort();
			await operation?.done;
		} finally {
			this.stopping--;
		}
	}
}
