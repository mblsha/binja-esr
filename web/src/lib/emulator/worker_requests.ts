/** Finite RPC lifetimes. Timing out is NOT acknowledgement of cancellation. */
export class WorkerRequests {
	private pending = new Map<
		number,
		{
			resolve: (value: any) => void;
			reject: (error: Error) => void;
			timer: ReturnType<typeof setTimeout>;
		}
	>();
	private failure: Error | null = null;

	constructor(
		private post: (message: any, transfer?: Transferable[]) => void,
		private onTimeout: (error: Error) => void,
	) {}

	get pendingCount(): number {
		return this.pending.size;
	}

	request<T>(message: { id: number; type: string }, transfer?: Transferable[], timeoutMs = 30_000): Promise<T> {
		if (this.failure) return Promise.reject(this.failure);
		if (this.pending.has(message.id)) return Promise.reject(new Error('Duplicate worker request ID'));
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(message.id);
				const error = new Error(`Worker ${message.type} request timed out; execution state is unconfirmed`);
				reject(error);
				this.onTimeout(error);
			}, timeoutMs);
			this.pending.set(message.id, { resolve, reject, timer });
			try {
				this.post(message, transfer);
			} catch (error) {
				clearTimeout(timer);
				this.pending.delete(message.id);
				reject(error instanceof Error ? error : new Error(String(error)));
			}
		});
	}

	reply(message: { id: number; ok: boolean; result?: any; error?: string }): void {
		const pending = this.pending.get(message.id);
		if (!pending) return; // Timed-out or fire-and-forget request.
		this.pending.delete(message.id);
		clearTimeout(pending.timer);
		if (message.ok) pending.resolve(message.result);
		else pending.reject(new Error(message.error ?? 'worker error'));
	}

	fail(error: Error): void {
		this.failure = error;
		for (const pending of this.pending.values()) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}
		this.pending.clear();
	}
}
