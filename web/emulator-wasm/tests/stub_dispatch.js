let stubPatch = null;
let stubError = null;
let lastRegs = null;
let lastFlags = null;

export function install_stub_dispatch() {
	globalThis.__sc62015_stub_dispatch = (stubId, regs, flags) => {
		lastRegs = regs;
		lastFlags = flags;
		if (stubError) {
			throw new Error(stubError);
		}
		return stubPatch;
	};
}

export function set_stub_patch(patch) {
	stubPatch = patch;
}

export function set_stub_error(message) {
	stubError = message;
}

export function clear_stub_state() {
	stubPatch = null;
	stubError = null;
	lastRegs = null;
	lastFlags = null;
}

export function last_regs() {
	return lastRegs;
}

export function last_flags() {
	return lastFlags;
}
