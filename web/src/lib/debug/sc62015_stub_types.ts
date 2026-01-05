export type StubReturn =
	| { kind: 'ret'; pc?: number }
	| { kind: 'retf'; pc?: number }
	| { kind: 'jump'; pc: number }
	| { kind: 'stay' };

export type StubMemory = {
	read8(addr: number): number;
	read16(addr: number): number;
	read24(addr: number): number;
};

export type StubPatch = {
	mem_writes?: Array<{ addr: number; value: number; size?: 1 | 2 | 3 }> | Record<string, number>;
	regs?: Record<string, number>;
	flags?: Record<'C' | 'Z', 0 | 1>;
	ret?: StubReturn;
};

export type StubHandler = (
	mem: StubMemory,
	regs: Record<string, number>,
	flags: { C: number; Z: number },
) => StubPatch | void;

export type StubRegistration = {
	id: number;
	pc: number;
	name?: string | null;
	handler: StubHandler;
};
