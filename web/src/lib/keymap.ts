export const KEY_TO_MATRIX_CODE: Record<string, number> = {
	// PF keys (ROM uses these for menus).
	F1: 0x56, // PF1 (col=10,row=6)
	F2: 0x55, // PF2 (col=10,row=5)

	// Cursor/navigation keys.
	ArrowUp: 0x1e, // ↑  (col=3,row=6)
	ArrowDown: 0x17, // ↓ (col=2,row=7)
	ArrowLeft: 0x27, // ◀ (col=4,row=7)
	ArrowRight: 0x26, // ▶ (col=4,row=6)
};

export function matrixCodeForKeyEvent(event: KeyboardEvent): number | null {
	const code = KEY_TO_MATRIX_CODE[event.code];
	return typeof code === 'number' ? code : null;
}
