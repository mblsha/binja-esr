export const IOCS_PUBLIC_ENTRY_ADDR = 0x00fffe8;

// IOCS function numbers for the PC-E500 ROM.
// These are the values placed in `IL` before calling `CALLF 0xFFFE8`.
export enum IOCS {
	// TRM: Standard character-device "write byte data" (0x000D).
	// Params: (cl)/(ch) select device/drive, A is the byte. Cursor positioning is device-defined.
	STDCHR_WRITE_BYTE = 0x0d,

	// TRM: Display level-1 "one character output to arbitrary position" (0x0041).
	// Params: (bl)=x, (bh)=y, A is the byte, (cx)=0.
	DISPLAY_PUTCHAR_XY = 0x41,

	// Back-compat alias (older scripts/tests).
	LCD_PUTC = 0x0d
}
