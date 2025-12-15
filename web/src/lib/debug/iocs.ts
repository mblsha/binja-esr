export const IOCS_PUBLIC_ENTRY_ADDR = 0x00fffe8;

// IOCS function numbers for the PC-E500 ROM.
// These are the values placed in `IL` before calling `CALLF 0xFFFE8`.
export enum IOCS {
	// Verified experimentally: writes one character to the LCD, with the character in `A`.
	LCD_PUTC = 0x0d
}

