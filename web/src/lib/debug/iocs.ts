export const IOCS_PUBLIC_ENTRY_ADDR = 0x00fffe8;

// IOCS function numbers for the PC-E500 ROM.
//
// Note: some IOCS families are selected via `I` (16-bit) per the TRM, while
// some ROM-specific entry points appear to be selected via `IL` (8-bit). The
// Web function runner exposes both patterns.
export enum IOCS {
	// Verified experimentally: writes one character to the LCD, with the character in `A`.
	// This is ROM-specific and is invoked by setting `IL=0x0D` then calling the public IOCS entry.
	LCD_PUTC = 0x0d,

	// TRM: "One character output to arbitrary position" (IOCS I=0x0041).
	DISPLAY_PUTCHAR_XY = 0x0041,
}
