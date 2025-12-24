## SC62015 Opcode Table (Overview)

*   `(x)`: **Internal** Memory (specific address depends on the `PRE`-prefix)
*   `[x]`: **External** Memory

| L\H   | 0          | 1         | 2                  | 3                  | 4                  | 5                  | 6                  | 7                  | 8                  | 9                  | A                  | B                  | C                  | D                  | E                  | F                  |
| :--   | :--------- | :-------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- |
| **0** | NOP        | JP (n)    |                    | PRE (BP+n)         | ADD A,n            | ADC A,n            | CMP A,n            | AND A,n            | MV A,(n)           | MV A,[r3]          | MV (n),A           | MV [r3],A          | EX (m),(n)         | MV (k),[lmn]       | MV (m),[r3]        | MV (m),[(n)]       |
| **1** | RETI       | JP r3     | PRE (BP+m)         | PRE (BP+PY)        | ADD (m),n          | ADC (m),n          | CMP (m),n          | AND (m),n          | MV IL,(n)          | MV IL,[r3]         | MV (n),IL          | MV [r3],IL         | EXW' (m),(n)       | MVW (k),[lmn]      | MVW (m),[r3]       | MVW (m),[(n)]      |
| **2** | JP mn      | JR +n     | PRE (BP+m)         | PRE (m)            | ADD A,(n)          | ADC A,(n)          | CMP [klm],n        | AND [klm],n        | MV BA,(n)          | MV BA,[r3]         | MV (n),BA          | MV [r3],BA         | EXP (m),(n)        | MVP (k),[lmn]      | MVP (m),[r3]       | MVP (m),[(n)]      |
| **3** | JPF lmn    | JR -n     | PRE (BP+m)         | PRE (PY+n)         | ADD (n),A          | ADC (n),A          | CMP (n),A          | AND (n),A          | MV I,(n)           | MV I,[r3]          | MV (n),I           | MV [r3],I          | EXL (m),(n)        | MVL (k),[lmn]      | MVL (m),[r3++]     | MVL (m),[(n)]      |
| **4** | CALL mn    | JPZ mn    | PRE (BP+PX) (BP+n) | PRE (PX+m) (BP+n)  | ADD r1,r2 r2,r2    | ADCL (m),(n)       | TEST A,n           | MV A,B             | MV X,(n)           | MV X,[r3]          | MV (n),X           | MV [r3],X          | DADL (m),(n)       | DSBL (m),(n)       | ROR A              | SHR A              |
| **5** | CALLF lmn  | JPNZ mn   | PRE (BP+PX) (BP+PY)| PRE (PX+m) (BP+PY) | ADD r3,r           | ADCL (m),A         | TEST (m),n         | MV B,A             | MV Y,(n)           | MV Y,[r3]          | MV (n),Y           | MV [r3],Y          | DADL (n),A         | DSBL (n),A         | ROR (n)            | SHR (m)            |
| **6** | RET        | JPC mn    | PRE (BP+PX) (n)    | PRE (PX+m) (n)     | ADD r1,r1          | MVL (m),[X+n]      | TEST [klm],n       | AND (m),(n)        | MV U,(n)           | MV U,[r3]          | MV (n),U           | MV [r3],U          | CMPW (m),(n)       | CMPW (m),r2        | ROL A              | SHL A              |
| **7** | RETF       | JPNC mn   | PRE (BP+PX) (PY+n) | PRE (PX+m) (PY+n)  | PMDF (m),n         | PMDF (m),A         | TEST (n),A         | AND A,(n)          | MV S,(n)           | SC (n)             | MV (n),S           | CMP (m),(n)        | CMPP (m),(n)       | CMPP (m),r3        | ROL (n)            | SHL (n)            |
| **8** | MV A,n     | JRZ +n    | PUSHU A            | POPU A             | SUB A,n            | SBC A,n            | XOR A,n            | OR A,n             | MV A,[lmn]         | MV A,[(n)]         | MV [lmn],A         | MV [(n)],A         | MV (m),(n)         | MV [lmn],(n)       | MV [r3],(n)        | MV [(n)],(n)       |
| **9** | MV IL,n    | JRZ -n    | PUSHU IL           | POPU IL            | SUB (m),n          | SBC (m),n          | XOR (m),n          | OR (m),n           | MV IL,[lmn]        | MV IL,[(n)]        | MV [lmn],IL        | MV [(n)],IL        | MVW (m),(n)        | MVW [lmn],(n)      | MVW [r3],(n)       | MVW [(n)],(n)      |
| **A** | MV BA,mn   | JRNZ +n   | PUSHU BA           | POPU BA            | SUB A,(n)          | SBC A,(n)          | XOR [klm],n        | OR [klm],n         | MV BA,[lmn]        | MV BA,[(n)]        | MV [lmn],BA        | MV [(n)],BA        | MVP (m),(n)        | MVP [lmn],(n)      | MVW [r3],(n)       | MVP [(n)],(n)      |
| **B** | MV I,mn    | JRNZ -n   | PUSHU I            | POPU I             | SUB (n),A          | SBC (n),A          | XOR (n),A          | OR (n),A           | MV I,[lmn]         | MV I,[(n)]         | MV [lmn],I         | MV [(n)],I         | MVL (m),(n)        | MVL [lmn],(n)      | MVL [r3++],(n)     | MVL [(n)],(n)      |
| **C** | MV X,lmn   | JRC +n    | PUSHU X            | POPU X             | SUB r2,r2          | SBCL (m),(n)       | INC r              | DEC r              | MV X,[lmn]         | MV X,[(n)]         | MV [lmn],X         | MV [(n)],X         | MV (m),n           | MVP (n),[lmn]      | DSLL (n)           | DSRL (n)           |
| **D** | MV Y,lmn   | JRC -n    | PUSHU Y            | POPU Y             | SUB r3,r           | SBCL (n),A         | INC (n)            | DEC (n)            | MV Y,[lmn]         | MV Y,[(n)]         | MV [lmn],Y         | MV [(n)],Y         | MVW (l),mn         | EX A,B             | EX r2,r2 r3,r3     | MV r2,r2 r3,r3     |
| **E** | MV U,lmn   | JRNC +n   | PUSHU F            | POPU F             | SUB r1,r1          | MVL [X+n],(m)      | XOR (m),(n)        | OR (m),(n)         | MV U,[lmn]         | MV U,[(m)]         | MV [lmn],U         | MV [(n)],U         | TCL                | HALT               | SWAP A             | IR                 |
| **F** | MV S,lmn   | JRNC -n   | PUSHU IMR          | POPU IMR           | PUSHS F            | POPS F             | XOR A,(n)          | OR A,(n)           | MV S,[lmn]         | RC                 | MV [lmn],S         |                    | MVLD (m),(n)       | OFF                | WAIT               | RESET              |

---

## CPU Registers

An overview of the SC62015 CPU registers:

| Register | Size (bits) | Description                               | Notes                                                                 |
| :------- | :---------- | :---------------------------------------- | :-------------------------------------------------------------------- |
| `A`      | 8           | Accumulator                               | Low byte of `BA`.                                                     |
| `B`      | 8           | General Purpose / BA High Byte            | High byte of `BA`.                                                    |
| **`BA`** | **16**      | **16-bit Register (B:A)**                 | Composite: `B` (MSB), `A` (LSB).                                      |
| `IL`     | 8           | Index Register Low                        | Low byte of `I`.                                                      |
| `IH`     | 8           | Index Register High                       | High byte of `I`.                                                     |
| **`I`**  | **16**      | **Index Register (IH:IL)**                | Composite: `IH` (MSB), `IL` (LSB).                                    |
| **`X`**  | **24**      | **Index Register X**                      | General purpose 24-bit register, often used for addressing.           |
| **`Y`**  | **24**      | **Index Register Y**                      | General purpose 24-bit register, often used for addressing.           |
| **`U`**  | **24**      | **User Stack Pointer**                    | Points to the top of the user stack in external RAM.                  |
| **`S`**  | **24**      | **System Stack Pointer**                  | Points to the top of the system stack in external RAM. Used for `CALL`/`RET` and interrupts. |
| **`PC`** | **20**      | **Program Counter**                       | Holds the 20-bit address of the next instruction. Stored in 3 bytes.  |
| **`F`**  | **8**       | **Flags Register**                        | Contains status flags reflecting results of arithmetic/logic operations. |
| ↳ `C`    | 1           | Carry Flag                                | Part of `F` (bit 0). Set on arithmetic carry-out or borrow-in.        |
| ↳ `Z`    | 1           | Zero Flag                                 | Part of `F` (bit 1). Set if an operation result is zero.              |

---

## Register Encoding Table

This table shows the mapping between register names, their size category (`r₁` to `r₄`), and the 3-bit binary code used to represent them in instruction opcodes.

| Size Group | Register | Opcode Value | Binary  |
| :--------- | :------- | :----------- | :------ |
| r₁         | A        | 0            | (000)   |
| r₁         | IL       | 1            | (001)   |
| r₂         | BA       | 2            | (010)   |
| r₂         | I        | 3            | (011)   |
| r₃         | S        | 7            | (111)   |
| r₄         | X        | 4            | (100)   |
| r₄         | Y        | 5            | (101)   |
| r₄         | U        | 6            | (110)   |

## Internal RAM Addressing Prefix Byte Table

This table defines the hexadecimal value of the `PRE` (Prefix) byte required for certain complex internal RAM addressing modes, based on the combination of addressing calculations used for the first and second address components.

| 1st op \ 2nd op | `(n)` | `(BP+n)` | `(PY+n)` | `(BP+PY)` |
| :-------------- | :---- | :------- | :------- | :-------- |
| **(n)**         | 32H   | 30H      | 33H      | 31H       |
| **(BP+n)**      | 22H   |          | 23H      | 21H       |
| **(PX+n)**      | 36H   | 34H      | 37H      | 35H       |
| **(BP+PX)**     | 26H   | 24H      | 27H      | 25H       |

*   Rows indicate the addressing mode calculation specified by the *first* operand byte (e.g., in `MV (m) (n)`, this corresponds to `(m)`).
*   Columns indicate the addressing mode calculation specified by the *second* operand byte (e.g., in `MV (m) (n)`, this corresponds to `(n)`).

---

### System Initialization and Data Retainment for HALT/OFF/RESET

| | HALT | OFF | RESET |
| :--- | :--- | :--- | :--- |
| **Registers** | All retained. | All retained. | The PC read the reset vector.<br>Others than the PC are all retained. |
| **Flag (C/Z)** | Undefined. | Undefined. | Retained. |
| **Internal memory** | USR (F8H) bits 0 to 2/5 are reset (to "0").<br>SSR (FFH) bit 2 and USR (F8H) bits 3 and 4 are set (to "1").<br>Others than the above are all retained. | USR (F8H) bits 0 to 2/5 are reset (to "0").<br>SSR (FFH) bit 2 and USR (F8H) bits 3 and 4 are set (to "1").<br>Others than the above are all retained. | ACM (FEH) bit 7, UCR (F7H), USR (F8H) bits 0 to 2/5, IMR (FCH), and SCR (FDH), are all reset (to "0").<br>SSR (FFH) bit 2 and USR (F8H) bits 3 and 4 are set (to "1").<br>Others than the above are all retained. |

### Internal Memory Map (IMEMRegisters, defined in opcodes.py)

| Name | Address (Hex) | Description |
| :--- | :--- | :--- |
| **BP** | 0xEC | RAM Base Pointer |
| **PX** | 0xED | RAM PX Pointer |
| **PY** | 0xEE | RAM PY Pointer |
| **AMC** | 0xEF | **ADR Modify Control**<br>Allows two discontinuous RAM card address areas (CE1 and CE0) to be virtually joined into one contiguous block.<br>**Bitfields:**<br>• `AME` (bit 7): 1 to enable address‐modify.<br>• `AM5–AM0` (bits 6–1): CE0 RAM size code (e.g., 000000 = 2 KB, 111111 = 128 KB). |
| **KOL** | 0xF0 | Key Output Buffer H (Controls KO0-KO7). |
| **KOH** | 0xF1 | Key Output Buffer L (Controls KO8-KO15). |
| **KIL** | 0xF2 | Key Input Buffer (Reads KI0-KI7). |
| **EOL** | 0xF3 | E Port Output Buffer H (Controls E0-E7). |
| **EOH** | 0xF4 | E Port Output Buffer L (Controls E8-E15). |
| **EIL** | 0xF5 | E Port Input Buffer H (Reads E0-E7). |
| **EIH** | 0xF6 | E Port Input Buffer L (Reads E8-E15). |
| **UCR** | 0xF7 | **UART Control Register**<br><pre>  7     6     5     4     3     2     1     0<br>+-----+-----+-----+-----+-----+-----+-----+-----+<br>| BOE | BR2 | BR1 | BR0 | PA1 | PA0 |  DL |  ST |<br>+-----+-----+-----+-----+-----+-----+-----+-----+</pre>• `BOE` (bit 7): Break Output Enable.<br>• `BR2–BR0` (bits 6–4): Baud Rate (300 to 19200 bps).<br>• `PA1–PA0` (bits 3–2): Parity Select (EVEN, ODD, NONE).<br>• `DL` (bit 1): Character Length (7 or 8 bits).<br>• `ST` (bit 0): Stop Bits (1 or 2). |
| **USR** | 0xF8 | **UART Status Register**<br><pre>  7     6     5     4     3     2     1     0<br>+-----+-----+-----+-----+-----+-----+-----+-----+<br>|     |     | RXR | TXE | TXR |  FE |  OE |  PE |<br>+-----+-----+-----+-----+-----+-----+-----+-----+</pre>• `RXR` (bit 5): Receiver Ready.<br>• `TXE` (bit 4): Transmitter Empty.<br>• `TXR` (bit 3): Transmitter Ready.<br>• `FE` (bit 2): Framing Error.<br>• `OE` (bit 1): Overrun Error.<br>• `PE` (bit 0): Parity Error. |
| **RXD** | 0xF9 | UART Receive Buffer. |
| **TXD** | 0xFA | UART Transmit Buffer. |
| **IMR** | 0xFB | **Interrupt Mask Register**<br><pre>  7     6      5       4       3      2     1     0<br>+-----+-----+------+-------+------+-----+-----+-----+<br>| IRM | EXM | RXRM | TXRM  | ONKM | KEYM| STM | MTM |<br>+-----+-----+------+-------+------+-----+-----+-----+</pre>• `IRM` (bit 7): Global interrupt mask.<br>• Individual bits (6-0) mask specific interrupts (External, UART, Key, Timer). |
| **ISR** | 0xFC | **Interrupt Status Register**<br><pre>  7    6     5      4      3       2     1     0<br>+----+-----+-----+------+-------+-----+-----+-----+<br>|    | EXI | RXRI| TXRI | ONKI  | KEYI| STI | MTI |<br>+----+-----+-----+------+-------+-----+-----+-----+</pre>• Individual bits (6-0) are set to '1' to indicate a specific interrupt has occurred. |
| **SCR** | 0xFD | **System Control Register**<br><pre>  7    6    5    4     3    2    1     0<br>+----+----+----+----+-----+----+----+-----+<br>| ISE| BZ2| BZ1| BZ0| VDDC| STS| MTS| DISC|<br>+----+----+----+----+-----+----+----+-----+</pre>• `ISE` (bit 7): IRQ Start Enable.<br>• `BZ2–BZ0` (bits 6–4): CO/CI pin control.<br>• `VDDC` (bit 3): VDD Control.<br>• `STS` (bit 2): SEC Timer Select.<br>• `MTS` (bit 1): MSEC Timer Select.<br>• `DISC` (bit 0): LCD Driver Control. |
| **LCC** | 0xFE | **LCD Contrast Control**<br><pre>  7    6    5    4    3     2     1      0<br>+----+----+----+----+----+----+-----+------+<br>|LCC4|LCC3|LCC2|LCC1|LCC0| KSD| STCL| MTCL |<br>+----+----+----+----+----+----+-----+------+</pre>• `LCC4–LCC0` (bits 7–3): Contrast level (0–31).<br>• `KSD` (bit 2): Key Strobe Disable.<br>• `STCL` (bit 1): SEC Timer Clear enable.<br>• `MTCL` (bit 0): MSEC Timer Clear enable. |
| **SSR** | 0xFF | **System Status Control** (renamed from source for clarity)<br><pre>  7    6    5    4     3    2    1      0<br>+----+----+----+----+----+----+----+------<br>|    |    |    |    | ONK| RSF| CI | TEST |<br>+----+----+----+----+----+----+----+------</pre>• `ONK` (bit 3): ON-Key input status.<br>• `RSF` (bit 2): Reset-Start Flag.<br>• `CI` (bit 1): CMT Input status.<br>• `TEST` (bit 0): Test Input status. |

### Logic Registers (FCS/IOCS scratch area)

The PC-E500 TRM describes a set of "logic registers" that exist at fixed internal RAM
addresses. ROM code (and higher-level services like FCS/IOCS) use this block as a
parameter/return scratch area when there aren't enough CPU registers.

In this emulator, these addresses are also named in `IMEMRegisters` so disassembly and
assembly can use `BL`/`BH`/`CL`/`CH`/`DL`/`DH`/`SI`/`DI` instead of raw hex.

| Name | Address (Hex) | Size | Meaning |
| :--- | :--- | :--- | :--- |
| **BL** | 0xD4 | 1 | `(bl)` – low byte of `(bx)` |
| **BH** | 0xD5 | 1 | `(bh)` – high byte of `(bx)` |
| **CL** | 0xD6 | 1 | `(cl)` – low byte of `(cx)` |
| **CH** | 0xD7 | 1 | `(ch)` – high byte of `(cx)` |
| **DL** | 0xD8 | 1 | `(dl)` – low byte of `(dx)` |
| **DH** | 0xD9 | 1 | `(dh)` – high byte of `(dx)` |
| **SI** | 0xDA | 3 | `(si)` – 24-bit pointer: `[SI]`, `[SI+1]`, `[SI+2]` |
| **DI** | 0xDD | 3 | `(di)` – 24-bit pointer: `[DI]`, `[DI+1]`, `[DI+2]` |

Common composites (little-endian layout):

- `(bx)` is the 16-bit value at `BH:BL` (`0xD5:0xD4`)
- `(cx)` is the 16-bit value at `CH:CL` (`0xD7:0xD6`)
- `(dx)` is the 16-bit value at `DH:DL` (`0xD9:0xD8`)
- IOCS workspace base pointer (PC-E500 ROM convention): `IOCS_WS` at `0xE6..0xE8`

---

## Opcode Information Details

**Notes:**

*   `r₁`, `r₂`, `r₃`, `r₄`: Represent registers of different sizes. `r` in opcode patterns is a placeholder.
*   `rᵢₗ`, `rᵢₕ`, `rᵢₓ`: Subscripts denote Low, High, and Extension bytes of register `rᵢ`.
*   `n`, `m`, `l`, `k`: Represent 8-bit immediate values or address bytes. `mn`, `lmn` form 16/20/24-bit values/addresses.
*   `(addr)`: **Internal** Memory (Direct address `addr`, or PRE-modified).
*   `[addr]`: **External** Memory (Direct address `addr`).
*   `[r'₃]`: **External** Memory (Register Indirect: address is in `r'₃` (X,Y,U,S)).
*   `[r'₃++]` / `[--r'₃]`: **External** Memory (Register Indirect w/ post-increment / pre-decrement).
*   `[r'₃±n]`: **External** Memory (Register Indirect w/ 8-bit signed offset `n`).
*   `[(addr)]`: **External** Memory (Memory Indirect: address is in internal memory location `addr`).
*   `[(m)±n]`: **External** Memory (Memory Indexed: base address in internal `(m)`, offset `n`).
*   Flags: `C` (Carry), `Z` (Zero). `○` indicates affected, `-` indicates not affected.
*   Cycle counts like `A:X/IL:Y` mean X cycles if Accumulator A is operand, Y if Index Low IL is operand. `X+Y×I` means X base cycles + Y cycles per iteration of loop counter I.

### Memory Transfer Instructions (MV, MVW, MVP, MVL, MVLD)

| Mnemonic             | Function (`←` denotes assignment)                                                                            | Flags (C Z) | Bytes | Cycles       | Opcode (Bin / Hex)                                    | Operand Type |
| :------------------- | :----------------------------------------------------------------------------------------------------------- | :---------- | :---- | :----------- | :---------------------------------------------------- | :----------- |
| **MV r, immediate**  |                                                                                                              |             |       |              |                                                       |              |
| `MV r₁, n`           | `r₁ ← n` (if `r₁=IL` then `IH←0`)                                                                            | `- -`       | 2     | A:2/IL:3     | `0000 1 r` / `0X` <br> `n`                            | `n`          |
| `MV r₂, mn`          | `r₂ₗ ← n`, `r₂ₕ ← m`                                                                                         | `- -`       | 3     | 3            | `0000 1 r` / `0X` <br> `n` <br> `m`                   | `n,m`       |
| `MV r₃, lmn`         | `r₃ₗ ← n`, `r₃ₕ ← m`, `r₃ₓ ← l` (24-bit for X,Y,U,S)                                                         | `- -`       | 4     | 4            | `0000 1 r` / `0X` <br> `n` <br> `m` <br> `l`          | `n,m,l`    |
| **MV r, (n) (From Internal Memory)** |                                                                                              |             |       |              |                                                       |              |
| `MV r₁, (n)`         | `r₁ ← (n)` (if `r₁=IL` then `IH←0`)                                                                          | `- -`       | 2     | A:3/IL:4     | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| `MV r₂, (n)`         | `r₂ₗ ← (n)`, `r₂ₕ ← (n+1)`                                                                                   | `- -`       | 2     | 4            | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| `MV r₃, (n)`         | `r₃ₗ ← (n)`, `r₃ₕ ← (n+1)`, `r₃ₓ ← (n+2)`                                                                    | `- -`       | 2     | 5            | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| **MV r, [lmn] (From External Memory Direct)** |                                                                                      |             |       |              |                                                       |              |
| `MV r₁, [lmn]`       | `r₁ ← [lmn]` (if `r₁=IL` then `IH←0`)                                                                        | `- -`       | 4     | 6            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n,m,l`    |
| `MV r₂, [lmn]`       | `r₂ₗ ← [lmn]`, `r₂ₕ ← [lmn+1]`                                                                               | `- -`       | 4     | 7            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n,m,l`    |
| `MV r₃, [lmn]`       | `r₃ₗ ← [lmn]`, `r₃ₕ ← [lmn+1]`, `r₃ₓ ← [lmn+2]`                                                              | `- -`       | 4     | 8            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n,m,l`    |
| **MV r, [r'₃] (From Ext. Mem Reg Indirect)** | `r'₃` is one of X,Y,U,S                                                              |             |       |              |                                                       |              |
| `MV r₁, [r'₃]`       | `r₁ ← [r'₃]` (if `r₁=IL` then `IH←0`)                                                                        | `- -`       | 2     | A:4/IL:5     | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| `MV r₂, [r'₃]`       | `r₂ₗ ← [r'₃]`, `r₂ₕ ← [r'₃+1]`                                                                               | `- -`       | 2     | 5            | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| `MV r₄, [r'₃]`       | `r₄ₗ ← [r'₃]`, `r₄ₕ ← [r'₃+1]`, `r₄ₓ ← [r'₃+2]` (r₄ is X,Y,U)                                                | `- -`       | 2     | 6            | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| **MV r, [r'₃++] (From Ext. Mem Reg Ind. Post-Inc)** |                                                                                |             |       |              |                                                       |              |
| `MV r₁, [r'₃++]`     | `r₁ ← [r'₃]`, `r'₃ ← r'₃+1` (if `r₁=IL` then `IH←0`)                                                         | `- -`       | 2     | A:4/IL:5     | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| `MV r₂, [r'₃++]`     | `r₂ₗ ← [r'₃]`, `r₂ₕ ← [r'₃+1]`, `r'₃ ← r'₃+2`                                                                | `- -`       | 2     | 5            | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| `MV r₄, [r'₃++]`     | `r₄ₗ ← [r'₃]`, `r₄ₕ ← [r'₃+1]`, `r₄ₓ ← [r'₃+2]`, `r'₃ ← r'₃+3`                                               | `- -`       | 2     | 7            | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| **MV r, [--r'₃] (From Ext. Mem Reg Ind. Pre-Dec)** |                                                                                |             |       |              |                                                       |              |
| `MV r₁, [--r'₃]`     | `r'₃ ← r'₃-1`, `r₁ ← [r'₃]` (if `r₁=IL` then `IH←0`)                                                         | `- -`       | 2     | A:5/IL:6     | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| `MV r₂, [--r'₃]`     | `r'₃ ← r'₃-2`, `r₂ₗ ← [r'₃-2]`, `r₂ₕ ← [r'₃-1]`                                                              | `- -`       | 2     | 6            | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| `MV r₄, [--r'₃]`     | `r'₃ ← r'₃-3`, `r₄ₗ ← [r'₃-3]`, `r₄ₕ ← [r'₃-2]`, `r₄ₓ ← [r'₃-1]`                                               | `- -`       | 2     | 8            | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| **MV r, [r'₃±n] (From Ext. Mem Reg Ind. Offset)** | `±` encoded in opcode byte 2                                                     |             |       |              |                                                       |              |
| `MV r₁, [r'₃±n]`     | `r₁ ← [r'₃±n]` (if `r₁=IL` then `IH←0`)                                                                      | `- -`       | 3     | A:6/IL:7     | `1001 0 r` / `9X` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV r₂, [r'₃±n]`     | `r₂ₗ ← [r'₃±n]`, `r₂ₕ ← [r'₃±n+1]`                                                                           | `- -`       | 3     | 7            | `1001 0 r` / `9X` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV r₄, [r'₃±n]`     | `r₄ₗ ← [r'₃±n]`, `r₄ₕ ← [r'₃±n+1]`, `r₄ₓ ← [r'₃±n+2]`                                                        | `- -`       | 3     | 8            | `1001 0 r` / `9X` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| **MV r, [(n)] (From Ext. Mem via Int. Mem Ind.)** |                                                                                |             |       |              |                                                       |              |
| `MV r₁, [(n)]`       | `r₁ ← [[(n)]]` (if `r₁=IL` then `IH←0`)                                                                      | `- -`       | 3     | A:9/IL:10    | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV r₂, [(n)]`       | `r₂ₗ ← [[(n)]]`, `r₂ₕ ← [[(n)]+1]`                                                                            | `- -`       | 3     | 10           | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV r₄, [(n)]`       | `r₄ₗ ← [[(n)]]`, `r₄ₕ ← [[(n)]+1]`, `r₄ₓ ← [[(n)]+2]`                                                        | `- -`       | 3     | 11           | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| **MV r, [(m)±n] (From Ext. Mem via Int. Mem Indexed)** | `±` encoded in opcode byte 2                                                 |             |       |              |                                                       |              |
| `MV r₁, [(m)±n]`     | `r₁ ← [[(m)]±n]` (if `r₁=IL` then `IH←0`)                                                                    | `- -`       | 4     | A:11/IL:12   | `1001 1 r` / `9X` <br> `1s00 0000` / `80/C0` <br> `m` <br> `n` | `m,n`      |
| `MV r₂, [(m)±n]`     | `r₂ₗ ← [[(m)]±n]`, `r₂ₕ ← [[(m)]±n+1]`                                                                        | `- -`       | 4     | 12           | `1001 1 r` / `9X` <br> `1s00 0000` / `80/C0` <br> `m` <br> `n` | `m,n`      |
| `MV r₄, [(m)±n]`     | `r₄ₗ ← [[(m)]±n]`, `r₄ₕ ← [[(m)]±n+1]`, `r₄ₓ ← [[(m)]±n+2]`                                                     | `- -`       | 4     | 13           | `1001 1 r` / `9X` <br> `1s00 0000` / `80/C0` <br> `m` <br> `n` | `m,n`      |
| **MV (n), r (To Internal Memory)** |                                                                                                |             |       |              |                                                       |              |
| `MV (n), r₁`         | `(n) ← r₁`                                                                                                   | `- -`       | 2     | 3            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| `MV (n), r₂`         | `(n) ← r₂ₗ`, `(n+1) ← r₂ₕ`                                                                                   | `- -`       | 2     | 4            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| `MV (n), r₃`         | `(n) ← r₃ₗ`, `(n+1) ← r₃ₕ`, `(n+2) ← r₃ₓ`                                                                    | `- -`       | 2     | 5            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| **MV [lmn], r (To External Memory Direct)** |                                                                                        |             |       |              |                                                       |              |
| `MV [lmn], r₁`       | `[lmn] ← r₁`                                                                                                 | `- -`       | 4     | 5            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n,m,l`      |
| `MV [lmn], r₂`       | `[lmn] ← r₂ₗ`, `[lmn+1] ← r₂ₕ`                                                                               | `- -`       | 4     | 6            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n,m,l`      |
| `MV [lmn], r₃`       | `[lmn] ← r₃ₗ`, `[lmn+1] ← r₃ₕ`, `[lmn+2] ← r₃ₓ`                                                              | `- -`       | 4     | 7            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n,m,l`      |
| **MV [r'₃], r (To Ext. Mem Reg Indirect)** | `r'₃` is one of X,Y,U,S                                                                |             |       |              |                                                       |              |
| `MV [r'₃], r₁`       | `[r'₃] ← r₁`                                                                                                 | `- -`       | 2     | 4            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| `MV [r'₃], r₂`       | `[r'₃] ← r₂ₗ`, `[r'₃+1] ← r₂ₕ`                                                                               | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| `MV [r'₃], r₄`       | `[r'₃] ← r₄ₗ`, `[r'₃+1] ← r₄ₕ`, `[r'₃+2] ← r₄ₓ` (r₄ is X,Y,U)                                                | `- -`       | 2     | 6            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| **MV [r'₃++], r (To Ext. Mem Reg Ind. Post-Inc)** |                                                                                  |             |       |              |                                                       |              |
| `MV [r'₃++], r₁`     | `[r'₃] ← r₁`, `r'₃ ← r'₃+1`                                                                                  | `- -`       | 2     | 4            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| `MV [r'₃++], r₂`     | `[r'₃] ← r₂ₗ`, `[r'₃+1] ← r₂ₕ`, `r'₃ ← r'₃+2`                                                                | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| `MV [r'₃++], r₄`     | `[r'₃] ← r₄ₗ`, `[r'₃+1] ← r₄ₕ`, `[r'₃+2] ← r₄ₓ`, `r'₃ ← r'₃+3`                                               | `- -`       | 2     | 7            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| **MV [--r'₃], r (To Ext. Mem Reg Ind. Pre-Dec)** |                                                                                  |             |       |              |                                                       |              |
| `MV [--r'₃], r₁`     | `r'₃ ← r'₃-1`, `[r'₃] ← r₁`                                                                                  | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| `MV [--r'₃], r₂`     | `r'₃ ← r'₃-2`, `[r'₃-2] ← r₂ₗ`, `[r'₃-1] ← r₂ₕ`                                                              | `- -`       | 2     | 6            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| `MV [--r'₃], r₄`     | `r'₃ ← r'₃-3`, `[r'₃-3] ← r₄ₗ`, `[r'₃-2] ← r₄ₕ`, `[r'₃-1] ← r₄ₓ`                                               | `- -`       | 2     | 8            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| **MV [r'₃±n], r (To Ext. Mem Reg Ind. Offset)** | `±` encoded in opcode byte 2                                                       |             |       |              |                                                       |              |
| `MV [r'₃±n], r₁`     | `[r'₃±n] ← r₁`                                                                                               | `- -`       | 3     | 6            | `1011 0 r` / `BX` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV [r'₃±n], r₂`     | `[r'₃±n] ← r₂ₗ`, `[r'₃±n+1] ← r₂ₕ`                                                                           | `- -`       | 3     | 7            | `1011 0 r` / `BX` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV [r'₃±n], r₄`     | `[r'₃±n] ← r₄ₗ`, `[r'₃±n+1] ← r₄ₕ`, `[r'₃±n+2] ← r₄ₓ`                                                        | `- -`       | 3     | 8            | `1011 0 r` / `BX` <br> `1s00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| **MV [(n)], r (To Ext. Mem via Int. Mem Ind.)** |                                                                                    |             |       |              |                                                       |              |
| `MV [(n)], r₁`       | `[[(n)]] ← r₁`                                                                                               | `- -`       | 3     | 9            | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV [(n)], r₂`       | `[[(n)]] ← r₂ₗ`, `[[(n)]+1] ← r₂ₕ`                                                                            | `- -`       | 3     | 10           | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV [(n)], r₄`       | `[[(n)]] ← r₄ₗ`, `[[(n)]+1] ← r₄ₕ`, `[[(n)]+2] ← r₄ₓ`                                                        | `- -`       | 3     | 11           | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| **MV (memory), (memory) (Various Addressing)** |                                                                                    |             |       |              |                                                       |              |
| `MV ((l)±m),(n)`     | `[((l))±m] ← (n)`                                                                                            | `- -`       | 5     | 13           | `1111 0000` / `F0` <br> `1s00 0000` / `80/C0` <br> `l` <br> `m` <br> `n` | `l,m,n` |
| `MVW ((l)±m),(n)`    | `[((l))±m] ← (n)`, `[((l))±m+1] ← (n+1)`                                                                    | `- -`       | 5     | 14           | `1111 0001` / `F1` <br> `1s00 0000` / `80/C0` <br> `l` <br> `m` <br> `n` | `l,m,n` |
| `MVP ((l)±m),(n)`    | `[((l))±m..+2] ← (n..+2)`                                                                                    | `- -`       | 5     | 15           | `1111 0010` / `F2` <br> `1s00 0000` / `80/C0` <br> `l` <br> `m` <br> `n` | `l,m,n` |
| `MVL ((l)±m),(n)`    | `d←((l))±m, s←(n)`. Loop `I` times: `[d++]←[s++]`                                                            | `- -`       | 5     | 12+2×I       | `1111 0011` / `F3` <br> `1s00 0000` / `80/C0` <br> `l` <br> `m` <br> `n` | `l,m,n` |
| `MV (m),(n)`         | `(m) ← (n)` (Internal mem to Internal mem)                                                                   | `- -`       | 3     | 6            | `1100 1000` / `C8` <br> `m` <br> `n`                   | `m,n`        |
| `MVW (m),(n)`        | `(m..m+1) ← (n..n+1)`                                                                                        | `- -`       | 3     | 8            | `1100 1001` / `C9` <br> `m` <br> `n`                   | `m,n`        |
| `MVP (m),(n)`        | `(m..m+2) ← (n..n+2)`                                                                                        | `- -`       | 3     | 10           | `1100 1010` / `CA` <br> `m` <br> `n`                   | `m,n`        |
| `MVL (m),(n)`        | Loop `I` times: `(m++) ← (n++)`                                                                              | `- -`       | 3     | 5+2×I        | `1100 1011` / `CB` <br> `m` <br> `n`                   | `m,n`        |
| `MVLD (m),(n)`       | Loop `I` times: `(m--) ← (n--)`                                                                              | `- -`       | 3     | 5+2×I        | `1100 1111` / `CF` <br> `m` <br> `n`                   | `m,n`        |
| `MV (k),(lmn)`       | `(k) ← [lmn]` (Ext mem to Int mem)                                                                           | `- -`       | 5     | 7            | `1101 0000` / `D0` <br> `k` <br> `n` <br> `m` <br> `l` | `k,n,m,l`    |
| `MVW (k),(lmn)`      | `(k..k+1) ← [lmn..lmn+1]`                                                                                    | `- -`       | 5     | 8            | `1101 0001` / `D1` <br> `k` <br> `n` <br> `m` <br> `l` | `k,n,m,l`    |
| `MVP (k),(lmn)`      | `(k..k+2) ← [lmn..lmn+2]`                                                                                    | `- -`       | 5     | 9            | `1101 0010` / `D2` <br> `k` <br> `n` <br> `m` <br> `l` | `k,n,m,l`    |
| `MVL (k),(lmn)`      | Loop `I` times: `(k++) ← [lmn++]`                                                                            | `- -`       | 5     | 6+2×I        | `1101 0011` / `D3` <br> `k` <br> `n` <br> `m` <br> `l` | `k,n,m,l`    |
| `MV (m),n`           | `(m) ← n` (Immediate to Internal memory)                                                                     | `- -`       | 3     | 3            | `1100 1100` / `CC` <br> `m` <br> `n`                   | `m,n`        |
| `MVW (l),mn`         | `(l) ← n`, `(l+1) ← m`                                                                                       | `- -`       | 4     | 4            | `1100 1101` / `CD` <br> `l` <br> `n` <br> `m`         | `l,n,m`      |
| `MVP (k),lmn`        | `(k) ← n`, `(k+1) ← m`, `(k+2) ← l`                                                                          | `- -`       | 5     | 5            | `1101 1100` / `DC` <br> `k` <br> `n` <br> `m` <br> `l` | `k,n,m,l`    |
| `MV (n),[r3]`        | `(n) ← [r3]` (Ext mem reg-indirect to Int mem)                                                               | `- -`       | 3     | 6            | `1110 0000` / `E0` <br> `0r'₃` <br> `n`               | `n`          |
| `MVW (n),[r3]`       | `(n..n+1) ← [r3..r3+1]`                                                                                      | `- -`       | 3     | 7            | `1110 0001` / `E1` <br> `0r'₃` <br> `n`               | `n`          |
| `MVP (n),[r3]`       | `(n..n+2) ← [r3..r3+2]`                                                                                      | `- -`       | 3     | 8            | `1110 0010` / `E2` <br> `0r'₃` <br> `n`               | `n`          |
| `MVL (n),[r3++]`     | `d←(n), s←[r3]`. Loop `I` times: `[d++]←[s++]`. `r3` updated.                                              | `- -`       | 3     | 7+2×I        | `1110 0011` / `E3` <br> `0010 0r'₃` <br> `n`          | `n`          |
| `MVL (n),[--r3]`     | `d←(n), s←[r3]`. Loop `I` times: `[d++]←[--s]`. `r3` updated.                                              | `- -`       | 3     | 7+2×I        | `1110 0011` / `E3` <br> `0011 0r'₃` <br> `n`          | `n`          |
| `MV [r3],(n)`        | `[r3] ← (n)` (Int mem to Ext mem reg-indirect)                                                               | `- -`       | 3     | 6            | `1110 1000` / `E8` <br> `0r'₃` <br> `n`               | `n`          |
| `MVW [r3],(n)`       | `[r3..r3+1] ← (n..n+1)`                                                                                      | `- -`       | 3     | 7            | `1110 1001` / `E9` <br> `0r'₃` <br> `n`               | `n`          |
| `MVP [r3],(n)`       | `[r3..r3+2] ← (n..n+2)`                                                                                      | `- -`       | 3     | 8            | `1110 1010` / `EA` <br> `0r'₃` <br> `n`               | `n`          |
| `MVL [r3++],(n)`     | `d←[r3], s←(n)`. Loop `I` times: `[d++]←[s++]`. `r3` updated.                                              | `- -`       | 3     | 5+2×I        | `1110 1011` / `EB` <br> `0010 0r'₃` <br> `n`          | `n`          |
| `MVL [--r3],(n)`     | `d←[r3], s←(n)`. Loop `I` times: `[--d]←[s++]`. `r3` updated.                                              | `- -`       | 3     | 7+2×I        | `1110 1011` / `EB` <br> `0011 0r'₃` <br> `n`          | `n`          |
| `MVL (m),[r3±n]`     | `d←(m), s←[r3±n]`. Loop `I` times: `d++ ← s++`. `r3` not changed by `±n` for loop.                            | `- -`       | 4     | 5+2×I        | `0101 0110` / `56` <br> `1s00 0r'₃` <br> `m` <br> `n` | `m,n`      |
| `MVL [r3±m],(n)`     | `d←[r3±m], s←(n)`. Loop `I` times: `d++ ← s++`. `r3` not changed by `±m` for loop.                            | `- -`       | 4     | 5+2×I        | `0101 1110` / `5E` <br> `1s00 0r'₃` <br> `n` <br> `m` | `n,m`      |
| `MV (m),[(n)]`       | `(m) ← [[(n)]]` (Ext mem via Int mem Ind. to Int mem)                                                        | `- -`       | 4     | 11           | `1111 0000` / `F0` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVW (m),[(n)]`      | `(m..m+1) ← [[(n)]..[(n)]+1]`                                                                                | `- -`       | 4     | 12           | `1111 0001` / `F1` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVP (m),[(n)]`      | `(m..m+2) ← [[(n)]..[(n)]+2]`                                                                                | `- -`       | 4     | 13           | `1111 0010` / `F2` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVL (m),[(n)]`      | `d←(m), s←[[(n)]]`. Loop `I` times: `[d++]←[s++]`.                                                           | `- -`       | 4     | 10+2×I       | `1111 0011` / `F3` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MV [(m)],(n)`       | `[[(m)]] ← (n)` (Int mem to Ext mem via Int mem Ind.)                                                        | `- -`       | 4     | 11           | `1111 1000` / `F8` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVW [(m)],(n)`      | `[[(m)]..[(m)]+1] ← (n..n+1)`                                                                                | `- -`       | 4     | 12           | `1111 1001` / `F9` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVP [(m)],(n)`      | `[[(m)]..[(m)]+2] ← (n..n+2)`                                                                                | `- -`       | 4     | 13           | `1111 1010` / `FA` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MVL [(m)],(n)`      | `d←[[(m)]], s←(n)`. Loop `I` times: `[d++]←[s++]`.                                                           | `- -`       | 4     | 10+2×I       | `1111 1011` / `FB` <br> `0000 0000` <br> `m` <br> `n` | `m,n`        |
| `MV B,A`             | `B ← A`                                                                                                      | `- -`       | 1     | 1            | `0111 0101` / `75`                                    |              |
| `MV A,B`             | `A ← B`                                                                                                      | `- -`       | 1     | 1            | `0111 0100` / `74`                                    |              |
| `MV r₂,r'₂`          | `r₂ ← r'₂`                                                                                                   | `- -`       | 2     | 2            | `1111 1101` / `FD` <br> `0r 0r'` / `XX`               | `rr'`        |
| `MV r₃,r'₃`          | `r₃ ← r'₃`                                                                                                   | `- -`       | 2     | 2            | `1111 1101` / `FD` <br> `0r 0r'` / `XX`               | `rr'`        |

### Exchange Instructions (EX, EXW, EXP, EXL)

| Mnemonic        | Function (`↔` denotes exchange)                                                                     | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :-------------------------------------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `EX (m),(n)`    | `(m) ↔ (n)` (Internal memory byte exchange)                                                         | `- -`       | 3     | 7        | `1100 0000` / `C0` <br> `m` <br> `n`              | `m,n`        |
| `EXW (m),(n)`   | `(m,m+1) ↔ (n,n+1)` (Word exchange)                                                                 | `- -`       | 3     | 10       | `1100 0001` / `C1` <br> `m` <br> `n`              | `m,n`        |
| `EXP (m),(n)`   | `(m..m+2) ↔ (n..n+2)` (24-bit pointer/data exchange)                                                | `- -`       | 3     | 13       | `1100 0010` / `C2` <br> `m` <br> `n`              | `m,n`        |
| `EXL (m),(n)`   | Loop `I` times: `(m++) ↔ (n++)` (Block exchange)                                                    | `- -`       | 3     | 5+3×I    | `1100 0011` / `C3` <br> `m` <br> `n`              | `m,n`        |
| `EX A,B`        | `A ↔ B`                                                                                             | `- -`       | 1     | 3        | `1101 1101` / `DD`                               |              |
| `EX r₂,r'₂`     | `r₂ ↔ r'₂`                                                                                          | `- -`       | 2     | 4        | `1110 1101` / `ED` <br> `0r 0r'` / `XX`          | `rr'`        |
| `EX r₃,r'₃`     | `r₃ ↔ r'₃`                                                                                          | `- -`       | 2     | 4        | `1110 1101` / `ED` <br> `0r 0r'` / `XX`          | `rr'`        |

### Arithmetic Instructions (ADD, SUB, ADC, SBC, ADCL, SBCL, DADL, DSBL, PMDF)

| Mnemonic        | Function                                                                                             | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :--------------------------------------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `ADD A,n`       | `A ← A+n`                                                                                            | `○ ○`       | 2     | 3        | `0100 0000` / `40` <br> `n`                       | `n`          |
| `ADD (m),n`     | `(m) ← (m)+n`                                                                                        | `○ ○`       | 3     | 4        | `0100 0001` / `41` <br> `m` <br> `n`              | `m,n`        |
| `ADD A,(n)`     | `A ← A+(n)`                                                                                          | `○ ○`       | 2     | 4        | `0100 0010` / `42` <br> `n`                       | `n`          |
| `ADD (n),A`     | `(n) ← (n)+A`                                                                                        | `○ ○`       | 2     | 4        | `0100 0011` / `43` <br> `n`                       | `n`          |
| `ADD r₁,r'₁`    | `r₁ ← r₁+r'₁`                                                                                        | `○ ○`       | 2     | 3        | `0100 0110` / `46` <br> `rr'` / `XX`              | `rr'`        |
| `ADD r₂,r'₁`    | `r₂ ← r₂+r'₁`                                                                                        | `○ ○`       | 2     | 5        | `0100 0100` / `44` <br> `rr'` / `XX`              | `rr'`        |
| `ADD r₂,r'₂`    | `r₂ ← r₂+r'₂`                                                                                        | `○ ○`       | 2     | 5        | `0100 0100` / `44` <br> `rr'` / `XX`              | `rr'`        |
| `ADD r₃,r'`     | `r₃ ← r₃+r'` (r' can be r1, r2, or r3 type)                                                          | `○ ○`       | 2     | 7        | `0100 0101` / `45` <br> `rr'` / `XX`              | `rr'`        |
| `SUB A,n`       | `A ← A-n`                                                                                            | `○ ○`       | 2     | 3        | `0100 1000` / `48` <br> `n`                       | `n`          |
| `SUB (m),n`     | `(m) ← (m)-n`                                                                                        | `○ ○`       | 3     | 4        | `0100 1001` / `49` <br> `m` <br> `n`              | `m,n`        |
| `SUB A,(n)`     | `A ← A-(n)`                                                                                          | `○ ○`       | 2     | 4        | `0100 1010` / `4A` <br> `n`                       | `n`          |
| `SUB (n),A`     | `(n) ← (n)-A`                                                                                        | `○ ○`       | 2     | 4        | `0100 1011` / `4B` <br> `n`                       | `n`          |
| `SUB r₁,r'₁`    | `r₁ ← r₁-r'₁`                                                                                        | `○ ○`       | 2     | 3        | `0100 1110` / `4E` <br> `rr'` / `XX`              | `rr'`        |
| `SUB r₂,r'₁`    | `r₂ ← r₂-r'₁`                                                                                        | `○ ○`       | 2     | 5        | `0100 1100` / `4C` <br> `rr'` / `XX`              | `rr'`        |
| `SUB r₂,r'₂`    | `r₂ ← r₂-r'₂`                                                                                        | `○ ○`       | 2     | 5        | `0100 1100` / `4C` <br> `rr'` / `XX`              | `rr'`        |
| `SUB r₃,r'`     | `r₃ ← r₃-r'`                                                                                         | `○ ○`       | 2     | 7        | `0100 1101` / `4D` <br> `rr'` / `XX`              | `rr'`        |
| `ADC A,n`       | `A ← A+n+C`                                                                                          | `○ ○`       | 2     | 3        | `0101 0000` / `50` <br> `n`                       | `n`          |
| `ADC (m),n`     | `(m) ← (m)+n+C`                                                                                      | `○ ○`       | 3     | 4        | `0101 0001` / `51` <br> `m` <br> `n`              | `m,n`        |
| `ADC A,(n)`     | `A ← A+(n)+C`                                                                                        | `○ ○`       | 2     | 4        | `0101 0010` / `52` <br> `n`                       | `n`          |
| `ADC (n),A`     | `(n) ← (n)+A+C`                                                                                      | `○ ○`       | 2     | 4        | `0101 0011` / `53` <br> `n`                       | `n`          |
| `SBC A,n`       | `A ← A-n-C`                                                                                          | `○ ○`       | 2     | 3        | `0101 1000` / `58` <br> `n`                       | `n`          |
| `SBC (m),n`     | `(m) ← (m)-n-C`                                                                                      | `○ ○`       | 3     | 4        | `0101 1001` / `59` <br> `m` <br> `n`              | `m,n`        |
| `SBC A,(n)`     | `A ← A-(n)-C`                                                                                        | `○ ○`       | 2     | 4        | `0101 1010` / `5A` <br> `n`                       | `n`          |
| `SBC (n),A`     | `(n) ← (n)-A-C`                                                                                      | `○ ○`       | 2     | 4        | `0101 1011` / `5B` <br> `n`                       | `n`          |
| `ADCL (m),(n)`  | Loop `I` times: `(m) ← (m)+(n)+C` (byte-wise, C propagates)                                          | `○ ○`       | 3     | 5+2×I    | `0101 0100` / `54` <br> `m` <br> `n`              | `m,n`        |
| `ADCL (n),A`    | Loop `I` times: `(n) ← (n)+A+C` (byte-wise, A is src for each byte, C propagates)                    | `○ ○`       | 2     | 4+1×I    | `0101 0101` / `55` <br> `n`                       | `n`          |
| `SBCL (m),(n)`  | Loop `I` times: `(m) ← (m)-(n)-C` (byte-wise, C propagates as borrow)                                | `○ ○`       | 3     | 5+2×I    | `0101 1100` / `5C` <br> `m` <br> `n`              | `m,n`        |
| `SBCL (n),A`    | Loop `I` times: `(n) ← (n)-A-C` (byte-wise, A is src for each byte, C propagates)                    | `○ ○`       | 2     | 4+1×I    | `0101 1101` / `5D` <br> `n`                       | `n`          |
| `DADL (m),(n)`  | BCD add with carry: `(m) ← (m)+(n)+C` (multi-byte, addresses dec.)                                   | `○ ○`       | 3     | 5+2×I    | `1100 0100` / `C4` <br> `m` <br> `n`              | `m,n`        |
| `DADL (n),A`    | BCD add with carry: `(n) ← (n)+A+C` (multi-byte, (n) addr dec.)                                      | `○ ○`       | 2     | 4+1×I    | `1100 0101` / `C5` <br> `n`                       | `n`          |
| `DSBL (m),(n)`  | BCD sub with borrow: `(m) ← (m)-(n)-C` (multi-byte, addresses dec.)                                  | `○ ○`       | 3     | 5+2×I    | `1101 0100` / `D4` <br> `m` <br> `n`              | `m,n`        |
| `DSBL (n),A`    | BCD sub with borrow: `(n) ← (n)-A-C` (multi-byte, (n) addr dec.)                                     | `○ ○`       | 2     | 4+1×I    | `1101 0101` / `D5` <br> `n`                       | `n`          |
| `PMDF (m),n`    | Packed BCD Modify: `(m) ← (m)+n` (special BCD operation)                                             | `- -`       | 3     | 4        | `0100 0111` / `47` <br> `m` <br> `n`              | `m,n`        |
| `PMDF (n),A`    | Packed BCD Modify: `(n) ← (n)+A`                                                                     | `- -`       | 2     | 4        | `0101 0111` / `57` <br> `n`                       | `n`          |

### Logical Instructions (AND, OR, XOR, TEST, SWAP)

| Mnemonic        | Function                                                               | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :--------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `AND A,n`       | `A ← A & n`                                                            | `- ○`       | 2     | 3        | `0111 0000` / `70` <br> `n`                       | `n`          |
| `AND (m),n`     | `(m) ← (m) & n`                                                        | `- ○`       | 3     | 4        | `0111 0001` / `71` <br> `m` <br> `n`              | `m,n`        |
| `AND [lmn],n`   | `[lmn] ← [lmn] & n`                                                    | `- ○`       | 5     | 7        | `0111 0010` / `72` <br> `lmn` <br> `n`           | `lmn,n`      |
| `AND (n),A`     | `(n) ← (n) & A`                                                        | `- ○`       | 2     | 4        | `0111 0011` / `73` <br> `n`                       | `n`          |
| `AND A,(n)`     | `A ← A & (n)`                                                          | `- ○`       | 2     | 4        | `0111 0111` / `77` <br> `n`                       | `n`          |
| `AND (m),(n)`   | `(m) ← (m) & (n)`                                                      | `- ○`       | 3     | 6        | `0111 0110` / `76` <br> `m` <br> `n`              | `m,n`        |
| `OR A,n`        | `A ← A \| n`                                                            | `- ○`       | 2     | 3        | `0111 1000` / `78` <br> `n`                       | `n`          |
| `OR (m),n`      | `(m) ← (m) \| n`                                                        | `- ○`       | 3     | 4        | `0111 1001` / `79` <br> `m` <br> `n`              | `m,n`        |
| `OR [lmn],n`    | `[lmn] ← [lmn] \| n`                                                    | `- ○`       | 5     | 7        | `0111 1010` / `7A` <br> `lmn` <br> `n`           | `lmn,n`      |
| `OR (n),A`      | `(n) ← (n) \| A`                                                        | `- ○`       | 2     | 4        | `0111 1011` / `7B` <br> `n`                       | `n`          |
| `OR A,(n)`      | `A ← A \| (n)`                                                          | `- ○`       | 2     | 4        | `0111 1111` / `7F` <br> `n`                       | `n`          |
| `OR (m),(n)`    | `(m) ← (m) \| (n)`                                                      | `- ○`       | 3     | 6        | `0111 1110` / `7E` <br> `m` <br> `n`              | `m,n`        |
| `XOR A,n`       | `A ← A ^ n`                                                            | `- ○`       | 2     | 3        | `0110 1000` / `68` <br> `n`                       | `n`          |
| `XOR (m),n`     | `(m) ← (m) ^ n`                                                        | `- ○`       | 3     | 4        | `0110 1001` / `69` <br> `m` <br> `n`              | `m,n`        |
| `XOR [lmn],n`   | `[lmn] ← [lmn] ^ n`                                                    | `- ○`       | 5     | 7        | `0110 1010` / `6A` <br> `lmn` <br> `n`           | `lmn,n`      |
| `XOR (n),A`     | `(n) ← (n) ^ A`                                                        | `- ○`       | 2     | 4        | `0110 1011` / `6B` <br> `n`                       | `n`          |
| `XOR A,(n)`     | `A ← A ^ (n)`                                                          | `- ○`       | 2     | 4        | `0110 1111` / `6F` <br> `n`                       | `n`          |
| `XOR (m),(n)`   | `(m) ← (m) ^ (n)`                                                      | `- ○`       | 3     | 6        | `0110 1110` / `6E` <br> `m` <br> `n`              | `m,n`        |
| `TEST A,n`      | `A & n` (sets Z flag)                                                  | `- ○`       | 2     | 3        | `0110 0100` / `64` <br> `n`                       | `n`          |
| `TEST (m),n`    | `(m) & n`                                                              | `- ○`       | 3     | 4        | `0110 0101` / `65` <br> `m` <br> `n`              | `m,n`        |
| `TEST [lmn],n`  | `[lmn] & n`                                                            | `- ○`       | 5     | 6        | `0110 0110` / `66` <br> `lmn` <br> `n`           | `lmn,n`      |
| `TEST (n),A`    | `(n) & A`                                                              | `- ○`       | 2     | 4        | `0110 0111` / `67` <br> `n`                       | `n`          |
| `SWAP A`        | `A₀₋₃ ↔ A₄₋₇` (Swap nibbles of A)                                      | `○ ○`       | 1     | 3        | `1110 1110` / `EE`                               |              |

### Compare Instructions (CMP, CMPW, CMPP)

| Mnemonic        | Function (`-` denotes comparison, sets flags)                         | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :-------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `CMP A,n`       | `A - n`                                                               | `○ ○`       | 2     | 3        | `0110 0000` / `60` <br> `n`                       | `n`          |
| `CMP (m),n`     | `(m) - n`                                                             | `○ ○`       | 3     | 4        | `0110 0001` / `61` <br> `m` <br> `n`              | `m,n`        |
| `CMP [lmn],n`   | `[lmn] - n`                                                           | `○ ○`       | 5     | 6        | `0110 0010` / `62` <br> `lmn` <br> `n`           | `lmn,n`      |
| `CMP (n),A`     | `(n) - A`                                                             | `○ ○`       | 2     | 4        | `0110 0011` / `63` <br> `n`                       | `n`          |
| `CMP (m),(n)`   | `(m) - (n)`                                                           | `○ ○`       | 3     | 6        | `1011 0111` / `B7` <br> `m` <br> `n`              | `m,n`        |
| `CMPW (m),(n)`  | `(m..m+1) - (n..n+1)` (Word compare)                                  | `○ ○`       | 3     | 8        | `1100 0110` / `C6` <br> `m` <br> `n`              | `m,n`        |
| `CMPW (m),r2`   | `(m..m+1) - r2`                                                       | `○ ○`       | 3     | 7        | `1101 0110` / `D6` <br> `0r` <br> `m`             | `m` (r in op)|
| `CMPP (m),(n)`  | `(m..m+2) - (n..n+2)` (24-bit compare)                                | `○ ○`       | 3     | 10       | `1100 0111` / `C7` <br> `m` <br> `n`              | `m,n`        |
| `CMPP (m),r3`   | `(m..m+2) - r3`                                                       | `○ ○`       | 3     | 9        | `1101 0111` / `D7` <br> `0r` <br> `m`             | `m` (r in op)|

### Shift and Rotate Instructions

| Mnemonic        | Function                                                              | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :-------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `ROR A`         | Rotate A Right (bit 0 to bit 7 and C)                                 | `○ ○`       | 1     | 2        | `1110 0100` / `E4`                               |              |
| `ROR (n)`       | Rotate (n) Right                                                      | `○ ○`       | 2     | 3        | `1110 0101` / `E5` <br> `n`                       | `n`          |
| `ROL A`         | Rotate A Left (bit 7 to bit 0 and C)                                  | `○ ○`       | 1     | 2        | `1110 0110` / `E6`                               |              |
| `ROL (n)`       | Rotate (n) Left                                                       | `○ ○`       | 2     | 3        | `1110 0111` / `E7` <br> `n`                       | `n`          |
| `SHR A`         | Shift A Right through Carry (C ← A₀, A₀ ← A₁, ..., A₇ ← C)            | `○ ○`       | 1     | 2        | `1111 0100` / `F4`                               |              |
| `SHR (n)`       | Shift (n) Right through Carry                                         | `○ ○`       | 2     | 3        | `1111 0101` / `F5` <br> `n`                       | `n`          |
| `SHL A`         | Shift A Left through Carry (C ← A₇, A₇ ← A₆, ..., A₀ ← C)             | `○ ○`       | 1     | 2        | `1111 0110` / `F6`                               |              |
| `SHL (n)`       | Shift (n) Left through Carry                                          | `○ ○`       | 2     | 3        | `1111 0111` / `F7` <br> `n`                       | `n`          |
| `DSRL (n)`      | Decimal Shift Right Logical (multi-byte, (n) is LSB addr, addrs inc.) | `- ○`       | 2     | 4+1×I    | `1111 1100` / `FC` <br> `n`                       | `n`          |
| `DSLL (n)`      | Decimal Shift Left Logical (multi-byte, (n) is MSB addr, addrs dec.)  | `- ○`       | 2     | 4+1×I    | `1110 1100` / `EC` <br> `n`                       | `n`          |

### Increment and Decrement Instructions

| Mnemonic        | Function                                  | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :---------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `INC r`         | `r ← r+1` (r can be r1, r2, r3)           | `- ○`       | 2     | 3        | `0110 1100` / `6C` <br> `0r` / `0X`               | `r`          |
| `INC (n)`       | `(n) ← (n)+1`                             | `- ○`       | 2     | 3        | `0110 1101` / `6D` <br> `n`                       | `n`          |
| `DEC r`         | `r ← r-1` (r can be r1, r2, r3)           | `- ○`       | 2     | 3        | `0111 1100` / `7C` <br> `0r` / `0X`               | `r`          |
| `DEC (n)`       | `(n) ← (n)-1`                             | `- ○`       | 2     | 3        | `0111 1101` / `7D` <br> `n`                       | `n`          |

### Jump, Call, and Return Instructions

| Mnemonic        | Function                                                                                             | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                                    | Operand Type |
| :-------------- | :--------------------------------------------------------------------------------------------------- | :---------- | :---- | :------- | :---------------------------------------------------- | :----------- |
| `JP (n)`        | `PC ← (n)` (address from internal memory `(n),(n+1),(n+2)`)                                          | `- -`       | 2     | 6        | `0001 0000` / `10` <br> `n`                            | `n`          |
| `JP r3`         | `PC ← r3` (r3 is X,Y,U,S)                                                                            | `- -`       | 2     | 4        | `0001 0001` / `11` <br> `0000 0r₃` / `0X`             |              |
| `JP mn`         | `PC ← mn` (effective 20-bit addr: current page + mn)                                                 | `- -`       | 3     | 4        | `0000 0010` / `02` <br> `n` <br> `m`                   | `n,m`        |
| `JPF lmn`       | `PC ← lmn`, `PS ← 1` (far jump)                                                                      | `- -`       | 4     | 5        | `0000 0011` / `03` <br> `n` <br> `m` <br> `l`          | `n,m,l`      |
| `JR +n`         | `PC ← PC+2+n` (relative jump forward)                                                                | `- -`       | 2     | 3        | `0001 0010` / `12` <br> `n`                            | `+n`         |
| `JR -n`         | `PC ← PC+2-n` (relative jump backward)                                                               | `- -`       | 2     | 3        | `0001 0011` / `13` <br> `n`                            | `-n`         |
| `JPZ mn`        | If Z=1, `PC ← mn` else `PC ← PC+3`                                                                   | `- -`       | 3     | 4/3      | `0001 0100` / `14` <br> `n` <br> `m`                   | `n,m`        |
| `JPNZ mn`       | If Z=0, `PC ← mn` else `PC ← PC+3`                                                                   | `- -`       | 3     | 4/3      | `0001 0101` / `15` <br> `n` <br> `m`                   | `n,m`        |
| `JPC mn`        | If C=1, `PC ← mn` else `PC ← PC+3`                                                                   | `- -`       | 3     | 4/3      | `0001 0110` / `16` <br> `n` <br> `m`                   | `n,m`        |
| `JPNC mn`       | If C=0, `PC ← mn` else `PC ← PC+3`                                                                   | `- -`       | 3     | 4/3      | `0001 0111` / `17` <br> `n` <br> `m`                   | `n,m`        |
| `JRZ ±n`        | If Z=1, `PC ← PC+2±n` else `PC ← PC+2`                                                               | `- -`       | 2     | 3/2      | `0001 100s` / `18/19` <br> `n`                       | `±n`         |
| `JRNZ ±n`       | If Z=0, `PC ← PC+2±n` else `PC ← PC+2`                                                               | `- -`       | 2     | 3/2      | `0001 101s` / `1A/1B` <br> `n`                       | `±n`         |
| `JRC ±n`        | If C=1, `PC ← PC+2±n` else `PC ← PC+2`                                                               | `- -`       | 2     | 3/2      | `0001 110s` / `1C/1D` <br> `n`                       | `±n`         |
| `JRNC ±n`       | If C=0, `PC ← PC+2±n` else `PC ← PC+2`                                                               | `- -`       | 2     | 3/2      | `0001 111s` / `1E/1F` <br> `n`                       | `±n`         |
| `CALL mn`       | `S ← S-2`, `[S] ← PC+3` (current page), `PC ← mn`                                                    | `- -`       | 3     | 6        | `0000 0100` / `04` <br> `n` <br> `m`                   | `n,m`        |
| `CALLF lmn`     | `S ← S-3`, `[S] ← PC+4`, `PS ← 1`, `PC ← lmn`                                                        | `- -`       | 4     | 8        | `0000 0101` / `05` <br> `n` <br> `m` <br> `l`          | `n,m,l`      |
| `RET`           | `PC ← [S]`, `S ← S+2`                                                                                | `- -`       | 1     | 4        | `0000 0110` / `06`                                   |              |
| `RETF`          | `PS ← [S+2]`, `PC ← [S]`, `S ← S+3`                                                                  | `- -`       | 1     | 5        | `0000 0111` / `07`                                   |              |
| `RETI`          | `IMR ← [S]`, `F ← [S+1]`, `PC ← [S+2]`, `PS ← [S+4]`, `S ← S+5`                                       | C Z restore | 1     | 7        | `0000 0001` / `01`                                   |              |

### Stack Instructions (PUSH, POP)

| Mnemonic        | Function (`U` is User Stack, `S` is System Stack)                                          | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :----------------------------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `PUSHU r₁`      | `U ← U-1`, `[U] ← r₁`                                                                      | `- -`       | 1     | 3        | `0010 1rr0` / `28/29`                            | `r₁`         |
| `PUSHU r₂`      | `U ← U-2`, `[U] ← r₂`                                                                      | `- -`       | 1     | 4        | `0010 1rr0` / `2A/2B`                            | `r₂`         |
| `PUSHU r₄`      | `U ← U-3`, `[U] ← r₄` (X,Y)                                                                | `- -`       | 1     | 5        | `0010 1rr0` / `2C/2D`                            | `r₄`         |
| `PUSHU F`       | `U ← U-1`, `[U] ← F`                                                                       | `- -`       | 1     | 3        | `0010 1110` / `2E`                               |              |
| `PUSHU IMR`     | `U ← U-1`, `[U] ← IMR`, `IMR₇ ← 0`                                                         | `- -`       | 1     | 3        | `0010 1111` / `2F`                               |              |
| `POPU r₁`       | `r₁ ← [U]`, `U ← U+1`                                                                      | `- -`       | 1     | A:2/IL:3 | `0011 1rr0` / `38/39`                            | `r₁`         |
| `POPU r₂`       | `r₂ ← [U]`, `U ← U+2`                                                                      | `- -`       | 1     | 3        | `0011 1rr0` / `3A/3B`                            | `r₂`         |
| `POPU r₄`       | `r₄ ← [U]`, `U ← U+3` (X,Y)                                                                | `- -`       | 1     | 4        | `0011 1rr0` / `3C/3D`                            | `r₄`         |
| `POPU F`        | `F ← [U]`, `U ← U+1`                                                                       | C Z restore | 1     | 2        | `0011 1110` / `3E`                               |              |
| `POPU IMR`      | `IMR ← [U]`, `U ← U+1`                                                                     | `- -`       | 1     | 2        | `0011 1111` / `3F`                               |              |
| `PUSHS F`       | `S ← S-1`, `[S] ← F`                                                                       | `- -`       | 1     | 3        | `0100 1111` / `4F`                               |              |
| `POPS F`        | `F ← [S]`, `S ← S+1`                                                                       | C Z restore | 1     | 2        | `0101 1111` / `5F`                               |              |

### Miscellaneous Instructions

| Mnemonic        | Function                                                                   | Flags (C Z) | Bytes | Cycles | Opcode (Bin / Hex)                               | Operand Type |
| :-------------- | :------------------------------------------------------------------------- | :---------- | :---- | :------- | :----------------------------------------------- | :----------- |
| `NOP`           | No Operation                                                               | `- -`       | 1     | 1        | `0000 0000` / `00`                               |              |
| `SC`            | Set Carry Flag (`C ← 1`)                                                   | `○ -`       | 1     | 1        | `1001 0111` / `97`                               |              |
| `RC`            | Reset Carry Flag (`C ← 0`)                                                 | `○ -`       | 1     | 1        | `1001 1111` / `9F`                               |              |
| `TCL`           | Timer Clear / Divider ← D                                                  | `- -`       | 1     | 1        | `1100 1110` / `CE`                               |              |
| `HALT`          | Halt CPU (System Clock Stop)                                               | `- -`       | 1     | Note 1   | `1101 1110` / `DE`                               |              |
| `OFF`           | Power Off CPU (System & Sub Clock Stop)                                    | `- -`       | 1     | Note 1   | `1101 1111` / `DF`                               |              |
| `WAIT`          | Wait Loop (`I` times)                                                      | `- -`       | 1     | 1+1×I    | `1110 1111` / `EF`                               |              |
| `IR`            | Software Interrupt                                                         | `- -`       | 1     | Note 2   | `1111 1110` / `FE`                               |              |
| `RESET`         | Software Reset                                                             | `- -`       | 1     | Note 2   | `1111 1111` / `FF`                               |              |

*Note 1: HALT/OFF cycle count is effectively "until interrupt/resume".*
*Note 2: IR/RESET involve stack operations and vector fetches, cycles are complex.*
