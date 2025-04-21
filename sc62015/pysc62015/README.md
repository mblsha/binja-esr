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

## Opcode Info (INCOMPLETE)

**Notes:**

*   `r₁`, `r₂`, `r₃`, `r₄`: Represent registers of different sizes. `r` in opcode patterns is a placeholder.
*   `rᵢₗ`, `rᵢₕ`, `rᵢₓ`: Subscripts denote Low, High, and Extension bytes of register `rᵢ`.
*   `n`, `m`, `l`: Represent 8-bit immediate values or address bytes. `mn`, `lmn` form 16/20-bit values/addresses.
*   `(addr)`: **Internal** Memory (Direct address `addr`).
*   `[addr]`: **External** Memory (Direct address `addr`).
*   `[r'₃]`: **External** Memory (Register Indirect: address is in `r'₃`).
*   `[r'₃++]` / `[--r'₃]`: **External** Memory (Register Indirect w/ post-increment / pre-decrement).
*   `[r'₃±n]`: **External** Memory (Register Indirect w/ 8-bit signed offset `n`).
*   `[(addr)]`: **External** Memory (Memory Indirect: address is in internal memory location `addr`).
*   `[(m)±n]`: **External** Memory (Memory Indexed: base address in internal `(m)`, offset `n`).
*   `r'₃`: Specific register used for indirect addressing modes shown here.

| Mnemonic             | Function (`←` denotes assignment)                                                                            | Flags (C Z) | Bytes | Cycles       | Opcode (Bin / Hex)                                    | Operand Type |
| :------------------- | :----------------------------------------------------------------------------------------------------------- | :---------- | :---- | :----------- | :---------------------------------------------------- | :----------- |
| **MV r, immediate**  |                                                                                                              |             |       |              |                                                       |              |
| `MV r₁, n`           | `r₁ ← n` (if `r₁=IL` then `IH←0`)                                                                            | `- -`       | 2     | A: 2 / IL: 3 | `0000 1 r` / `0X` <br> `n`                            | `n`          |
| `MV r₂, mn`          | `r₂ₗ ← n`, `r₂ₕ ← m`                                                                                         | `- -`       | 3     | 3            | `0000 1 r` / `0X` <br> `n` <br> `m`                   | `n, m`       |
| `MV r₃, lmn`         | `r₃ₗ ← n`, `r₃ₕ ← m`, `r₃ₓ ← l`                                                                              | `- -`       | 4     | 4            | `0000 1 r` / `0X` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| **MV r, (n)**        | **From Internal Memory (Direct)**                                                                            |             |       |              |                                                       |              |
| `MV r₁, (n)`         | `r₁ ← (n)` (if `r₁=IL` then `IH←0`)                                                                          | `- -`       | 2     | A: 3 / IL: 4 | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| `MV r₂, (n)`         | `r₂ₗ ← (n)`, `r₂ₕ ← (n+1)`                                                                                   | `- -`       | 2     | 4            | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| `MV r₃, (n)`         | `r₃ₗ ← (n)`, `r₃ₕ ← (n+1)`, `r₃ₓ ← (n+2)`                                                                    | `- -`       | 2     | 5            | `1000 0 r` / `8X` <br> `n`                            | `n`          |
| **MV r, [lmn]**      | **From External Memory (Direct)**                                                                            |             |       |              |                                                       |              |
| `MV r₁, [lmn]`       | `r₁ ← [lmn]` (if `r₁=IL` then `IH←0`)                                                                        | `- -`       | 4     | 6            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| `MV r₂, [lmn]`       | `r₂ₗ ← [lmn]`, `r₂ₕ ← [lmn+1]`                                                                               | `- -`       | 4     | 7            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| `MV r₃, [lmn]`       | `r₃ₗ ← [lmn]`, `r₃ₕ ← [lmn+1]`, `r₃ₓ ← [lmn+2]`                                                              | `- -`       | 4     | 8            | `1000 1 r` / `8X` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| **MV r, [r'₃]**      | **From External Memory (Register Indirect)**                                                                 |             |       |              |                                                       |              |
| `MV r₁, [r'₃]`       | `r₁ ← [r'₃]` (if `r₁=IL` then `IH←0`)                                                                        | `- -`       | 2     | A: 4 / IL: 5 | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| `MV r₂, [r'₃]`       | `r₂ₗ ← [r'₃]`, `r₂ₕ ← [r'₃+1]`                                                                               | `- -`       | 2     | 5            | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| `MV r₄, [r'₃]`       | `r₄ₗ ← [r'₃]`, `r₄ₕ ← [r'₃+1]`, `r₄ₓ ← [r'₃+2]`                                                              | `- -`       | 2     | 6            | `1001 0 r` / `9X` <br> `0000 0 r'₃` / `0X`            |              |
| **MV r, [r'₃++]**    | **From External Memory (Register Indirect w/ Post-Increment)**                                               |             |       |              |                                                       |              |
| `MV r₁, [r'₃++]`     | `r₁ ← [r'₃]` (if `r₁=IL` then `IH←0`), `r'₃ ← r'₃+1`                                                         | `- -`       | 2     | A: 4 / IL: 5 | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| `MV r₂, [r'₃++]`     | `r₂ₗ ← [r'₃]`, `r₂ₕ ← [r'₃+1]`, `r'₃ ← r'₃+2`                                                                | `- -`       | 2     | 5            | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| `MV r₄, [r'₃++]`     | `r₄ₗ ← [r'₃]`, `r₄ₕ ← [r'₃+1]`, `r₄ₓ ← [r'₃+2]`, `r'₃ ← r'₃+3`                                               | `- -`       | 2     | 7            | `1001 0 r` / `9X` <br> `0010 0 r'₃` / `2X`            |              |
| **MV r, [--r'₃]**    | **From External Memory (Register Indirect w/ Pre-Decrement)**                                                |             |       |              |                                                       |              |
| `MV r₁, [--r'₃]`     | `r'₃ ← r'₃-1`, `r₁ ← [r'₃]` (if `r₁=IL` then `IH←0`)                                                         | `- -`       | 2     | A: 5 / IL: 6 | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| `MV r₂, [--r'₃]`     | `r'₃ ← r'₃-2`, `r₂ₗ ← [r'₃]`, `r₂ₕ ← [r'₃+1]`                                                                | `- -`       | 2     | 6            | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| `MV r₄, [--r'₃]`     | `r'₃ ← r'₃-3`, `r₄ₗ ← [r'₃]`, `r₄ₕ ← [r'₃+1]`, `r₄ₓ ← [r'₃+2]`                                               | `- -`       | 2     | 8            | `1001 0 r` / `9X` <br> `0011 0 r'₃` / `3X`            |              |
| **MV r, [r'₃±n]**    | **From External Memory (Register Indirect w/ Offset)**                                                       |             |       |              |                                                       |              |
| `MV r₁, [r'₃±n]`     | `r₁ ← [r'₃±n]` (if `r₁=IL` then `IH←0`)                                                                      | `- -`       | 3     | A: 6 / IL: 7 | `1001 0 r` / `9X` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV r₂, [r'₃±n]`     | `r₂ₗ ← [r'₃±n]`, `r₂ₕ ← [r'₃±n+1]`                                                                           | `- -`       | 3     | 7            | `1001 0 r` / `9X` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV r₄, [r'₃±n]`     | `r₄ₗ ← [r'₃±n]`, `r₄ₕ ← [r'₃±n+1]`, `r₄ₓ ← [r'₃±n+2]`                                                        | `- -`       | 3     | 8            | `1001 0 r` / `9X` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| **MV r, [(n)]**      | **From External Memory (Memory Indirect)**                                                                   |             |       |              |                                                       |              |
| `MV r₁, [(n)]`       | `r₁ ← [(n)]` (if `r₁=IL` then `IH←0`)                                                                        | `- -`       | 3     | A: 9 / IL: 10 | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV r₂, [(n)]`       | `r₂ₗ ← [(n)]`, `r₂ₕ ← [(n)+1]`                                                                               | `- -`       | 3     | 10           | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV r₄, [(n)]`       | `r₄ₗ ← [(n)]`, `r₄ₕ ← [(n)+1]`, `r₄ₓ ← [(n)+2]`                                                              | `- -`       | 3     | 11           | `1001 1 r` / `9X` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| **MV r, [(m)±n]**    | **From External Memory (Memory Indexed)**                                                                    |             |       |              |                                                       |              |
| `MV r₁, [(m)±n]`     | `r₁ ← [(m)±n]` (if `r₁=IL` then `IH←0`)                                                                      | `- -`       | 4     | A:11 / IL:12 | `1001 1 r` / `9X` <br> `1[0/1]00 0000` / `80/C0` <br> `m` <br> `n` | `m, n`       |
| `MV r₂, [(m)±n]`     | `r₂ₗ ← [(m)±n]`, `r₂ₕ ← [(m)±n+1]`                                                                           | `- -`       | 4     | 12           | `1001 1 r` / `9X` <br> `1[0/1]00 0000` / `80/C0` <br> `m` <br> `n` | `m, n`       |
| `MV r₄, [(m)±n]`     | `r₄ₗ ← [(m)±n]`, `r₄ₕ ← [(m)±n+1]`, `r₄ₓ ← [(m)±n+2]`                                                        | `- -`       | 4     | 13           | `1001 1 r` / `9X` <br> `1[0/1]00 0000` / `80/C0` <br> `m` <br> `n` | `m, n`       |
| **MV (n), r**        | **To Internal Memory (Direct)**                                                                              |             |       |              |                                                       |              |
| `MV (n), r₁`         | `(n) ← r₁`                                                                                                   | `- -`       | 2     | 3            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| `MV (n), r₂`         | `(n) ← r₂ₗ`, `(n+1) ← r₂ₕ`                                                                                   | `- -`       | 2     | 4            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| `MV (n), r₃`         | `(n) ← r₃ₗ`, `(n+1) ← r₃ₕ`, `(n+2) ← r₃ₓ`                                                                    | `- -`       | 2     | 5            | `1010 0 r` / `AX` <br> `n`                            | `n`          |
| **MV [lmn], r**      | **To External Memory (Direct)**                                                                              |             |       |              |                                                       |              |
| `MV [lmn], r₁`       | `[lmn] ← r₁`                                                                                                 | `- -`       | 4     | 5            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| `MV [lmn], r₂`       | `[lmn] ← r₂ₗ`, `[lmn+1] ← r₂ₕ`                                                                               | `- -`       | 4     | 6            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| `MV [lmn], r₃`       | `[lmn] ← r₃ₗ`, `[lmn+1] ← r₃ₕ`, `[lmn+2] ← r₃ₓ`                                                              | `- -`       | 4     | 7            | `1010 1 r` / `AX` <br> `n` <br> `m` <br> `l`          | `n, m, l`    |
| **MV [r'₃], r**      | **To External Memory (Register Indirect)**                                                                   |             |       |              |                                                       |              |
| `MV [r'₃], r₁`       | `[r'₃] ← r₁`                                                                                                 | `- -`       | 2     | 4            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| `MV [r'₃], r₂`       | `[r'₃] ← r₂ₗ`, `[r'₃+1] ← r₂ₕ`                                                                               | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| `MV [r'₃], r₄`       | `[r'₃] ← r₄ₗ`, `[r'₃+1] ← r₄ₕ`, `[r'₃+2] ← r₄ₓ`                                                              | `- -`       | 2     | 6            | `1011 0 r` / `BX` <br> `0000 0 r'₃` / `0X`            |              |
| **MV [r'₃++], r**    | **To External Memory (Register Indirect w/ Post-Increment)**                                                 |             |       |              |                                                       |              |
| `MV [r'₃++], r₁`     | `[r'₃] ← r₁`, `r'₃ ← r'₃+1`                                                                                  | `- -`       | 2     | 4            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| `MV [r'₃++], r₂`     | `[r'₃] ← r₂ₗ`, `[r'₃+1] ← r₂ₕ`, `r'₃ ← r'₃+2`                                                                | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| `MV [r'₃++], r₄`     | `[r'₃] ← r₄ₗ`, `[r'₃+1] ← r₄ₕ`, `[r'₃+2] ← r₄ₓ`, `r'₃ ← r'₃+3`                                               | `- -`       | 2     | 7            | `1011 0 r` / `BX` <br> `0010 0 r'₃` / `2X`            |              |
| **MV [--r'₃], r**    | **To External Memory (Register Indirect w/ Pre-Decrement)**                                                  |             |       |              |                                                       |              |
| `MV [--r'₃], r₁`     | `r'₃ ← r'₃-1`, `[r'₃] ← r₁`                                                                                  | `- -`       | 2     | 5            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| `MV [--r'₃], r₂`     | `r'₃ ← r'₃-2`, `[r'₃] ← r₂ₗ`, `[r'₃+1] ← r₂ₕ`                                                                | `- -`       | 2     | 6            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| `MV [--r'₃], r₄`     | `r'₃ ← r'₃-3`, `[r'₃] ← r₄ₗ`, `[r'₃+1] ← r₄ₕ`, `[r'₃+2] ← r₄ₓ`                                               | `- -`       | 2     | 8            | `1011 0 r` / `BX` <br> `0011 0 r'₃` / `3X`            |              |
| **MV [r'₃±n], r**    | **To External Memory (Register Indirect w/ Offset)**                                                         |             |       |              |                                                       |              |
| `MV [r'₃±n], r₁`     | `[r'₃±n] ← r₁`                                                                                               | `- -`       | 3     | 6            | `1011 0 r` / `BX` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV [r'₃±n], r₂`     | `[r'₃±n] ← r₂ₗ`, `[r'₃±n+1] ← r₂ₕ`                                                                           | `- -`       | 3     | 7            | `1011 0 r` / `BX` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| `MV [r'₃±n], r₄`     | `[r'₃±n] ← r₄ₗ`, `[r'₃±n+1] ← r₄ₕ`, `[r'₃±n+2] ← r₄ₓ`                                                        | `- -`       | 3     | 8            | `1011 0 r` / `BX` <br> `1[0/1]00 0 r'₃` / `8X/CX` <br> `n` | `n`          |
| **MV [(n)], r**      | **To External Memory (Memory Indirect)**                                                                     |             |       |              |                                                       |              |
| `MV [(n)], r₁`       | `[(n)] ← r₁`                                                                                                 | `- -`       | 3     | 9            | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV [(n)], r₂`       | `[(n)] ← r₂ₗ`, `[(n)+1] ← r₂ₕ`                                                                               | `- -`       | 3     | 10           | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |
| `MV [(n)], r₄`       | `[(n)] ← r₄ₗ`, `[(n)+1] ← r₄ₕ`, `[(n)+2] ← r₄ₓ`                                                              | `- -`       | 3     | 11           | `1011 1 r` / `BX` <br> `0000 0000` / `00` <br> `n`      | `n`          |

