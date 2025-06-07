# Missing Instruction Support in the Assembler

This document catalogs classes of SC62015 instructions that are not yet recognized by the assembler grammar. The list can serve as a roadmap for implementing the remaining instruction forms.

## 1. Memory Transfer Instructions
Only a few `MV` forms are parsed today (`MV A,B`, `MV B,A`, and `MV (imem),(imem)` which requires a `PRE` prefix). All other data movement variants remain unhandled:

- **Move Immediate to Register** – e.g. `MV A, 0x42`, `MV BA, 0x1234`, `MV X, 0x56789`.
- **Move Memory to Register** – direct internal `(n)`, direct external `[lmn]`, register indirect external `[r'3]` and variations, and memory indirect external `[(n)]`.
- **Move Register to Memory** – direct internal, direct external, register indirect external, and memory indirect external forms.
- **Block and Multi-byte Moves** – instructions such as `MVW`, `MVP`, `MVL`, `MVLD`.
- **Register to Register Moves** – generic forms like `MV r2, r'2` or `MV r3, r'3`.

## 2. Arithmetic Instructions
No arithmetic operations are present in the grammar. The missing set includes:

- **Addition**: `ADD`, `ADC`.
- **Subtraction**: `SUB`, `SBC`.
- **Multi-byte Arithmetic**: `ADCL`, `SBCL`.
- **BCD Arithmetic**: `DADL`, `DSBL`.
- **Packed BCD Modify**: `PMDF`.

## 3. Program Flow Instructions
Besides `RET`, `RETI`, and `RETF`, jump instructions are absent:

- **Unconditional Jumps**: `JP`, `JR` and their far or register forms.
- **Conditional Jumps**: `JPZ`, `JPNZ`, `JPC`, `JPNC`.
- **Conditional Relative Jumps**: `JRZ`, `JRNZ`, `JRC`, `JRNC`.

## 4. Logical and Compare Instructions
None of the logical, test, or compare operations are handled:

- **Logical Ops**: `AND`, `OR`, `XOR` with all addressing modes.
- **Test**: `TEST` in all forms.
- **Compare**: `CMP`, `CMPW`, `CMPP`.

## 5. Increment, Decrement, and Exchange Instructions
Only `EX A,B` is implemented. Missing variants include:

- **Increment**: `INC` for registers and memory locations.
- **Decrement**: `DEC` for registers and memory locations.
- **Exchange**: `EX`, `EXW`, `EXP`, `EXL` with register and memory operands.

## 6. Shift and Rotate Instructions
The grammar supports the accumulator forms (`ROR A`, `ROL A`, `SHR A`, `SHL A`) but omits:

- **Memory Forms**: `ROR (n)`, `ROL (n)`, `SHR (n)`, `SHL (n)`.
- **Decimal Shifts**: `DSRL`, `DSLL`.

## 7. Stack Instructions
Stack operations are limited to the `F` and `IMR` registers. Missing user stack operations include:

- `PUSHU r`
- `POPU r`

for `A`, `IL`, `BA`, `I`, `X`, and `Y`.

