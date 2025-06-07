# Missing Instruction Support in the Assembler

This document catalogs classes of SC62015 instructions that are not yet recognized by the assembler grammar. The list can serve as a roadmap for implementing the remaining instruction forms.

## 1. Memory Transfer Instructions
Only a few `MV` forms are parsed today (`MV A,B`, `MV B,A`, and `MV (imem),(imem)` which requires a `PRE` prefix). All other data movement variants remain unhandled:

- **Move Immediate to Register** – e.g. `MV A, 0x42`, `MV BA, 0x1234`, `MV X, 0x56789`.
- **Move Memory to Register** – direct internal `(n)`, direct external `[lmn]`, register indirect external `[r'3]` and variations, and memory indirect external `[(n)]`.
- **Move Register to Memory** – direct internal, direct external, register indirect external, and memory indirect external forms.
- **Block and Multi-byte Moves** – instructions such as `MVW`, `MVP`, `MVL`, `MVLD`.
- **Register to Register Moves** – generic forms like `MV r2, r'2` or `MV r3, r'3`.

## 2. Arithmetic Instructions: supposedly all implemented

## 3. Program Flow Instructions
All jump, call, and return instructions are now supported by the assembler.

## 4. Logical and Compare Instructions
All logical, test, and compare instructions are now implemented by the assembler.

## 5. Increment, Decrement, and Exchange Instructions
`EX A,B`, register-to-register forms, and memory-to-memory forms are implemented.
Missing variants include:

- **Exchange**: register-to-memory combinations for `EX`, `EXW`, `EXP`, `EXL`.

## 6. Shift and Rotate Instructions
All rotate, shift, and decimal shift forms are now supported by the assembler.

## 7. Stack Instructions
All user stack operations are now supported in the assembler.

