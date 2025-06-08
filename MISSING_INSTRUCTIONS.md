# Missing Instruction Support in the Assembler

This document catalogs classes of SC62015 instructions that are not yet recognized by the assembler grammar. The list can serve as a roadmap for implementing the remaining instruction forms.

## 1. Memory Transfer Instructions
All `MV` variants are now recognized by the assembler. This includes immediate
loads into registers, register/memory transfers, external memory forms, block
and multi-byte moves (`MVW`, `MVP`, `MVL`, `MVLD`), and generic register to
register moves.

## 2. Arithmetic Instructions: supposedly all implemented

## 3. Program Flow Instructions
All jump, call, and return instructions are now supported by the assembler.

## 4. Logical and Compare Instructions
All logical, test, and compare instructions are now implemented by the assembler.

## 5. Increment, Decrement, and Exchange Instructions
`EX A,B`, register-to-register forms, and memory-to-memory forms are implemented.
Earlier versions of this document incorrectly listed "register-to-memory"
exchange instructions as missing. The SC62015 architecture does not provide
such opcodes, so there is nothing to implement in the assembler.

## 6. Shift and Rotate Instructions
All rotate, shift, and decimal shift forms are now supported by the assembler.

## 7. Stack Instructions
All user stack operations are now supported in the assembler.

