# SC62015 Assembler Improvement Proposal

This proposal focuses on the concrete assembler defects reproduced in the current tree and the smallest set of fixes that will make normal SC62015 source assemble reliably.

## Goals

- Make bare labels work naturally in immediate and memory operands.
- Make page-local `CALL` and `JP*` work correctly on high ROM pages.
- Improve error messages so assembler failures point at the real source line.
- Add regression coverage for the syntax we expect card ROM and test programs to use.

## Non-Goals

- Redesign the whole assembler language.
- Add a macro system.
- Change existing valid syntax unless there is no compatible alternative.

## Problem Summary

The current assembler has four practical issues:

1. Bare identifiers are ambiguous between labels and registers.
2. Bracketed identifiers are ambiguous between absolute external memory and register-indirect external memory.
3. `CALL`, `JP`, `JPZ`, `JPNZ`, `JPC`, and `JPNC` reject same-page targets above `0xFFFF` even though the instruction semantics already treat those forms as page-local.
4. Error reporting can blame the wrong line because blank lines and comment-only lines are dropped before pass two.

These issues force callers into avoidable workarounds such as hardcoded addresses and manually truncated jump targets.

## Proposed Fixes

### 1. Make register names explicit in the grammar

Current issue:

- `reg: CNAME` accepts any identifier as a register candidate.
- `MV X, message` is parsed as register-to-register instead of register-to-immediate.
- `JP loop` is routed through the register form first and only works because `jp_reg()` has a special fallback.

Proposal:

- Replace generic `reg: CNAME` with explicit register tokens for the real SC62015 register set:
  - `A`, `IL`, `BA`, `I`, `X`, `Y`, `U`, `S`
  - plus the existing special cases `B`, `F`, `IMR`
- Keep labels as generic `CNAME`.
- Remove parser ambiguity instead of depending on transformer-side recovery.

Implementation notes:

- Update `asm.lark` to use explicit register terminals or a dedicated `REGISTER` token.
- Update `AsmTransformer.reg()` to accept only real register names.
- Remove the symbol fallback in `jp_reg()` once bare labels no longer reach that rule.

Syntax this should enable:

```asm
message:
    defm "HELLO"
    defb 0

start:
    MV X, message
    MV BA, message
    JP start
    JPZ message_done
```

### 2. Disambiguate `[label]` as absolute external memory

Current issue:

- `[message_ptr]` is parsed as `[reg]` because `emem_reg_operand` competes with `emem_addr`, and `reg` currently accepts any identifier.
- This breaks normal pointer-table usage such as `MV X, [message_ptr]`.

Proposal:

- Once registers are explicit, make `[label]` parse as `emem_addr`.
- Keep `[X]`, `[Y]`, `[U]`, `[S]`, `[X++]`, `[--Y]`, and `[X+4]` as register-indirect forms.

Implementation notes:

- Fixing the register grammar should remove most of this ambiguity automatically.
- Add targeted tests for absolute-label and register-indirect bracket forms so the distinction stays stable.

Syntax this should enable:

```asm
message_ptr:
    defl message

message:
    defm "HELLO"
    defb 0

start:
    MV X, [message_ptr]
    MV A, [uart_data]
    MV [uart_data], A
```

### 3. Encode page-local `CALL` and `JP*` targets correctly

Current issue:

- The assembler builds near `CALL` and `JP*` with `Imm16`.
- If a label resolves to `0x10100`, encoding fails because `Imm16.encode()` rejects values above `0xFFFF`.
- The instruction semantics already treat these as page-local by combining the current page with the 16-bit operand.

Proposal:

- For 16-bit control-flow operands only, resolve the symbol in pass two and:
  - encode `target & 0xFFFF` when `target` is on the current `0xFF0000` page
  - raise a clear error when the target is on a different page, telling the user to use `CALLF` or `JPF`
- Do not silently truncate cross-page targets.

Implementation notes:

- Add a page-aware normalization step in the assembler before `Imm16.encode()` runs for near control-flow instructions.
- Keep `CALLF` and `JPF` as the explicit far forms.
- Add tests for same-page success and cross-page failure.

Syntax this should enable:

```asm
.ORG 0x10100

start:
    JP loop

loop:
    MV A, [X++]
    CMP A, 0
    JPZ done
    CALL emit_char
    JP loop

done:
    RET

emit_char:
    MV [0x1FFF1], A
    RET
```

Expected cross-page behaviour:

```asm
.ORG 0x10100
    JP same_page_label      ; valid
    CALL same_page_sub      ; valid
    JPF far_page_label      ; valid
    CALLF far_page_sub      ; valid

    ; JP far_page_label     ; assembler error: target is not on current page
    ; CALL far_page_sub     ; assembler error: use CALLF
```

### 4. Preserve original source line numbers in diagnostics

Current issue:

- The transformer drops blank and comment-only lines.
- Pass two reports errors using the compacted AST index against the original source text.
- This can point at a nearby `.ORG` or comment instead of the real failing instruction.

Proposal:

- Preserve the original source line number on every `LineNode`.
- Use that stored line number in both passes and in error reporting.

Implementation notes:

- Record line metadata in the transformer instead of rebuilding it later from list position.
- Include both line number and source text in `AssemblerError`.

Syntax this should make debuggable:

```asm
; comments and blank lines should not shift diagnostics

.ORG 0x10100

start:
    JP missing_label
```

Expected error shape:

```text
on line 6: Undefined symbol: missing_label
>     JP missing_label
```

### 5. Turn the missing regression coverage into real tests

Current issue:

- The current low-page label tests pass.
- The important high-page and ambiguous-label cases are not covered.
- `test_asm_e2e.py` returns before asserting anything, so it is not protecting behaviour today.

Proposal:

- Add focused tests for:
  - `MV X, label`
  - `MV BA, label`
  - `MV A, [label]`
  - `MV [label], A`
  - `MV X, [ptr_label]` where `ptr_label` is a `defl`
  - `.ORG 0x10100` with `JP label`, `JPZ label`, `CALL label`
  - cross-page rejection for near control-flow
  - line-accurate error reporting
- Remove the early `return` from `test_asm_e2e.py`.

Syntax this should keep working long-term:

```asm
table_ptr:
    defl table

table:
    defb 1, 2, 3, 0

start:
    MV X, [table_ptr]
read_next:
    MV A, [X++]
    CMP A, 0
    JPZ done
    JP read_next
done:
    RET
```

## Recommended Order

1. Fix register and `[label]` parsing.
2. Fix page-local control-flow encoding.
3. Fix diagnostics.
4. Add and enable regression tests.

This order keeps the parser and encoder changes small and makes each step independently testable.

## Acceptance Criteria

- `MV X, message` assembles without a workaround.
- `MV X, [message_ptr]` assembles when `message_ptr` is defined with `defl`.
- `JP label`, `JPZ label`, and `CALL label` assemble correctly on a high `.ORG` when the target is on the same page.
- Near control-flow to a different page fails with a specific far-jump/far-call hint.
- Reported source lines match the actual failing instruction even with comments and blank lines.
- The new syntax examples above are covered by automated tests.

## Optional Follow-On Work

These are useful, but not required for the fixes above:

- Add a small expression language for `label + 4`, `label - 1`, and similar relocatable forms.
- Add clearer wording around immediate versus memory operands in the assembler README.
- Add an assembler round-trip fixture based on the UART hello example so card ROM development keeps exercising the natural source style.
