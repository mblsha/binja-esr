# binja_helpers

Utility modules for developing and testing Binary Ninja plugins without requiring
Binary Ninja itself.  Importing ``binja_helpers.binja_api`` provides a minimal
stub of the ``binaryninja`` Python API when Binary Ninja is not installed.

## Usage

1. **Import the stub helper**

   Import ``binja_helpers.binja_api`` before any ``binaryninja`` imports.  The
   module attempts to locate a local Binary Ninja installation and installs a
   lightweight stub when none is found.  This makes unit tests runnable on
   systems without a Binary Ninja license.

2. **Construct mock LLIL**

   ``mock_llil`` implements ``MockLowLevelILFunction`` and helper constructors
   like ``mllil`` and ``mreg``.  They mirror the real LLIL API closely enough to
   let tests build and inspect instructions.  Example from the SC62015
   architecture tests:

   ```python
   from binja_helpers import binja_api  # noqa: F401
   from binja_helpers.mock_llil import MockLowLevelILFunction
   from sc62015.arch import SC62015

   def test_get_instruction_low_level_il_with_bytes() -> None:
       arch = SC62015()
       il = MockLowLevelILFunction()
       length = arch.get_instruction_low_level_il(b"\x00", 0, il)
       assert length == 1
       assert il.ils[0].op == "NOP"
   ```

3. **Verify analysis information**

   ``mock_analysis.MockAnalysisInfo`` mimics ``InstructionInfo`` so branch
   targets and instruction lengths can be checked offline:

   ```python
   info = MockAnalysisInfo()
   instr.analyze(info, 0x1000)
   assert info.mybranches == [(BranchType.FalseBranch, 0x1002)]
   ```

4. **Evaluate LLIL**

   ``eval_llil`` contains a small interpreter for mock LLIL expressions.  It is
   useful for emulator-style tests:

   ```python
   from binja_helpers.eval_llil import evaluate_llil, Memory, State

   result, flags = evaluate_llil(il_expr, regs, Memory(read_mem, write_mem), State())
   ```

5. **Compare disassembly**

   Use ``asm_str`` to convert token lists returned by ``render()`` into plain
   strings for assertions.  This mirrors how the SC62015 tests verify
   decoding results:

   ```python
   from binja_helpers.tokens import asm_str  # noqa: F401

   instr = arch.decode_instruction(0x00)
   assert asm_str(instr.render()) == "NOP"
   ```

The SC62015 plugin in this repository uses these helpers extensively to test its
instruction lifter and emulator logic without depending on a Binary Ninja
install.  New plugins can adopt the same approach to keep their unit tests fast
and self-contained.
