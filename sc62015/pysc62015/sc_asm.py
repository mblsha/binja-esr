#!/usr/bin/env python3
import sys
import copy
from typing import Dict, List, Any, Optional, cast, Tuple

import bincopy  # type: ignore[import-untyped]
from plumbum import cli  # type: ignore[import-untyped]

# Assuming the provided library files are in a package named 'sc62015'
from .asm import AsmTransformer, asm_parser, ParsedInstruction
from binja_test_mocks.coding import Encoder
from .instr.opcode_table import OPCODES
from .instr import (
    Instruction,
    Opts,
    IMemOperand,
    IMem8,
    ImmOperand,
    Imm20,
    ImmOffset,
    EMemReg,
    RegIMemOffset,
    EMemIMemOffset,
    Reg3,
    RegPair,
    REVERSE_PRE_TABLE,
    SINGLE_OPERAND_PRE_LOOKUP,
    AddressingMode,
)

# A simple cache for the reverse lookup table
REVERSE_OPCODES_CACHE: Dict[str, List[Dict[str, Any]]] = {}


class AssemblerError(Exception):
    pass


class Assembler:
    """
    A two-pass assembler for the SC62015 architecture that understands sections.
    """

    # --- Sane Defaults for Section Base Addresses ---
    SECTION_BASE_ADDRESSES: Dict[str, int] = {
        "code": 0x00000,  # Executable code in ROM
        "text": 0x00000,  # Alias for .code
        "data": 0x80000,  # Initialized R/W data in RAM
        "bss": 0x90000,  # Uninitialized R/W data in RAM (a default, will follow .data)
    }

    DEFAULT_SECTION = "code"

    def __init__(self) -> None:
        self.symbols: Dict[str, int] = {}
        self.section_pointers: Dict[str, int] = {}
        self.current_address: int = 0
        # Cache built instructions from pass 1 to use in pass 2
        self.instructions_cache: Dict[int, Instruction] = {}

        if not REVERSE_OPCODES_CACHE:
            REVERSE_OPCODES_CACHE.update(self._build_reverse_opcodes())
        self._reverse_opcodes = REVERSE_OPCODES_CACHE

    def _build_reverse_opcodes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Creates a reverse lookup table from mnemonic to instruction templates."""
        reverse_map: Dict[str, List[Dict[str, Any]]] = {}
        for opcode, definition in OPCODES.items():
            cls, opts = (
                definition if isinstance(definition, tuple) else (definition, Opts())
            )
            name = (opts.name or cls.__name__.split("_")[0]).upper()
            if name not in reverse_map:
                reverse_map[name] = []
            reverse_map[name].append({"opcode": opcode, "class": cls, "opts": opts})
        return reverse_map

    def _evaluate_operand(self, value: str) -> int:
        """Evaluates a string that can be a number or a symbol."""
        value = value.strip()
        key = value.upper()
        if key in self.symbols:
            return self.symbols[key]
        try:
            # Allow hex (0x...), binary (0b...), octal (0o...), and decimal
            return int(value, 0)
        except ValueError:
            raise AssemblerError(f"Undefined symbol or invalid number: {value}")

    def _apply_location(
        self,
        stmt: Dict[str, Any],
        pointers: Dict[str, int],
        current_section: str,
        first_pass: bool,
    ) -> Tuple[str, bool]:
        """Update section/address based on SECTION or ORG directives.

        Returns the possibly updated section name and a flag indicating the
        statement was fully handled (and should not be processed further).
        """

        consumed = False

        if stmt.get("section"):
            new_section = stmt["section"].lower()
            if first_pass:
                if new_section not in pointers:
                    last_addr = max(pointers.values()) if pointers else 0
                    pointers[new_section] = last_addr
            else:
                if new_section not in pointers:
                    raise AssemblerError(
                        f"Undefined section '{new_section}' encountered in second pass."
                    )
            current_section = new_section
            consumed = True

        if stmt.get("type") == "org":
            if first_pass:
                try:
                    new_addr = int(str(stmt["args"]), 0)
                except ValueError:
                    new_addr = 0
            else:
                new_addr = self._evaluate_operand(str(stmt["args"]))
                self.current_address = new_addr

            pointers[current_section] = new_addr
            consumed = True

        return current_section, consumed

    def _build_instruction(self, parsed_instr: ParsedInstruction) -> Instruction:
        """Builds an Instruction object from a parsed instruction node from the AST."""
        instr_class = parsed_instr["instr_class"]
        instr_opts = parsed_instr["instr_opts"]

        mnemonic = (instr_opts.name or instr_class.__name__.split("_")[0]).upper()
        provided_ops = instr_opts.ops or []
        provided_cond = instr_opts.cond

        if mnemonic not in self._reverse_opcodes:
            raise AssemblerError(f"Unknown mnemonic: {mnemonic}")

        # For the unambiguous grammar, we can match on class and operand representation
        for template in self._reverse_opcodes[mnemonic]:
            if template["class"] is not instr_class:
                continue

            if provided_cond is not None and template["opts"].cond != provided_cond:
                continue

            template_ops = template["opts"].ops or []

            # Compare operands. Using repr is a simple way for the basic operand types.
            if len(provided_ops) == len(template_ops):
                converted_match = True
                for p_op, t_op in zip(provided_ops, template_ops):
                    if isinstance(t_op, IMem8) and isinstance(p_op, IMemOperand):
                        continue
                    if isinstance(t_op, ImmOffset) and isinstance(p_op, ImmOffset):
                        if t_op.sign != p_op.sign:
                            converted_match = False
                            break
                        continue
                    if isinstance(t_op, RegIMemOffset) and isinstance(p_op, RegIMemOffset):
                        if t_op.order != p_op.order:
                            converted_match = False
                            break
                        allowed_modes = t_op.allowed_modes
                        if allowed_modes is not None and p_op.mode not in allowed_modes:
                            converted_match = False
                            break
                        continue
                    if isinstance(t_op, EMemReg) and isinstance(p_op, EMemReg):
                        if t_op.width != p_op.width:
                            converted_match = False
                            break
                        # Template does not specify the addressing mode, so any
                        # provided mode should be accepted when ``t_op.mode`` is
                        # ``None``.
                        t_mode = getattr(t_op, "mode", None)
                        p_mode = getattr(p_op, "mode", None)
                        if t_mode is not None and t_mode != p_mode:
                            converted_match = False
                            break
                        allowed_modes = getattr(t_op, "allowed_modes", None)
                        if allowed_modes is not None and p_mode not in allowed_modes:
                            converted_match = False
                            break
                        continue
                    if isinstance(t_op, EMemIMemOffset) and isinstance(p_op, EMemIMemOffset):
                        if t_op.order != p_op.order:
                            converted_match = False
                            break
                        continue
                    if (
                        isinstance(t_op, ImmOperand)
                        and isinstance(p_op, ImmOperand)
                        and not isinstance(t_op, ImmOffset)
                    ):
                        if type(p_op) is type(t_op):
                            continue
                    if isinstance(t_op, RegPair) and isinstance(p_op, RegPair):
                        if (
                            t_op.size is not None
                            and p_op.size is not None
                            and t_op.size != p_op.size
                        ):
                            converted_match = False
                            break
                        continue
                    if isinstance(t_op, Reg3) and isinstance(p_op, Reg3):
                        continue
                    if repr(p_op) != repr(t_op):
                        converted_match = False
                        break
                if converted_match:
                    instr = instr_class(
                        name=mnemonic,
                        operands=provided_ops,
                        cond=template["opts"].cond,
                        ops_reversed=template["opts"].ops_reversed,
                    )
                    instr.opcode = template["opcode"]

                    # Determine if a PRE prefix byte is required based on
                    # internal memory operands.
                    operands_list = list(instr.operands())
                    imem_ops = [op for op in operands_list if isinstance(op, IMemOperand)]

                    pre_byte: Optional[int] = None
                    if len(imem_ops) == 2:
                        pre_byte = REVERSE_PRE_TABLE.get((imem_ops[0].mode, imem_ops[1].mode))
                        if pre_byte is None:
                            raise AssemblerError(
                                f"Invalid addressing mode combination for {mnemonic}: "
                                f"{imem_ops[0].mode.value} and {imem_ops[1].mode.value}"
                            )
                    elif len(imem_ops) == 1:
                        mode = imem_ops[0].mode
                        if mode != AddressingMode.N:
                            pre_byte = SINGLE_OPERAND_PRE_LOOKUP.get(mode)
                            if pre_byte is None:
                                raise AssemblerError(
                                    f"Unsupported addressing mode {mode.value} for {mnemonic}"
                                )

                    if pre_byte is not None:
                        instr._pre = pre_byte

                    # Use a copy for length calculation so that symbolic
                    # operands remain unresolved for pass two.
                    instr_for_size = copy.deepcopy(instr)
                    if pre_byte is not None:
                        instr_for_size._pre = pre_byte
                    for op in instr_for_size.operands():
                        if hasattr(op, "value") and isinstance(getattr(op, "value"), str):
                            try:
                                setattr(op, "value", int(getattr(op, "value"), 0))
                            except ValueError:
                                setattr(op, "value", 0)
                        offset = getattr(op, "offset", None)
                        if (
                            isinstance(offset, ImmOffset)
                            and isinstance(offset.value, str)
                        ):
                            try:
                                offset.value = int(offset.value, 0)
                            except ValueError:
                                offset.value = 0
                        if hasattr(op, "extra_hi") and getattr(op, "value", None) is not None:
                            setattr(op, "extra_hi", (int(getattr(op, "value")) >> 16) & 0xFF)
                        if isinstance(op, Imm20) and isinstance(op.value, int):
                            op.extra_hi = (op.value >> 16) & 0xFF

                    encoder = Encoder()
                    try:
                        instr_for_size.encode(encoder, 0)  # address doesn't matter
                    except ValueError as e:
                        raise AssemblerError(str(e)) from e
                    instr.set_length(len(encoder.buf))

                    return instr

        raise AssemblerError(
            f"Could not find a matching opcode for {mnemonic} with operands {provided_ops}"
        )

    def _get_statement_size(self, statement: Dict[str, Any], line_num: int) -> int:
        """Calculates the size of a statement in bytes in the first pass."""
        stmt_type = statement.get("type")
        args = statement.get("args")
        if stmt_type == "defs":
            return int(str(args))
        elif stmt_type == "defb":
            return sum(
                len(arg[1:-1]) if isinstance(arg, str) and arg.startswith('"') else 1
                for arg in (args or [])
            )
        elif stmt_type == "defw":
            return len(args or []) * 2
        elif stmt_type == "defl":
            return len(args or []) * 3
        elif stmt_type == "defm":
            return len(args or "")
        elif "instruction" in statement:
            instr = self._build_instruction(
                cast(ParsedInstruction, statement["instruction"])
            )
            self.instructions_cache[line_num] = instr  # Cache for second pass
            return instr.length()
        return 0

    def _first_pass(self, program_ast: Dict[str, Any]) -> None:
        """Pass 1: Build the symbol table and calculate section sizes."""
        self.symbols = {}
        self.section_pointers = self.SECTION_BASE_ADDRESSES.copy()
        current_section = self.DEFAULT_SECTION
        self.instructions_cache.clear()

        for i, line in enumerate(program_ast["lines"]):
            consumed = False
            if "statement" in line:
                current_section, consumed = self._apply_location(
                    line["statement"],
                    self.section_pointers,
                    current_section,
                    True,
                )

            addr = self.section_pointers[current_section]

            if "label" in line:
                label_name = line["label"].upper()
                if label_name in self.symbols:
                    raise AssemblerError(f"Duplicate label definition: {line['label']}")
                self.symbols[label_name] = addr

            if "statement" in line and not consumed:
                stmt = line["statement"]
                size = self._get_statement_size(stmt, i)
                self.section_pointers[current_section] += size

        # Layout .bss after .data if both exist
        if "data" in self.section_pointers and "bss" in self.section_pointers:
            self.section_pointers["bss"] = self.section_pointers["data"]

    def _second_pass(self, program_ast: Dict[str, Any]) -> bincopy.BinFile:
        """Pass 2: Generate machine code using the symbol table."""
        binfile = bincopy.BinFile()
        current_section_pointers = self.SECTION_BASE_ADDRESSES.copy()
        if "data" in self.section_pointers and "bss" in current_section_pointers:
            current_section_pointers["bss"] = self.section_pointers["data"]

        current_section = self.DEFAULT_SECTION

        for i, line in enumerate(program_ast["lines"]):
            consumed = False
            if "statement" in line:
                current_section, consumed = self._apply_location(
                    line["statement"],
                    current_section_pointers,
                    current_section,
                    False,
                )

            self.current_address = current_section_pointers[current_section]

            if "statement" in line and not consumed:
                stmt = line["statement"]
                try:
                    encoded_bytes = self._encode_statement(stmt, i)
                    if encoded_bytes:
                        if current_section == "bss":
                            # .bss section only reserves space, no data in file
                            pass
                        else:
                            binfile.add_binary(encoded_bytes, self.current_address)

                    current_section_pointers[current_section] += len(encoded_bytes)
                except Exception as e:
                    source_line = program_ast["source_text"].splitlines()[i]
                    raise AssemblerError(f"on line {i+1}: {e}\n> {source_line}") from e
        return binfile

    def _encode_statement(self, statement: Dict[str, Any], line_num: int) -> bytearray:
        """Encodes a single statement into a bytearray."""
        if "instruction" in statement:
            # Retrieve the already-built instruction from the cache
            instr = self.instructions_cache[line_num]

            for op in instr.operands():
                if hasattr(op, "value") and isinstance(getattr(op, "value"), str):
                    val = self._evaluate_operand(getattr(op, "value"))
                    setattr(op, "value", val)
                    if hasattr(op, "extra_hi"):
                        setattr(op, "extra_hi", (val >> 16) & 0xFF)
                if hasattr(op, "value"):
                    val = getattr(op, "value")
                    if isinstance(val, str):
                        val = self._evaluate_operand(val)
                        setattr(op, "value", val)
                    if isinstance(op, Imm20) and isinstance(val, int):
                        op.extra_hi = (val >> 16) & 0xFF
                offset = getattr(op, "offset", None)
                if (
                    isinstance(offset, ImmOffset)
                    and isinstance(offset.value, str)
                ):
                    offset.value = self._evaluate_operand(offset.value)
            encoder = Encoder()
            instr.encode(encoder, self.current_address)
            return encoder.buf

        stmt_type = statement.get("type")
        if not stmt_type:
            return bytearray()

        args = statement["args"]
        if stmt_type == "org":
            return bytearray()
        if stmt_type == "defs":
            size = self._evaluate_operand(str(args))
            return bytearray(size)
        elif stmt_type == "defm":
            if isinstance(args, str):
                return bytearray(args.encode("ascii"))
            return bytearray()

        encoder = Encoder()
        for arg in args:
            if isinstance(arg, str) and arg.startswith('"') and arg.endswith('"'):
                for char in arg[1:-1]:
                    encoder.unsigned_byte(ord(char))
            else:
                value = self._evaluate_operand(arg)
                if stmt_type == "defb":
                    encoder.unsigned_byte(value & 0xFF)
                elif stmt_type == "defw":
                    encoder.unsigned_word_le(value & 0xFFFF)
                elif stmt_type == "defl":
                    encoder.unsigned_byte(value & 0xFF)
                    encoder.unsigned_byte((value >> 8) & 0xFF)
                    encoder.unsigned_byte((value >> 16) & 0xFF)
        return encoder.buf

    def assemble(self, source_text: str) -> bincopy.BinFile:
        """Assembles the given source text."""
        if not source_text.endswith("\n"):
            source_text += "\n"

        try:
            tree = asm_parser.parse(source_text)
            program_ast = AsmTransformer().transform(tree)
        except Exception as e:
            raise AssemblerError(f"Parsing failed: {e}") from e

        program_ast["source_text"] = source_text

        self._first_pass(program_ast)
        bin_file = self._second_pass(program_ast)

        return bin_file


class SC62015_AssemblerCLI(cli.Application):
    """An assembler for the SC62015 microprocessor that outputs Intel HEX format."""

    PROGNAME = "sc_asm"
    VERSION = "1.0.0"

    output_file = cli.SwitchAttr(
        ["-o", "--output"], str, help="Output file path for ihex (default: stdout)"
    )

    def main(self, input_file: cli.ExistingFile) -> None:  # type: ignore[type-arg]
        """Main entry point for the CLI application."""
        print(f"Assembling '{input_file}'...")
        try:
            with open(input_file, "r") as f:
                source_code = f.read()

            assembler = Assembler()
            bin_file = assembler.assemble(source_code)
            ihex_data = bin_file.as_ihex()

            if self.output_file:
                with open(self.output_file, "w") as f:
                    f.write(ihex_data)
                print(f"Assembly successful. Output written to '{self.output_file}'.")
            else:
                print("\n--- BEGIN IHEX OUTPUT ---")
                print(ihex_data.strip())
                print("--- END IHEX OUTPUT ---")

        except AssemblerError as e:
            print(f"\nAssembly Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    SC62015_AssemblerCLI.run()
