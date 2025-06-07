#!/usr/bin/env python3
import re
import sys
from typing import Dict, List, Any

import bincopy  # type: ignore[import-untyped]
from plumbum import cli  # type: ignore[import-untyped]

# Assuming the provided library files are in a package named 'sc62015'
from .asm import AsmTransformer, asm_parser
from .coding import Encoder
from .instr import REG_NAMES
from .instr import (
    Instruction,
    OPCODES,
    Operand,
    Opts,
    Reg,
    Reg3,
    EMemAddr,
    Imm8,
    Imm16,
    Imm20,
    IMem8,
    IMem16,
    IMem20,
    encode,
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
        if value in self.symbols:
            return self.symbols[value]
        try:
            # Allow hex (0x...), binary (0b...), octal (0o...), and decimal
            return int(value, 0)
        except ValueError:
            raise AssemblerError(f"Undefined symbol or invalid number: {value}")

    def _parse_and_build_instruction(self, instr_text: str) -> Instruction:
        """Parses a raw instruction string and attempts to build an Instruction object."""
        parts = re.split(r"[, ]+", instr_text, 1)
        mnemonic = parts[0].upper()
        operands_str = parts[1] if len(parts) > 1 else ""
        operands = (
            [op.strip() for op in operands_str.split(",")] if operands_str else []
        )

        if mnemonic not in self._reverse_opcodes:
            raise AssemblerError(f"Unknown mnemonic: {mnemonic}")

        for template in self._reverse_opcodes[mnemonic]:
            opts = template["opts"]
            op_templates: List[Operand] = opts.ops or []

            if len(op_templates) != len(operands):
                continue

            try:
                built_operands: List[Operand] = []
                for op_str, op_template in zip(operands, op_templates):
                    val = self._evaluate_operand(op_str)
                    instance: Operand

                    if isinstance(op_template, EMemAddr):
                        instance = EMemAddr(width=op_template.width())
                        instance.value = val
                        instance.extra_hi = (val >> 16) & 0xF
                        built_operands.append(instance)
                    elif isinstance(op_template, Imm20):
                        instance = Imm20()
                        instance.value = val
                        instance.extra_hi = (val >> 16) & 0xF
                        built_operands.append(instance)
                    elif isinstance(op_template, (Imm8, IMem8, Imm16, IMem16, IMem20)):
                        instance = op_template.__class__()
                        instance.value = val
                        built_operands.append(instance)
                    elif isinstance(op_template, Reg3):

                        instance = Reg3()
                        try:
                            op_str_upper = op_str.upper()
                            # Find the RegisterName object from the list of available registers
                            reg_enum = next(
                                r for r in REG_NAMES if str(r) == op_str_upper
                            )
                            idx = REG_NAMES.index(reg_enum)
                            instance.reg = reg_enum
                            instance.reg_raw = idx
                            instance.high4 = 0
                        except StopIteration:
                            raise AssemblerError(
                                f"Invalid register name for instruction: {op_str}"
                            ) from None
                        built_operands.append(instance)
                    elif isinstance(op_template, Reg):
                        instance = Reg(op_str.upper())
                        built_operands.append(instance)
                    else:
                        raise NotImplementedError(
                            f"Operand template not implemented: {type(op_template)}"
                        )

                instr: Instruction = template["class"](
                    name=mnemonic,
                    operands=built_operands,
                    cond=opts.cond,
                    ops_reversed=opts.ops_reversed,
                )
                instr.opcode = template["opcode"]
                return instr
            except (ValueError, NotImplementedError, AssemblerError):
                continue
        raise AssemblerError(f"No valid operand combination found for: {instr_text}")

    def _get_statement_size(self, statement: Dict[str, Any]) -> int:
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
            try:
                instr = self._parse_and_build_instruction(statement["instruction"])
                return len(encode(instr, 0))
            except AssemblerError:
                # Can't resolve symbols yet, assume max size.
                return 5
        return 0

    def _first_pass(self, program_ast: Dict[str, Any]) -> None:
        """Pass 1: Build the symbol table and calculate section sizes."""
        self.symbols = {}
        self.section_pointers = self.SECTION_BASE_ADDRESSES.copy()
        current_section = self.DEFAULT_SECTION

        for line in program_ast["lines"]:
            if "statement" in line and line["statement"].get("section"):
                current_section = line["statement"]["section"].lower()
                if current_section not in self.section_pointers:
                    # If a new section is found, place it after the previous one
                    last_addr = (
                        max(self.section_pointers.values())
                        if self.section_pointers
                        else 0
                    )
                    self.section_pointers[current_section] = last_addr

            addr = self.section_pointers[current_section]

            if "label" in line:
                if line["label"] in self.symbols:
                    raise AssemblerError(f"Duplicate label definition: {line['label']}")
                self.symbols[line["label"]] = addr

            if "statement" in line:
                size = self._get_statement_size(line["statement"])
                self.section_pointers[current_section] += size

        # Layout .bss after .data if both exist
        if "data" in self.section_pointers and "bss" in self.SECTION_BASE_ADDRESSES:
            data_end_addr = self.section_pointers["data"]
            self.section_pointers["bss"] = data_end_addr

    def _second_pass(self, program_ast: Dict[str, Any]) -> bincopy.BinFile:
        """Pass 2: Generate machine code using the symbol table."""
        binfile = bincopy.BinFile()

        # Reset pointers to their base addresses for code generation
        self.section_pointers = self.SECTION_BASE_ADDRESSES.copy()
        if "data" in self.section_pointers and "bss" in self.SECTION_BASE_ADDRESSES:
            # Recalculate data section size to place bss correctly
            data_size = sum(
                self._get_statement_size(line["statement"])
                for line in program_ast["lines"]
                if line.get("statement", {}).get("section", self.DEFAULT_SECTION)
                == "data"
            )
            self.section_pointers["bss"] = (
                self.SECTION_BASE_ADDRESSES["data"] + data_size
            )

        current_section = self.DEFAULT_SECTION

        for i, line in enumerate(program_ast["lines"]):
            if "statement" in line and line["statement"].get("section"):
                current_section = line["statement"]["section"].lower()
                if current_section not in self.section_pointers:
                    raise AssemblerError(
                        f"Undefined section '{current_section}' encountered in second pass."
                    )

            self.current_address = self.section_pointers[current_section]

            if "statement" in line:
                try:
                    encoded_bytes = self._encode_statement(line["statement"])
                    if encoded_bytes:
                        if current_section == "bss":
                            # .bss section only reserves space, no data in file
                            pass
                        else:
                            binfile.add_binary(encoded_bytes, self.current_address)

                    self.section_pointers[current_section] += len(encoded_bytes)
                except Exception as e:
                    raise AssemblerError(
                        f"on line {i+1}: {e}\n> {program_ast['source_text'].splitlines()[i]}"
                    )
        return binfile

    def _encode_statement(self, statement: Dict[str, Any]) -> bytearray:
        """Encodes a single statement into a bytearray."""
        stmt_type = statement.get("type")
        if "instruction" in statement:
            instr = self._parse_and_build_instruction(statement["instruction"])
            return encode(instr, self.current_address)

        if not stmt_type:
            return bytearray()

        args = statement["args"]
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

        tree = asm_parser.parse(source_text)
        program_ast = AsmTransformer().transform(tree)
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

    def main(self, input_file: cli.ExistingFile) -> None:
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
