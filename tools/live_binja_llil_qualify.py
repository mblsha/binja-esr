#!/usr/bin/env python3
"""Qualify the current SC62015 checkout against a live Binary Ninja API.

Run this file through ``binja-cli python -f`` while any BinaryView is open.  It
loads the checkout under an isolated package name, registers an ephemeral
architecture name derived from the source hash, and exercises real Binary
Ninja instruction metadata, text, and LLIL builders.  It never changes or
saves the open BinaryView.

The ordinary test suite intentionally uses lightweight Binary Ninja mocks.
This harness is the complementary integration check that catches LLIL width,
intrinsic-signature, and finalization errors accepted by those mocks.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

from binaryninja import LowLevelILFunction


def _find_source_root() -> Path:
    explicit = os.environ.get("SC62015_SOURCE_ROOT")
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = []

    script_file = globals().get("__file__")
    if script_file:
        candidates.append(Path(script_file).resolve().parents[1])

    binary_view = globals().get("bv")
    if binary_view is not None:
        view_path = Path(binary_view.file.filename).resolve()
        for parent in view_path.parents:
            candidates.extend((parent, parent / "public-src"))

    for candidate in candidates:
        if (candidate / "sc62015" / "pysc62015" / "opcodes.txt").is_file():
            return candidate
    raise RuntimeError(
        "could not locate the current checkout; set SC62015_SOURCE_ROOT in "
        "Binary Ninja's environment or open a BinaryView beneath the paired "
        "private repository"
    )


SOURCE_ROOT = _find_source_root()
SC62015_ROOT = SOURCE_ROOT / "sc62015"
OPCODE_MANIFEST = SC62015_ROOT / "pysc62015" / "opcodes.txt"
TEST_ADDRESS = 0x1234
READ_LENGTH = 16
MAX_FUNCTION_ENTRY_SAMPLES = 128


def _load_current_architecture() -> tuple[Any, Any, str]:
    source_digest = hashlib.sha256(
        (SC62015_ROOT / "arch.py").read_bytes()
        + (SC62015_ROOT / "pysc62015" / "instr" / "opcodes.py").read_bytes()
        + (SC62015_ROOT / "pysc62015" / "instr" / "instructions.py").read_bytes()
    ).hexdigest()[:12]
    package_name = f"sc62015_live_qualification_{source_digest}"

    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            SC62015_ROOT / "__init__.py",
            submodule_search_locations=[str(SC62015_ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not construct the isolated SC62015 package")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    package = sys.modules[package_name]

    arch_module = importlib.import_module(f"{package_name}.arch")
    instr_module = importlib.import_module(f"{package_name}.pysc62015.instr")

    architecture_name = f"SC62015Qualification_{source_digest}_v2"
    architecture = getattr(package, "_qualification_architecture", None)
    if architecture is None:
        qualification_class = type(
            architecture_name,
            (arch_module.SC62015,),
            {"name": architecture_name},
        )
        architecture = qualification_class.register()
        package._qualification_architecture = architecture

    return architecture, instr_module, source_digest


def _manifest_rows() -> list[tuple[int, bytes, str]]:
    rows: list[tuple[int, bytes, str]] = []
    for line_number, raw_line in enumerate(
        OPCODE_MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        byte_text, separator, description = line.partition(":")
        if not separator:
            raise ValueError(
                f"malformed opcode manifest line {line_number}: {raw_line}"
            )
        rows.append((line_number, bytes.fromhex(byte_text), description.strip()))
    return rows


def _is_canonical(instr_module: Any, data: bytes, address: int) -> bool:
    try:
        decoded = instr_module.decode(data, address, instr_module.OPCODES)
        if decoded is None or isinstance(
            decoded, (instr_module.PRE, instr_module.UnknownInstruction)
        ):
            return False
        encoded = bytes(instr_module.encode(decoded, address))
        return encoded == data[: decoded.length()]
    except (AssertionError, instr_module.InvalidInstruction):
        return False


def _qualify_one(architecture: Any, data: bytes, address: int) -> dict[str, Any]:
    info = architecture.get_instruction_info(data, address)
    text = architecture.get_instruction_text(data, address)
    il = LowLevelILFunction(architecture)
    lifted_length = architecture.get_instruction_low_level_il(data, address, il)

    if info is None or text is None or lifted_length is None:
        return {
            "accepted": False,
            "info": info is not None,
            "text": text is not None,
            "llil": lifted_length is not None,
        }

    rendered_tokens, text_length = text
    rendered = "".join(token.text for token in rendered_tokens).strip()
    if not rendered:
        raise ValueError("Binary Ninja returned empty instruction text")
    if not (info.length == text_length == lifted_length):
        raise ValueError(
            "metadata/text/LLIL length mismatch: "
            f"{info.length}, {text_length}, {lifted_length}"
        )

    il.finalize()
    operations = [il[index].operation.name for index in range(len(il))]
    if "LLIL_UNIMPL" in operations:
        raise ValueError("real Binary Ninja LLIL contains LLIL_UNIMPL")

    return {
        "accepted": True,
        "length": lifted_length,
        "rendered": rendered,
        "llil_expressions": il.get_expr_count(),
        "llil_instructions": len(il),
        "llil_operations": operations,
    }


def _qualify_manifest(architecture: Any, instr_module: Any) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    llil_expressions = 0
    llil_instructions = 0
    operation_names: set[str] = set()

    for line_number, data, description in _manifest_rows():
        expected_accepted = _is_canonical(instr_module, data, TEST_ADDRESS)
        try:
            result = _qualify_one(architecture, data, TEST_ADDRESS)
            if result["accepted"] != expected_accepted:
                raise ValueError(
                    f"expected accepted={expected_accepted}, got {result['accepted']}"
                )
            if result["accepted"]:
                accepted += 1
                llil_expressions += result["llil_expressions"]
                llil_instructions += result["llil_instructions"]
                operation_names.update(result["llil_operations"])
            else:
                rejected += 1
        except Exception as exc:
            failures.append(
                {
                    "line": line_number,
                    "bytes": data.hex().upper(),
                    "description": description,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "rows": accepted + rejected + len(failures),
        "accepted": accepted,
        "rejected": rejected,
        "llil_expressions": llil_expressions,
        "llil_instructions": llil_instructions,
        "llil_operations": sorted(operation_names),
        "failures": failures,
    }


def _qualify_open_view(
    architecture: Any, instr_module: Any, binary_view: Any
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    addresses: set[int] = set()
    accepted = 0
    rejected = 0
    skipped_functions = 0

    for function in binary_view.functions:
        if getattr(function, "analysis_skipped", False):
            skipped_functions += 1
            continue
        addresses.add(function.start)

    candidate_addresses = sorted(addresses)
    if len(candidate_addresses) <= MAX_FUNCTION_ENTRY_SAMPLES:
        sampled_addresses = candidate_addresses
    else:
        final_index = len(candidate_addresses) - 1
        sampled_addresses = sorted(
            {
                candidate_addresses[
                    sample_index * final_index // (MAX_FUNCTION_ENTRY_SAMPLES - 1)
                ]
                for sample_index in range(MAX_FUNCTION_ENTRY_SAMPLES)
            }
        )

    for address in sampled_addresses:
        data = bytes(binary_view.read(address, READ_LENGTH))
        expected_accepted = _is_canonical(instr_module, data, address)
        try:
            result = _qualify_one(architecture, data, address)
            if result["accepted"] != expected_accepted:
                raise ValueError(
                    f"expected accepted={expected_accepted}, got {result['accepted']}"
                )
            if result["accepted"]:
                accepted += 1
            else:
                rejected += 1
        except Exception as exc:
            failures.append(
                {
                    "address": f"0x{address:05X}",
                    "bytes": data.hex().upper(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "functions": len(binary_view.functions),
        "analysis_skipped_functions": skipped_functions,
        "candidate_function_entry_addresses": len(candidate_addresses),
        "sampled_function_entry_addresses": len(sampled_addresses),
        "accepted": accepted,
        "rejected": rejected,
        "failures": failures,
    }


def main() -> None:
    binary_view = globals().get("bv")
    if binary_view is None:
        raise RuntimeError("run this script through binja-cli with an open BinaryView")

    architecture, instr_module, source_digest = _load_current_architecture()
    manifest_result = _qualify_manifest(architecture, instr_module)
    # LowLevelILInstruction wrappers retain their anonymous IL function.  Drop
    # the manifest batch before constructing the ROM-entry batch so the live
    # core never observes wrappers whose function has already been finalized.
    gc.collect()
    open_view_result = _qualify_open_view(architecture, instr_module, binary_view)

    result = {
        "source_root": str(SOURCE_ROOT),
        "source_digest": source_digest,
        "architecture": architecture.name,
        "binary_view": {
            "file": binary_view.file.filename,
            "view_type": binary_view.view_type,
            "architecture": binary_view.arch.name if binary_view.arch else None,
        },
        "manifest": manifest_result,
        "open_view": open_view_result,
    }
    result["passed"] = not (
        result["manifest"]["failures"] or result["open_view"]["failures"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if globals().get("bv") is not None:
    main()
