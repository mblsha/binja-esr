#!/usr/bin/env python3
"""Compare two Perfetto traces and report the earliest SC62015 divergence."""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sc62015.pysc62015.constants import (  # noqa: E402
    INTERNAL_MEMORY_LENGTH,
    INTERNAL_MEMORY_START,
)

from retrobus_perfetto.proto import perfetto_pb2  # noqa: E402


@dataclass(frozen=True)
class TraceEvent:
    """Simplified representation of a Perfetto TrackEvent."""

    track: str
    name: str
    timestamp: int
    annotations: Dict[str, object]


def _annotation_value(annotation: perfetto_pb2.DebugAnnotation) -> object | None:
    """Extract the user-supplied value from a DebugAnnotation."""

    # Resolve interned names/values if present to handle traces that use iid/name_iid.
    name = annotation.name or ""
    if not name and annotation.HasField("name_iid"):
        name = annotation.name_iid

    if annotation.HasField("int_value"):
        return annotation.int_value
    if annotation.HasField("uint_value"):
        return annotation.uint_value
    if annotation.HasField("pointer_value"):
        return annotation.pointer_value
    if annotation.HasField("double_value"):
        return annotation.double_value
    if annotation.HasField("bool_value"):
        return annotation.bool_value
    if annotation.HasField("string_value"):
        return annotation.string_value
    if annotation.HasField("string_value_iid"):
        return annotation.string_value_iid
    if annotation.HasField("legacy_json_value"):
        return annotation.legacy_json_value
    return None


def _load_trace(path: Path) -> List[TraceEvent]:
    """Load and decode all TrackEvents from a Perfetto trace."""

    trace = perfetto_pb2.Trace()
    trace.ParseFromString(path.read_bytes())

    track_names: Dict[int, str] = {}
    name_intern: Dict[int, str] = {}
    value_intern: Dict[int, object] = {}
    events: List[TraceEvent] = []

    has_interned = "interned_data" in perfetto_pb2.TracePacket.DESCRIPTOR.fields_by_name

    for packet in trace.packet:
        if has_interned and packet.HasField("interned_data"):
            for entry in packet.interned_data.debug_annotation_name:
                if entry.HasField("iid") and entry.HasField("name"):
                    name_intern[entry.iid] = entry.name
            for entry in packet.interned_data.debug_annotation_string_value:
                if entry.HasField("iid") and entry.HasField("string_value"):
                    value_intern[entry.iid] = entry.string_value

        if packet.HasField("track_descriptor"):
            desc = packet.track_descriptor
            name = (
                desc.thread.thread_name
                or desc.name
                or desc.process.process_name
                or f"track_{desc.uuid}"
            )
            track_names[desc.uuid] = name

        if packet.HasField("track_event"):
            track_uuid = packet.track_event.track_uuid
            track_name = track_names.get(track_uuid, f"track_{track_uuid}")
            annotations = {}
            for ann in packet.track_event.debug_annotations:
                name = ann.name
                if not name:
                    name = name_intern.get(ann.name_iid, "")
                if not name:
                    continue
                value = _annotation_value(ann)
                if isinstance(value, int) and ann.HasField("string_value_iid"):
                    value = value_intern.get(value, value)
                annotations[name] = value
            events.append(
                TraceEvent(
                    track=track_name,
                    name=packet.track_event.name,
                    timestamp=packet.timestamp,
                    annotations=annotations,
                )
            )

    # Sort to guarantee deterministic comparison even if timestamps match
    events.sort(key=lambda evt: (evt.timestamp, evt.track, evt.name))
    return events


def _index_instruction_events(events: Sequence[TraceEvent]) -> Dict[int, TraceEvent]:
    """Map op_index -> TraceEvent for InstructionTrace events."""

    indexed: Dict[int, TraceEvent] = {}
    fallback_index = 0
    for evt in events:
        if evt.track != "InstructionTrace":
            continue
        op_index = evt.annotations.get("op_index")
        if op_index is None:
            # Fallback: assign sequential index when op_index annotation is missing.
            op_index = fallback_index
            fallback_index += 1
        if isinstance(op_index, int):
            indexed[op_index] = evt
    return indexed


def _format_reg_dump(annotations: Dict[str, object]) -> str:
    """Pretty-print register annotations for diagnostics."""

    regs = sorted(
        (key, value) for key, value in annotations.items() if key.startswith("reg_")
    )
    if not regs:
        return "<no registers>"
    parts = []
    for key, value in regs:
        if isinstance(value, int):
            parts.append(f"{key[4:].upper()}=0x{value:X}")
        else:
            parts.append(f"{key[4:].upper()}={value}")
    return " ".join(parts)


def _memory_write_events(events: Sequence[TraceEvent]) -> List[TraceEvent]:
    """Filter Perfetto events down to internal-memory writes."""

    filtered = [
        evt
        for evt in events
        if evt.track in {"MemoryWrites", "Memory_Internal", "Memory"}
    ]
    filtered.sort(key=lambda evt: evt.timestamp)
    return filtered


def _irq_events(events: Sequence[TraceEvent]) -> List[TraceEvent]:
    """Filter events to IRQ/timer related tracks."""

    filtered = [
        evt
        for evt in events
        if evt.track.startswith("irq")
        or evt.name.startswith(("MTI", "STI", "KEYI", "TimerFired", "IRQ_"))
    ]
    filtered.sort(key=lambda evt: evt.timestamp)
    return filtered


def _load_snapshot_internal(path: Path) -> bytearray:
    with zipfile.ZipFile(path, "r") as zf:
        blob = zf.read("internal_ram.bin")
    data = bytearray(blob[:INTERNAL_MEMORY_LENGTH])
    if len(data) < INTERNAL_MEMORY_LENGTH:
        data.extend(b"\x00" * (INTERNAL_MEMORY_LENGTH - len(data)))
    return data


def _replay_internal_memory(
    snapshot: Path, events: Sequence[TraceEvent], stop_timestamp: Optional[int]
) -> bytearray:
    memory = _load_snapshot_internal(snapshot)
    for evt in events:
        if stop_timestamp is not None and evt.timestamp > stop_timestamp:
            break
        space = evt.annotations.get("space")
        if not isinstance(space, str) or "internal" not in space.lower():
            continue
        addr = evt.annotations.get("address")
        value = evt.annotations.get("value")
        size = evt.annotations.get("size", 1)
        if not isinstance(addr, int) or not isinstance(value, int):
            continue
        if not isinstance(size, int) or size <= 0:
            size = 1
        if not (
            INTERNAL_MEMORY_START
            <= addr
            < INTERNAL_MEMORY_START + INTERNAL_MEMORY_LENGTH
        ):
            continue
        offset = addr - INTERNAL_MEMORY_START
        for idx in range(size):
            target = offset + idx
            if not (0 <= target < INTERNAL_MEMORY_LENGTH):
                break
            byte = (value >> (idx * 8)) & 0xFF
            memory[target] = byte
    return memory


def _compare_internal_memory(
    lhs: bytearray, rhs: bytearray, limit: int
) -> List[Tuple[int, int, int]]:
    diffs: List[Tuple[int, int, int]] = []
    for idx, (va, vb) in enumerate(zip(lhs, rhs)):
        if va != vb:
            diffs.append((INTERNAL_MEMORY_START + idx, va, vb))
            if len(diffs) >= limit > 0:
                break
    return diffs


def compare_instruction_traces(
    lhs: Dict[int, TraceEvent], rhs: Dict[int, TraceEvent]
) -> Tuple[Optional[int], Optional[TraceEvent], Optional[TraceEvent], List[str]]:
    """Return earliest differing op_index and the associated events."""

    all_indices = sorted(set(lhs.keys()) & set(rhs.keys()))
    # Always compare PC + opcode, plus IMR/ISR snapshots so WAIT/timer parity regressions
    # show up directly at the divergent instruction.
    keys_of_interest = {"pc", "opcode", "mem_imr", "mem_isr"}

    for index in all_indices:
        evt_a = lhs.get(index)
        evt_b = rhs.get(index)
        if evt_a is None or evt_b is None:
            return index, evt_a, evt_b, ["missing-event"]

        ann_a = dict(evt_a.annotations)
        ann_b = dict(evt_b.annotations)

        # Fallback: parse pc/opcode/op_index from event name when annotations are missing.
        def _parse_name_into(ann: Dict[str, object], name: str) -> None:
            if "pc" in ann and "opcode" in ann and "op_index" in ann:
                return
            if name.startswith("Exec@0x"):
                try:
                    parts = name.split("/")
                    pc_part = parts[0].split("@")[1]
                    ann.setdefault("pc", int(pc_part, 16))
                    for part in parts[1:]:
                        if part.startswith("op="):
                            ann.setdefault("opcode", int(part.split("=", 1)[1], 16))
                        if part.startswith("idx="):
                            ann.setdefault("op_index", int(part.split("=", 1)[1]))
                except Exception:
                    pass

        _parse_name_into(ann_a, evt_a.name if evt_a else "")
        _parse_name_into(ann_b, evt_b.name if evt_b else "")
        fields = set(ann_a.keys()) | set(ann_b.keys())
        reg_fields = {f for f in fields if f.startswith("reg_")}
        compare_fields = keys_of_interest | reg_fields | {"op_index"}

        mismatches: List[str] = []
        for key in sorted(compare_fields):
            if key in ann_a and key in ann_b and ann_a.get(key) != ann_b.get(key):
                mismatches.append(key)

        if mismatches:
            return index, evt_a, evt_b, mismatches

    return None, None, None, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Perfetto traces emitted by the SC62015 cores and "
        "report the earliest divergence."
    )
    parser.add_argument("trace_a", type=Path, help="Reference trace (e.g., python)")
    parser.add_argument("trace_b", type=Path, help="Trace to compare (e.g., rust)")
    parser.add_argument(
        "--snapshot-a",
        type=Path,
        help="Optional snapshot (.pcsnap) used to seed Trace A memory state.",
    )
    parser.add_argument(
        "--snapshot-b",
        type=Path,
        help="Optional snapshot (.pcsnap) used to seed Trace B memory state.",
    )
    parser.add_argument(
        "--memory-diff-limit",
        type=int,
        default=16,
        help="Maximum number of differing internal-memory bytes to display "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--compare-irq",
        action="store_true",
        help="Also diff IRQ/timer events (experimental).",
    )
    args = parser.parse_args()

    events_a = _load_trace(args.trace_a)
    events_b = _load_trace(args.trace_b)

    instr_a = _index_instruction_events(events_a)
    instr_b = _index_instruction_events(events_b)

    mismatch_index, evt_a, evt_b, mismatch_fields = compare_instruction_traces(
        instr_a, instr_b
    )

    divergence_detected = mismatch_index is not None
    exit_code = 1 if divergence_detected else 0

    if divergence_detected:
        print(f"✗ Divergence detected at op_index={mismatch_index}")
        if evt_a is None:
            print(f"  Trace A missing instruction #{mismatch_index}")
        else:
            ann = evt_a.annotations
            pc_a = ann.get("pc")
            opcode_a = ann.get("opcode")
            print(
                f"  Trace A: pc={pc_a:#06X} opcode={opcode_a:#04X} ({evt_a.track})"
                if isinstance(pc_a, int) and isinstance(opcode_a, int)
                else f"  Trace A: {evt_a}"
            )
            print(f"           {_format_reg_dump(ann)}")
        if evt_b is None:
            print(f"  Trace B missing instruction #{mismatch_index}")
        else:
            ann = evt_b.annotations
            pc_b = ann.get("pc")
            opcode_b = ann.get("opcode")
            print(
                f"  Trace B: pc={pc_b:#06X} opcode={opcode_b:#04X} ({evt_b.track})"
                if isinstance(pc_b, int) and isinstance(opcode_b, int)
                else f"  Trace B: {evt_b}"
            )
            print(f"           {_format_reg_dump(ann)}")

        if mismatch_fields:
            print("  Differing fields:", ", ".join(mismatch_fields))

        if args.snapshot_a and args.snapshot_b:
            memory_events_a = _memory_write_events(events_a)
            memory_events_b = _memory_write_events(events_b)
            limit = max(0, int(args.memory_diff_limit))

            mem_a = _replay_internal_memory(
                args.snapshot_a, memory_events_a, evt_a.timestamp if evt_a else None
            )
            mem_b = _replay_internal_memory(
                args.snapshot_b, memory_events_b, evt_b.timestamp if evt_b else None
            )

            diffs = _compare_internal_memory(mem_a, mem_b, limit)
            if diffs:
                print("  Internal memory differences at divergence:")
                for addr, va, vb in diffs:
                    offset = addr - INTERNAL_MEMORY_START
                    print(f"    (0x{offset:02X}) TraceA=0x{va:02X} TraceB=0x{vb:02X}")
            else:
                print("  Internal memory identical at divergence.")

    irq_mismatch: Optional[str] = None
    if args.compare_irq:
        irq_a = _irq_events(events_a)
        irq_b = _irq_events(events_b)
        len_irq_a = len(irq_a)
        len_irq_b = len(irq_b)
        for idx, (ea, eb) in enumerate(zip_longest(irq_a, irq_b)):
            if ea is None or eb is None:
                missing_trace = "Trace A" if ea is None else "Trace B"
                irq_mismatch = (
                    f"{missing_trace} missing IRQ/timer event at index {idx} "
                    f"(TraceA={len_irq_a} TraceB={len_irq_b})"
                )
                break
            if ea.name != eb.name:
                irq_mismatch = f"{ea.name} vs {eb.name} at irq index {idx}"
                break
            if ea.annotations != eb.annotations:
                irq_mismatch = f"{ea.name} annotations differ at irq index {idx}"
                break
        if irq_mismatch:
            print("\nIRQ/timer differences:")
            print(f"  {irq_mismatch}")
            exit_code = 1
        else:
            print(f"\nIRQ/timer events match ({len_irq_a} events).")

    if not divergence_detected:
        if irq_mismatch:
            print("✗ Instruction traces match but IRQ/timer events differ.")
        else:
            print(
                f"✓ No divergence detected across {min(len(instr_a), len(instr_b))} instructions."
            )

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
