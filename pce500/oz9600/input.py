"""Physical contacts; static printed mode labels use raw tablet input."""

import json
from pathlib import Path

KEYS = json.loads(
    (
        Path(__file__).resolve().parents[2] / "sc62015/core/data/physical_keys.json"
    ).read_text()
)["oz-9600"]


def parse_replay(data):
    document = json.loads(data)
    if not isinstance(document, dict) or set(document) != {"steps"}:
        raise ValueError("physical replay requires only a steps array")
    steps = document["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 4096:
        raise ValueError("physical replay requires 1-4096 steps")
    for step in steps:
        if set(step) - {"boundaries", "contact", "tablet", "on_key", "label"}:
            raise ValueError("unknown physical step field")
        if (
            type(step.get("boundaries")) is not int
            or not 0 <= step["boundaries"] <= 10_000_000
        ):
            raise ValueError("physical replay contact/budget out of range")
        if step.get("on_key") is not None and type(step["on_key"]) is not bool:
            raise ValueError("physical ON contact must be a boolean")
        for name, coordinates in (
            ("contact", {"column": 10, "row": 7}),
            ("tablet", {"raw_x": 1023, "raw_y": 1023}),
        ):
            contact = step.get(name)
            if contact is None:
                continue
            if (
                set(contact) != set(coordinates) | {"pressed"}
                or type(contact["pressed"]) is not bool
                or any(
                    type(contact.get(c)) is not int or not 0 <= contact[c] <= limit
                    for c, limit in coordinates.items()
                )
            ):
                raise ValueError("physical replay contact/budget out of range")
        if step.get("label") is not None and not isinstance(step["label"], str):
            raise ValueError("physical replay label must be a string")
    return steps
