"""PI raw contact replay validation, mirrored by the public native runner."""

import json


def parse_replay(data):
    document = json.loads(data)
    if not isinstance(document, dict) or set(document) != {"steps"}:
        raise ValueError("PI replay requires a steps document")
    steps = document["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 4096:
        raise ValueError("PI contact replay/budget out of range")
    for step in steps:
        if not isinstance(step, dict) or set(step) - {
            "boundaries",
            "tablet",
            "on_key",
            "label",
        }:
            raise ValueError("unknown PI replay step member")
        if (
            type(step.get("boundaries")) is not int
            or not 0 <= step["boundaries"] <= 10000000
        ):
            raise ValueError("PI contact replay/budget out of range")
        if step.get("on_key") is not None and type(step["on_key"]) is not bool:
            raise ValueError("PI ON contact must be boolean")
        if step.get("label") is not None and not isinstance(step["label"], str):
            raise ValueError("PI replay label must be text")
        contact = step.get("tablet")
        if contact is not None:
            if not isinstance(contact, dict) or set(contact) != {
                "raw_x",
                "raw_y",
                "pressed",
            }:
                raise ValueError("PI raw contact fields mismatch")
            if (
                any(
                    type(contact[n]) is not int or not 0 <= contact[n] <= 1023
                    for n in ("raw_x", "raw_y")
                )
                or type(contact["pressed"]) is not bool
            ):
                raise ValueError("PI contact replay/budget out of range")
    return steps
