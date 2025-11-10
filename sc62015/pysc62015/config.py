from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Set

_VALID_MODES = {"off", "shadow", "prod"}


def _normalize_mode(mode: str | None) -> str:
    value = (mode or "shadow").strip().lower()
    if value not in _VALID_MODES:
        return "off"
    return value


def _parse_csv(value: str | None, *, split_commas: bool = False) -> Set[str]:
    if not value:
        return set()
    normalized = value.replace("\n", ";")
    tokens = normalized.split(";") if ";" in normalized else [normalized]
    if split_commas:
        expanded: list[str] = []
        for token in tokens:
            expanded.extend(token.split(","))
        tokens = expanded
    return {item.strip() for item in tokens if item.strip()}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "off", ""}


@dataclass(frozen=True)
class ScilConfig:
    mode: str
    allow: Set[str]
    block: Set[str]
    families: Set[str]
    strict_compare: bool
    trace: bool

    def is_allowed(self, mnemonic: str, family: str | None) -> bool:
        if mnemonic in self.block:
            return False
        if mnemonic in self.allow:
            return True
        if family and family in self.families:
            return True
        return False


def load_scil_config() -> ScilConfig:
    return ScilConfig(
        mode=_normalize_mode(os.getenv("BN_USE_SCIL")),
        allow=_parse_csv(os.getenv("BN_SCIL_ALLOW")),
        block=_parse_csv(os.getenv("BN_SCIL_BLOCK")),
        families=_parse_csv(os.getenv("BN_SCIL_FAMILIES"), split_commas=True),
        strict_compare=_env_flag("BN_SCIL_STRICT_COMPARE", default=True),
        trace=_env_flag("BN_SCIL_TRACE", default=False),
    )


__all__ = ["ScilConfig", "load_scil_config"]
