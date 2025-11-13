from __future__ import annotations

from dataclasses import dataclass
import os


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    return normalized not in {"0", "false", "off", ""}


@dataclass(frozen=True)
class ScilConfig:
    allow_legacy: bool
    trace: bool


def load_scil_config() -> ScilConfig:
    return ScilConfig(
        allow_legacy=_env_flag("BN_ALLOW_LEGACY", default=False),
        trace=_env_flag("BN_SCIL_TRACE", default=False),
    )


__all__ = ["ScilConfig", "load_scil_config"]
