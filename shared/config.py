"""Central runtime configuration loaded from ``config.toml`` (+ env overrides).

Read once at import; everything in the repo gets its device/seed/dtype from here
rather than hardcoding. Unit tests pin CPU for determinism; notebooks and
experiments call ``configure()`` to run on the resolved device (mps on Apple
Silicon).
"""
from __future__ import annotations

import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

from shared.devices import get_device

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

_VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}
_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the whole repo."""

    device: str = "auto"
    seed: int = 0
    dtype: str = "float32"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``config.toml`` (``[runtime]`` table), then apply env overrides.

    Env overrides: ``AIEM_DEVICE``, ``AIEM_SEED``, ``AIEM_DTYPE``.

    Raises:
        ValueError: If device or dtype is not a recognized value.
    """
    path = path or _CONFIG_PATH
    data: dict = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f).get("runtime", {})

    device = os.environ.get("AIEM_DEVICE", data.get("device", "auto"))
    raw_seed = os.environ.get("AIEM_SEED", data.get("seed", 0))
    try:
        seed = int(raw_seed)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"AIEM_SEED must be an integer, got {raw_seed!r}") from exc
    dtype = os.environ.get("AIEM_DTYPE", data.get("dtype", "float32"))

    if device not in _VALID_DEVICES:
        raise ValueError(f"device must be one of {sorted(_VALID_DEVICES)}, got {device!r}")
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    return Settings(device=device, seed=seed, dtype=dtype)


def resolve_device(settings: Settings | None = None) -> torch.device:
    """Resolve the configured device to a concrete ``torch.device``."""
    settings = settings or load_settings()
    if settings.device == "auto":
        return get_device()
    return get_device(settings.device)


def resolve_dtype(settings: Settings | None = None) -> torch.dtype:
    """Resolve the configured dtype string to a ``torch.dtype``."""
    settings = settings or load_settings()
    return _DTYPES[settings.dtype]


def seed_everything(seed: int | None = None) -> int:
    """Seed Python and torch RNGs for reproducibility; returns the seed used."""
    if seed is None:
        seed = load_settings().seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    return seed


def configure(settings: Settings | None = None) -> torch.device:
    """Apply settings (seed + default dtype) and return the resolved device.

    Intended for notebooks/experiments::

        from shared.config import configure
        device = configure()

    Warning:
        Calls ``torch.set_default_dtype``, which mutates process-global PyTorch
        state. Call this only at a process entry point (a notebook or training
        script) — never from library code or unit tests.
    """
    settings = settings or load_settings()
    seed_everything(settings.seed)
    torch.set_default_dtype(resolve_dtype(settings))
    return resolve_device(settings)


# Computed once at import. Tests should call load_settings() directly;
# monkeypatching env vars does NOT retroactively update these module-level constants.
SETTINGS = load_settings()
DEVICE = resolve_device(SETTINGS)
DTYPE = resolve_dtype(SETTINGS)
