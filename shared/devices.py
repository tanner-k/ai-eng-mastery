"""Device selection helpers (cuda > mps > cpu)."""
from __future__ import annotations

import torch

_VALID_DEVICES = {"cuda", "mps", "cpu"}


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: Optional explicit device string ("cuda", "mps", "cpu"). When
            provided it is validated and used as-is.

    Returns:
        The selected ``torch.device`` (cuda if available, else mps, else cpu).

    Raises:
        ValueError: If ``prefer`` is not one of "cuda", "mps", "cpu".
    """
    if prefer is not None:
        if prefer not in _VALID_DEVICES:
            raise ValueError(f"prefer must be one of {sorted(_VALID_DEVICES)}, got {prefer!r}")
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
