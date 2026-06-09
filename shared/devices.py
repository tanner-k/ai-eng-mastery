"""Device selection helpers (cuda > mps > cpu)."""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: Optional explicit device string ("cuda", "mps", "cpu"). When
            provided it is used as-is.

    Returns:
        The selected ``torch.device`` (cuda if available, else mps, else cpu).
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
