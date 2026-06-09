"""Reusable plotting helpers for optimization/learning visualizations."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe for tests and CI
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def plot_loss_surface(
    loss_fn: Callable[[float, float], "float | torch.Tensor"],
    w_range: tuple[float, float],
    b_range: tuple[float, float],
    steps: int = 50,
    ax: "plt.Axes | None" = None,
    levels: int = 30,
) -> "plt.Axes":
    """Draw a filled contour of a 2-parameter loss surface.

    ``loss_fn`` is evaluated per grid point (a Python-level ``steps * steps``
    loop), which is fine for the small 2-parameter surfaces used here; for large
    grids, prefer a vectorized loss that accepts tensors. When ``ax`` is None a
    new figure is created and the caller owns its lifecycle (``plt.close``).

    Args:
        loss_fn: Callable ``(w, b) -> loss`` (scalar float or 0-d tensor).
        w_range: (min, max) for the first parameter axis.
        b_range: (min, max) for the second parameter axis.
        steps: Grid resolution per axis.
        ax: Optional existing axes to draw on.
        levels: Number of contour levels.

    Returns:
        The axes the surface was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    ws = torch.linspace(*w_range, steps)
    bs = torch.linspace(*b_range, steps)
    grid = torch.tensor([[float(loss_fn(float(w), float(b))) for w in ws] for b in bs])
    ax.contourf(ws.numpy(), bs.numpy(), grid.numpy(), levels=levels)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    return ax


def plot_descent_path(
    path: Sequence[tuple[float, float]],
    ax: "plt.Axes | None" = None,
    **kwargs: Any,
) -> "plt.Axes":
    """Plot a parameter trajectory ``[(w, b), ...]`` as a connected line.

    When ``ax`` is None a new figure is created and the caller owns its
    lifecycle (``plt.close``).

    Args:
        path: Sequence of (w, b) points visited during optimization.
        ax: Optional existing axes to draw on.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The axes the path was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, marker="o", **kwargs)
    return ax
