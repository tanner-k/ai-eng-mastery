"""Linear regression with hand-derived gradients (MSE loss)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearModel:
    """``y = X @ w + b`` with manual MSE gradients.

    Immutable: optimization produces new ``LinearModel`` instances rather than
    mutating ``w``/``b`` in place.
    """

    w: torch.Tensor
    b: torch.Tensor

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((self.predict(X) - y) ** 2)

    def gradients(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic gradients of the MSE loss wrt ``w`` and ``b``."""
        n = X.shape[0]
        residual = self.predict(X) - y
        grad_w = (2.0 / n) * (X.T @ residual)
        grad_b = (2.0 / n) * residual.sum()
        return grad_w, grad_b
