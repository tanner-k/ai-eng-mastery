"""From-scratch first-order optimizers.

Each ``step(params, grads)`` returns NEW parameter tensors; the input params are
never mutated in place. The optimizers are intentionally *stateful*: running
moments (``velocity`` / ``sq`` / ``m`` / ``v``) and the timestep ``t`` are
updated on the instance each call — that internal state is the one place where
mutation is by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

Params = list[torch.Tensor]


@dataclass
class SGD:
    """Vanilla (stochastic) gradient descent."""

    lr: float = 0.01

    def step(self, params: Params, grads: Params) -> Params:
        return [p - self.lr * g for p, g in zip(params, grads)]


@dataclass
class Momentum:
    """SGD with an EMA-of-gradients velocity term."""

    lr: float = 0.01
    beta: float = 0.9
    velocity: Params | None = None

    def step(self, params: Params, grads: Params) -> Params:
        if self.velocity is None:
            self.velocity = [torch.zeros_like(p) for p in params]
        self.velocity = [self.beta * v + (1 - self.beta) * g for v, g in zip(self.velocity, grads)]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


@dataclass
class RMSProp:
    """Per-coordinate adaptive learning rate via an EMA of squared gradients."""

    lr: float = 0.01
    beta: float = 0.99
    eps: float = 1e-8
    sq: Params | None = None

    def step(self, params: Params, grads: Params) -> Params:
        if self.sq is None:
            self.sq = [torch.zeros_like(p) for p in params]
        self.sq = [self.beta * s + (1 - self.beta) * g**2 for s, g in zip(self.sq, grads)]
        return [
            p - self.lr * g / (torch.sqrt(s) + self.eps)
            for p, g, s in zip(params, grads, self.sq)
        ]


@dataclass
class Adam:
    """Adam (Kingma & Ba, 2015): momentum + RMSProp with bias correction."""

    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: Params | None = None
    v: Params | None = None
    t: int = 0

    def step(self, params: Params, grads: Params) -> Params:
        if self.m is None:
            self.m = [torch.zeros_like(p) for p in params]
            self.v = [torch.zeros_like(p) for p in params]
        self.t += 1
        new_params: Params = []
        new_m: Params = []
        new_v: Params = []
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            new_params.append(p - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))
            new_m.append(m)
            new_v.append(v)
        self.m, self.v = new_m, new_v
        return new_params
