"""带单纯形约束的批量凸 QP；固定展开次数，保留输入梯度。"""
from __future__ import annotations

import math
import torch


def simplex_projection(values):
    ordered = values.sort(dim=-1, descending=True).values
    cssv = ordered.cumsum(-1) - 1
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    valid = ordered - cssv / ranks > 0
    rho = valid.sum(-1, keepdim=True).clamp_min(1)
    theta = cssv.gather(-1, rho - 1) / rho
    return (values - theta).clamp_min(0)


def solve_simplex_qp(gram, linear, constant, steps=200):
    """min w'Aw - 2b'w + c；A 已包含 eta I。"""
    if steps < 1:
        raise ValueError('steps must be positive')
    if gram.ndim == 2:
        gram = gram.expand(linear.shape[0], -1, -1)
    # A is fixed reference data; only b and c depend on the tested input.
    lipschitz = 2 * torch.linalg.eigvalsh(gram.detach()).amax(-1).clamp_min(1e-12)
    w = torch.full_like(linear, 1 / linear.shape[-1])
    extrapolated = w
    momentum_time = 1.
    for _ in range(steps):
        grad = 2 * (torch.bmm(gram, extrapolated.unsqueeze(-1)).squeeze(-1) - linear)
        next_w = simplex_projection(extrapolated - grad / lipschitz[:, None])
        next_time = (1 + math.sqrt(1 + 4*momentum_time**2))/2
        extrapolated = next_w + ((momentum_time-1)/next_time)*(next_w-w)
        w, momentum_time = next_w, next_time
    grad = 2 * (torch.bmm(gram, w.unsqueeze(-1)).squeeze(-1) - linear)
    objective = (w * (torch.bmm(gram, w.unsqueeze(-1)).squeeze(-1) - 2*linear)).sum(-1) + constant
    fw_gap = (grad * w).sum(-1) - grad.amin(-1)
    return w, objective, fw_gap
