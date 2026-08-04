"""VICReg variance and covariance terms.

Applied to the **context encoder output**, not to the predictor output: the
collapse being guarded against is the encoder folding every patch onto one
constant vector, which would make every residual zero and every surprise map
flat.

Var(z) is monitored per dimension rather than aggregated. A collapse affecting
only some dimensions leaves the mean variance looking healthy.
"""

from __future__ import annotations

import torch

EPS = 1e-4


def _valid_tokens(tokens: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Flatten (B, N, D) tokens to (M, D), dropping padded slots."""
    return tokens[keep]


def variance_loss(tokens: torch.Tensor, keep: torch.Tensor, target_std: float = 1.0):
    """Hinge on per-dimension standard deviation; also returns the per-dim std."""
    flat = _valid_tokens(tokens, keep)
    if flat.shape[0] < 2:
        zero = tokens.sum() * 0.0
        return zero, torch.zeros(tokens.shape[-1], device=tokens.device)
    std = torch.sqrt(flat.var(dim=0) + EPS)
    loss = torch.relu(target_std - std).mean()
    return loss, std.detach()


def covariance_loss(tokens: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariances, normalized by dimension."""
    flat = _valid_tokens(tokens, keep)
    n, dim = flat.shape
    if n < 2:
        return tokens.sum() * 0.0
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n - 1)
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diagonal.pow(2).sum() / dim
