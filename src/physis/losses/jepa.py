"""Prediction residual and the L_pred term.

    s(p, a) = || g_phi(z_ctx, pos_p, c[age <- a]) - z_p ||^2

SPEC writes a squared L2. It is computed here as a **mean** over the 384 feature
dimensions rather than a sum, so that the margin `m = 0.10` and the residual live
on the same scale. A sum would put s in the hundreds and make any margin below
1.0 vacuously satisfied at step 0. The choice is a fixed rescaling of every
residual, so it changes nothing downstream except the numeric value of m.
"""

from __future__ import annotations

import torch


def patch_residual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-patch residual s(p, a); (B, N) from (B, N, D) inputs."""
    return (pred - target).pow(2).mean(dim=-1)


def masked_mean(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over entries where `keep` is True.

    Padded slots hold real numbers computed from a duplicated patch index, so
    they must be excluded rather than merely down-weighted.
    """
    keep = keep.to(values.dtype)
    total = (values * keep).sum()
    count = keep.sum().clamp_min(1.0)
    return total / count


def prediction_loss(residual: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return masked_mean(residual, keep)
