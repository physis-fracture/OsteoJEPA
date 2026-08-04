"""Margin loss: force the predictor to actually use the age channel.

The failure this prevents: a 60% context block already carries skeletal
maturation signal, so explicit age is redundant and optimization happily pushes
gamma(c) -> 1, beta(c) -> 0. The residual curve then goes flat against age, and
every Stage B quantity - a_hat, Delta, the whole score - loses its basis while
the training loss keeps falling.

    L_margin = mean_p mean_{a' in D} max(0, m - (s(p, a') - s(p, a_rec)))

D holds two distractor ages sampled **symmetrically**, one from each side. With
one-sided sampling the distractor is always older for toddlers and always
younger for teenagers, so the sign of the residual difference alone wins the
margin: the predictor learns the direction of the age ordering and never touches
anatomy.

Cost control: the distractor pass reuses z_ctx and z. Only the predictor runs
again, so per-step cost rises by about a quarter. Calling either encoder a second
time roughly doubles it, and nothing in the loss value would reveal that.
"""

from __future__ import annotations

import numpy as np
import torch

from .jepa import masked_mean, patch_residual


def age_grid(age_min: float, age_max: float, step: float) -> np.ndarray:
    """The candidate age set A = {0.5, 1.0, ..., 19.0}."""
    n = int(round((age_max - age_min) / step)) + 1
    return np.round(age_min + step * np.arange(n), 6)


def sample_distractor_ages(
    recorded: np.ndarray,
    grid: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    n_distractors: int = 2,
) -> np.ndarray:
    """(B, n_distractors) distractor ages, symmetric around the recorded age.

    One from {a <= a_rec - tau} and one from {a >= a_rec + tau}. When one side is
    empty - a two-year-old has no younger distractor, a seventeen-year-old has no
    older one - both are taken from the available side.
    """
    out = np.zeros((len(recorded), n_distractors), dtype=np.float32)
    for b, a_rec in enumerate(recorded):
        low = grid[grid <= a_rec - tau]
        high = grid[grid >= a_rec + tau]
        if len(low) and len(high):
            picks = [rng.choice(low), rng.choice(high)]
            extra = n_distractors - 2
            if extra > 0:
                pool = np.concatenate([low, high])
                picks += list(rng.choice(pool, size=extra))
        elif len(low):
            picks = list(rng.choice(low, size=n_distractors, replace=len(low) < n_distractors))
        elif len(high):
            picks = list(rng.choice(high, size=n_distractors, replace=len(high) < n_distractors))
        else:  # unreachable for tau=3 on a 0.5-19.0 grid; kept so it fails loudly
            raise ValueError(f"no distractor age available for a_rec={a_rec} at tau={tau}")
        out[b] = np.asarray(picks[:n_distractors], dtype=np.float32)
    return out


def margin_loss(
    model,
    *,
    z_ctx: torch.Tensor,
    ctx_idx: torch.Tensor,
    ctx_keep: torch.Tensor,
    tgt_idx: torch.Tensor,
    tgt_keep: torch.Tensor,
    z_tgt: torch.Tensor,
    meta: dict,
    residual_recorded: torch.Tensor,
    distractor_ages: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, s_distractor) with s_distractor of shape (B, n_d, N_tgt)."""
    hinges = []
    residuals = []
    for d in range(distractor_ages.shape[1]):
        cond = model.condition_vector(distractor_ages[:, d], meta)
        # z_ctx and z_tgt are reused as-is: neither encoder runs again here.
        pred = model.predict(z_ctx, ctx_idx, ctx_keep, tgt_idx, tgt_keep, cond)
        s_distractor = patch_residual(pred, z_tgt)
        residuals.append(s_distractor)
        hinge = torch.relu(margin - (s_distractor - residual_recorded))
        hinges.append(masked_mean(hinge, tgt_keep))
    return torch.stack(hinges).mean(), torch.stack(residuals, dim=1)


def age_sensitivity(residuals: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """V(p) = Var_a s(p, a) over the ages available in `residuals`.

    Returned per patch so it can be reported spatially. A margin loss can be
    satisfied by a uniform age-dependent offset that contains no anatomy, and the
    median of V(p) cannot tell that apart from a real mechanism; the spatial map
    can.
    """
    variance = residuals.var(dim=1, unbiased=False)
    return torch.where(keep, variance, torch.full_like(variance, float("nan")))
