"""The full Stage A objective, assembled in one place.

    L = L_pred + lambda_v * L_var + lambda_c * L_cov + lambda_m * L_margin

Training and the per-step benchmark both call this, so the "with margin" and
"without margin" numbers M2 is accepted on differ in exactly one term and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from ..models.osteojepa import gather_tokens
from .jepa import patch_residual, prediction_loss
from .margin import age_sensitivity, margin_loss, sample_distractor_ages
from .vicreg import covariance_loss, variance_loss


@dataclass
class ObjectiveOutput:
    total: torch.Tensor
    pred: torch.Tensor
    var: torch.Tensor
    cov: torch.Tensor
    margin: torch.Tensor
    std_per_dim: torch.Tensor
    v_patch: torch.Tensor | None
    predictor_calls: int
    margin_block: int


def compute_objective(
    model,
    *,
    images: torch.Tensor,
    ages: torch.Tensor,
    ages_cpu: np.ndarray,
    meta: dict,
    masks: dict,
    cfg: DictConfig,
    rng: np.random.Generator,
    grid: np.ndarray,
    use_margin: bool = True,
    margin_block: int | None = None,
) -> ObjectiveOutput:
    """One forward pass of the whole objective.

    The encoders run exactly twice per step regardless of `use_margin`: once for
    the target encoding and once for the context. Distractor ages reuse both, so
    turning the margin on adds predictor passes only. If per-step time nearly
    doubles when the margin is switched on, an encoder has been re-invoked.
    """
    ctx_idx, ctx_keep = masks["ctx_idx"], masks["ctx_keep"]
    tgt_idx, tgt_keep = masks["tgt_idx"], masks["tgt_keep"]
    n_blocks = tgt_idx.shape[1]
    if margin_block is None:
        margin_block = int(rng.integers(n_blocks))

    z_full = model.encode_targets(images)                 # encoder call 1 (no grad)
    z_ctx = model.encode_context(images, ctx_idx, ctx_keep)  # encoder call 2
    cond = model.condition_vector(ages, meta)

    loss_pred = images.new_zeros(())
    residual_for_margin = None
    z_tgt_for_margin = None
    predictor_calls = 0
    for block in range(n_blocks):
        idx_b, keep_b = tgt_idx[:, block], tgt_keep[:, block]
        z_tgt = gather_tokens(z_full, idx_b)
        pred = model.predict(z_ctx, ctx_idx, ctx_keep, idx_b, keep_b, cond)
        predictor_calls += 1
        residual = patch_residual(pred, z_tgt)
        loss_pred = loss_pred + prediction_loss(residual, keep_b)
        if block == margin_block:
            residual_for_margin = residual
            z_tgt_for_margin = z_tgt
    loss_pred = loss_pred / n_blocks

    loss_var, std_per_dim = variance_loss(z_ctx, ctx_keep)
    loss_cov = covariance_loss(z_ctx, ctx_keep)

    lambda_margin = float(cfg.loss.lambda_margin)
    v_patch = None
    if use_margin and lambda_margin > 0:
        distractors = sample_distractor_ages(
            ages_cpu, grid, float(cfg.loss.distractor_tau), rng
        )
        loss_margin, s_distractor = margin_loss(
            model,
            z_ctx=z_ctx,
            ctx_idx=ctx_idx,
            ctx_keep=ctx_keep,
            tgt_idx=tgt_idx[:, margin_block],
            tgt_keep=tgt_keep[:, margin_block],
            z_tgt=z_tgt_for_margin,
            meta=meta,
            residual_recorded=residual_for_margin,
            distractor_ages=torch.from_numpy(distractors).to(images.device),
            margin=float(cfg.loss.margin_m),
        )
        predictor_calls += distractors.shape[1]
        all_ages = torch.cat([residual_for_margin.unsqueeze(1), s_distractor], dim=1).detach()
        v_patch = age_sensitivity(all_ages, tgt_keep[:, margin_block])
    else:
        loss_margin = images.new_zeros(())

    total = (
        loss_pred
        + float(cfg.loss.lambda_var) * loss_var
        + float(cfg.loss.lambda_cov) * loss_cov
        + lambda_margin * loss_margin
    )
    return ObjectiveOutput(
        total=total,
        pred=loss_pred,
        var=loss_var,
        cov=loss_cov,
        margin=loss_margin,
        std_per_dim=std_per_dim,
        v_patch=v_patch,
        predictor_calls=predictor_calls,
        margin_block=margin_block,
    )
