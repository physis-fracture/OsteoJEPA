"""Stage B: the age sweep (SPEC section 6, Algorithm 1 of the paper).

Training samples context blocks at random. At inference that would make the
surprise map of one image differ between runs, and patches that happened to land
in the context would carry no residual at all. A triage system cannot behave
that way, so patches are partitioned into K complementary interleaved groups:
for group k the context is every valid patch outside P_k and the target is P_k
itself. Every valid patch gets exactly one residual and repeated calls return
identical output.

Two details that are easy to lose:

* The recorded age is always evaluated, whether or not it lands on the coarse
  grid (Algorithm 1, line 1). Without it `s(p, a_rec)` - the whole basis of the
  score - would have to be interpolated.
* The refinement window is a union over the patches *of one image*. Taking the
  union across a batch would make one image's result depend on which images it
  was batched with, which is exactly the non-determinism the partition exists to
  remove. The sweep therefore evaluates the batch-wide union but only lets each
  image minimise over its own ages.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from omegaconf import DictConfig

from ..data.masking import mask_to_indices, pad_index_batch, partition_interleaved
from ..losses.jepa import patch_residual
from ..losses.margin import age_grid
from ..models.osteojepa import gather_tokens


def coarse_ages(cfg: DictConfig) -> np.ndarray:
    return age_grid(
        float(cfg.inference.age_min),
        float(cfg.inference.age_max),
        float(cfg.inference.age_grid_coarse_step),
    )


def fine_ages(cfg: DictConfig) -> np.ndarray:
    return age_grid(
        float(cfg.inference.age_min),
        float(cfg.inference.age_max),
        float(cfg.inference.age_grid_fine_step),
    )


def _partition_indices(valid_np: np.ndarray, k: int, device) -> list[tuple]:
    """Padded (idx, keep) pairs for the target group and its complement context."""
    out = []
    for group in range(k):
        tgt_lists, ctx_lists = [], []
        for valid in valid_np:
            groups = partition_interleaved(valid, k)
            tgt_lists.append(mask_to_indices(groups[group]))
            ctx_lists.append(mask_to_indices(valid & ~groups[group]))
        out.append((pad_index_batch(tgt_lists, device), pad_index_batch(ctx_lists, device)))
    return out


@torch.no_grad()
def sweep_batch(model, batch: dict, cfg: DictConfig, device: torch.device) -> dict:
    """Run the two-stage age sweep over one batch.

    Returns numpy arrays shaped (B, grid, grid), NaN at padding patches:
        s_rec, s_min, a_hat, delta, v_patch
    """
    model.eval()
    k = int(cfg.inference.context_partition_k)
    images = batch["image"].to(device)
    valid = batch["valid_mask"].to(device)
    valid_np = valid.detach().cpu().numpy()
    batch_size, grid_h, grid_w = valid_np.shape
    n_tokens = grid_h * grid_w

    age_recorded = batch["age"].to(device)
    meta = {key: batch[key].to(device) for key in ("gender", "view", "laterality")}

    z_full = model.encode_targets(images)                      # once
    partitions = _partition_indices(valid_np, k, device)
    context_encodings = [
        model.encoder(images, keep_idx=ctx[0], keep_mask=ctx[1]) for _, ctx in partitions
    ]                                                          # K times, age-independent

    def residual_at(age_values: torch.Tensor) -> np.ndarray:
        """Residual of every valid patch at one age per sample; (B, n_tokens)."""
        cond = model.condition_vector(age_values, meta)
        out = np.full((batch_size, n_tokens), np.nan, dtype=np.float32)
        for group, ((tgt_idx, tgt_keep), (ctx_idx, ctx_keep)) in enumerate(partitions):
            z_tgt = gather_tokens(z_full, tgt_idx)
            pred = model.predict(
                context_encodings[group], ctx_idx, ctx_keep, tgt_idx, tgt_keep, cond
            )
            residual = patch_residual(pred, z_tgt).float().cpu().numpy()
            idx = tgt_idx.cpu().numpy()
            keep = tgt_keep.cpu().numpy()
            for b in range(batch_size):
                out[b, idx[b][keep[b]]] = residual[b][keep[b]]
        return out

    # --- coarse pass: A_c, plus the recorded age of every sample -------------
    coarse = coarse_ages(cfg)
    s_coarse = np.stack([residual_at(_full(age, batch_size, device)) for age in coarse], axis=1)
    s_recorded = residual_at(age_recorded)
    recorded_np = batch["age"].numpy().astype(np.float64)

    a_star = np.full((batch_size, n_tokens), np.nan, dtype=np.float32)
    for b in range(batch_size):
        ages_b = np.append(coarse, recorded_np[b])
        stacked = _nan_to_inf(np.concatenate([s_coarse[b], s_recorded[b][None]], axis=0))
        a_star[b] = ages_b[np.argmin(stacked, axis=0)]

    # --- refinement: per-image union of windows around the coarse minimum ----
    window = float(cfg.inference.refine_window)
    full_grid = fine_ages(cfg)
    coarse_set = set(np.round(coarse, 6))
    candidate_grid = np.array([a for a in full_grid if round(a, 6) not in coarse_set])

    allowed_extra = np.zeros((batch_size, len(candidate_grid)), dtype=bool)
    for b in range(batch_size):
        centres = a_star[b][~np.isnan(a_star[b])]
        if centres.size and candidate_grid.size:
            distance = np.abs(candidate_grid[None, :] - centres[:, None])
            allowed_extra[b] = (distance <= window).any(axis=0)

    extra_used = np.flatnonzero(allowed_extra.any(axis=0))
    extra_ages = candidate_grid[extra_used]
    s_extra = (
        np.stack([residual_at(_full(age, batch_size, device)) for age in extra_ages], axis=1)
        if extra_ages.size
        else np.zeros((batch_size, 0, n_tokens), dtype=np.float32)
    )

    # --- minimise over each image's own candidate set ------------------------
    s_min = np.full((batch_size, n_tokens), np.nan, dtype=np.float32)
    a_hat = np.full((batch_size, n_tokens), np.nan, dtype=np.float32)
    v_patch = np.full((batch_size, n_tokens), np.nan, dtype=np.float32)
    for b in range(batch_size):
        keep_extra = allowed_extra[b][extra_used] if extra_used.size else np.zeros(0, dtype=bool)
        ages_b = np.concatenate([coarse, extra_ages[keep_extra], [recorded_np[b]]])
        s_b = np.concatenate([s_coarse[b], s_extra[b][keep_extra], s_recorded[b][None]], axis=0)
        best = np.argmin(_nan_to_inf(s_b), axis=0)
        s_min[b] = s_b[best, np.arange(n_tokens)]
        a_hat[b] = ages_b[best]
        # Padding columns are NaN for every age; nanvar warns on such slices and
        # the result is discarded below anyway.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v_patch[b] = np.nanvar(s_b, axis=0)

    invalid = ~valid_np.reshape(batch_size, n_tokens)
    for array in (s_min, a_hat, v_patch):
        array[invalid] = np.nan
    s_recorded[invalid] = np.nan

    delta = s_recorded - s_min
    shape = (batch_size, grid_h, grid_w)
    return {
        "s_rec": s_recorded.reshape(shape),
        "s_min": s_min.reshape(shape),
        "a_hat": a_hat.reshape(shape),
        "delta": delta.reshape(shape),
        "v_patch": v_patch.reshape(shape),
        "valid": valid_np,
    }


def score_map(s_rec: np.ndarray, delta: np.ndarray, lam: float) -> np.ndarray:
    """score(p) = s(p, a_rec) - lambda * Delta(p)."""
    return s_rec - lam * delta


def _full(age: float, batch_size: int, device) -> torch.Tensor:
    return torch.full((batch_size,), float(age), dtype=torch.float32, device=device)


def _nan_to_inf(array: np.ndarray) -> np.ndarray:
    """Padding patches carry NaN; +inf makes them lose every argmin cleanly."""
    out = array.copy()
    out[np.isnan(out)] = np.inf
    return out
