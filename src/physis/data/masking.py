"""Context and target sampling, and the deterministic inference partition.

Two rules govern everything here.

*Padding never enters.* Blocks are sampled from the valid region only. Beyond
wasting compute, padding patches leak age: the aspect ratio of a radiograph
tracks body size and body size tracks age, so the number of padding patches
hands the predictor a way to guess age without touching anatomy. The margin loss
does not close that hole, because padding patches satisfy the margin too.

*Training samples blocks at random; inference must not.* Random context at
inference would make the surprise map of one image differ between runs, and
patches that happened to land in the context would carry no residual at all.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omegaconf import DictConfig

MIN_OVERLAP_FRACTION = 0.5  # a block landing mostly in padding teaches nothing


def _sample_block(
    valid: np.ndarray,
    area_fraction: float,
    rng: np.random.Generator,
    *,
    aspect_range: tuple[float, float] = (0.75, 1.5),
    max_tries: int = 20,
) -> np.ndarray:
    """One rectangular block, intersected with the valid region.

    `area_fraction` is measured against the number of *valid* patches, not
    against the 576-token canvas, so an image with a lot of padding still gets a
    context of the intended size.
    """
    grid_h, grid_w = valid.shape
    n_valid = int(valid.sum())
    target_area = max(1.0, area_fraction * n_valid)

    best, best_overlap = None, -1
    for _ in range(max_tries):
        aspect = rng.uniform(*aspect_range)
        h = int(np.clip(round(math.sqrt(target_area / aspect)), 1, grid_h))
        w = int(np.clip(round(math.sqrt(target_area * aspect)), 1, grid_w))
        j0 = int(rng.integers(0, grid_h - h + 1))
        i0 = int(rng.integers(0, grid_w - w + 1))

        block = np.zeros_like(valid)
        block[j0 : j0 + h, i0 : i0 + w] = True
        block &= valid

        overlap = int(block.sum())
        if overlap > best_overlap:
            best, best_overlap = block, overlap
        if overlap >= MIN_OVERLAP_FRACTION * h * w and overlap > 0:
            return block
    return best if best_overlap > 0 else valid.copy()


def sample_context_and_targets(
    valid: np.ndarray, cfg: DictConfig, rng: np.random.Generator
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sample one context block and `n_target_blocks` target blocks.

    Reading of the config numbers: 60% context plus 4 x 15% targets sums past
    100%, so they cannot all be disjoint slices of the same canvas. Following
    I-JEPA, target blocks are drawn first and may overlap *each other*; the
    context block is drawn at 60% and then has the target union removed from it.
    The 60% is therefore the size before removal, and `allow_overlap: false`
    governs context against targets, which is the overlap that would leak the
    answer.

    Returns boolean (grid, grid) masks indexed [j, i].
    """
    n_blocks = int(cfg.masking.n_target_blocks)
    target_masks = [
        _sample_block(valid, float(cfg.masking.target_block_ratio), rng) for _ in range(n_blocks)
    ]
    target_union = np.zeros_like(valid)
    for block in target_masks:
        target_union |= block

    context = _sample_block(valid, float(cfg.masking.context_ratio), rng)
    if not bool(cfg.masking.allow_overlap):
        context = context & ~target_union

    if not context.any():
        # Every context patch fell inside a target. Fall back to the whole valid
        # complement rather than returning an empty context.
        context = valid & ~target_union
    if not context.any():
        # Pathological: targets cover all valid patches. Give one block back.
        target_masks = target_masks[:1]
        context = valid & ~target_masks[0]

    target_masks = [block for block in target_masks if block.any()]
    assert target_masks, "no non-empty target block was sampled"
    assert not (context & target_union).any() or bool(cfg.masking.allow_overlap)
    assert not (context & ~valid).any(), "context escaped the valid region"
    return context, target_masks


def mask_to_indices(mask: np.ndarray) -> np.ndarray:
    """Flat token indices of a [j, i] mask, in row-major (j * grid + i) order."""
    return np.flatnonzero(mask.reshape(-1)).astype(np.int64)


def partition_interleaved(valid: np.ndarray, k: int = 4) -> list[np.ndarray]:
    """Split valid patches into `k` complementary interleaved groups.

    Group id of patch (i, j) is (j % m) * m + (i % m) with m = sqrt(k), so each
    group is spread evenly across the image instead of occupying one region.
    For group k the context is every valid patch outside P_k and the target is
    P_k itself, which gives every valid patch exactly one residual and makes
    repeated calls byte-identical. Padding patches are in no group.
    """
    m = int(round(math.sqrt(k)))
    assert m * m == k, f"k must be a perfect square for an interleaved grid, got {k}"
    grid_h, grid_w = valid.shape
    jj, ii = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    group_id = (jj % m) * m + (ii % m)
    return [valid & (group_id == g) for g in range(k)]


def pad_index_batch(
    index_lists: list[np.ndarray], device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ragged index arrays into (B, L) tensors plus a keep mask.

    N_ctx and N_tgt vary per sample because the valid region does. Padded slots
    carry index 0 and `keep = False`; every consumer must respect the keep mask,
    since token 0 is a real patch for some other sample in the batch.
    """
    lengths = [len(x) for x in index_lists]
    max_len = max(lengths) if lengths else 0
    batch = len(index_lists)
    idx = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    keep = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    for b, arr in enumerate(index_lists):
        n = len(arr)
        if n:
            idx[b, :n] = torch.from_numpy(np.asarray(arr)).to(device)
            keep[b, :n] = True
    return idx, keep


def build_batch_masks(
    valid_masks: torch.Tensor, cfg: DictConfig, rng: np.random.Generator
) -> dict:
    """Sample context and target blocks for a whole batch.

    Returns padded index tensors:
        ctx_idx  (B, Lc)      ctx_keep  (B, Lc)
        tgt_idx  (B, K, Lt)   tgt_keep  (B, K, Lt)
    where K is the number of target blocks. Blocks stay separate rather than
    being merged into one target set: the predictor is run once per block so
    that mask tokens of different blocks cannot attend to each other, which is
    what I-JEPA does and what keeps each block a genuine prediction from context
    alone.
    """
    device = valid_masks.device
    valid_np = valid_masks.detach().cpu().numpy()
    n_blocks = int(cfg.masking.n_target_blocks)

    ctx_lists: list[np.ndarray] = []
    tgt_lists: list[list[np.ndarray]] = [[] for _ in range(n_blocks)]
    for valid in valid_np:
        context, targets = sample_context_and_targets(valid, cfg, rng)
        ctx_lists.append(mask_to_indices(context))
        for b in range(n_blocks):
            block = targets[b % len(targets)]
            tgt_lists[b].append(mask_to_indices(block))

    ctx_idx, ctx_keep = pad_index_batch(ctx_lists, device)
    per_block = [pad_index_batch(lists, device) for lists in tgt_lists]
    max_len = max(t[0].shape[1] for t in per_block)
    batch = valid_masks.shape[0]
    tgt_idx = torch.zeros((batch, n_blocks, max_len), dtype=torch.long, device=device)
    tgt_keep = torch.zeros((batch, n_blocks, max_len), dtype=torch.bool, device=device)
    for b, (idx, keep) in enumerate(per_block):
        tgt_idx[:, b, : idx.shape[1]] = idx
        tgt_keep[:, b, : keep.shape[1]] = keep

    return {"ctx_idx": ctx_idx, "ctx_keep": ctx_keep, "tgt_idx": tgt_idx, "tgt_keep": tgt_keep}
