"""ImageNet weight adaptation, on a synthetic state dict so no download is needed.

Pitfall 6 in SPEC: loading ImageNet weights by slicing one channel instead of
summing three. The model trains either way and the loss still falls, so this is
tested rather than trusted.
"""

import torch

from physis.models.vit import _interpolate_pos_embed, adapt_imagenet_state_dict

DIM = 8
PATCH = 16


def fake_state_dict(depth: int = 12, grid: int = 14) -> dict:
    weight = torch.stack(
        [torch.full((DIM, PATCH, PATCH), float(c + 1)) for c in range(3)], dim=1
    )  # channel c holds the constant c + 1
    state = {
        "cls_token": torch.zeros(1, 1, DIM),
        "pos_embed": torch.randn(1, grid * grid + 1, DIM),
        "patch_embed.proj.weight": weight,
        "patch_embed.proj.bias": torch.zeros(DIM),
        "norm.weight": torch.ones(DIM),
        "norm.bias": torch.zeros(DIM),
        "head.weight": torch.zeros(10, DIM),
        "head.bias": torch.zeros(10),
    }
    for block in range(depth):
        state[f"blocks.{block}.norm1.weight"] = torch.ones(DIM)
    return state


def test_patch_embed_is_summed_not_sliced():
    adapted = adapt_imagenet_state_dict(fake_state_dict(), dst_grid=24, depth=12)
    weight = adapted["patch_embed.proj.weight"]
    assert weight.shape == (DIM, 1, PATCH, PATCH)
    # Summing 1 + 2 + 3 gives 6; slicing any single channel would give 1, 2 or 3.
    assert torch.allclose(weight, torch.full_like(weight, 6.0))


def test_pos_embed_is_interpolated_and_loses_the_class_token():
    adapted = adapt_imagenet_state_dict(fake_state_dict(), dst_grid=24, depth=12)
    assert adapted["pos_embed"].shape == (1, 24 * 24, DIM)
    assert "cls_token" not in adapted


def test_blocks_beyond_the_configured_depth_are_dropped():
    adapted = adapt_imagenet_state_dict(fake_state_dict(depth=12), dst_grid=24, depth=2)
    kept = sorted(k for k in adapted if k.startswith("blocks."))
    assert kept == ["blocks.0.norm1.weight", "blocks.1.norm1.weight"]


def test_classifier_head_is_dropped():
    adapted = adapt_imagenet_state_dict(fake_state_dict(), dst_grid=24, depth=12)
    assert not any(k.startswith("head") for k in adapted)


def test_pos_embed_interpolation_preserves_a_constant_field():
    pos = torch.full((1, 14 * 14, DIM), 3.0)
    out = _interpolate_pos_embed(pos, 14, 24)
    assert out.shape == (1, 24 * 24, DIM)
    assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-5)


def test_pos_embed_without_a_class_token_is_handled():
    state = fake_state_dict()
    state["pos_embed"] = torch.randn(1, 14 * 14, DIM)
    adapted = adapt_imagenet_state_dict(state, dst_grid=24, depth=12)
    assert adapted["pos_embed"].shape == (1, 24 * 24, DIM)
