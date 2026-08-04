"""ViT backbone with ImageNet initialization and variable-length token input.

Two adjustments are needed to put ImageNet weights on a 1-channel 384px model,
and both fail silently if done wrong:

1. `patch_embed.proj.weight` ships as (384, 3, 16, 16). It is **summed** across
   the channel dimension to (384, 1, 16, 16). Slicing out one channel would
   throw away two thirds of the learned filter energy, and the model would still
   train.
2. `pos_embed` is trained at 14x14 for 224px input and is bicubic-interpolated
   to 24x24 for 384px input, with the class token dropped rather than blended
   into the grid.

The class token is removed entirely: I-JEPA predicts patch representations and
never pools, so a class token would only be dead weight carrying a position
embedding nothing consumes.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


def _interpolate_pos_embed(pos_embed: torch.Tensor, src_grid: int, dst_grid: int) -> torch.Tensor:
    """Bicubic-resize a (1, src_grid^2, dim) position embedding to dst_grid^2."""
    if src_grid == dst_grid:
        return pos_embed
    dim = pos_embed.shape[-1]
    grid = pos_embed.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(dst_grid, dst_grid), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, dst_grid * dst_grid, dim)


def adapt_imagenet_state_dict(
    state_dict: dict, *, dst_grid: int, depth: int, in_chans: int = 1
) -> dict:
    """Convert a pretrained 3-channel 224px ViT state dict to our configuration."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("head") or key in {"cls_token", "pre_logits.fc.weight", "pre_logits.fc.bias"}:
            continue
        if key.startswith("blocks."):
            block_index = int(key.split(".")[1])
            if block_index >= depth:
                continue
        if key == "patch_embed.proj.weight":
            assert value.ndim == 4, f"unexpected patch_embed shape {tuple(value.shape)}"
            if in_chans == 1 and value.shape[1] != 1:
                value = value.sum(dim=1, keepdim=True)
            assert value.shape[1] == in_chans
        if key == "pos_embed":
            n_tokens = value.shape[1]
            src_grid = int(round((n_tokens - 1) ** 0.5))
            if src_grid * src_grid == n_tokens - 1:
                value = value[:, 1:]  # drop the class-token position
            else:
                src_grid = int(round(n_tokens**0.5))
                assert src_grid * src_grid == n_tokens, f"pos_embed length {n_tokens} is not a grid"
            value = _interpolate_pos_embed(value, src_grid, dst_grid)
        out[key] = value
    return out


class PhysisViT(nn.Module):
    """ViT over patch tokens, able to run on an arbitrary subset of positions.

    `forward(x, keep_idx, keep_mask)` embeds all patches, adds position
    embeddings, then gathers the requested subset. Gathering after the position
    embedding is what lets the context encoder see 60% of the patches while each
    token still knows where on the canvas it came from.

    N_ctx varies per sample, so subsets arrive padded. Padded slots are excluded
    from attention through an additive mask rather than being dropped, since
    they are real patch indices for other samples in the batch.
    """

    def __init__(
        self,
        *,
        timm_name: str,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 1,
        depth: int | None = None,
        init: str = "imagenet",
    ):
        super().__init__()
        kwargs = dict(
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
            class_token=False,
            global_pool="",
        )
        if depth is not None:
            kwargs["depth"] = depth
        self.net = timm.create_model(timm_name, **kwargs)

        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.embed_dim = self.net.embed_dim
        assert self.net.pos_embed.shape[1] == self.num_patches, (
            f"pos_embed holds {self.net.pos_embed.shape[1]} tokens, expected {self.num_patches}"
        )

        if init == "imagenet":
            self._load_imagenet(timm_name, depth=depth or len(self.net.blocks), in_chans=in_chans)
        elif init != "random":
            raise ValueError(f"unknown encoder init {init!r}")

    def _load_imagenet(self, timm_name: str, *, depth: int, in_chans: int) -> None:
        source = timm.create_model(timm_name, pretrained=True, num_classes=0)
        adapted = adapt_imagenet_state_dict(
            source.state_dict(), dst_grid=self.grid, depth=depth, in_chans=in_chans
        )
        missing, unexpected = self.net.load_state_dict(adapted, strict=False)
        assert not unexpected, f"unexpected keys when loading ImageNet weights: {unexpected}"
        # `missing` is legitimately non-empty only for parameters the source has
        # no counterpart for; anything in the patch embedding or the blocks we
        # kept would mean the adaptation silently skipped a tensor.
        suspicious = [k for k in missing if k.startswith(("patch_embed.", "pos_embed", "blocks."))]
        assert not suspicious, f"ImageNet weights did not cover: {suspicious}"

        weight = self.net.patch_embed.proj.weight
        assert weight.shape[1] == in_chans, "patch_embed was not adapted to 1 channel"

    def forward(
        self,
        images: torch.Tensor,
        keep_idx: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B, N, dim) token features for the requested positions."""
        tokens = self.net.patch_embed(images)
        tokens = tokens + self.net.pos_embed

        if keep_idx is not None:
            index = keep_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, 1, index)

        attn_mask = build_attn_mask(keep_mask, tokens.dtype) if keep_mask is not None else None
        for block in self.net.blocks:
            tokens = block(tokens, attn_mask=attn_mask)
        return self.net.norm(tokens)


def build_attn_mask(keep_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive (B, 1, 1, L) attention mask: 0 for real tokens, -inf for padding.

    Additive rather than boolean so it behaves identically on timm's fused and
    unfused attention paths.
    """
    mask = torch.zeros(keep_mask.shape, dtype=dtype, device=keep_mask.device)
    mask = mask.masked_fill(~keep_mask, float("-inf"))
    return mask[:, None, None, :]


def build_encoder(cfg: DictConfig) -> PhysisViT:
    return PhysisViT(
        timm_name=str(cfg.encoder.timm_name),
        img_size=int(cfg.encoder.img_size),
        patch_size=int(cfg.image.patch),
        in_chans=int(cfg.encoder.in_chans),
        depth=int(cfg.encoder.depth) if cfg.encoder.get("depth") else None,
        init=str(cfg.encoder.init),
    )
