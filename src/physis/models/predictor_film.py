"""FiLM-conditioned predictor.

This is the one place the condition vector is allowed to enter. The encoder must
stay age-blind so that z = f_bar(x) is a fixed reference: sweeping candidate ages
then moves only the prediction, and the residual curve measures the effect of
age alone. Injecting c into the encoder instead makes the target move together
with the sweep, which is what E1b is built to demonstrate.

FiLM per layer, following SPEC section 2:

    h <- gamma(c) * LayerNorm(h) + beta(c)

The last layer of each FiLM MLP is zero-initialized so gamma = 1 and beta = 0 at
step 0. That point is exactly the degenerate solution the method has to avoid —
a predictor that ignores age entirely — which is why the margin loss runs from
step 0 with no warmup.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from timm.layers import Mlp, trunc_normal_
from timm.models.vision_transformer import Attention
from torch import nn

from .vit import build_attn_mask


class FiLMBlock(nn.Module):
    """Transformer block whose pre-attention norm is modulated by c.

    One (gamma, beta) pair per layer, applied at the attention norm, which is the
    literal reading of the SPEC equation. The MLP branch keeps an ordinary
    LayerNorm.
    """

    def __init__(self, dim: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.film = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        gamma = (1.0 + scale).unsqueeze(1)
        beta = shift.unsqueeze(1)
        x = x + self.attn(gamma * self.norm1(x) + beta, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class FiLMPredictor(nn.Module):
    """Predict target-patch encodings from context encodings, conditioned on c."""

    def __init__(
        self,
        *,
        encoder_dim: int,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        cond_dim: int = 128,
        num_patches: int = 576,
    ):
        super().__init__()
        self.ctx_proj = nn.Linear(encoder_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [FiLMBlock(dim, heads, cond_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, encoder_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

    def _gather_pos(self, idx: torch.Tensor) -> torch.Tensor:
        pos = self.pos_embed.expand(idx.shape[0], -1, -1)
        return torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))

    def forward(
        self,
        z_ctx: torch.Tensor,
        ctx_idx: torch.Tensor,
        ctx_keep: torch.Tensor,
        tgt_idx: torch.Tensor,
        tgt_keep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, N_tgt, encoder_dim) predictions for the target positions."""
        ctx_tokens = self.ctx_proj(z_ctx) + self._gather_pos(ctx_idx)
        tgt_tokens = self.mask_token + self._gather_pos(tgt_idx)

        tokens = torch.cat([ctx_tokens, tgt_tokens], dim=1)
        keep = torch.cat([ctx_keep, tgt_keep], dim=1)
        attn_mask = build_attn_mask(keep, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, cond, attn_mask)

        n_target = tgt_idx.shape[1]
        return self.out_proj(self.norm(tokens[:, -n_target:]))


def build_predictor(cfg: DictConfig, encoder_dim: int, num_patches: int) -> FiLMPredictor:
    assert str(cfg.predictor.film_init) == "zero_last", (
        "film_init must stay zero_last; the margin loss is calibrated to start there"
    )
    return FiLMPredictor(
        encoder_dim=encoder_dim,
        dim=int(cfg.predictor.dim),
        depth=int(cfg.predictor.depth),
        heads=int(cfg.predictor.heads),
        cond_dim=int(cfg.condition.dim),
        num_patches=num_patches,
    )
