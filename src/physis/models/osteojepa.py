"""OsteoJEPA: context encoder, EMA target encoder, FiLM predictor."""

from __future__ import annotations

import copy

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .condition import build_condition_encoder
from .predictor_film import build_predictor
from .vit import build_encoder


def momentum_at(step: int, total_steps: int, start: float, end: float) -> float:
    """EMA momentum, linear from `start` to `end` over the steps actually run.

    The I-JEPA schedule assumes 300-800 epochs. Reusing its original length on a
    100-epoch budget drives the momentum to 1.0 early and freezes the target
    encoder before the predictor has had a chance to use it.
    """
    if total_steps <= 1:
        return end
    fraction = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + (end - start) * fraction


class OsteoJEPA(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if str(cfg.condition.inject) != "predictor":
            raise NotImplementedError(
                f"condition.inject={cfg.condition.inject!r} is the E1b variant. "
                "It is scheduled for M8 and deliberately not part of the skeleton."
            )

        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.condition = build_condition_encoder(cfg)
        self.predictor = build_predictor(
            cfg, encoder_dim=self.encoder.embed_dim, num_patches=self.encoder.num_patches
        )

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def encode_context(
        self, images: torch.Tensor, ctx_idx: torch.Tensor, ctx_keep: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(images, keep_idx=ctx_idx, keep_mask=ctx_keep)

    @torch.no_grad()
    def encode_targets(self, images: torch.Tensor) -> torch.Tensor:
        """Full-image target encoding z = f_bar(x); (B, 576, dim), no gradients."""
        self.target_encoder.eval()
        return self.target_encoder(images)

    def condition_vector(self, age: torch.Tensor, meta: dict) -> torch.Tensor:
        return self.condition(age, meta["gender"], meta["view"], meta["laterality"])

    def predict(
        self,
        z_ctx: torch.Tensor,
        ctx_idx: torch.Tensor,
        ctx_keep: torch.Tensor,
        tgt_idx: torch.Tensor,
        tgt_keep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(z_ctx, ctx_idx, ctx_keep, tgt_idx, tgt_keep, cond)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)
        for target, source in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            target.copy_(source)


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, N, D) tokens at (B, L) indices."""
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def load_osteojepa(cfg: DictConfig, checkpoint_path, device) -> OsteoJEPA:
    """Rebuild a model and load a checkpoint into it.

    The encoder is built with random weights first: every parameter is about to
    be overwritten, so downloading the ImageNet checkpoint here would only cost
    time and hide a genuinely incomplete state dict behind pretrained values.
    """
    eval_cfg = OmegaConf.merge(cfg, OmegaConf.create({"encoder": {"init": "random"}}))
    model = OsteoJEPA(eval_cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    assert not missing and not unexpected
    return model.to(device).eval()
