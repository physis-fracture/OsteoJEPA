"""Supervised fracture classifier — the model the product ranks by.

Not a contribution and not presented as one. OsteoJEPA returned a null result
(see docs/EXPERIMENT_REVISION.md), so the triage score comes from a model
trained on fracture labels instead. What that costs the paper is the
annotation-free adoption claim, and the honest move is to say so rather than to
keep the old name on a supervised score.

Two things carry over from the JEPA path and both matter:

* The same `PhysisViT`, so ImageNet weights still arrive with `patch_embed`
  summed across channels and `pos_embed` interpolated to 24x24.
* Padding patches still never enter anything. Pooling is a masked mean over
  valid tokens; letting 43% of the canvas into the average would dilute every
  image by a different amount depending on its aspect ratio.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from .condition import build_condition_encoder
from .vit import build_encoder


class PhysisClassifier(nn.Module):
    """ViT-S over valid patches, optional age conditioning, one logit out.

    `use_condition` is the E2a switch. Concatenating the same condition vector
    the predictor used keeps the comparison honest: the two arms differ in
    whether age reaches the head, and in nothing else.
    """

    def __init__(self, cfg: DictConfig, *, use_condition: bool = True):
        super().__init__()
        self.encoder = build_encoder(cfg)
        self.use_condition = use_condition
        dim = self.encoder.embed_dim
        if use_condition:
            self.condition = build_condition_encoder(cfg)
            dim += int(cfg.condition.dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, images: torch.Tensor, valid_mask: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        tokens = self.encoder(images)                       # (B, 576, D)
        keep = valid_mask.flatten(1).unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

        if self.use_condition:
            cond = self.condition(
                meta["age"], meta["gender"], meta["view"], meta["laterality"]
            )
            pooled = torch.cat([pooled, cond], dim=-1)
        return self.head(self.norm(pooled)).squeeze(-1)


def load_backbone(model: PhysisClassifier, checkpoint_path: str) -> dict:
    """Initialize the encoder from a Stage A checkpoint instead of ImageNet.

    Used for the ablation that asks whether Stage A pretraining bought anything.
    If the two arms land in the same place, that is independent confirmation of
    the null - measured through a downstream task rather than through the skill
    ratio alone.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = state["model"] if "model" in state else state
    encoder_weights = {
        key[len("encoder.") :]: value
        for key, value in weights.items()
        if key.startswith("encoder.")
    }
    assert encoder_weights, f"no encoder.* weights found in {checkpoint_path}"
    missing, unexpected = model.encoder.load_state_dict(encoder_weights, strict=False)
    assert not unexpected, f"unexpected keys from the Stage A checkpoint: {unexpected}"
    return {"loaded": len(encoder_weights), "missing": len(missing)}


def build_classifier(cfg: DictConfig) -> PhysisClassifier:
    return PhysisClassifier(cfg, use_condition=bool(cfg.classifier.use_condition))
