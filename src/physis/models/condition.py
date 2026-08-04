"""The condition vector c.

    c = [ FourierFeatures(age, 16) ; Embed(sex) ; Embed(view) ; Embed(laterality) ]

Fourier features give the predictor high resolution on a low-dimensional
continuous variable, so it can tell 7.0 years from 7.5 years without discrete
bins. Frequencies are log-spaced between 1 and 20 cycles over the 0-19 year
range: 20 cycles resolves roughly half a year, which matches the 0.5-year fine
grid of the inference sweep. Pushing to the usual 2^15 would put most bands
below the resolution the data can support and turn them into noise.

Each categorical field carries a trained `unknown` embedding, so an image with
missing metadata is scored rather than silently dropped.
"""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig
from torch import nn

from ..data.dataset import GENDER_UNKNOWN, LATERALITY_UNKNOWN, VIEW_UNKNOWN

MAX_AGE_CYCLES = 20.0


class FourierAgeFeatures(nn.Module):
    """Sin/cos features of age at `bands` log-spaced frequencies. Output: 2*bands."""

    def __init__(self, bands: int = 16, age_max: float = 19.0):
        super().__init__()
        self.age_max = float(age_max)
        freqs = torch.logspace(0.0, math.log2(MAX_AGE_CYCLES), bands, base=2.0)
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 2 * bands

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        normalized = (age / self.age_max).unsqueeze(-1)
        phase = 2.0 * math.pi * normalized * self.freqs
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class ConditionEncoder(nn.Module):
    """Metadata to a single (B, dim) condition vector."""

    def __init__(
        self,
        *,
        dim: int = 128,
        age_bands: int = 16,
        age_max: float = 19.0,
        embed_dim: int = 16,
        unknown_embedding: bool = True,
    ):
        super().__init__()
        assert unknown_embedding, "the unknown embedding is not optional; see SPEC section 2"
        self.age = FourierAgeFeatures(age_bands, age_max=age_max)
        # +1 slot for the `unknown` category of each field.
        self.gender = nn.Embedding(GENDER_UNKNOWN + 1, embed_dim)
        self.view = nn.Embedding(VIEW_UNKNOWN + 1, embed_dim)
        self.laterality = nn.Embedding(LATERALITY_UNKNOWN + 1, embed_dim)
        self.proj = nn.Linear(self.age.out_dim + 3 * embed_dim, dim)
        self.dim = dim

    def forward(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
        view: torch.Tensor,
        laterality: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            self.age(age),
            self.gender(gender),
            self.view(view),
            self.laterality(laterality),
        ]
        return self.proj(torch.cat(parts, dim=-1))


def build_condition_encoder(cfg: DictConfig) -> ConditionEncoder:
    return ConditionEncoder(
        dim=int(cfg.condition.dim),
        age_bands=int(cfg.condition.age_fourier_bands),
        age_max=float(cfg.inference.age_max),
        unknown_embedding=bool(cfg.condition.unknown_embedding),
    )
