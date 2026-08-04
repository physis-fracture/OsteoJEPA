"""Loss-term behaviour, especially the two that can be satisfied the wrong way."""

import numpy as np
import pytest
import torch

from physis.losses.jepa import masked_mean, patch_residual
from physis.losses.margin import age_grid, age_sensitivity, sample_distractor_ages
from physis.losses.vicreg import covariance_loss, variance_loss
from physis.models.osteojepa import momentum_at

GRID = age_grid(0.5, 19.0, 0.5)
TAU = 3.0


def test_age_grid_is_the_candidate_set_from_the_paper():
    assert GRID[0] == 0.5 and GRID[-1] == 19.0
    assert len(GRID) == 38
    assert np.allclose(np.diff(GRID), 0.5)


def test_distractors_are_symmetric_when_both_sides_exist():
    rng = np.random.default_rng(0)
    ages = np.full(64, 10.0, dtype=np.float32)
    picks = sample_distractor_ages(ages, GRID, TAU, rng)
    assert picks.shape == (64, 2)
    assert (picks[:, 0] <= 10.0 - TAU).all(), "the first distractor must come from below"
    assert (picks[:, 1] >= 10.0 + TAU).all(), "the second distractor must come from above"


def test_both_distractors_come_from_above_for_a_toddler():
    """A two-year-old has no younger distractor, so both are taken from above."""
    rng = np.random.default_rng(1)
    picks = sample_distractor_ages(np.full(32, 2.0, dtype=np.float32), GRID, TAU, rng)
    assert (picks >= 2.0 + TAU).all()


def test_both_distractors_come_from_below_for_a_late_teenager():
    rng = np.random.default_rng(2)
    picks = sample_distractor_ages(np.full(32, 17.5, dtype=np.float32), GRID, TAU, rng)
    assert (picks <= 17.5 - TAU).all()


def test_distractors_respect_the_minimum_distance():
    rng = np.random.default_rng(3)
    ages = np.linspace(0.5, 19.0, 40).astype(np.float32)
    picks = sample_distractor_ages(ages, GRID, TAU, rng)
    assert (np.abs(picks - ages[:, None]) >= TAU - 1e-6).all()


def test_distractor_sampling_is_seed_reproducible():
    first = sample_distractor_ages(np.full(8, 11.0), GRID, TAU, np.random.default_rng(5))
    second = sample_distractor_ages(np.full(8, 11.0), GRID, TAU, np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_residual_is_a_mean_over_the_feature_dimension():
    pred = torch.zeros(2, 3, 384)
    target = torch.ones(2, 3, 384)
    assert torch.allclose(patch_residual(pred, target), torch.ones(2, 3))


def test_masked_mean_ignores_padded_slots():
    values = torch.tensor([[1.0, 100.0], [3.0, 100.0]])
    keep = torch.tensor([[True, False], [True, False]])
    assert float(masked_mean(values, keep)) == 2.0


def test_variance_loss_fires_on_a_collapsed_representation():
    collapsed = torch.ones(4, 10, 8)
    keep = torch.ones(4, 10, dtype=torch.bool)
    loss, std = variance_loss(collapsed, keep)
    assert float(loss) > 0.9, "a constant representation must be penalised"
    assert float(std.max()) < 0.05


def test_variance_loss_is_silent_on_a_healthy_representation():
    torch.manual_seed(0)
    healthy = torch.randn(16, 20, 8) * 2.0
    keep = torch.ones(16, 20, dtype=torch.bool)
    loss, std = variance_loss(healthy, keep)
    assert float(loss) == 0.0
    assert float(std.min()) > 1.0


def test_variance_is_reported_per_dimension_not_aggregated():
    """A collapse confined to one dimension must be visible."""
    torch.manual_seed(0)
    tokens = torch.randn(32, 10, 4) * 3.0
    tokens[..., 2] = 1.0  # one dead dimension among healthy ones
    keep = torch.ones(32, 10, dtype=torch.bool)
    _, std = variance_loss(tokens, keep)
    assert float(std.min()) < 0.05
    assert float(std.mean()) > 1.0, "the mean hides the dead dimension"


def test_covariance_loss_punishes_correlated_dimensions():
    torch.manual_seed(0)
    base = torch.randn(256, 1)
    correlated = torch.cat([base, base, base, base], dim=1)[None]
    independent = torch.randn(1, 256, 4)
    keep = torch.ones(1, 256, dtype=torch.bool)
    assert float(covariance_loss(correlated, keep)) > float(covariance_loss(independent, keep))


def test_age_sensitivity_is_nan_where_the_patch_is_masked():
    residuals = torch.tensor([[[1.0, 2.0], [3.0, 9.0]]])  # (B=1, ages=2, patches=2)
    keep = torch.tensor([[True, False]])
    v = age_sensitivity(residuals, keep)
    assert not torch.isnan(v[0, 0])
    assert torch.isnan(v[0, 1])


def test_age_sensitivity_is_zero_for_a_flat_residual_curve():
    """The failure the margin loss exists to prevent."""
    flat = torch.ones(1, 5, 3)
    keep = torch.ones(1, 3, dtype=torch.bool)
    assert float(age_sensitivity(flat, keep).max()) == 0.0


@pytest.mark.parametrize(
    "step,total,expected", [(0, 100, 0.996), (99, 100, 1.0), (49, 100, 0.996 + 0.004 * 49 / 99)]
)
def test_ema_momentum_spans_the_steps_actually_run(step, total, expected):
    assert momentum_at(step, total, 0.996, 1.0) == pytest.approx(expected)


def test_ema_momentum_does_not_saturate_early_on_a_short_budget():
    """The I-JEPA schedule would freeze the target encoder before the predictor
    can use it; the schedule must stretch over the real step count."""
    assert momentum_at(500, 1000, 0.996, 1.0) < 0.999
