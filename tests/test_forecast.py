"""ForecastDistribution math + uncertainty decomposition properties.

These tests are the quant floor: if they don't hold, probabilistic reasoning
downstream is broken."""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_edge.forecast import (
    ForecastDistribution,
    bernoulli_entropy,
    decompose_binary_uncertainty,
)
from kalshi_edge.forecasters.base import bayesian_model_average


def test_bernoulli_entropy_bounds() -> None:
    assert bernoulli_entropy(0.5) == pytest.approx(np.log(2), rel=1e-9)
    assert bernoulli_entropy(0.0) == pytest.approx(0.0, abs=1e-9)
    assert bernoulli_entropy(1.0) == pytest.approx(0.0, abs=1e-9)


def test_decomposition_identity() -> None:
    # Total = aleatoric + epistemic by construction.
    rng = np.random.default_rng(0)
    p = rng.beta(2, 5, size=10_000)
    d = decompose_binary_uncertainty(p)
    assert d["total"] == pytest.approx(d["aleatoric"] + d["epistemic"], rel=1e-6)
    # Epistemic >= 0 always.
    assert d["epistemic"] >= -1e-9


def test_decomposition_zero_epistemic_when_degenerate() -> None:
    # If all samples are the same p, no parameter uncertainty.
    p = np.full(5_000, 0.37)
    d = decompose_binary_uncertainty(p)
    assert d["epistemic"] == pytest.approx(0.0, abs=1e-9)


def test_parametric_normal_quantile_and_cdf() -> None:
    d = ForecastDistribution(kind="parametric", family="normal", params={"loc": 60, "scale": 4})
    # P(T > 64) = 1 - Phi(1) ~ 0.1587
    assert d.prob_above(64) == pytest.approx(0.1587, abs=1e-3)
    assert d.mean() == pytest.approx(60.0)
    assert d.std() == pytest.approx(4.0)


def test_samples_distribution_matches_empirical() -> None:
    rng = np.random.default_rng(1)
    s = rng.normal(60, 4, size=100_000)
    d = ForecastDistribution(kind="samples", samples=s)
    assert d.prob_above(64) == pytest.approx(0.1587, abs=3e-3)
    assert d.quantile(0.5) == pytest.approx(60.0, abs=0.1)


def test_bma_mixture_mean_equals_weighted_mean() -> None:
    rng = np.random.default_rng(2)
    a = ForecastDistribution(kind="parametric", family="normal", params={"loc": 50, "scale": 3})
    b = ForecastDistribution(kind="parametric", family="normal", params={"loc": 70, "scale": 3})
    bma = bayesian_model_average([a, b], [0.3, 0.7], n_samples=200_000, rng=rng)
    assert bma.mean() == pytest.approx(0.3 * 50 + 0.7 * 70, abs=0.2)


def test_bma_rejects_bad_weights() -> None:
    a = ForecastDistribution(kind="parametric", family="normal", params={"loc": 0, "scale": 1})
    with pytest.raises(ValueError):
        bayesian_model_average([a], [-1.0])
    with pytest.raises(ValueError):
        bayesian_model_average([a], [0.0])
