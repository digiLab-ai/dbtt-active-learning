"""Tests for recovering the AHT parameters from measured data."""

import numpy as np
import pytest

from dbtt import CharpySimulator, CharpySimulatorConfig
from dbtt.recovery import (
    MIN_WIDTH,
    compare_width_models,
    estimate_noise_std,
    fit_shelves,
    recover,
    tanh_basis,
)

DENSE = np.array(
    [-120, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 20, 40, 60, 80],
    dtype=float,
)


def noiseless(temperatures=DENSE, **curve):
    cfg = CharpySimulatorConfig(noise={"base": 1e-9, "relative": 0.0}, curve=curve)
    sim = CharpySimulator(cfg)
    return sim, sim.simulate(temperatures).y


def test_tanh_basis_is_zero_at_the_dbtt():
    assert tanh_basis([-40.0], -40.0, 30.0)[0, 0] == pytest.approx(0.0)


def test_tanh_basis_switches_width_at_the_dbtt():
    basis = tanh_basis([-60.0, -20.0], -40.0, 40.0, 10.0).reshape(-1)
    assert basis[0] == pytest.approx(np.tanh(-20 / 40))
    assert basis[1] == pytest.approx(np.tanh(20 / 10))


def test_noiseless_recovery_is_exact():
    """The whole pipeline must invert its own forward map."""
    sim, y = noiseless()
    result = recover(DENSE, y, profile_dbtt=False)
    assert result.dbtt == pytest.approx(sim.config.curve.dbtt, abs=0.1)
    assert result.width_lower == pytest.approx(sim.config.curve.width_lower, abs=0.2)
    assert result.shelves.lower_shelf == pytest.approx(
        sim.config.curve.lower_shelf, abs=0.2
    )
    assert result.shelves.upper_shelf == pytest.approx(
        sim.config.curve.upper_shelf, abs=0.5
    )


def test_noiseless_asymmetric_recovery_is_exact():
    sim, y = noiseless(width_lower=55.0, width_upper=15.0)
    result = recover(DENSE, y, symmetric=False, profile_dbtt=False)
    assert result.width_lower == pytest.approx(55.0, abs=1.0)
    assert result.width_upper == pytest.approx(15.0, abs=1.0)


def test_recovery_is_close_on_noisy_data():
    y = CharpySimulator(CharpySimulatorConfig(seed=2)).simulate(DENSE).y
    result = recover(DENSE, y, profile_dbtt=False)
    assert result.dbtt == pytest.approx(-40.0, abs=12.0)
    assert result.shelves.upper_shelf == pytest.approx(200.0, abs=35.0)


def test_predict_matches_the_fitted_parameters():
    sim, y = noiseless()
    result = recover(DENSE, y, profile_dbtt=False)
    np.testing.assert_allclose(result.predict(DENSE), sim.true_curve(DENSE), atol=1.0)


def test_profile_interval_brackets_the_estimate():
    y = CharpySimulator(CharpySimulatorConfig(seed=2)).simulate(DENSE).y
    result = recover(DENSE, y)
    lower, upper = result.dbtt_interval
    assert lower is not None and upper is not None
    assert lower < result.dbtt < upper


def test_widths_respect_the_physical_floor():
    y = CharpySimulator(CharpySimulatorConfig(seed=5)).simulate(DENSE).y
    result = recover(DENSE, y, symmetric=False, profile_dbtt=False)
    assert result.width_lower >= MIN_WIDTH
    assert result.width_upper >= MIN_WIDTH


def test_weighting_is_reported():
    y = CharpySimulator().simulate(DENSE).y
    assert recover(DENSE, y, profile_dbtt=False).trace["heteroscedastic"] is True
    assert (
        recover(DENSE, y, profile_dbtt=False, heteroscedastic=False).trace[
            "heteroscedastic"
        ]
        is False
    )


def test_fixed_weights_make_evidence_comparable():
    """Evidence at two shapes must be comparable under one fixed weighting."""
    y = CharpySimulator().simulate(DENSE).y
    w = np.full(len(DENSE), 0.1)
    a = fit_shelves(DENSE, y, -40.0, 30.0, weights=w)
    b = fit_shelves(DENSE, y, -40.0, 20.0, weights=w)
    assert a.log_evidence != b.log_evidence
    # The true shape should score better than a badly wrong one.
    assert a.log_evidence > fit_shelves(DENSE, y, 60.0, 3.0, weights=w).log_evidence


def test_estimate_noise_std_tracks_growing_scatter():
    fitted = np.linspace(5, 200, 60)
    rng = np.random.default_rng(0)
    residual = rng.normal(0.0, 3.0 + 0.1 * fitted)
    sigma = estimate_noise_std(fitted, residual)
    assert sigma[-1] > sigma[0]
    assert np.all(sigma > 0)


def test_estimate_noise_std_falls_back_when_homoscedastic():
    fitted = np.linspace(5, 200, 60)
    rng = np.random.default_rng(0)
    sigma = estimate_noise_std(fitted, rng.normal(0.0, 4.0, size=60))
    assert np.all(sigma > 0)


def test_bic_penalises_the_extra_width():
    sim, y = noiseless()
    constant = recover(DENSE, y, profile_dbtt=False)
    asymmetric = recover(DENSE, y, symmetric=False, profile_dbtt=False)
    assert constant.n_shape_params == 2
    assert asymmetric.n_shape_params == 3
    assert asymmetric.bic > 2.0 * -asymmetric.log_evidence


def test_compare_width_models_prefers_constant_for_a_symmetric_truth():
    sim, y = noiseless()
    comparison = compare_width_models(DENSE, y)
    assert comparison["constant"].symmetric
    assert not comparison["asymmetric"].symmetric
    assert comparison["loo_rmse"]["constant"] < 1.0
