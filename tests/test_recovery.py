"""Tests for recovering the AHT parameters from measured data."""

import numpy as np
import pytest

from dbtt import CharpySimulator, CharpySimulatorConfig
from dbtt.recovery import (
    MIN_WIDTH,
    compare_width_models,
    dbtt_from_curve,
    dbtt_posterior,
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


# --- the DBTT posterior ----------------------------------------------------
def _campaign(seed=0, n=18):
    """A noisy campaign across the testable range, and the simulator behind it."""
    from dbtt import CharpySimulator, CharpySimulatorConfig
    from dbtt.simulator import ENERGY_COL, TEMPERATURE_COL

    simulator = CharpySimulator(CharpySimulatorConfig(seed=seed))
    frame = simulator.run(list(np.linspace(-120, 100, n)), start_index=1)
    return (
        frame[TEMPERATURE_COL].to_numpy(),
        frame[ENERGY_COL].to_numpy(),
        simulator,
    )


def test_dbtt_posterior_is_a_normalised_density():
    t, y, _ = _campaign()
    post = dbtt_posterior(t, y)

    assert np.all(post.density >= 0)
    assert np.trapezoid(post.density, post.grid) == pytest.approx(1.0, abs=1e-6)
    # Mass has to be interior, or the grid clipped the posterior.
    assert post.density[0] < post.density.max() * 1e-3
    assert post.density[-1] < post.density.max() * 1e-3


def test_dbtt_posterior_brackets_the_truth():
    t, y, simulator = _campaign()
    post = dbtt_posterior(t, y)

    low, high = post.interval
    assert low < high
    assert low <= simulator.dbtt <= high
    assert post.mean == pytest.approx(simulator.dbtt, abs=3 * post.std)


def test_dbtt_posterior_summaries_agree_with_the_grid():
    t, y, _ = _campaign()
    post = dbtt_posterior(t, y)

    assert post.mode == pytest.approx(post.grid[int(np.argmax(post.density))])
    assert post.std > 0
    # Equal-tailed, so the mean sits inside its own interval.
    assert post.interval[0] < post.mean < post.interval[1]


def test_dbtt_posterior_narrows_with_more_data():
    sparse = dbtt_posterior(*_campaign(n=8)[:2])
    dense = dbtt_posterior(*_campaign(n=30)[:2])
    assert dense.std < sparse.std


def test_dbtt_posterior_reuses_the_fitted_noise_weights():
    t, y, _ = _campaign()
    result = recover(t, y, symmetric=True)
    assert result.trace["weights"] is not None

    post = dbtt_posterior(t, y, result=result)
    # Same fixed weights as the shape search, so the grid evidences are
    # comparable with the evidence `recover` reported.
    assert post.log_evidence.shape == (post.grid.size, post.width_grid.size)
    assert np.isfinite(post.log_evidence).any()


def test_dbtt_posterior_accepts_explicit_grids():
    t, y, _ = _campaign()
    grid = np.linspace(-80.0, 0.0, 41)
    post = dbtt_posterior(t, y, dbtt_grid=grid, width_grid=np.array([20.0, 30.0]))
    assert np.array_equal(post.grid, grid)
    assert post.width_grid.size == 2


# --- reading the DBTT off a fitted curve -----------------------------------
def test_dbtt_from_curve_finds_the_transition_on_the_true_curve():
    _, _, simulator = _campaign()
    t = np.linspace(-120.0, 100.0, 201)
    read = dbtt_from_curve(t, simulator.true_curve(t))

    # Exact to the grid spacing, since the crossing is interpolated linearly.
    assert read.dbtt == pytest.approx(simulator.dbtt, abs=1.5)
    assert read.n_crossings == 1
    assert read.std is None  # no uncertainty supplied, none invented


def test_dbtt_from_curve_propagates_uncertainty_through_the_slope():
    _, _, simulator = _campaign()
    t = np.linspace(-120.0, 100.0, 201)
    y = simulator.true_curve(t)

    narrow = dbtt_from_curve(t, y, sd=np.full(t.size, 4.0))
    wide = dbtt_from_curve(t, y, sd=np.full(t.size, 8.0))

    # Delta method: twice the vertical spread is twice the horizontal spread.
    assert wide.std == pytest.approx(2 * narrow.std, rel=1e-6)
    assert narrow.std == pytest.approx(4.0 / abs(narrow.gradient), rel=1e-6)


def test_dbtt_from_curve_takes_the_steepest_crossing():
    # A transition with a small dip that re-crosses the mid-shelf level.
    t = np.linspace(-100.0, 100.0, 401)
    y = 100.0 + 95.0 * np.tanh((t + 40.0) / 25.0)
    level = 0.5 * (y.min() + y.max())
    y = y + 1.2 * (y > level) * np.sin((t - 20.0) / 6.0) * (np.abs(t - 40.0) < 20)

    read = dbtt_from_curve(t, y)
    assert read.dbtt == pytest.approx(-40.0, abs=3.0)


def test_dbtt_from_curve_refuses_a_curve_with_no_transition():
    t = np.linspace(-100.0, 100.0, 50)
    with pytest.raises(ValueError, match="never crosses"):
        dbtt_from_curve(t, np.full(t.size, 5.0))


def test_dbtt_from_curve_checks_its_inputs():
    t = np.linspace(-100.0, 100.0, 50)
    y = 100.0 + 95.0 * np.tanh((t + 40.0) / 25.0)
    with pytest.raises(ValueError, match="differ in length"):
        dbtt_from_curve(t, y[:-1])
    with pytest.raises(ValueError, match="sd differs in length"):
        dbtt_from_curve(t, y, sd=np.ones(3))
    with pytest.raises(ValueError, match="at least three"):
        dbtt_from_curve([1.0, 2.0], [1.0, 2.0])


def test_dbtt_from_curve_agrees_with_the_data_posterior():
    t, y, simulator = _campaign(n=30)
    posterior = dbtt_posterior(t, y)

    grid = np.linspace(-120.0, 100.0, 201)
    read = dbtt_from_curve(grid, simulator.true_curve(grid))

    # Two routes to the same quantity, so they should agree to within the
    # spread the measurements support. Not interval containment: the data
    # posterior is tight here and mildly biased, and at 95% nominal coverage it
    # is allowed to exclude the truth — which is exactly what it does on some
    # seeds.
    assert read.dbtt == pytest.approx(simulator.dbtt, abs=2.0)
    assert abs(read.dbtt - posterior.mean) < 3 * posterior.std + 2.0
