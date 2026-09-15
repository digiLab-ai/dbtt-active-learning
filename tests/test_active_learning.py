"""Tests for the Bayesian-optimisation active-learning loop."""

import matplotlib
import numpy as np
import pytest

from dbtt import CharpySimulator, CharpySimulatorConfig
from dbtt.active_learning import ACQUISITIONS, DBTTActiveLearner, initial_design
from dbtt.plotting import plot_gp

matplotlib.use("Agg")

# Keep the acquisition argmax cheap: these tests check plumbing, not convergence.
LIGHT = {"raw_samples": 32, "num_restarts": 3}


@pytest.fixture
def learner():
    return DBTTActiveLearner(CharpySimulator(), acquisition="PSD", **LIGHT)


def test_initial_design_is_the_two_limits():
    np.testing.assert_allclose(initial_design((-150.0, 100.0)), [-150.0, 100.0])


def test_initial_design_fills_in_evenly():
    np.testing.assert_allclose(initial_design((-100.0, 100.0), n=5),
                               [-100.0, -50.0, 0.0, 50.0, 100.0])


def test_initial_design_rejects_bad_input():
    with pytest.raises(ValueError, match="high > low"):
        initial_design((100.0, -100.0))
    with pytest.raises(ValueError, match="at least the two limits"):
        initial_design((-100.0, 100.0), n=1)


def test_learner_starts_from_the_initial_design(learner):
    np.testing.assert_allclose(sorted(learner.temperatures), [-150.0, 100.0])
    assert len(learner.energies) == 2
    assert learner.history == []


def test_unknown_acquisition_is_rejected():
    with pytest.raises(ValueError, match="acquisition must be one of"):
        DBTTActiveLearner(CharpySimulator(), acquisition="EI")


def test_every_declared_acquisition_can_propose():
    for acquisition in ACQUISITIONS:
        al = DBTTActiveLearner(CharpySimulator(), acquisition=acquisition, **LIGHT)
        temperature, _ = al.propose()
        assert al.bounds[0] <= temperature <= al.bounds[1]


def test_step_adds_one_observation(learner):
    before = len(learner.temperatures)
    record = learner.step()
    assert len(learner.temperatures) == before + 1
    assert record.n_observations == before + 1
    assert learner.bounds[0] <= record.temperature <= learner.bounds[1]
    assert record.energy >= 0.0


def test_run_records_history(learner):
    steps = learner.run(3)
    assert len(steps) == 3
    assert [s.index for s in steps] == [0, 1, 2]
    assert len(learner.temperatures) == 5


def test_proposals_stay_inside_the_bounds():
    al = DBTTActiveLearner(CharpySimulator(), bounds=(-80.0, 20.0),
                           acquisition="PSD", **LIGHT)
    for step in al.run(3):
        assert -80.0 <= step.temperature <= 20.0


def test_active_learning_reduces_uncertainty(learner):
    before = learner.mean_posterior_std()
    learner.run(5)
    assert learner.mean_posterior_std() < before


def test_posterior_shapes_match_the_grid(learner):
    grid = np.linspace(-150, 100, 50)
    x, mean, std = learner.posterior(grid)
    assert x.shape == mean.shape == std.shape == (50,)
    assert np.all(std > 0)


def test_dbtt_estimate_is_inside_the_bounds(learner):
    learner.run(4)
    assert learner.bounds[0] <= learner.dbtt_estimate() <= learner.bounds[1]


def test_plot_gp_draws_the_observations(learner):
    learner.run(2)
    ax = plot_gp(learner, simulator=learner.simulator, proposed=-30.0,
                 grid=np.linspace(-150, 100, 50))
    assert ax.get_xlabel() == "Test temperature [C]"
    assert ax.collections  # the band and the scatter
