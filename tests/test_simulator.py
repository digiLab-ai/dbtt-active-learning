import numpy as np
import pytest

from dbtt.interface import SimulationResult, Simulator

from dbtt import CharpySimulator, CharpySimulatorConfig
from dbtt.simulator import COLUMNS, ENERGY_COL, TEMPERATURE_COL

CONFIG_PATH = "configs/eurofer_eaf.yaml"


def test_curve_hits_midpoint_at_dbtt():
    sim = CharpySimulator()
    c = sim.config.curve
    assert sim.true_curve(c.dbtt) == pytest.approx(0.5 * (c.lower_shelf + c.upper_shelf))


def test_curve_approaches_shelves():
    sim = CharpySimulator()
    c = sim.config.curve
    assert sim.true_curve(-1000) == pytest.approx(c.lower_shelf, abs=1e-6)
    assert sim.true_curve(1000) == pytest.approx(c.upper_shelf, abs=1e-6)


def test_curve_is_monotonic():
    sim = CharpySimulator()
    y = sim.true_curve(np.linspace(-150, 100, 500))
    assert np.all(np.diff(y) > 0)


def test_asymmetry_uses_the_right_width():
    # Equal offsets either side of the DBTT give unequal departures from the
    # midpoint when the widths differ.
    sim = CharpySimulator(CharpySimulatorConfig(curve={"width_lower": 60.0, "width_upper": 20.0}))
    c = sim.config.curve
    mid = 0.5 * (c.lower_shelf + c.upper_shelf)
    assert (sim.true_curve(c.dbtt + 20) - mid) > (mid - sim.true_curve(c.dbtt - 20))


def test_default_config_is_symmetric():
    c = CharpySimulatorConfig.from_yaml(CONFIG_PATH).curve
    assert c.width_lower == c.width_upper


def test_noise_has_base_and_relative_components():
    sim = CharpySimulator()
    n = sim.config.noise
    sigma = sim.noise_std(sim.dbtt)
    assert sigma == pytest.approx(n.base + n.relative * sim.true_curve(sim.dbtt))


def test_measurements_are_seeded_and_reproducible():
    temps = [-100, -40, 20]
    a = CharpySimulator().simulate(temps).y
    b = CharpySimulator().simulate(temps).y
    np.testing.assert_allclose(a, b)


def test_repeat_tests_differ():
    sim = CharpySimulator()
    assert sim.simulate(-40).y[0] != sim.simulate(-40).y[0]


def test_energy_never_negative():
    cfg = CharpySimulatorConfig(noise={"base": 50.0, "relative": 0.0})
    sim = CharpySimulator(cfg)
    assert np.all(sim.simulate(np.full(500, -120.0)).y >= 0.0)


def test_simulate_returns_a_simulation_result():
    result = CharpySimulator().simulate(-45.0)
    assert isinstance(result, SimulationResult)
    assert result.y.shape == (1,)
    assert set(result.outputs) == {
        "temperature",
        "true_energy",
        "noise_std",
        "dbtt",
        "units",
    }


def test_simulate_outputs_carry_the_ground_truth():
    sim = CharpySimulator()
    result = sim.simulate([-100.0, 0.0])
    np.testing.assert_allclose(result.outputs["true_energy"], sim.true_curve([-100.0, 0.0]))
    np.testing.assert_allclose(result.outputs["noise_std"], sim.noise_std([-100.0, 0.0]))
    assert result.outputs["dbtt"] == sim.dbtt


def test_forward_is_the_raw_map_used_by_simulate():
    a = CharpySimulator()._forward([-40.0])
    b = CharpySimulator().simulate([-40.0]).y
    np.testing.assert_allclose(a, b)


def test_simulator_conforms_to_the_interface():
    assert issubclass(CharpySimulator, Simulator)
    assert CharpySimulator.data_type() is np.ndarray
    described = CharpySimulator.describe()
    assert described["name"] == "CharpySimulator"
    assert described["data_type"] == "ndarray"


def test_run_returns_full_table():
    rows = CharpySimulator().run([-100, -40, 20], start_index=5)
    assert list(rows.columns) == COLUMNS
    assert len(rows) == 3
    assert rows["Test house specimen ID"].tolist() == [5, 6, 7]
    assert rows["NEURONE specimen ID"].iloc[0] == "NRN2025-0005"
    assert rows[TEMPERATURE_COL].tolist() == [-100, -40, 20]
    assert rows[ENERGY_COL].notna().all()


def test_run_table_carries_no_ground_truth():
    """The dataset must hold only what a real test house would report."""
    rows = CharpySimulator().run([-40, 0])
    leaked = [c for c in rows.columns if "curve" in c.lower() or "true" in c.lower()]
    assert leaked == []


def test_out_of_range_temperature_rejected():
    with pytest.raises(ValueError, match="outside the configured range"):
        CharpySimulator().simulate(500.0)


def test_config_roundtrip(tmp_path):
    cfg = CharpySimulatorConfig.from_yaml(CONFIG_PATH)
    out = tmp_path / "cfg.yaml"
    cfg.to_yaml(out)
    assert CharpySimulatorConfig.from_yaml(out) == cfg


def test_config_rejects_bad_shelves():
    with pytest.raises(ValueError):
        CharpySimulatorConfig(curve={"lower_shelf": 200.0, "upper_shelf": 5.0})


def test_config_rejects_unknown_field():
    with pytest.raises(ValueError):
        CharpySimulatorConfig(curve={"nonsense": 1.0})
