"""Tests for the SDK workflow builders.

The builder tests need the SDK, which cannot share an environment with the
in-process library, so they skip unless run in the SDK environment:

    .venv-sdk/bin/python -m pytest tests/test_sdk_workflows.py
"""

import os

import pytest

from dbtt import backend
from dbtt import sdk_workflows as W

needs_sdk = pytest.mark.skipif(
    not backend.has_sdk(), reason="requires the uncertainty-engine SDK environment"
)


# --- these run anywhere ----------------------------------------------------
def test_acquisition_names_map_to_sdk_values():
    assert W.ACQUISITIONS["qNIPV"] == "MonteCarloNegativeIntegratedPosteriorVariance"
    assert W.ACQUISITIONS["PSD"] == "PosteriorStandardDeviation"


def test_campaign_to_frames_splits_x_and_y():
    x, y = W.campaign_to_frames([-100.0, 0.0], [5.0, 190.0])
    assert list(x.columns) == ["temperature_C"]
    assert list(y.columns) == ["energy_J"]
    assert len(x) == len(y) == 2


def test_temperature_grid_spans_the_bounds():
    grid = W.temperature_grid((-150.0, 100.0), n=11)
    assert len(grid) == 11
    assert grid["temperature_C"].iloc[0] == -150.0
    assert grid["temperature_C"].iloc[-1] == 100.0


def test_credentials_present_reads_the_environment(monkeypatch):
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert not W.credentials_present()
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.setenv(key, "placeholder")
    assert W.credentials_present()


def test_load_credentials_reports_missing(monkeypatch, tmp_path):
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert W.load_credentials(tmp_path / "absent.env") == "missing"


def test_load_credentials_reads_a_dotenv(monkeypatch, tmp_path):
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("UE_USERNAME=someone\nUE_PASSWORD=placeholder\n")
    assert W.load_credentials(env) == "dotenv"
    assert os.environ["UE_USERNAME"] == "someone"


def test_backend_detection_is_one_of_three():
    assert backend.detect_backend() in ("library", "sdk", "none")
    assert backend.has_library() != backend.has_sdk() or backend.detect_backend() == "none"


def test_backend_describe_is_consistent():
    described = backend.describe()
    assert described["in_process_modelling"] == (described["backend"] == "library")
    assert described["remote_workflows"] == (described["backend"] == "sdk")


def test_bad_acquisition_is_rejected_before_touching_the_sdk():
    with pytest.raises((ValueError, ImportError)):
        W.build_recommendation_workflow(None, "p", "m", acquisition="nope")


# --- these need the SDK ----------------------------------------------------
@needs_sdk
def test_training_workflow_wires_config_and_data():
    wf = W.build_training_workflow(None, "proj", "x-id", "y-id", "dbtt_gp")
    nodes = wf.graph["nodes"]
    assert set(nodes) == {
        "GP config", "Load temperatures", "Load energies", "Train GP", "Save GP",
    }
    train = nodes["Train GP"]["inputs"]
    assert train["config"]["node_name"] == "GP config"
    assert train["inputs"]["node_name"] == "Load temperatures"
    assert train["outputs"]["node_name"] == "Load energies"
    assert nodes["Save GP"]["inputs"]["data"]["node_name"] == "Train GP"


@needs_sdk
def test_prediction_workflow_predicts_from_the_saved_model():
    wf = W.build_prediction_workflow(None, "proj", "dbtt_gp", "grid-id")
    nodes = wf.graph["nodes"]
    assert set(nodes) == {"Load GP", "Load grid", "Predict curve"}
    predict = nodes["Predict curve"]["inputs"]
    assert predict["model"]["node_name"] == "Load GP"
    assert predict["dataset"]["node_name"] == "Load grid"


@needs_sdk
def test_recommendation_workflow_uses_the_trained_model():
    wf = W.build_recommendation_workflow(None, "proj", "dbtt_gp", "qNIPV", 2)
    nodes = wf.graph["nodes"]
    assert "Recommend (qNIPV)" in nodes
    assert nodes["Recommend (qNIPV)"]["inputs"]["model"]["node_name"] == "Load GP"


@needs_sdk
def test_every_acquisition_builds():
    for acquisition in W.ACQUISITIONS:
        wf = W.build_recommendation_workflow(None, "proj", "m", acquisition)
        assert f"Recommend ({acquisition})" in wf.graph["nodes"]


@needs_sdk
def test_connect_refuses_without_credentials(monkeypatch):
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="credentials are not set"):
        W.connect("AnyProject")
