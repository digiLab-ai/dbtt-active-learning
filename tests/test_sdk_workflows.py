"""Tests for the SDK workflow builders.

The builder tests need the SDK, which cannot share an environment with the
in-process library, so they skip unless run in the SDK environment:

    .venv-sdk/bin/python -m pytest tests/test_sdk_workflows.py

They build every graph with ``client=None``, which skips schema resolution and
validation. That checks the *wiring* — labels, edges and requested outputs —
against the workflows exported from the Engine GUI, without a network call. It
cannot check the inputs against the live node schemas; only running against the
Engine does that.
"""

import os

import pytest

from dbtt import backend
from dbtt import sdk_workflows as W

needs_sdk = pytest.mark.skipif(
    not backend.has_sdk(), reason="requires the uncertainty-engine SDK environment"
)


def nodes(workflow):
    return workflow.graph["nodes"]


def edge(workflow, label, key):
    """The (node_name, node_handle) an input of ``label`` is wired to."""
    wired = nodes(workflow)[label]["inputs"][key]
    return wired["node_name"], wired["node_handle"]


# --- these run anywhere ----------------------------------------------------
def test_acquisition_names_map_to_sdk_values():
    assert W.ACQUISITIONS["qNIPV"] == "MonteCarloNegativeIntegratedPosteriorVariance"
    assert W.ACQUISITIONS["PSD"] == "PosteriorStandardDeviation"


def test_columns_match_what_the_simulator_emits():
    from dbtt.simulator import ENERGY_COL, TEMPERATURE_COL

    assert W.INPUT_COLUMNS == [TEMPERATURE_COL]
    assert W.OUTPUT_COLUMNS == [ENERGY_COL]


def test_temperature_grid_spans_the_bounds():
    grid = W.temperature_grid((-150.0, 100.0), n=11)
    assert len(grid) == 11
    assert grid[W.INPUT_COLUMNS[0]].iloc[0] == -150.0
    assert grid[W.INPUT_COLUMNS[0]].iloc[-1] == 100.0


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


def test_project_name_prefers_the_override(monkeypatch):
    monkeypatch.setenv(W.PROJECT_KEY, "FromEnv")
    assert W.project_name() == "FromEnv"
    assert W.project_name("Explicit") == "Explicit"


def test_project_name_refuses_when_unset(monkeypatch):
    monkeypatch.delenv(W.PROJECT_KEY, raising=False)
    with pytest.raises(RuntimeError, match="No Engine project named"):
        W.project_name()


def test_backend_detection_is_one_of_three():
    # "library" means the in-process distribution is installed where the SDK is
    # wanted — it shadows the SDK, and this repo cannot use it.
    assert backend.detect_backend() in ("sdk", "library", "none")
    assert backend.has_sdk() == (backend.detect_backend() == "sdk")


def test_backend_describe_is_consistent():
    described = backend.describe()
    assert described["remote_workflows"] == (described["backend"] == "sdk")
    assert "in_process_modelling" not in described


def test_bad_acquisition_is_rejected_before_touching_the_sdk():
    with pytest.raises((ValueError, ImportError)):
        W.build_recommendation_workflow(None, "p", "m", "pts", acquisition="nope")


# --- these need the SDK ----------------------------------------------------
@needs_sdk
def test_training_workflow_splits_one_dataset_into_x_and_y():
    wf = W.build_training_workflow(None, "proj", "data-id", "dbtt_gp")
    assert set(nodes(wf)) == {
        "Experimental Data", "Temp", "Energy", "Gaussian Process",
        "Train Model", "Save",
    }
    # Both filters read the single uploaded dataset, whose handle is `file`.
    assert edge(wf, "Temp", "dataset") == ("Experimental Data", "file")
    assert edge(wf, "Energy", "dataset") == ("Experimental Data", "file")
    assert edge(wf, "Train Model", "config") == ("Gaussian Process", "config")
    assert edge(wf, "Train Model", "inputs") == ("Temp", "dataset")
    assert edge(wf, "Train Model", "outputs") == ("Energy", "dataset")
    assert edge(wf, "Save", "data") == ("Train Model", "model")


@needs_sdk
def test_training_workflow_filters_the_simulator_columns():
    wf = W.build_training_workflow(None, "proj", "data-id", "dbtt_gp")
    inputs = wf.inputs
    assert inputs["Temp_columns"] == W.INPUT_COLUMNS
    assert inputs["Energy_columns"] == W.OUTPUT_COLUMNS
    assert inputs["Gaussian Process_kernel"] == "LIN+RBF"
    assert inputs["Save_file_id"] == "dbtt_gp"


@needs_sdk
def test_recommendation_workflow_saves_the_new_points():
    wf = W.build_recommendation_workflow(None, "proj", "model-id", "next_point")
    assert set(nodes(wf)) == {"Model", "Active Learning", "Save"}
    assert edge(wf, "Active Learning", "model") == ("Model", "file")
    assert edge(wf, "Save", "data") == ("Active Learning", "recommended_points")
    assert wf.inputs["Active Learning_acquisition_function"] == W.ACQUISITIONS["qNIPV"]
    assert set(wf.requested_output) == {
        "Save_saved_resource", "Active Learning_recommended_points",
    }


@needs_sdk
def test_every_acquisition_builds():
    for acquisition in W.ACQUISITIONS:
        wf = W.build_recommendation_workflow(None, "proj", "m", "pts", acquisition)
        assert wf.inputs["Active Learning_acquisition_function"] == W.ACQUISITIONS[acquisition]


@needs_sdk
def test_gp_plot_workflow_slices_the_model_input_column():
    wf = W.build_gp_plot_workflow(None, "proj", "model-id", (-80.0, 100.0))
    assert set(nodes(wf)) == {"Model", "Slice", "Predict", "Uncertainty Plot"}
    assert edge(wf, "Predict", "model") == ("Model", "file")
    assert edge(wf, "Predict", "dataset") == ("Slice", "data")
    assert edge(wf, "Uncertainty Plot", "X") == ("Slice", "data")
    assert edge(wf, "Uncertainty Plot", "prediction") == ("Predict", "prediction")
    assert edge(wf, "Uncertainty Plot", "uncertainty") == ("Predict", "uncertainty")
    # The slice variable is the trained input column, not the filter's label.
    assert wf.inputs["Slice_variable"] == W.INPUT_COLUMNS[0]
    assert wf.inputs["Slice_num_points"] == 101
    assert set(wf.requested_output) == {"Uncertainty Plot_data", "Slice_data"}


@needs_sdk
def test_prediction_workflow_saves_mean_and_uncertainty_separately():
    wf = W.build_prediction_workflow(
        None, "proj", "model-id", (-80.0, 100.0), "gp_mean", "gp_sd"
    )
    assert set(nodes(wf)) == {
        "Model", "Slice", "Predict", "Save Mean", "Save Uncertainty",
    }
    assert edge(wf, "Predict", "model") == ("Model", "file")
    assert edge(wf, "Predict", "dataset") == ("Slice", "data")
    # PredictModel emits mean and uncertainty as separate handles, so each
    # needs its own Save: there is no node here that joins two datasets.
    assert edge(wf, "Save Mean", "data") == ("Predict", "prediction")
    assert edge(wf, "Save Uncertainty", "data") == ("Predict", "uncertainty")
    assert wf.inputs["Save Mean_file_id"] == "gp_mean"
    assert wf.inputs["Save Uncertainty_file_id"] == "gp_sd"
    assert set(wf.requested_output) == {
        "Save Mean_saved_resource", "Save Uncertainty_saved_resource", "Slice_data",
    }


def test_saved_prediction_joins_the_grid_with_the_downloads(monkeypatch):
    import pandas as pd

    class Response:
        outputs = {"outputs": {
            "Slice_data": {"csv": "Test temperature (C)\n10.0\n-10.0\n"},
            "Save Mean_saved_resource": {"id": "mean-id"},
            "Save Uncertainty_saved_resource": {"id": "sd-id"},
        }}

    frames = {
        "mean-id": pd.DataFrame({"prediction": [2.0, 1.0]}),
        "sd-id": pd.DataFrame({"uncertainty": [0.5, 0.4]}),
    }
    monkeypatch.setattr(W, "download_dataset", lambda c, p, rid: frames[rid])
    monkeypatch.setattr(W, "require_sdk", lambda: None)

    curve = W.saved_prediction(None, "proj", Response())
    assert list(curve.columns) == ["Test temperature (C)", "mean", "sd"]
    # Sorted by the input variable, so it can be plotted as a band directly.
    assert curve["Test temperature (C)"].tolist() == [-10.0, 10.0]
    assert curve["mean"].tolist() == [1.0, 2.0]
    assert curve["sd"].tolist() == [0.4, 0.5]


@needs_sdk
def test_validation_workflow_scores_against_held_out_truth():
    wf = W.build_validation_workflow(None, "proj", "model-id", "holdout-id")
    assert set(nodes(wf)) == {
        "Model", "Validation Data", "Temp", "Energy",
        "Predict", "Score", "Parity Plot",
    }
    assert edge(wf, "Predict", "dataset") == ("Temp", "dataset")
    assert edge(wf, "Score", "predictions") == ("Predict", "prediction")
    assert edge(wf, "Score", "truth") == ("Energy", "dataset")
    assert edge(wf, "Score", "predictions_uncertainty") == ("Predict", "uncertainty")
    assert edge(wf, "Parity Plot", "expected") == ("Energy", "dataset")
    assert wf.inputs["Score_metrics"] == ["MSE", "RMSE", "R2"]
    assert set(wf.requested_output) == {"Score_scores", "Parity Plot_plot"}


def test_outputs_unwraps_the_nested_payload():
    class Response:
        outputs = {"outputs": {"Save_saved_resource": {"id": "abc"}}}

    assert W.outputs(Response()) == {"Save_saved_resource": {"id": "abc"}}


def test_outputs_tolerates_an_unwrapped_payload():
    class Response:
        outputs = {"Save_saved_resource": {"id": "abc"}}

    assert W.outputs(Response()) == {"Save_saved_resource": {"id": "abc"}}


def test_as_frame_parses_a_csv_payload():
    frame = W.as_frame({"csv": "metric_name,score\nRMSE,1.62\nR2,0.99\n"})
    assert list(frame.columns) == ["metric_name", "score"]
    assert frame.loc[frame["metric_name"] == "R2", "score"].item() == 0.99


def test_as_frame_passes_records_through():
    assert W.as_frame([{"a": 1}, {"a": 2}])["a"].tolist() == [1, 2]


def test_uncertainty_curve_reads_the_plot_payload():
    class Response:
        outputs = {"outputs": {"Uncertainty Plot_data": {
            "x_label": "Test temperature (C)",
            "x_vals": [10.0, -10.0],
            "mean": [2.0, 1.0],
            "std": [0.5, 0.4],
        }}}

    curve = W.uncertainty_curve(Response())
    assert list(curve.columns) == ["Test temperature (C)", "mean", "sd"]
    # Sorted by temperature, so the band can be drawn without reordering.
    assert curve["Test temperature (C)"].tolist() == [-10.0, 10.0]
    assert curve["mean"].tolist() == [1.0, 2.0]


@needs_sdk
def test_duplicate_labels_are_refused():
    from uncertainty_engine.exceptions import GraphValidationError
    from uncertainty_engine.nodes.base import Node

    graph = W._graph()
    for _ in range(2):
        node = Node(node_name="Save", version="0.6.0", label="Save", client=None)
        if _ == 0:
            graph.add_node(node)
        else:
            with pytest.raises(GraphValidationError):
                graph.add_node(node)


@needs_sdk
def test_connect_refuses_without_credentials(monkeypatch):
    for key in W.CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="credentials are not set"):
        W.connect("AnyProject")
