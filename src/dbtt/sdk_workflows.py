"""Build DBTT active-learning workflows on the remote Uncertainty Engine.

The GP and the acquisition run on the Engine, not here: this module assembles
workflow graphs, saves them to the user's project workspace, and runs them
remotely through the **SDK**.

Four workflows, matching the ones exported from the Engine GUI:

1. :func:`build_training_workflow` — load the campaign, split it into input and
   output columns, configure a GP and train it, saving the fitted model.
2. :func:`build_recommendation_workflow` — ask the trained model for the next
   test temperature(s) under an uncertainty-reduction acquisition.
3. :func:`build_gp_plot_workflow` — predict across a temperature slice, which is
   the *model visualisation*: posterior mean and uncertainty over the whole
   testable range.
4. :func:`build_validation_workflow` — score the model against held-out data and
   produce a parity plot.

Node schemas are resolved from the Node Registry at run time (SDK >= 0.15), so
the builders follow whatever version the Engine currently serves rather than a
version pinned here. :data:`_NODE_VERSIONS` is only the fallback for building a
graph with no client, which is what the tests do.

Requires the SDK — see :mod:`dbtt.backend`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .backend import require_sdk

#: Our acquisition names mapped onto the SDK's `AvailableAcquisitions`.
ACQUISITIONS: Dict[str, str] = {
    "qNIPV": "MonteCarloNegativeIntegratedPosteriorVariance",
    "PSD": "PosteriorStandardDeviation",
    "EI": "ExpectedImprovement",
    "LogEI": "LogExpectedImprovement",
}

#: Environment variables the SDK reads for credentials, plus the project to use.
CREDENTIAL_KEYS = ("UE_USERNAME", "UE_PASSWORD")
PROJECT_KEY = "UE_PROJECT_NAME"

#: The campaign columns, as `dbtt.simulator` emits them.
INPUT_COLUMNS: List[str] = ["Test temperature (C)"]
OUTPUT_COLUMNS: List[str] = ["Energy absorbed (J)"]

#: Node versions for the clientless build path, taken from the workflows
#: exported from the Engine GUI. With a client these are ignored: the Node
#: Registry picks the default version for each node.
_NODE_VERSIONS: Dict[str, str] = {
    "CreateDatasetSlice": "0.3.0",
    "FilterDataset": "0.4.0",
    "LoadDataset": "0.5.0",
    "LoadModel": "0.4.0",
    "ModelConfig": "0.5.0",
    "ParityPlot": "0.4.0",
    "PredictModel": "0.5.0",
    "Recommend": "0.3.0",
    "Save": "0.6.0",
    "ScoreModel": "0.3.0",
    "TrainModel": "0.7.0",
    "UncertaintyPlot": "0.3.0",
}


# ---------------------------------------------------------------------------
# credentials and connection
# ---------------------------------------------------------------------------
def load_credentials(env_path: str | Path = ".env") -> str:
    """Load Uncertainty Engine credentials into the process environment.

    Tries Google Colab's secret store first, then a local ``.env`` file. Values
    are never returned or logged — only the source is reported, so a notebook can
    show where its credentials came from without showing what they are.

    Returns
    -------
    str
        ``"colab"``, ``"dotenv"``, ``"environment"`` or ``"missing"``.
    """
    keys = CREDENTIAL_KEYS + (PROJECT_KEY,)
    try:  # Google Colab secrets
        from google.colab import userdata  # type: ignore

        found = False
        for key in keys:
            try:
                os.environ[key] = userdata.get(key)
                found = True
            except Exception:
                pass
        if found:
            return "colab"
    except ImportError:
        pass

    path = Path(env_path)
    if path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(path)
            return "dotenv"
        except ImportError as error:  # pragma: no cover - depends on extras
            raise ImportError(
                "python-dotenv is needed to read a .env file: pip install python-dotenv"
            ) from error

    if all(os.environ.get(key) for key in CREDENTIAL_KEYS):
        return "environment"
    return "missing"


def credentials_present() -> bool:
    """Whether every credential key is set. Does not read the values."""
    return all(os.environ.get(key) for key in CREDENTIAL_KEYS)


def project_name(override: Optional[str] = None) -> str:
    """The Engine project to work in: ``override``, else ``$UE_PROJECT_NAME``."""
    name = override or os.environ.get(PROJECT_KEY)
    if not name:
        raise RuntimeError(
            f"No Engine project named. Set {PROJECT_KEY} in your .env (see "
            ".env.example) or pass project_name=... explicitly. Projects are "
            "created in the Uncertainty Engine GUI, not from the SDK."
        )
    return name


def connect(name: Optional[str] = None):
    """Authenticate and resolve a project id.

    Projects are created in the Uncertainty Engine GUI, not from the SDK, so the
    project must already exist.

    Returns
    -------
    tuple
        ``(client, project_id)``.
    """
    require_sdk()
    from uncertainty_engine import Client

    if not credentials_present():
        raise RuntimeError(
            "Uncertainty Engine credentials are not set. Put "
            f"{' and '.join(CREDENTIAL_KEYS)} in a .env file (see .env.example) "
            "or in Colab secrets, then call load_credentials()."
        )

    client = Client()
    client.authenticate()
    return client, client.projects.get_project_id_by_name(project_name(name))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def temperature_grid(
    bounds: tuple[float, float], n: int = 101
) -> pd.DataFrame:
    """The input space to visualise the model over, as a local frame.

    The Engine builds its own grid with ``CreateDatasetSlice``; this mirrors it
    for plotting in the notebook.
    """
    return pd.DataFrame(
        {INPUT_COLUMNS[0]: np.linspace(float(bounds[0]), float(bounds[1]), n)}
    )


def resource_id(
    client, project_id: str, name: str, resource_type: str = "dataset"
) -> Optional[str]:
    """The id of a project resource by name, or ``None`` if there is no such one."""
    require_sdk()
    records = client.resources.list_resources(project_id, resource_type)
    matches = [r for r in records if getattr(r, "name", None) == name]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} {resource_type} resources are named {name!r} in this "
            "project; rename or delete the duplicates."
        )
    return matches[0].id


def upload_dataset(client, project_id: str, frame: pd.DataFrame, name: str) -> str:
    """Write a frame to the project workspace and return its resource id.

    Resources are uploaded from a file rather than in memory, so the frame is
    staged in a temporary CSV first.

    Creating and versioning are *different calls*: ``upload`` refuses a name the
    project already holds (409 Conflict), and ``update`` adds a new version to
    an existing resource, keeping its id. The active-learning loop depends on
    that id staying put — the saved workflows point at it — so this picks the
    right call rather than assuming either.
    """
    require_sdk()
    import tempfile

    existing = resource_id(client, project_id, name)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.csv"
        frame.to_csv(path, index=False)
        if existing is None:
            client.resources.upload(
                project_id=project_id,
                name=name,
                resource_type="dataset",
                file_path=str(path),
            )
        else:
            client.resources.update(
                project_id=project_id,
                resource_type="dataset",
                resource_id=existing,
                file_path=str(path),
            )

    return existing or client.resources.get_resource_id_by_name(
        resource_type="dataset", project_id=project_id, name=name
    )


# ---------------------------------------------------------------------------
# node and workflow plumbing
# ---------------------------------------------------------------------------
def _node(client, node_name: str, label: str, **inputs):
    """Build a node by name.

    With a client the schema is resolved from the Node Registry and the inputs
    are validated against it. Without one — as the builder tests do, to check
    the wiring offline — the node is built at a fallback version, unvalidated.
    """
    if client is not None:
        return client.nodes(node_name, label=label, **inputs)

    from uncertainty_engine.nodes.base import Node

    return Node(
        node_name=node_name,
        version=_NODE_VERSIONS[node_name],
        label=label,
        client=None,
        **inputs,
    )


def _resource(resource_id: str) -> Dict[str, str]:
    """A resource id in the serialised form the load nodes take."""
    from uncertainty_engine_types import ResourceID

    return ResourceID(id=resource_id).model_dump()


def _workflow(graph, requested_output: Dict[str, Any], client=None):
    """Wrap a populated graph as an executable workflow."""
    from uncertainty_engine.nodes.workflow import Workflow

    return Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        requested_output=requested_output,
        external_input_id=graph.external_input_id,
        client=client,
    )


def _graph():
    """A graph that refuses duplicate labels, since every label here is fixed."""
    from uncertainty_engine.graph import Graph

    return Graph(prevent_node_overwrite=True)


def _output(label: str, handle: str) -> Dict[str, Dict[str, str]]:
    """One entry of a workflow's ``requested_output`` map."""
    return {f"{label}_{handle}": {"node_name": label, "node_handle": handle}}


def _load_dataset(client, project_id: str, resource_id: str, label: str):
    """A ``LoadDataset`` node. Its output handle is ``file``, not ``dataset``."""
    return _node(
        client, "LoadDataset", label,
        project_id=project_id, file_id=_resource(resource_id),
    )


def _load_model(client, project_id: str, resource_id: str, label: str):
    """A ``LoadModel`` node. Its output handle is ``file``, not ``model``."""
    return _node(
        client, "LoadModel", label,
        project_id=project_id, file_id=_resource(resource_id),
    )


def _filter(client, source, label: str, columns: Sequence[str]):
    """A ``FilterDataset`` node selecting ``columns`` from a loaded dataset."""
    return _node(
        client, "FilterDataset", label,
        dataset=source.make_handle("file"),
        columns=list(columns),
        sample_fraction=1,
    )


def _save(client, source, handle: str, project_id: str, file_name: str, label: str):
    """A ``Save`` node. The node input is ``file_id``; it holds the *name*."""
    return _node(
        client, "Save", label,
        data=source.make_handle(handle),
        file_id=file_name,
        project_id=project_id,
        account_id=None,
    )


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------
def build_training_workflow(
    client,
    project_id: str,
    dataset_resource_id: str,
    model_name: str,
    kernel: str = "LIN+RBF",
    model_type: str = "SingleTaskGPTorch",
    seed: Optional[int] = 0,
    input_columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
):
    """Train a GP on the campaign so far and save the fitted model.

    The campaign is uploaded as *one* dataset; ``FilterDataset`` splits out the
    input and output columns, so the loop re-uploads a single CSV per iteration.
    The saved model becomes a project resource the other workflows load by id.
    """
    require_sdk()
    graph = _graph()

    data = _load_dataset(client, project_id, dataset_resource_id, "Experimental Data")
    temp = _filter(client, data, "Temp", input_columns or INPUT_COLUMNS)
    energy = _filter(client, data, "Energy", output_columns or OUTPUT_COLUMNS)
    config = _node(
        client, "ModelConfig", "Gaussian Process",
        model_type=model_type, kernel=kernel, warp_inputs=False, seed=seed,
    )
    for node in (data, temp, energy, config):
        graph.add_node(node)

    train = _node(
        client, "TrainModel", "Train Model",
        config=config.make_handle("config"),
        inputs=temp.make_handle("dataset"),
        outputs=energy.make_handle("dataset"),
    )
    save = _save(client, train, "model", project_id, model_name, "Save")
    graph.add_node(train)
    graph.add_node(save)

    return _workflow(graph, _output("Save", "saved_resource"), client)


def build_recommendation_workflow(
    client,
    project_id: str,
    model_resource_id: str,
    points_name: str,
    acquisition: str = "qNIPV",
    num_points: int = 1,
):
    """Ask the trained GP for the next test temperature(s).

    ``acquisition`` takes our short names (``"qNIPV"``, ``"PSD"``), which are
    translated to the SDK's `AvailableAcquisitions` values.
    """
    require_sdk()
    if acquisition not in ACQUISITIONS:
        raise ValueError(
            f"acquisition must be one of {sorted(ACQUISITIONS)}, got {acquisition!r}"
        )
    graph = _graph()

    model = _load_model(client, project_id, model_resource_id, "Model")
    graph.add_node(model)

    recommend = _node(
        client, "Recommend", "Active Learning",
        model=model.make_handle("file"),
        number_of_points=str(num_points),
        acquisition_function=ACQUISITIONS[acquisition],
    )
    save = _save(
        client, recommend, "recommended_points", project_id, points_name, "Save"
    )
    graph.add_node(recommend)
    graph.add_node(save)

    # `recommended_points` comes back as an S3 reference, not as values, so the
    # saved resource is what the loop actually reads: `Save_saved_resource`
    # carries the id to download.
    requested = {
        **_output("Save", "saved_resource"),
        **_output("Active Learning", "recommended_points"),
    }
    return _workflow(graph, requested, client)


def build_gp_plot_workflow(
    client,
    project_id: str,
    model_resource_id: str,
    bounds: tuple[float, float],
    num_points: int = 101,
    variable: Optional[str] = None,
):
    """Predict across the temperature slice — the model visualisation.

    ``variable`` is the *model's input column*, which is whatever
    :func:`build_training_workflow` filtered on, not the label of the filter
    node that produced it.
    """
    require_sdk()
    graph = _graph()

    model = _load_model(client, project_id, model_resource_id, "Model")
    slice_ = _node(
        client, "CreateDatasetSlice", "Slice",
        variable=variable or INPUT_COLUMNS[0],
        min=str(bounds[0]),
        max=str(bounds[1]),
        fixed_values={},
        num_points=num_points,
    )
    graph.add_node(model)
    graph.add_node(slice_)

    predict = _node(
        client, "PredictModel", "Predict",
        model=model.make_handle("file"),
        dataset=slice_.make_handle("data"),
    )
    plot = _node(
        client, "UncertaintyPlot", "Uncertainty Plot",
        X=slice_.make_handle("data"),
        prediction=predict.make_handle("prediction"),
        uncertainty=predict.make_handle("uncertainty"),
    )
    graph.add_node(predict)
    graph.add_node(plot)

    # `UncertaintyPlot` returns the whole curve inline — x_vals, mean, std,
    # lower, upper — which is the only posterior that comes back as values.
    # `PredictModel`'s own prediction/uncertainty handles return S3 references
    # that would each need a separate download, so they are not requested.
    # `Slice_data` is inline CSV, kept as the record of the grid asked for.
    requested = {
        **_output("Uncertainty Plot", "data"),
        **_output("Slice", "data"),
    }
    return _workflow(graph, requested, client)


def build_prediction_workflow(
    client,
    project_id: str,
    model_resource_id: str,
    bounds: tuple[float, float],
    mean_name: str,
    uncertainty_name: str,
    num_points: int = 101,
    variable: Optional[str] = None,
):
    """Predict across a slice and **save** the posterior as project datasets.

    The same shape as :func:`build_gp_plot_workflow`, but where that one returns
    a plot payload that lives only as long as the job, this writes the posterior
    mean and uncertainty to the project as datasets. They can then be downloaded
    (see :func:`saved_prediction`), plotted alongside anything else, compared
    between iterations, or picked up in the GUI.

    Mean and uncertainty are separate resources because `PredictModel` emits them
    as separate handles and there is no node here that joins two datasets.
    """
    require_sdk()
    graph = _graph()

    model = _load_model(client, project_id, model_resource_id, "Model")
    slice_ = _node(
        client, "CreateDatasetSlice", "Slice",
        variable=variable or INPUT_COLUMNS[0],
        min=str(bounds[0]),
        max=str(bounds[1]),
        fixed_values={},
        num_points=num_points,
    )
    graph.add_node(model)
    graph.add_node(slice_)

    predict = _node(
        client, "PredictModel", "Predict",
        model=model.make_handle("file"),
        dataset=slice_.make_handle("data"),
    )
    save_mean = _save(
        client, predict, "prediction", project_id, mean_name, "Save Mean"
    )
    save_sd = _save(
        client, predict, "uncertainty", project_id, uncertainty_name, "Save Uncertainty"
    )
    for node in (predict, save_mean, save_sd):
        graph.add_node(node)

    # `Slice_data` is inline CSV, so the grid the posterior is over comes back
    # with the job and needs no download of its own.
    requested = {
        **_output("Save Mean", "saved_resource"),
        **_output("Save Uncertainty", "saved_resource"),
        **_output("Slice", "data"),
    }
    return _workflow(graph, requested, client)


def build_validation_workflow(
    client,
    project_id: str,
    model_resource_id: str,
    dataset_resource_id: str,
    metrics: Sequence[str] = ("MSE", "RMSE", "R2"),
    input_columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
):
    """Score the model against a held-out dataset and draw a parity plot."""
    require_sdk()
    graph = _graph()

    model = _load_model(client, project_id, model_resource_id, "Model")
    data = _load_dataset(client, project_id, dataset_resource_id, "Validation Data")
    temp = _filter(client, data, "Temp", input_columns or INPUT_COLUMNS)
    energy = _filter(client, data, "Energy", output_columns or OUTPUT_COLUMNS)
    for node in (model, data, temp, energy):
        graph.add_node(node)

    predict = _node(
        client, "PredictModel", "Predict",
        model=model.make_handle("file"),
        dataset=temp.make_handle("dataset"),
    )
    score = _node(
        client, "ScoreModel", "Score",
        predictions=predict.make_handle("prediction"),
        truth=energy.make_handle("dataset"),
        predictions_uncertainty=predict.make_handle("uncertainty"),
        metrics=list(metrics),
    )
    parity = _node(
        client, "ParityPlot", "Parity Plot",
        predictions=predict.make_handle("prediction"),
        expected=energy.make_handle("dataset"),
        uncertainties=predict.make_handle("uncertainty"),
    )
    graph.add_node(predict)
    graph.add_node(score)
    graph.add_node(parity)

    requested = {
        **_output("Score", "scores"),
        **_output("Parity Plot", "plot"),
    }
    return _workflow(graph, requested, client)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
def workflow_id(client, project_id: str, name: str) -> Optional[str]:
    """The id of a saved workflow by name, or ``None`` if there is no such one."""
    require_sdk()
    records = client.workflows.list_workflows(project_id)
    matches = [r for r in records if getattr(r, "name", None) == name]
    return matches[0].id if matches else None


def save_workflow(client, project_id: str, workflow, name: str) -> str:
    """Save a workflow to the project workspace and return its id.

    It then appears in the Uncertainty Engine GUI and can be re-run, versioned or
    edited there, which is the point of building it programmatically rather than
    only executing it.

    Workflow names are unique within a project, so saving over an existing name
    updates that workflow as a new version instead of failing. Re-running the
    notebook therefore revises its four workflows rather than duplicating them.
    """
    require_sdk()
    existing = workflow_id(client, project_id, name)
    if existing is not None:
        return client.workflows.save(
            project_id=project_id, workflow=workflow, workflow_id=existing
        )
    return client.workflows.save(
        project_id=project_id, workflow=workflow, workflow_name=name
    )


def _completed(response) -> Any:
    """Return a finished job, or raise with the response body if it failed."""
    if response.status.value != "completed":
        raise RuntimeError(
            f"Workflow did not complete (status {response.status.value}): "
            f"{response.model_dump()}"
        )
    return response


def run(client, workflow) -> Any:
    """Execute a workflow object directly, without saving it first."""
    require_sdk()
    return _completed(client.run_node(workflow))


def run_saved(
    client,
    project_id: str,
    workflow_id: str,
    overrides: Optional[Dict[tuple[str, str], Any]] = None,
) -> Any:
    """Run a workflow already saved in the project, by id.

    This is the loop's workhorse: the graphs are built and saved once, then
    re-run each iteration. Nothing needs overriding for that, because every
    iteration re-uploads its dataset under the same resource name — a new
    *version* of the same resource id, which the saved graph already points at.

    ``overrides`` maps ``(node_label, input_handle)`` to a value, for the cases
    where you do want to vary something, such as the acquisition function.
    """
    require_sdk()
    from uncertainty_engine_types import OverrideWorkflowInput

    inputs = [
        OverrideWorkflowInput(node_label=label, input_handle=handle, value=value)
        for (label, handle), value in (overrides or {}).items()
    ]
    return _completed(
        client.run_workflow(
            project_id=project_id,
            workflow_id=workflow_id,
            inputs=inputs or None,
        )
    )


def download_dataset(client, project_id: str, resource_id: str) -> pd.DataFrame:
    """Download a dataset resource and parse it as a frame.

    ``resources.download`` returns raw bytes when given no file path.
    """
    require_sdk()
    import io

    content = client.resources.download(
        project_id=project_id, resource_type="dataset", resource_id=resource_id
    )
    return pd.read_csv(io.BytesIO(content))


def outputs(response) -> Dict[str, Any]:
    """The requested outputs of a finished job, keyed as the builders named them.

    ``JobInfo.outputs`` wraps the map in another ``"outputs"`` key; this unwraps
    it, so callers see the ``"<label>_<handle>"`` keys the builders asked for.
    """
    payload = dict(response.outputs or {})
    inner = payload.get("outputs")
    return dict(inner) if isinstance(inner, dict) else payload


def as_frame(payload: Any) -> pd.DataFrame:
    """A node output that carries a CSV payload, as a frame.

    Several nodes (``ScoreModel``, ``CreateDatasetSlice``) return their result
    as ``{"csv": "..."}`` rather than as records.
    """
    import io

    if isinstance(payload, dict) and "csv" in payload:
        return pd.read_csv(io.StringIO(payload["csv"]))
    return pd.DataFrame(payload)


def uncertainty_curve(response) -> pd.DataFrame:
    """The posterior from a GP-plot job: grid, mean and standard deviation.

    ``UncertaintyPlot`` is the one node here that returns the posterior as
    values rather than an S3 reference, so it is what the notebook plots.
    """
    data = outputs(response)["Uncertainty Plot_data"]
    return pd.DataFrame(
        {
            data.get("x_label", INPUT_COLUMNS[0]): data["x_vals"],
            "mean": data["mean"],
            "sd": data["std"],
        }
    ).sort_values(data.get("x_label", INPUT_COLUMNS[0]), ignore_index=True)


def saved_prediction(client, project_id: str, response) -> pd.DataFrame:
    """The posterior a prediction job saved, as one frame.

    Downloads the two saved datasets and puts them beside the grid, which came
    back inline with the job. Columns: the input variable, ``mean`` and ``sd``.
    """
    require_sdk()
    out = outputs(response)
    grid = as_frame(out["Slice_data"])
    mean = download_dataset(client, project_id, out["Save Mean_saved_resource"]["id"])
    sd = download_dataset(
        client, project_id, out["Save Uncertainty_saved_resource"]["id"]
    )

    frame = pd.DataFrame(
        {
            grid.columns[0]: grid.iloc[:, 0].to_numpy(dtype=float),
            "mean": mean.iloc[:, 0].to_numpy(dtype=float),
            "sd": sd.iloc[:, 0].to_numpy(dtype=float),
        }
    )
    return frame.sort_values(frame.columns[0], ignore_index=True)


def recommended_points(client, project_id: str, response) -> pd.DataFrame:
    """The points a recommendation job chose, downloaded from its saved resource.

    The ``recommended_points`` handle is an S3 reference; the ``Save`` node's
    ``saved_resource`` carries the resource id, which is what can be read back.
    """
    saved = outputs(response)["Save_saved_resource"]
    return download_dataset(client, project_id, saved["id"])
