"""Build DBTT active-learning workflows on the remote Uncertainty Engine.

The in-process route (`dbtt.active_learning`) runs the GP and the acquisition on
this machine. This module does the same job through the **SDK**: it assembles
workflow graphs, saves them to the user's project workspace, and runs them on
the Engine.

Three workflows, mirroring the gold-miner example:

1. :func:`build_training_workflow` — configure a GP and train it on the campaign
   so far, saving the fitted model as a project resource.
2. :func:`build_prediction_workflow` — predict across a temperature grid, which
   is the *model visualisation*: posterior mean and uncertainty over the whole
   testable range.
3. :func:`build_recommendation_workflow` — ask the trained model for the next
   test temperature(s) under an uncertainty-reduction acquisition.

Requires the SDK, which cannot share an environment with the in-process library
— see :mod:`dbtt.backend`.
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

#: Environment variables the SDK reads for credentials.
CREDENTIAL_KEYS = ("UE_USERNAME", "UE_PASSWORD")


# ---------------------------------------------------------------------------
# credentials
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
    try:  # Google Colab secrets
        from google.colab import userdata  # type: ignore

        found = False
        for key in CREDENTIAL_KEYS:
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


def connect(project_name: str):
    """Authenticate and resolve a project id.

    Projects are created in the Uncertainty Engine GUI, not from the SDK, so
    ``project_name`` must already exist.

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
    return client, client.projects.get_project_id_by_name(project_name)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def campaign_to_frames(
    temperatures: Sequence[float], energies: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a campaign into the separate X and y frames the Engine expects."""
    x = pd.DataFrame({"temperature_C": np.asarray(temperatures, dtype=float)})
    y = pd.DataFrame({"energy_J": np.asarray(energies, dtype=float)})
    return x, y


def temperature_grid(bounds: tuple[float, float], n: int = 100) -> pd.DataFrame:
    """The input space to visualise the model over."""
    return pd.DataFrame(
        {"temperature_C": np.linspace(float(bounds[0]), float(bounds[1]), n)}
    )


def upload_dataset(client, project_id: str, frame: pd.DataFrame, name: str) -> str:
    """Write a frame to the project workspace and return its resource id.

    Resources are uploaded from a file rather than in memory, so the frame is
    staged in a temporary CSV first.
    """
    require_sdk()
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.csv"
        frame.to_csv(path, index=False)
        client.resources.upload(
            project_id=project_id,
            name=name,
            resource_type="dataset",
            file_path=str(path),
        )
    return client.resources.get_resource_id_by_name(
        resource_type="dataset", project_id=project_id, name=name
    )


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------
#: Node versions for the clientless build path. SDK 0.15 dropped its per-node
#: classes: a node is now built off the client, which resolves its schema from
#: the Node Registry on demand. With no client there is nothing to resolve
#: against, so the version has to be named here — these are the versions the
#: 0.14 node classes pinned.
_NODE_VERSIONS: Dict[str, str] = {
    "LoadDataset": "0.2.0",
    "LoadModel": "0.2.0",
    "ModelConfig": "0.2.0",
    "PredictModel": "0.2.0",
    "Recommend": "0.2.0",
    "Save": "0.2.0",
    "TrainModel": "0.2.0",
}


def _node(client, node_name: str, label: str, **inputs):
    """Build a node by name.

    With a client, the schema is resolved from the Node Registry and the inputs
    are validated against it. Without one — as the builder tests do, to check
    the wiring offline — the node is built at a pinned version, unvalidated.
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


def _workflow(graph, client=None):
    """Wrap a populated graph as an executable workflow."""
    from uncertainty_engine.nodes.workflow import Workflow

    return Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        client=client,
    )


def build_training_workflow(
    client,
    project_id: str,
    x_resource_id: str,
    y_resource_id: str,
    model_name: str,
    seed: Optional[int] = 0,
):
    """Configure a GP, train it on the campaign, and save the fitted model.

    The saved model becomes a project resource that the prediction and
    recommendation workflows load by name.
    """
    require_sdk()
    from uncertainty_engine.graph import Graph

    graph = Graph()

    config = _node(
        client, "ModelConfig", "GP config",
        model_type="SingleTaskGPTorch",
        kernel=None,
        warp_inputs=False,
        seed=seed,
        input_variance=None,
        input_retained_dimensions=None,
        output_variance=None,
        output_retained_dimensions=None,
    )
    load_x = _node(
        client, "LoadDataset", "Load temperatures",
        project_id=project_id, file_id=_resource(x_resource_id),
    )
    load_y = _node(
        client, "LoadDataset", "Load energies",
        project_id=project_id, file_id=_resource(y_resource_id),
    )
    for node in (config, load_x, load_y):
        graph.add_node(node)

    train = _node(
        client, "TrainModel", "Train GP",
        config=config.make_handle("model_config"),
        inputs=load_x.make_handle("dataset"),
        outputs=load_y.make_handle("dataset"),
    )
    save = _node(
        client, "Save", "Save GP",
        data=train.make_handle("model"),
        # The node input is `file_id`, but it holds the human-readable name.
        file_id=model_name,
        project_id=project_id,
    )
    graph.add_node(train)
    graph.add_node(save)
    return _workflow(graph, client)


def build_prediction_workflow(
    client, project_id: str, model_resource_id: str, grid_resource_id: str
):
    """Predict across the temperature grid — the model visualisation.

    Returns the posterior mean and uncertainty over the whole testable range,
    which is what gets plotted as the GP band.
    """
    require_sdk()
    from uncertainty_engine.graph import Graph

    graph = Graph()
    load_model = _node(
        client, "LoadModel", "Load GP",
        project_id=project_id, file_id=_resource(model_resource_id),
    )
    load_grid = _node(
        client, "LoadDataset", "Load grid",
        project_id=project_id, file_id=_resource(grid_resource_id),
    )
    graph.add_node(load_model)
    graph.add_node(load_grid)

    predict = _node(
        client, "PredictModel", "Predict curve",
        model=load_model.make_handle("model"),
        dataset=load_grid.make_handle("dataset"),
    )
    graph.add_node(predict)
    return _workflow(graph, client)


def build_recommendation_workflow(
    client,
    project_id: str,
    model_resource_id: str,
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
    from uncertainty_engine.graph import Graph

    graph = Graph()
    load_model = _node(
        client, "LoadModel", "Load GP",
        project_id=project_id, file_id=_resource(model_resource_id),
    )
    graph.add_node(load_model)

    recommend = _node(
        client, "Recommend", f"Recommend ({acquisition})",
        acquisition_function=ACQUISITIONS[acquisition],
        model=load_model.make_handle("model"),
        num_of_points=num_points,
    )
    graph.add_node(recommend)
    return _workflow(graph, client)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
def save_workflow(client, project_id: str, workflow, name: str) -> str:
    """Save a workflow to the project workspace and return its id.

    It then appears in the Uncertainty Engine GUI and can be re-run, versioned or
    edited there, which is the point of building it programmatically rather than
    only executing it.
    """
    require_sdk()
    return client.workflows.save(
        project_id=project_id, workflow=workflow, workflow_name=name
    )


def run(client, workflow) -> Any:
    """Execute a workflow, raising with the response body if it fails."""
    require_sdk()
    response = client.run_node(workflow)
    if response.status.value != "completed":
        raise RuntimeError(
            f"Workflow did not complete (status {response.status.value}): "
            f"{response.model_dump()}"
        )
    return response
