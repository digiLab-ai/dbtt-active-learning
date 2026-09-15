"""Locating and loading DBTT dataset CSVs.

Creating and growing a dataset is deliberately *not* handled here — that is the
notebook's job, done directly with pandas (``simulator.run`` -> ``pd.concat`` ->
``DataFrame.to_csv``) so the campaign history is an ordinary DataFrame you can
inspect and slice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def dataset_path(name: str | Path) -> Path:
    """Resolve ``name`` to a CSV path inside the repository ``data/`` folder.

    A bare name gets the ``.csv`` suffix and the ``data/`` prefix; anything that
    already looks like a path is left alone.
    """
    path = Path(name)
    if path.suffix != ".csv":
        path = path.with_suffix(".csv")
    if path.parent == Path("."):
        path = DATA_DIR / path
    return path


def load(name: str | Path) -> pd.DataFrame:
    """Load a dataset CSV by name or path."""
    return pd.read_csv(dataset_path(name))
