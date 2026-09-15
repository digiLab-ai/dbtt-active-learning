"""Plotting helpers for DBTT datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from . import dataset as _dataset
from .simulator import ENERGY_COL, TEMPERATURE_COL


def plot_dataset(
    data: pd.DataFrame | str | Path,
    ax: Optional[plt.Axes] = None,
    label: str = "Measurements",
    **scatter_kwargs,
) -> plt.Axes:
    """Scatter energy absorbed against test temperature.

    Parameters
    ----------
    data:
        A dataset DataFrame, or the name/path of a dataset CSV.
    ax:
        Axes to draw on. A new figure is created if omitted.
    label:
        Legend label for the points.
    **scatter_kwargs:
        Passed through to :meth:`matplotlib.axes.Axes.scatter`.
    """
    if not isinstance(data, pd.DataFrame):
        data = _dataset.load(data)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    style = {"s": 45, "color": "k", "zorder": 3}
    style.update(scatter_kwargs)
    ax.scatter(data[TEMPERATURE_COL], data[ENERGY_COL], label=label, **style)

    ax.set_xlabel("Test temperature [C]")
    ax.set_ylabel("Energy absorbed [J]")
    ax.set_title("Charpy impact energy vs temperature")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return ax
