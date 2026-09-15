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


def plot_gp(
    learner,
    simulator=None,
    proposed: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    grid: Optional["np.ndarray"] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot an active learner's GP: posterior mean, +/- 2 sigma band, observations.

    Parameters
    ----------
    learner:
        A :class:`dbtt.active_learning.DBTTActiveLearner`.
    simulator:
        Optional. Overlays the true curve and true DBTT — simulator-only
        information, for judging the fit. Omit for a real experiment.
    proposed:
        Optional temperature [C] to mark as the next proposal.
    """
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    x, mean, std = learner.posterior(grid)

    ax.fill_between(
        x, mean - 2 * std, mean + 2 * std,
        color="tab:blue", alpha=0.18, label=r"GP $\pm 2\sigma$",
    )
    ax.plot(x, mean, color="tab:blue", lw=2, label="GP mean")

    if simulator is not None:
        ax.plot(x, simulator.true_curve(x), color="k", lw=1.6, ls="--", label="True curve")
        ax.axvline(simulator.dbtt, color="k", ls=":", lw=1.2,
                   label=f"True DBTT {simulator.dbtt:g} C")

    ax.scatter(learner.temperatures, learner.energies, s=45, color="k",
               zorder=3, label=f"Tested (n={len(learner.temperatures)})")

    if proposed is not None:
        ax.axvline(proposed, color="tab:red", lw=1.8,
                   label=f"Next proposal {proposed:.1f} C")

    ax.set_xlabel("Test temperature [C]")
    ax.set_ylabel("Energy absorbed [J]")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")
    return ax
