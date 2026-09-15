"""Charpy impact-test simulator based on an asymmetric hyperbolic tangent."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from .config import CharpySimulatorConfig
from .interface import SimulationResult, Simulator

#: Column headers of the test-result table, matching the test-house spreadsheet.
COLUMNS = [
    "Material",
    "Condition",
    "Processing",
    "Orientation",
    "NEURONE specimen ID",
    "Test house specimen ID",
    "Specimen Design",
    "Specimen location",
    "Specimen Type",
    "Notch type",
    "Test temperature (C)",
    "Energy absorbed (J)",
]

TEMPERATURE_COL = "Test temperature (C)"
ENERGY_COL = "Energy absorbed (J)"


class CharpySimulator(Simulator[np.ndarray]):
    """Charpy impact test: test temperature [C] -> energy absorbed [J].

    The forward map is an asymmetric hyperbolic tangent plus heteroscedastic
    Gaussian noise. It is *stochastic*: one ``simulate`` call is one realisation
    of the forward map, i.e. one broken specimen, not a summary of the output
    law. Repeated calls at the same temperature give different energies, as in a
    real campaign.

    Parameters
    ----------
    config:
        Simulator configuration. Defaults to the built-in EUROFER EAF setup.
    """

    def __init__(self, config: Optional[CharpySimulatorConfig] = None) -> None:
        super().__init__(config or CharpySimulatorConfig())
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # the forward map
    # ------------------------------------------------------------------
    def _forward(self, X: float | Iterable[float]) -> np.ndarray:
        """Draw noisy energy-absorbed measurements [J] at temperature(s) ``X`` [C].

        The raw ``X -> y`` map. Advances the internal random state.
        """
        t = np.atleast_1d(np.asarray(X, dtype=float))
        self._validate_temperatures(t)
        draw = self._rng.normal(loc=self.true_curve(t), scale=self.noise_std(t))
        return np.clip(draw, self.config.noise.energy_floor, None)

    def build_result(self, X: Any, y: Any) -> SimulationResult:
        """Wrap the measurement, carrying the ground truth as auxiliary payload.

        ``outputs["true_energy"]`` and ``outputs["noise_std"]`` are simulator-only
        information with no counterpart in a real experiment. They are here to
        score the active-learning loop, and must not be fed to the GP.
        """
        t = np.atleast_1d(np.asarray(X, dtype=float))
        return SimulationResult(
            y=y,
            outputs={
                "temperature": t,
                "true_energy": self.true_curve(t),
                "noise_std": self.noise_std(t),
                "dbtt": self.dbtt,
                "units": {"temperature": "C", "energy": "J"},
            },
        )

    # ------------------------------------------------------------------
    # ground truth (simulator-only; unavailable in a real experiment)
    # ------------------------------------------------------------------
    def true_curve(self, temperature: float | Iterable[float]) -> np.ndarray:
        """Noise-free energy absorbed [J] at the given temperature(s) [C]."""
        curve = self.config.curve
        t = np.asarray(temperature, dtype=float)
        width = np.where(t < curve.dbtt, curve.width_lower, curve.width_upper)
        midpoint = 0.5 * (curve.lower_shelf + curve.upper_shelf)
        amplitude = 0.5 * (curve.upper_shelf - curve.lower_shelf)
        return midpoint + amplitude * np.tanh((t - curve.dbtt) / width)

    def noise_std(self, temperature: float | Iterable[float]) -> np.ndarray:
        """Measurement standard deviation [J] at the given temperature(s) [C]."""
        noise = self.config.noise
        return noise.base + noise.relative * self.true_curve(temperature)

    @property
    def dbtt(self) -> float:
        """The ground-truth DBTT [C] the active-learning loop is chasing."""
        return self.config.curve.dbtt

    # ------------------------------------------------------------------
    # test-campaign convenience
    # ------------------------------------------------------------------
    def run(
        self,
        temperatures: Sequence[float] | float,
        start_index: int = 1,
    ) -> pd.DataFrame:
        """Test specimens at ``temperatures`` and return a test-result table.

        A thin wrapper over :meth:`simulate` that attaches the specimen
        provenance from the config and numbers the specimens. The table carries
        only what a real test house would report: the noise-free truth stays on
        the :class:`SimulationResult` and never reaches the dataset.

        Parameters
        ----------
        temperatures:
            Test temperature(s) [C].
        start_index:
            Number of the first specimen; subsequent specimens follow
            consecutively.
        """
        result = self.simulate(temperatures)
        t = result.outputs["temperature"]
        spec = self.config.specimen
        numbers = np.arange(start_index, start_index + t.size)

        return pd.DataFrame(
            {
                "Material": spec.material,
                "Condition": spec.condition,
                "Processing": spec.processing,
                "Orientation": spec.orientation,
                "NEURONE specimen ID": [
                    f"{spec.neurone_id_prefix}{n:0{spec.neurone_id_digits}d}"
                    for n in numbers
                ],
                "Test house specimen ID": numbers,
                "Specimen Design": spec.specimen_design,
                "Specimen location": spec.specimen_location,
                "Specimen Type": spec.specimen_type,
                "Notch type": spec.notch_type,
                TEMPERATURE_COL: t,
                ENERGY_COL: np.round(result.y, 2),
            },
            columns=COLUMNS,
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @classmethod
    def data_type(cls) -> Type:
        return np.ndarray

    @classmethod
    def capabilities(cls) -> List[str]:
        return []

    # ------------------------------------------------------------------
    def _validate_temperatures(self, t: np.ndarray) -> None:
        lo, hi = self.config.temperature_range
        if np.any(t < lo) or np.any(t > hi):
            bad = t[(t < lo) | (t > hi)]
            raise ValueError(
                f"Test temperature(s) {bad.tolist()} outside the configured "
                f"range [{lo}, {hi}] C."
            )
