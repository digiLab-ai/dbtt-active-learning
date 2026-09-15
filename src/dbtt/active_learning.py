"""Choose Charpy test temperatures by Bayesian optimisation.

The goal is not to find a maximum: it is to *reduce uncertainty* about the whole
transition curve, so the DBTT can be located from as few broken specimens as
possible. That is an active-learning objective, and the engine exposes it through
BoTorch's uncertainty-reduction acquisitions:

- ``qNIPV`` — negative integrated posterior variance. Picks the temperature that
  most reduces the GP's variance *averaged over the whole range*. Global.
- ``PSD`` / ``qPSD`` — posterior standard deviation. Picks wherever the GP is
  currently least certain. Local, greedy, and cheaper.

Both are wrapped by `DBTTActiveLearner`, a thin ask / run-the-simulator / tell
loop over the engine's `BayesianOptimisation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from uncertainty_engine.models.kernel.gp import SingleTaskGPTorch, SingleTaskGPTorchConfig
from uncertainty_engine.optimisers.bo import (
    BayesianOptimisation,
    BayesianOptimisationConfig,
)

from .simulator import CharpySimulator

ACQUISITIONS = ("qNIPV", "PSD", "qPSD")
DTYPE = torch.float64


def initial_design(bounds: Tuple[float, float], n: int = 2) -> np.ndarray:
    """Propose the opening test temperatures.

    The conventional start is the two limits of the testable range: they pin the
    lower and upper shelves, which is what the transition is measured *between*.
    Asking for more than two fills in evenly between them.
    """
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        raise ValueError("bounds must be (low, high) with high > low")
    if n < 2:
        raise ValueError("an initial design needs at least the two limits")
    return np.linspace(low, high, n)


@dataclass
class Step:
    """One active-learning iteration."""

    index: int
    temperature: float
    energy: float
    acquisition_value: float
    n_observations: int


class DBTTActiveLearner:
    """Active learning for the transition curve, over the engine's BO.

    Parameters
    ----------
    simulator:
        Stands in for the test house. Its ``simulate`` is called once per
        proposal, i.e. one proposal is one broken specimen.
    bounds:
        Testable temperature range [C]. Defaults to the simulator's own.
    acquisition:
        ``"qNIPV"`` (global uncertainty reduction), ``"PSD"`` or ``"qPSD"``.
    initial_temperatures:
        Opening design. Defaults to the two limits of ``bounds``.
    """

    def __init__(
        self,
        simulator: CharpySimulator,
        bounds: Optional[Tuple[float, float]] = None,
        acquisition: str = "qNIPV",
        initial_temperatures: Optional[Sequence[float]] = None,
        num_restarts: int = 10,
        raw_samples: int = 128,
    ) -> None:
        if acquisition not in ACQUISITIONS:
            raise ValueError(
                f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}"
            )
        self.simulator = simulator
        self.bounds = tuple(bounds or simulator.config.temperature_range)
        self.acquisition = acquisition
        self.history: List[Step] = []

        temperatures = np.asarray(
            initial_design(self.bounds)
            if initial_temperatures is None
            else initial_temperatures,
            dtype=float,
        )
        energies = simulator.simulate(temperatures).y

        self._optimiser = BayesianOptimisation(
            BayesianOptimisationConfig(
                surrogate_config=SingleTaskGPTorchConfig(
                    X=torch.tensor(temperatures, dtype=DTYPE).reshape(-1, 1),
                    y=torch.tensor(energies, dtype=DTYPE).reshape(-1, 1),
                ),
                objective_config={
                    "objective_type": "BoTorchAcquisition",
                    "acq_func": acquisition,
                },
                lower_bounds=torch.tensor([self.bounds[0]], dtype=DTYPE),
                upper_bounds=torch.tensor([self.bounds[1]], dtype=DTYPE),
                num_points=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
        )

    # ------------------------------------------------------------------
    @property
    def temperatures(self) -> np.ndarray:
        """Test temperatures observed so far [C]."""
        return self._optimiser.config.surrogate_config.X.detach().numpy().reshape(-1)

    @property
    def energies(self) -> np.ndarray:
        """Energies absorbed so far [J]."""
        return self._optimiser.config.surrogate_config.y.detach().numpy().reshape(-1)

    def propose(self) -> Tuple[float, float]:
        """Ask the acquisition for the next test temperature.

        Returns the temperature [C] and the batch acquisition value.
        """
        result = self._optimiser.ask(1)
        return float(result.candidates[0]["value"][0]), float(result.batch_score)

    def step(self) -> Step:
        """Propose a temperature, break a specimen at it, and refit the GP."""
        result = self._optimiser.ask(1)
        candidate = result.candidates[0]["value"]
        energy = float(self.simulator.simulate(np.asarray(candidate)).y[0])
        self._optimiser.tell(
            result.candidates, [[energy]], batch_score=result.batch_score
        )
        record = Step(
            index=len(self.history),
            temperature=float(candidate[0]),
            energy=energy,
            acquisition_value=float(result.batch_score),
            n_observations=int(self._optimiser.config.surrogate_config.X.shape[0]),
        )
        self.history.append(record)
        return record

    def run(self, n_steps: int) -> List[Step]:
        """Run ``n_steps`` iterations and return them."""
        return [self.step() for _ in range(n_steps)]

    # ------------------------------------------------------------------
    def posterior(self, grid: Optional[np.ndarray] = None):
        """GP posterior mean and standard deviation on ``grid``.

        Refits a GP on the current data, because the optimiser holds the data on
        its surrogate config rather than a live fitted model.

        The spread is the **predictive** standard deviation: it includes the
        observation noise, so it answers "what would the next specimen at this
        temperature read?". The latent-only spread would be the purer measure of
        what active learning reduces, but this GP's latent covariance is
        numerically singular and Cholesky fails on it at any grid size. A
        practical consequence: the band plateaus at the test-to-test scatter
        rather than shrinking towards zero, because no amount of testing removes
        that scatter.
        """
        if grid is None:
            grid = np.linspace(self.bounds[0], self.bounds[1], 200)
        model = SingleTaskGPTorch(self._optimiser.config.surrogate_config)
        model.fit()
        result = model.predict(
            torch.tensor(grid, dtype=DTYPE).reshape(-1, 1), observation_noise=True
        )
        return (
            np.asarray(grid, dtype=float),
            result.prediction.detach().numpy().reshape(-1),
            result.uncertainty.detach().numpy().reshape(-1),
        )

    def mean_posterior_std(self, grid: Optional[np.ndarray] = None) -> float:
        """Average GP predictive standard deviation over the range [J].

        The quantity active learning is driving down. It plateaus at the
        measurement noise rather than reaching zero — see :meth:`posterior`.
        """
        _, _, std = self.posterior(grid)
        return float(np.mean(std))

    def dbtt_estimate(self, grid: Optional[np.ndarray] = None) -> float:
        """Temperature where the GP mean crosses the mid-energy level.

        The mid-energy definition of the DBTT: halfway between the GP's lowest
        and highest predicted energy. Read off the posterior mean, so it carries
        no uncertainty of its own — use it as a running indicator, not a result.
        """
        grid, mean, _ = self.posterior(grid)
        midpoint = 0.5 * (mean.min() + mean.max())
        crossing = np.argmin(np.abs(mean - midpoint))
        return float(grid[crossing])
