"""A local mirror of the uncertainty-engine `Simulator` noun.

Deliberately dependency-light: it reproduces the shape of
``uncertainty_engine.interface.simulator`` (config-driven construction, a leaf
``_forward`` implementing the raw ``X -> y`` map, a public ``simulate`` that
wraps it in a ``SimulationResult``, and an overridable ``build_result``) without
importing the engine, so this repo stays installable on its own. Anything
written against it should port across with a change of import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DataType = TypeVar("DataType")


class SimulationResult(BaseModel):
    """Structured result returned by :meth:`Simulator.simulate`.

    ``y`` carries the canonical output of the forward map; ``outputs`` holds
    auxiliary machine-readable payload a leaf chooses to expose.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    y: Any = Field(None, description="Canonical output(s) of the forward map X -> y.")
    outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary payload: diagnostics, provenance, units, ground truth.",
    )


class SimulatorConfig(BaseModel):
    """Base configuration for :class:`Simulator` leaves.

    Concrete simulators subclass this with their own fields; by the engine's
    convention the subclass name ends in ``Config``.
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(
        None, description="Optional human-readable name for this simulator instance."
    )
    description: Optional[str] = Field(
        None, description="Optional free-text note on what this simulator computes."
    )


class Simulator(Generic[DataType], ABC):
    """Abstract base for all simulators: the ``X -> y`` forward map.

    Construction is config-driven; ``X`` is an argument to ``simulate``, never
    stored on the config::

        sim = CharpySimulator(config)
        result = sim.simulate(X)     # SimulationResult: y plus metadata
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    # ----- the one universal verb -----

    @abstractmethod
    def _forward(self, X: DataType) -> Any:
        """Evaluate the raw forward map ``X -> y``.

        Leaves implement this and return the raw output; the public
        :meth:`simulate` wraps it in a :class:`SimulationResult`.
        """

    def simulate(self, X: DataType) -> SimulationResult:
        """Evaluate the forward map and wrap the output in a result.

        One call is one evaluation of the forward map. For a stochastic
        simulator that is a single realisation, not a summary of the output law.
        """
        y = self._forward(X)
        return self.build_result(X, y)

    def build_result(self, X: DataType, y: Any) -> SimulationResult:
        """Wrap a raw forward output in a :class:`SimulationResult`.

        The extension point every ``simulate`` goes through: override it to
        attach diagnostics, units or provenance.
        """
        return SimulationResult(y=y)

    # ----- introspection -----

    @classmethod
    def data_type(cls) -> Optional[Type]:
        """Report the input type this simulator expects."""
        return None

    @classmethod
    def capabilities(cls) -> List[str]:
        """Names of the optional capability mixins this leaf declares."""
        return []

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        """Introspect the leaf without instantiating it."""
        return {
            "name": cls.__name__,
            "summary": (cls.__doc__ or "").strip().split("\n")[0],
            "data_type": getattr(cls.data_type(), "__name__", None),
            "capabilities": cls.capabilities(),
        }


__all__ = ["DataType", "SimulationResult", "Simulator", "SimulatorConfig"]
