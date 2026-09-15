"""Configuration models for the Charpy / DBTT simulator.

Follows the uncertainty-engine pattern: every simulator input is a declared,
validated, documented pydantic field, and a whole run is reproducible from a
single YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .interface import SimulatorConfig


class AHTParams(BaseModel):
    """Asymmetric hyperbolic tangent (AHT) transition curve.

    Energy absorbed as a function of test temperature::

        Y(T) = (LS + US) / 2 + (US - LS) / 2 * tanh((T - DBTT) / C(T))

    where ``C(T) = width_lower`` below the DBTT and ``width_upper`` above it.
    Setting the two widths equal recovers the symmetric hyperbolic tangent.
    """

    model_config = ConfigDict(extra="forbid")

    lower_shelf: float = Field(
        5.0, ge=0.0, description="Lower (brittle) shelf energy, LS [J]."
    )
    upper_shelf: float = Field(
        200.0, gt=0.0, description="Upper (ductile) shelf energy, US [J]."
    )
    dbtt: float = Field(
        -40.0, description="Ductile-brittle transition temperature [C]."
    )
    width_lower: float = Field(
        30.0,
        gt=0.0,
        description="Transition width C below the DBTT [C]. Larger is more gradual.",
    )
    width_upper: float = Field(
        30.0,
        gt=0.0,
        description="Transition width C above the DBTT [C]. Larger is more gradual.",
    )

    @model_validator(mode="after")
    def _check_shelves(self) -> "AHTParams":
        if self.upper_shelf <= self.lower_shelf:
            raise ValueError("upper_shelf must be greater than lower_shelf")
        return self


class NoiseParams(BaseModel):
    """Heteroscedastic measurement noise.

    The standard deviation has a constant floor plus a component proportional
    to the true energy::

        sigma(T) = base + relative * Y(T)
    """

    model_config = ConfigDict(extra="forbid")

    base: float = Field(
        3.0, ge=0.0, description="Constant noise standard deviation [J]."
    )
    relative: float = Field(
        0.10,
        ge=0.0,
        description="Noise standard deviation as a fraction of the true energy [-].",
    )
    energy_floor: float = Field(
        0.0,
        ge=0.0,
        description="Sampled energies are clipped at this value [J]; a Charpy "
        "test cannot absorb negative energy.",
    )


class SpecimenParams(BaseModel):
    """Specimen provenance carried through to the test-result table."""

    model_config = ConfigDict(extra="forbid")

    material: str = Field("EUROFER EAF", description="Material designation.")
    condition: str = Field("Base", description="Material condition.")
    processing: str = Field("Rolled", description="Processing route.")
    orientation: str = Field("RD", description="Specimen orientation.")
    specimen_design: str = Field("ASTM", description="Specimen design standard.")
    specimen_location: str = Field(
        "EAF 1a Plate 1", description="Location the specimen was taken from."
    )
    specimen_type: str = Field("Charpy", description="Test specimen type.")
    notch_type: str = Field("V notch", description="Notch geometry.")
    neurone_id_prefix: str = Field(
        "NRN2025-", description="Prefix for generated NEURONE specimen IDs."
    )
    neurone_id_digits: int = Field(
        4, ge=1, description="Zero-padded width of the NEURONE ID number."
    )


class CharpySimulatorConfig(SimulatorConfig):
    """Configuration for :class:`dbtt.simulator.CharpySimulator`."""

    name: str = Field("eurofer_eaf", description="Short name for this campaign.")
    curve: AHTParams = Field(default_factory=AHTParams)
    noise: NoiseParams = Field(default_factory=NoiseParams)
    specimen: SpecimenParams = Field(default_factory=SpecimenParams)
    seed: Optional[int] = Field(
        0, description="Random seed. None gives non-reproducible draws."
    )
    temperature_range: tuple[float, float] = Field(
        (-150.0, 100.0),
        description="Physically sensible test-temperature bounds [C]. Requests "
        "outside this range are rejected.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> "CharpySimulatorConfig":
        lo, hi = self.temperature_range
        if hi <= lo:
            raise ValueError("temperature_range must be (low, high) with high > low")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CharpySimulatorConfig":
        """Load a configuration from a YAML file."""
        with open(path) as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})

    def to_yaml(self, path: str | Path) -> None:
        """Write this configuration to a YAML file."""
        with open(path, "w") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
