# ukaea-dbtt-active-learning

Active learning for ductile-brittle transition temperature (DBTT) testing.

The end goal is to recommend Charpy test temperatures and return a fitted GP that can be
studied to locate the DBTT. This stage provides the **simulator** that stands in for the
real test campaign, plus dataset and plotting utilities.

## Model

Energy absorbed follows an asymmetric hyperbolic tangent (AHT):

```
Y(T) = (LS + US)/2 + (US - LS)/2 * tanh((T - DBTT) / C(T))
```

with `C(T) = width_lower` below the DBTT and `width_upper` above it. Setting the two
widths equal recovers the symmetric tanh. Measurement noise is heteroscedastic, with a
base and a relative component:

```
sigma(T) = noise.base + noise.relative * Y(T)
```

Sampled energies are clipped at `noise.energy_floor` (0 J by default).

### Default parameters

`configs/eurofer_eaf.yaml` ships `LS=5 J, US=200 J, DBTT=-40 C, C=30 C` (both sides),
describing the EUROFER EAF plate-1 campaign. The curve is symmetric; the asymmetric form
is available by setting the two widths differently.

Noise defaults (`base=3 J`, `relative=0.10`) are hand-set, not fitted. Real Charpy
scatter in the transition region is wider than this — raise `relative` towards ~0.2 for
a more realistic spread.

## Layout

| Path | Purpose |
| --- | --- |
| `src/dbtt/interface.py` | Local mirror of the uncertainty-engine `Simulator` noun (`_forward` / `simulate` / `SimulationResult`) |
| `src/dbtt/config.py` | Pydantic config models (`AHTParams`, `NoiseParams`, `SpecimenParams`, `CharpySimulatorConfig`), YAML load/save |
| `src/dbtt/simulator.py` | `CharpySimulator` — `simulate`, `run`, `true_curve`, `noise_std` |
| `src/dbtt/dataset.py` | Resolve and load campaign CSVs in `data/` |
| `src/dbtt/plotting.py` | `plot_dataset` — scatter of energy absorbed vs test temperature |
| `configs/eurofer_eaf.yaml` | Default configuration |
| `data/*.csv` | Simulated campaigns (gitignored) |
| `notebooks/01_simulate_dataset.ipynb` | Build and visualise a dataset |
| `tests/` | pytest suite |

## Simulator API

The simulator mirrors the uncertainty-engine `Simulator` noun: config-driven
construction, a leaf `_forward(X)` implementing the raw `X -> y` map, and the public verb
`simulate(X)` returning a `SimulationResult`.

```python
result = simulator.simulate(-45.0)
result.y                         # array([87.86])  — the measurement
result.outputs["true_energy"]    # array([86.40])  — ground truth, simulator-only
result.outputs["noise_std"]      # array([11.64])
```

The forward map is **stochastic**: one `simulate` call is one broken specimen, a single
realisation rather than a summary of the output law. `simulator.run(temperatures)` wraps
`simulate` and attaches specimen provenance and numbering.

`src/dbtt/interface.py` is a dependency-light local copy of that contract, so this repo
installs on its own; code written against it ports to the engine with a change of import.

### Ground truth stays out of the dataset

`true_curve`, `noise_std` and `result.outputs` are simulator-only — they have no
counterpart in a real experiment. The CSV carries only what a test house would report
(specimen details, test temperature, energy absorbed), so ground truth cannot leak into
the GP. Use it only to score how well active learning recovers the DBTT.

## Setup

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

## Usage

```python
import pandas as pd

from dbtt import CharpySimulator, CharpySimulatorConfig, plot_dataset
from dbtt.simulator import COLUMNS

config = CharpySimulatorConfig.from_yaml("configs/eurofer_eaf.yaml")
simulator = CharpySimulator(config)

# The campaign is an ordinary DataFrame; you own it.
df = pd.DataFrame(columns=COLUMNS)

for batch in ([-120, -80, -40, 0, 40], [-60, -20]):
    rows = simulator.run(batch, start_index=len(df) + 1)   # numbering continues
    df = pd.concat([df, rows], ignore_index=True)

df.to_csv("data/my_campaign.csv", index=False)
plot_dataset(df)
```

`dbtt.dataset.load("my_campaign")` is a convenience wrapper over `pd.read_csv` that
resolves bare names against `data/`; creating and growing a campaign is done in the
notebook with plain pandas.

## Notes

- CSV headers use `(C)` rather than `(°C)` to keep the files encoding-safe; column order
  otherwise matches the test-house spreadsheet. The constant `DBTT (°C)` column is
  dropped (it is a property of the campaign, held in the config, not of each specimen).
- `CharpySimulator.measure` advances its random state on every call, so repeated tests at
  the same temperature give different energies — as in a real campaign.
- Requested temperatures outside `temperature_range` are rejected rather than silently
  extrapolated.

## Tests

```bash
.venv/bin/python -m pytest -q
```
