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
| `src/dbtt/recovery.py` | Recover `(LS, US, DBTT, C)` from data via engine BLR + a shape search |
| `notebooks/01_simulate_dataset.ipynb` | Build and visualise a dataset |
| `src/dbtt/active_learning.py` | `DBTTActiveLearner` — engine BO with qNIPV / PSD over test temperature |
| `notebooks/02_recover_dbtt.ipynb` | Fit the AHT parameters and test whether `C` varies with `T` |
| `notebooks/03_bayesian_optimisation.ipynb` | Choose test temperatures by uncertainty reduction |
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

## The uncertainty-engine

`src/dbtt/recovery.py` uses the engine's `BayesianLinearRegression`. The engine is a
sister repo, installed editable rather than from an index:

```bash
uv pip install -e "../uncertainty-engine[gptorch,metrics]"
```

### Recovering the AHT parameters

Conditional on the shape `(DBTT, C)`, the AHT is *linear* in the shelves —
`Y = a + b·tanh((T−DBTT)/C)` with `a=(LS+US)/2`, `b=(US−LS)/2`. So BLR gives a
closed-form shelf posterior plus a log marginal likelihood, and only the two nonlinear
shape parameters need an outer search.

Two things the fit has to get right, both found by the fit failing first:

- **Heteroscedastic noise.** Charpy scatter grows with absorbed energy. A homoscedastic
  fit over-weights the quiet lower shelf and collapses onto degenerate narrow widths
  (recovering `C=16.6` against a true `30`). `recover` whitens by a `sigma(T)` estimated
  from a first pass.
- **Fixed weights.** Those weights are estimated once and held fixed. Re-estimating them
  per candidate shape makes the evidences incomparable and the optimiser games it. The
  evidence carries the `sum(log w)` Jacobian term so it stays a density over the original
  energies.

Calibration is measured, not assumed: noiseless recovery is exact to 4 decimal places,
and the nominal 95% DBTT profile interval achieves ~88% empirical coverage (mildly
over-confident, because the estimated weights are treated as known).

### Active learning

`DBTTActiveLearner` wraps the engine's `BayesianOptimisation` in an ask / run-the-
simulator / tell loop over a single input, test temperature:

```python
from dbtt.active_learning import DBTTActiveLearner

learner = DBTTActiveLearner(simulator, bounds=(-150, 100), acquisition="qNIPV")
for step in learner.run(8):
    print(step.temperature, step.energy)
```

The initial design is the two limits of the testable range: they pin the shelves the
transition sits between, without which the curve is unidentifiable. Then `qNIPV`
(integrated posterior variance, global) or `PSD`/`qPSD` (posterior standard deviation,
greedy) chooses each next temperature.

Two things worth knowing:

- The plotted band is the **predictive** standard deviation, so it plateaus at the
  material's test-to-test scatter rather than shrinking to zero. The latent-only spread
  would be the purer measure of what active learning reduces, but this GP's latent
  covariance is numerically singular and Cholesky fails on it at any grid size.
- `dbtt_estimate()` reads the DBTT off the GP posterior mean and carries **no uncertainty
  of its own**. For a DBTT with an interval, fit the AHT with `recovery.recover`.

### Is `C` really `C(T)`?

No — for this material it is a constant. `compare_width_models` fits both and compares by
profile evidence, BIC and leave-one-out RMSE. Over 10 noise seeds it picks the constant
model 8/10 when the truth is constant and detects real asymmetry 8/10 when it is there,
so the answer is not just a preference for the simpler model. The asymmetric form stays
available for welds and irradiated material.

## Tests

```bash
.venv/bin/python -m pytest -q
```
