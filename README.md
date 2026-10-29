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
| `src/dbtt/recovery.py` | Recover `(LS, US, DBTT, C)` from data; the DBTT posterior |
| `notebooks/05_engine_active_learning.ipynb` | Active learning as remote Engine workflows |
| `notebooks/06_dbtt_from_the_model.ipynb` | The DBTT and its uncertainty, two ways |
| `src/dbtt/backend.py` | Whether the uncertainty-engine SDK is installed |
| `src/dbtt/sdk_workflows.py` | Build/save/run DBTT workflows on the remote Engine via the SDK |
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
uv venv --python 3.12 && uv pip install -e ".[dev,sdk]"
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

## The DBTT, with its uncertainty

The campaign exists to locate a transition temperature, so the deliverable is a DBTT
*with* an uncertainty. `dbtt.recovery` provides two routes, and
`notebooks/06_dbtt_from_the_model.ipynb` runs both.

### 1. A posterior from the measurements

Fit the AHT form, `Y(T) = (LS+US)/2 + (US-LS)/2 · tanh((T−DBTT)/C)`. Written as
`Y = a + b·h(T)`, it is **linear in the shelves** once the shape `(DBTT, C)` is fixed, so
the shelves and the noise precision integrate out in closed form and the marginal
likelihood is a function of two parameters only.

`dbtt_posterior` evaluates that on a grid, normalises, and marginalises out the width:

```python
from dbtt.recovery import dbtt_posterior, recover

fit = recover(temperatures, energies, symmetric=True)
posterior = dbtt_posterior(temperatures, energies, result=fit)
print(posterior)          # DBTTPosterior(mean=-45.11, std=2.33, 95% CI=[-49.93, -39.91])
```

That is `p(DBTT | data)` itself, not a profile or a curvature approximation. Priors are
flat in the DBTT and flat in `log C` — proper only because the grids are bounded, so a
grid that clips the posterior will distort it.

Two things the fit has to get right, both found by it failing first:

- **Heteroscedastic noise.** Charpy scatter grows with absorbed energy. A homoscedastic
  fit over-weights the quiet lower shelf and collapses onto degenerately narrow widths.
  `recover` whitens by a `sigma(T)` estimated from a first pass.
- **Fixed weights.** Those weights are estimated once and held fixed; re-estimating them
  per candidate shape makes the evidences incomparable and the optimiser games it. The
  posterior grid reuses the same fixed weights for the same reason.

**Calibration is measured, not assumed.** Over 25 noise seeds the nominal 95% credible
interval contained the truth 24 times (96%), against 92% for the profile-likelihood
interval it replaces; bias −0.66 °C, RMSE 6.59 °C on 18-specimen campaigns.

### 2. Reading it off a fitted model

`dbtt_from_curve` takes a model's posterior across a temperature slice — the mean and
standard deviation the Engine's prediction workflow saves — finds where it crosses the
mid-shelf energy, and propagates the uncertainty through the slope by the delta method,
`sd_DBTT ≈ sd_Y(T*) / |dY/dT|(T*)`.

The two spreads measure different things and should not be read as checks on each other.
The AHT posterior is the uncertainty *given the tanh form*. The curve reading divides a
**predictive** standard deviation — measurement scatter included — by the transition
gradient, so on a shallow transition it is the wider of the two. The delta method also
linearises at the crossing and ignores correlation between neighbouring temperatures,
which a draw-based estimate would capture — the Engine exposes no posterior draws, so
that is not available.

### Is `C` really `C(T)`?

No — for this material it is a constant. `compare_width_models` fits both the symmetric
and the asymmetric form and compares by profile evidence, BIC and leave-one-out RMSE. The
asymmetric form stays available for welds and irradiated material. `dbtt_posterior`
supports the symmetric model only: with two widths the grid becomes three-dimensional.

## The SDK

Everything that models — the GP, the acquisition, the scoring — runs on the **remote
Uncertainty Engine**, reached through its SDK (`uncertainty-engine` on PyPI, pinned
`>=0.15,<0.16`). This repo holds the simulator that stands in for the test house, and the
code that builds Engine workflows and drives them.

0.15 is a breaking change against earlier SDKs: it dropped the per-node classes in favour
of building nodes by name off the client, `client.nodes.TrainModel(...)`, with each schema
resolved from the Node Registry on demand. `sdk_workflows` builds its nodes that way and
pins no versions when a client is present.

One environment is enough:

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,sdk]"
```

An unpublished distribution of the same name provides an *in-process* modelling library
and shadows the SDK if installed alongside it. This repo no longer uses it;
`dbtt.backend.describe()` reports which one you have, and `require_sdk()` fails with the
install command rather than an opaque `ImportError`.

### Credentials

Copy `.env.example` to `.env` and fill in `UE_USERNAME` / `UE_PASSWORD`; on Colab use the
Secrets panel with the same keys. `load_credentials()` reads Colab secrets first, then
`.env`, and reports only *where* credentials came from, never their values. `.env` is
gitignored.

### Workflows

`sdk_workflows` builds five graphs — the four exported from the Engine GUI, plus one that
saves the posterior as a reusable dataset:

| Workflow | Nodes | Purpose |
| --- | --- | --- |
| `build_training_workflow` | `LoadDataset` → `FilterDataset` ×2 → `ModelConfig` → `TrainModel` → `Save` | Fit a GP on the campaign, save it as a project resource |
| `build_recommendation_workflow` | `LoadModel` → `Recommend` → `Save` | Active learning: the next test temperature |
| `build_gp_plot_workflow` | `LoadModel` + `CreateDatasetSlice` → `PredictModel` → `UncertaintyPlot` | Model visualisation, returned inline with the job |
| `build_prediction_workflow` | `LoadModel` + `CreateDatasetSlice` → `PredictModel` → `Save` ×2 | The same posterior, **saved** as project datasets |
| `build_validation_workflow` | `LoadModel` + `LoadDataset` → `FilterDataset` ×2 → `PredictModel` → `ScoreModel` + `ParityPlot` | Score against held-out data, with a parity plot |

The campaign is uploaded as **one** dataset and `FilterDataset` splits out the input and
output columns, so each active-learning iteration re-uploads a single CSV.

`save_workflow` puts each into the project workspace, so it appears in the GUI and can be
re-run, versioned and edited — the point of building them programmatically. The loop then
calls them back by id with `run_saved`.

Two things worth knowing about reading results:

- **Only some outputs are values.** `UncertaintyPlot` returns the posterior inline;
  `PredictModel`'s and `Recommend`'s own handles return S3 references. That is why the
  loop plots the first and reads recommended points from the resource `Save` wrote, and
  why `build_prediction_workflow` exists at all — a plot payload dies with the job, a
  saved dataset does not.
- **Creating and versioning are different calls.** `resources.upload` refuses a name the
  project already holds (409); `resources.update` adds a version and keeps the resource
  id. The saved workflows point at that id, so `upload_dataset` picks the right call and
  the graphs never need editing between iterations. Workflow names collide the same way,
  so `save_workflow` updates an existing one rather than failing.

**Status:** run end to end against a live Engine project — four iterations from a
two-point seed design, scoring R² 0.9996 (a *training* score: the validation workflow
scores against the campaign itself, so point it at a held-out resource for a
generalisation estimate). `notebooks/05_engine_active_learning.ipynb` is committed with
its outputs.

## Tests

```bash
.venv/bin/python -m pytest -q
```

`tests/test_recovery.py` is slow — around eight minutes — because the calibration tests
fit the AHT over many noise seeds. `-k "not coverage"` skips the worst of it.

The workflow-graph tests build every graph with no client, which checks the wiring —
labels, edges and requested outputs — against the workflows exported from the Engine GUI
without a network call. They cannot check inputs against the live node schemas; only
running against the Engine does that.


```bash
.venv/bin/python -m pytest -q
```
