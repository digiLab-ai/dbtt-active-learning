# dbtt-active-learning

Active learning for materials test campaigns, against the remote
[Uncertainty Engine](https://uncertaintyengine.ai).

Three self-contained notebooks, one per test type. Each one simulates a test
house, drives an active-learning loop on the Engine, reads a **decision number
with an interval** off the fitted model, and then asks whether choosing where to
test actually beat a fixed design.

| Notebook | The question | Inputs | |
| --- | --- | --- | --- |
| [`DBTT_active_learning.ipynb`](DBTT_active_learning.ipynb) | Where is the ductile–brittle transition temperature? | temperature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digiLab-ai/dbtt-active-learning/blob/main/DBTT_active_learning.ipynb) |
| [`Creep_active_learning.ipynb`](Creep_active_learning.ipynb) | What stress gives 100,000 h life at the design temperature? | stress **and** temperature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digiLab-ai/dbtt-active-learning/blob/main/Creep_active_learning.ipynb) |
| [`Fatigue_active_learning.ipynb`](Fatigue_active_learning.ipynb) | What amplitude survives 10⁷ cycles? | stress amplitude | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digiLab-ai/dbtt-active-learning/blob/main/Fatigue_active_learning.ipynb) |

They share a structure, deliberately, so that reading one teaches you all three:

| Section | What it does |
| --- | --- |
| Settings | Everything meant to be changed — the material, the noise, the initial design, the kernel, the acquisition, the plot colours |
| 1. A simulated campaign | A simulator stands in for the test house, so every estimate can be scored against a known truth |
| 2. Active learning on the Engine | Workflows — train, recommend, visualise, predict, validate — saved to the project and driven in a loop |
| 3. The answer, with its uncertainty | The decision number read off the trained model, with the uncertainty that comes with it |
| 4. Was the active learning worth it? | Three fixed designs through the same machinery, for comparison |

Each notebook is **self-contained**: it imports nothing from this repository, so
the Colab badges work without a checkout. Supporting code sits in collapsed
cells, each titled with what it covers, and everything meant to be changed is
gathered in a single **Settings** cell at the top.

## The common method

All three estimate the same way, and the method is the point.

**The answer is a level crossing.** A DBTT is where the transition curve
crosses the mid-shelf energy; a creep design stress is where the life curve
crosses 100,000 hours; a fatigue strength is where the S–N curve crosses 10⁷
cycles. In every case a fitted GP posterior is cut by a horizontal line.

**The interval is that band inverted.** Draw the level across the credible band;
the positions at which the band still contains it are the positions at which the
curve could still be at the level. This needs no linearisation and never divides
by the curve's gradient, so a shallow, poorly resolved curve *widens* the
interval instead of being flattered by it — which the delta method,
`sd(x*) / |dy/dx|(x*)`, does not. The result is asymmetric in general, and an
end comes back as `None` when the data does not bound the answer on that side.

Two caveats travel with it, and are repeated in each notebook's notes. The band
is **pointwise**, so inverting it gives approximately rather than exactly the
nominal coverage. And it is **predictive** — it carries the material's
specimen-to-specimen scatter — so it answers "where could a fresh specimen read
the level?" rather than "where does the underlying curve cross it?". Read it as
the conservative of the two.

The DBTT notebook shares its read-off with the other two: `dbtt_from_curve`
derives the level from the curve's own shelves, while `crossing_from_curve` in
the creep and fatigue notebooks takes the level as an argument. The band
inversion underneath is the same code.

## What each one is for

**DBTT** is the simplest: one input, a transition curve with two shelves, and a
level the curve supplies itself.

**Creep** is two-dimensional, and that is why it is here. Rupture life depends
on stress *and* temperature, and the Larson–Miller parameter says a hotter,
shorter test carries the same information as a cooler, longer one. A campaign
can buy long-life knowledge at high temperature instead of buying it with time,
and the acquisition is in a position to make that trade without being told the
parameter exists. `CreateDatasetSlice` holds temperature fixed to read the
design line; a separately uploaded grid gives the whole surface.

**Fatigue** is one-dimensional again but carries the awkward feature the other
two do not: **runouts**. A specimen that survives the rig's cycle limit has not
told you its life, only that it exceeds the limit. That is a censored
observation, and a GP regression forces you to enter it at the limit as though
it were a measurement — which biases the fitted strength optimistic, in a way no
credible band widens to cover. The notebook does this, says so, and explains
what a censored likelihood would do instead.

## One knob is deliberately missing

The GP's **mean function** is not exposed by the Engine: `ModelConfig` takes a
kernel, not a mean, and `TrainModel` takes only `(config, inputs, outputs)`. The
kernel is the lever: the `LIN` term in the default `LIN+RBF` is what lets the
fit carry a linear trend, which is most of what a non-constant mean would buy
you here.

## Running them

**Google Colab.** Click a badge. The repository is public, so no GitHub
authorisation is needed. Put your credentials in the Secrets panel (the 🔑 key
icon in the left sidebar) as `UE_USERNAME`, `UE_PASSWORD` and
`UE_PROJECT_NAME`, with *Notebook access* switched on for each. The first cell
installs whatever the runtime is missing.

**VS Code, or any local Jupyter.**

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill it in
```

Pick the `.venv` kernel in VS Code's top-right kernel picker. `.env` is
gitignored and must never be committed; copy the example rather than editing it,
so the template keeps its keys blank.

`UE_PROJECT_NAME` must name a project that **already exists** — projects are
created in the Engine GUI, not from the SDK. `load_credentials()` tries Colab
secrets first, then `.env`, then the environment, and reports only *where* the
credentials came from, never what they are, so the notebooks' outputs are safe
to commit.

All three notebooks write their resources and workflows into the same project
under distinct names (`dbtt_*`, `creep_*`, `fatigue_*`), so they do not collide
and can be run in any order.

## The models

**Charpy impact.** Energy absorbed follows an asymmetric hyperbolic tangent:

```
Y(T) = (LS + US)/2 + (US - LS)/2 * tanh((T - DBTT) / C(T))
```

with `C(T) = width_lower` below the DBTT and `width_upper` above it. Noise is
heteroscedastic, `sigma(T) = base + relative * Y(T)`. The defaults describe the
EUROFER EAF plate-1 campaign — `LS=5 J, US=200 J, DBTT=-40 C, C=30 C` — and the
noise parameters are hand-set, not fitted.

**Creep rupture.** A Larson–Miller master curve, quadratic in log stress:

```
LMP(x) = a0 + a1*x + a2*x^2,   x = log10(stress / MPa)
log10(t_r) = LMP(x) / (T + 273.15) - C
```

Scatter is in log life and widens with it. The coefficients are pinned through
three anchors of a P91-like 9Cr steel, the first of which — 100 MPa / 550 °C /
100,000 h — is the answer the notebook recovers. The quadratic turns over at
29.1 MPa, below which it stops being physical, and the config refuses a stress
range that reaches it.

**Fatigue.** A Stromeyer S–N curve — Basquin's law with an endurance limit
underneath:

```
log10(N_f) = A - B * log10(S_a - S_e),   S_a > S_e
```

Life runs away as the amplitude approaches `S_e`, so the curve never crosses it.
The constants are pinned so the curve passes through 250 MPa at 10⁷ cycles,
which is the answer the notebook recovers; `S_e` itself is 230 MPa, and the
notes explain why the notebook asks for the former and not the latter.

Ground truth has no counterpart in a real experiment. It never reaches the
dataset the GP trains on; it is there only to score how well the active learning
recovers the answer.

## Two things worth knowing

**Resource versioning is what makes the loop cheap.** `resources.upload` refuses
a name the project already holds (409); `resources.update` adds a version and
keeps the resource id. The saved workflows point at that id, so each iteration
re-uploads one CSV and no graph ever needs editing. Workflow names collide the
same way, so re-running a notebook revises its workflows rather than duplicating
them.

**`Recommend` takes no bounds.** It searches the domain the trained model has
seen, so the *seed design* is what tells the acquisition how far it may roam —
not the plotting bounds. That is why the DBTT and fatigue notebooks seed at both
ends of their range and the creep notebook seeds at all four corners of its box.

## History

Until `main` was streamlined to notebooks, the same work lived in an installed
`dbtt` package (simulator, recovery, SDK workflow builders), a pytest suite and
three notebooks. That is all still in the history — see the commit that removed
it — if you need the tests or want to import the code rather than run it.
