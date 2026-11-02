# ukaea-dbtt-active-learning

Active learning for ductile–brittle transition temperature (DBTT) testing, against
the remote [Uncertainty Engine](https://uncertaintyengine.ai).

Everything lives in one notebook:

**[`DBTT_active_learning.ipynb`](DBTT_active_learning.ipynb)** &nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digiLab-ai/ukaea-dbtt-active-learning/blob/main/DBTT_active_learning.ipynb)

| Section | What it does |
| --- | --- |
| 1. A simulated test campaign | A Charpy simulator stands in for the test house, so every estimate can be scored against a known truth |
| 2. Active learning on the Engine | Five workflows — train, recommend, visualise, predict, validate — saved to the project and driven in a loop |
| 3. The DBTT, with its uncertainty | Two routes to a *posterior* over the DBTT, not a point estimate |

The notebook is **self-contained**: it imports nothing from this repository, so the
Colab badge works without a checkout. Its supporting code sits in collapsed cells, each
titled with what it covers.

## Running it

**Google Colab.** Click the badge. Because the repository is internal you will be asked
to authorise Colab's GitHub access the first time. Put your credentials in the Secrets
panel (the 🔑 key icon in the left sidebar) as `UE_USERNAME`, `UE_PASSWORD` and
`UE_PROJECT_NAME`, with *Notebook access* switched on for each. The first cell installs
whatever the runtime is missing.

**VS Code, or any local Jupyter.**

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill it in
```

Pick the `.venv` kernel in VS Code's top-right kernel picker. `.env` is gitignored and
must never be committed; copy the example rather than editing it, so the template keeps
its keys blank.

`UE_PROJECT_NAME` must name a project that **already exists** — projects are created in
the Engine GUI, not from the SDK. `load_credentials()` tries Colab secrets first, then
`.env`, then the environment, and reports only *where* the credentials came from, never
what they are, so the notebook's outputs are safe to commit.

## The model

Energy absorbed follows an asymmetric hyperbolic tangent (AHT):

```
Y(T) = (LS + US)/2 + (US - LS)/2 * tanh((T - DBTT) / C(T))
```

with `C(T) = width_lower` below the DBTT and `width_upper` above it; setting the two
equal recovers the symmetric tanh. Measurement noise is heteroscedastic,
`sigma(T) = base + relative * Y(T)`. The defaults describe the EUROFER EAF plate-1
campaign — `LS=5 J, US=200 J, DBTT=-40 C, C=30 C` — and the noise parameters are
hand-set, not fitted.

Ground truth (`true_curve`, `noise_std`, `result.outputs`) has no counterpart in a real
experiment. It never reaches the dataset the GP trains on; it is there only to score how
well the active learning recovers the answer.

## Two things worth knowing

**The DBTT posterior is the deliverable.** Written as `Y = a + b·h(T)`, the AHT is
*linear in the shelves* once the shape `(DBTT, C)` is fixed, so the shelves and the noise
precision integrate out in closed form and the evidence is a function of two parameters
only. Sweeping those on a grid and marginalising the width gives `p(DBTT | data)` itself,
rather than a profile or a curvature approximation. Calibration was measured rather than
assumed: over 25 noise seeds the nominal 95% credible interval contained the truth 24
times (96%), against 92% for the profile-likelihood interval it replaced.

**Resource versioning is what makes the loop cheap.** `resources.upload` refuses a name
the project already holds (409); `resources.update` adds a version and keeps the resource
id. The saved workflows point at that id, so each iteration re-uploads one CSV and no
graph ever needs editing. Workflow names collide the same way, so re-running the notebook
revises its five workflows rather than duplicating them.

## History

Until `main` was streamlined to this single notebook, the same work lived in an installed
`dbtt` package (simulator, recovery, SDK workflow builders), a pytest suite and three
notebooks. That is all still in the history — see the commit that removed it — if you
need the tests or want to import the code rather than run it.
