# dbtt-active-learning

Active learning for ductile–brittle transition temperature (DBTT) testing, against
the remote [Uncertainty Engine](https://uncertaintyengine.ai).

Everything lives in one notebook:

**[`DBTT_active_learning.ipynb`](DBTT_active_learning.ipynb)** &nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digiLab-ai/dbtt-active-learning/blob/main/DBTT_active_learning.ipynb)

| Section | What it does |
| --- | --- |
| Settings | Everything meant to be changed — the material, the noise, the initial design, the kernel, the acquisition, the plot colours |
| 1. A simulated test campaign | A Charpy simulator stands in for the test house, so every estimate can be scored against a known truth |
| 2. Active learning on the Engine | Five workflows — train, recommend, visualise, predict, validate — saved to the project and driven in a loop |
| 3. The DBTT, with its uncertainty | The transition read off the trained model, with the uncertainty that comes with it |

The notebook is **self-contained**: it imports nothing from this repository, so the
Colab badge works without a checkout. Its supporting code sits in collapsed cells, each
titled with what it covers, and everything meant to be changed is gathered in a single
**Settings** cell at the top — nothing further down needs editing to run it on a
different material, a different campaign or a different model.

One knob is deliberately missing. The GP's **mean function** is not exposed by the
Engine: `ModelConfig` takes a kernel, not a mean, and `TrainModel` takes only
`(config, inputs, outputs)`. The kernel is the lever: the default `M32` carries no
trend term, and adding one — as in `LIN+M32` — lets the fit carry a linear trend,
which is most of what a non-constant mean would buy you here.

## Running it

**Google Colab.** Click the badge. The repository is public, so no GitHub
authorisation is needed. Put your credentials in the Secrets
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

**The interval is the deliverable, not the point estimate.** The DBTT is read off the
trained GP's posterior curve — the crossing of the mid-shelf energy `(LS + US) / 2` — and
its uncertainty comes from the delta method, `sd_DBTT ≈ sd(T*) / |dY/dT|(T*)`. Read that
spread with care. The numerator is a *predictive* standard deviation, so it includes the
material's test-to-test scatter; the denominator is the transition gradient, which is
precisely what active learning spends its specimens resolving. A steep gradient makes the
interval tight, and tight is not the same as right. A second, model-free route — fitting
the AHT to the measurements and marginalising the shelves and noise out in closed form,
which gives an asymmetric `p(DBTT | data)` rather than a ± number — was removed when the
notebook was cut back to the GP route, and is recoverable from git history.

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
