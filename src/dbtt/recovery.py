"""Recover the AHT parameters (LS, US, C, DBTT) from measured Charpy data.

The fit exploits a structural fact about the AHT. Write it as

    Y(T) = a + b * h(T),    h(T) = tanh((T - DBTT) / C(T))

with ``a = (LS + US) / 2`` and ``b = (US - LS) / 2``. Conditional on the *shape*
parameters ``(DBTT, C)``, the model is **linear** in ``(a, b)``. So the shelves
have a closed-form Gaussian posterior — see :func:`_blr`, which also reports the
log marginal likelihood — and only the two (or three) nonlinear shape parameters
need an outer search.

Because the shelves and the noise precision are marginalised analytically, the
evidence is a function of ``(DBTT, C)`` alone. :func:`dbtt_posterior` exploits
that: evaluate it on a grid, normalise, and marginalise out the width, and you
have ``p(DBTT | data)`` itself rather than an interval approximating it.

The outer search maximises that marginal likelihood, i.e. it is a *profile*
evidence: the shelves and the noise precision are marginalised, the shape
parameters are optimised. That asymmetry matters when comparing models of
different shape dimension, which is why :func:`compare_width_models` penalises
the profiled parameters with BIC and also reports held-out error.

**Heteroscedasticity matters here.** Charpy scatter grows with absorbed energy,
so a homoscedastic fit over-weights the quiet lower shelf, under-weights the
noisy upper shelf, and collapses onto degenerately narrow transition widths. The
fit therefore whitens the rows by an estimated ``sigma(T)`` before handing them
to BLR, re-estimating ``sigma`` from the residuals a few times (IRLS). Pass
``weights=None`` to opt out and see the unweighted behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

def tanh_basis(
    temperature: np.ndarray,
    dbtt: float,
    width_lower: float,
    width_upper: Optional[float] = None,
) -> np.ndarray:
    """The single basis function ``h(T) = tanh((T - DBTT) / C(T))``.

    ``width_upper=None`` gives the symmetric model, ``C(T) = width_lower``.
    """
    t = np.asarray(temperature, dtype=float)
    if width_upper is None:
        width = float(width_lower)
    else:
        width = np.where(t < dbtt, width_lower, width_upper)
    return np.tanh((t - dbtt) / width).reshape(-1, 1)


@dataclass
class ShelfPosterior:
    """Closed-form posterior over the shelves, conditional on the shape."""

    lower_shelf: float
    upper_shelf: float
    lower_shelf_std: float
    upper_shelf_std: float
    noise_std: float
    log_evidence: float

    def as_dict(self) -> dict:
        return {
            "lower_shelf": self.lower_shelf,
            "upper_shelf": self.upper_shelf,
            "lower_shelf_std": self.lower_shelf_std,
            "upper_shelf_std": self.upper_shelf_std,
            "noise_std": self.noise_std,
        }


MIN_WIDTH = 2.0
"""Narrowest transition width [C] the search will consider.

A Charpy transition spanning less than a couple of degrees is not physical; it is
the signature of the fit chasing a single noisy specimen.
"""


@dataclass
class _BLRSolution:
    """Posterior mean, posterior covariance and log marginal likelihood."""

    coef: np.ndarray
    posterior_covariance: np.ndarray
    log_evidence: float
    alpha: float
    beta: float


def _blr(X: np.ndarray, y: np.ndarray, iterations: int = 200, tol: float = 1e-9):
    """Bayesian linear regression on an explicit design, with the evidence.

    Closed-form Gaussian posterior over the weights under an isotropic Gaussian
    prior, with the prior precision ``alpha`` and the noise precision ``beta``
    set by Type-II maximum likelihood — the standard evidence-maximisation
    iteration. The constant column is expected to be in ``X`` already, so no
    intercept is fitted separately.

    This replaces the engine library's `BayesianLinearRegression`, which is what
    this module used before the library was dropped. The algorithm is the same,
    so the numbers should match to the tolerance of the stopping rule, but they
    are not guaranteed identical: the calibration figures in the README were
    re-measured against this implementation rather than carried over.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = X.shape

    xtx = X.T @ X
    xty = X.T @ y
    eigenvalues = np.linalg.eigvalsh(xtx)

    alpha, beta = 1.0, 1.0 / max(np.var(y), 1e-12)
    coef = np.zeros(m)
    covariance = np.eye(m)

    for _ in range(iterations):
        precision = alpha * np.eye(m) + beta * xtx
        covariance = np.linalg.inv(precision)
        coef = beta * covariance @ xty

        residual = y - X @ coef
        rss = float(residual @ residual)
        weight_norm = float(coef @ coef)

        # gamma: the number of parameters the data actually determines.
        gamma = float(np.sum(beta * eigenvalues / (alpha + beta * eigenvalues)))
        new_alpha = gamma / max(weight_norm, 1e-12)
        new_beta = max(n - gamma, 1e-12) / max(rss, 1e-12)

        if abs(new_alpha - alpha) < tol * alpha and abs(new_beta - beta) < tol * beta:
            alpha, beta = new_alpha, new_beta
            break
        alpha, beta = new_alpha, new_beta

    precision = alpha * np.eye(m) + beta * xtx
    covariance = np.linalg.inv(precision)
    coef = beta * covariance @ xty
    residual = y - X @ coef

    log_evidence = 0.5 * (
        m * np.log(alpha)
        + n * np.log(beta)
        - beta * float(residual @ residual)
        - alpha * float(coef @ coef)
        - np.linalg.slogdet(precision)[1]
        - n * np.log(2 * np.pi)
    )

    return _BLRSolution(
        coef=coef,
        posterior_covariance=covariance,
        log_evidence=float(log_evidence),
        alpha=float(alpha),
        beta=float(beta),
    )


def estimate_noise_std(fitted: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Estimate ``sigma(T) = base + relative * Y(T)`` from residual magnitudes.

    Regresses ``|residual| * sqrt(pi/2)`` (the unbiased scale estimate for a
    half-normal) on the fitted energy. Falls back to a constant when the slope
    comes out negative, which happens on small or genuinely homoscedastic data.
    """
    scale = np.abs(residual) * np.sqrt(np.pi / 2.0)
    A = np.column_stack([np.ones_like(fitted), fitted])
    coef, *_ = np.linalg.lstsq(A, scale, rcond=None)
    sigma = A @ coef
    if coef[1] < 0 or np.any(sigma <= 0):
        sigma = np.full_like(fitted, max(scale.mean(), 1e-6))
    return np.clip(sigma, max(1e-3, 0.05 * scale.mean()), None)


def fit_shelves(
    temperature: np.ndarray,
    energy: np.ndarray,
    dbtt: float,
    width_lower: float,
    width_upper: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> ShelfPosterior:
    """Bayesian linear regression for ``(a, b)`` at fixed shape parameters.

    Uses :func:`_blr`, whose Type-II marginal likelihood both fits the noise
    precision and scores the shape.

    Parameters
    ----------
    weights:
        Per-observation ``1 / sigma_i`` used to whiten the rows. These must be
        held **fixed** across a shape search: re-estimating them per shape makes
        the evidences incomparable and the outer optimisation degenerates. The
        reported evidence carries the ``sum(log w)`` Jacobian term, so it stays a
        density over the original energies and remains comparable across shapes.
    """
    y = np.asarray(energy, dtype=float)
    h = tanh_basis(temperature, dbtt, width_lower, width_upper)
    design = np.hstack([np.ones_like(h), h])

    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)

    solution = _blr(design * w[:, None], y * w)

    coef = solution.coef
    a, b = float(coef[0]), float(coef[1])

    # The shelves are LS = a - b and US = a + b, so their variances follow from
    # the (a, b) posterior covariance by the usual linear-combination rule.
    cov = solution.posterior_covariance
    var_ls = cov[0, 0] - 2 * cov[0, 1] + cov[1, 1]
    var_us = cov[0, 0] + 2 * cov[0, 1] + cov[1, 1]

    residual = y - design @ coef
    log_evidence = solution.log_evidence
    log_evidence += float(np.sum(np.log(w)))  # change of variables y -> w*y

    return ShelfPosterior(
        lower_shelf=a - b,
        upper_shelf=a + b,
        lower_shelf_std=float(np.sqrt(max(var_ls, 0.0))),
        upper_shelf_std=float(np.sqrt(max(var_us, 0.0))),
        noise_std=float(np.sqrt(np.mean(residual**2))),
        log_evidence=log_evidence,
    )


@dataclass
class RecoveryResult:
    """Recovered AHT parameters and the evidence that selected them."""

    dbtt: float
    width_lower: float
    width_upper: Optional[float]
    shelves: ShelfPosterior
    log_evidence: float
    n_shape_params: int
    n_observations: int
    dbtt_interval: Optional[tuple] = None
    trace: dict = field(default_factory=dict)

    @property
    def symmetric(self) -> bool:
        return self.width_upper is None

    @property
    def bic(self) -> float:
        """BIC on the profile evidence, penalising the profiled shape parameters.

        The marginal likelihood already integrates out the shelves and the noise
        precision, but *not* the shape parameters, which are optimised. This adds
        the standard penalty for those so models of different shape dimension can
        be compared. Lower is better.
        """
        return self.n_shape_params * np.log(self.n_observations) - 2.0 * self.log_evidence

    def params(self) -> dict:
        out = {
            "lower_shelf": self.shelves.lower_shelf,
            "upper_shelf": self.shelves.upper_shelf,
            "dbtt": self.dbtt,
            "width_lower": self.width_lower,
        }
        if self.width_upper is not None:
            out["width_upper"] = self.width_upper
        return out

    def predict(self, temperature) -> np.ndarray:
        """Fitted curve at the given temperature(s)."""
        h = tanh_basis(temperature, self.dbtt, self.width_lower, self.width_upper)
        a = 0.5 * (self.shelves.lower_shelf + self.shelves.upper_shelf)
        b = 0.5 * (self.shelves.upper_shelf - self.shelves.lower_shelf)
        return (a + b * h).reshape(-1)


def _neg_log_evidence(theta: np.ndarray, t, y, symmetric: bool, weights=None) -> float:
    dbtt = theta[0]
    widths = np.exp(theta[1:])
    if np.any(widths < MIN_WIDTH) or np.any(widths > 1e4):
        return np.inf
    try:
        wl = float(widths[0])
        wu = None if symmetric else float(widths[1])
        return -fit_shelves(t, y, dbtt, wl, wu, weights=weights).log_evidence
    except Exception:
        return np.inf


def _search_shape(t, y, symmetric, weights, dbtt_grid, width_grid):
    """Coarse grid then Nelder-Mead over the shape parameters, at fixed weights."""
    best = (np.inf, None)
    for d0 in dbtt_grid:
        for w0 in width_grid:
            theta = [d0, np.log(w0)] + ([np.log(w0)] if not symmetric else [])
            value = _neg_log_evidence(np.array(theta), t, y, symmetric, weights)
            if value < best[0]:
                best = (value, theta)
    if best[1] is None:
        raise RuntimeError("No feasible shape found; check the data and MIN_WIDTH.")
    return minimize(
        _neg_log_evidence,
        np.array(best[1]),
        args=(t, y, symmetric, weights),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000},
    )


def recover(
    temperature: Sequence[float],
    energy: Sequence[float],
    symmetric: bool = True,
    dbtt_grid: Optional[Sequence[float]] = None,
    width_grid: Optional[Sequence[float]] = None,
    profile_dbtt: bool = True,
    heteroscedastic: bool = True,
) -> RecoveryResult:
    """Recover ``(LS, US, DBTT, C)`` by maximising the marginal likelihood.

    A coarse grid over the shape parameters initialises a Nelder-Mead refinement,
    because the profile evidence is multimodal in the DBTT when the data does not
    bracket the transition.

    Parameters
    ----------
    symmetric:
        Fit one width (``C``) rather than separate widths either side.
    profile_dbtt:
        Also compute an approximate 95% profile-likelihood interval for the DBTT.
    heteroscedastic:
        Run a second pass with the rows whitened by a ``sigma(T)`` estimated from
        the first pass's residuals. Charpy scatter grows with absorbed energy, so
        a single homoscedastic pass biases the width downwards. The weights are
        estimated once and then held fixed, which keeps the evidences comparable
        across shapes.
    """
    t = np.asarray(temperature, dtype=float)
    y = np.asarray(energy, dtype=float)

    if dbtt_grid is None:
        dbtt_grid = np.linspace(t.min(), t.max(), 25)
    if width_grid is None:
        span = max(t.max() - t.min(), 1.0)
        width_grid = np.geomspace(max(MIN_WIDTH, span / 40), span, 12)

    opt = _search_shape(t, y, symmetric, None, dbtt_grid, width_grid)
    weights = None

    if heteroscedastic and len(t) >= 6:
        widths = np.exp(opt.x[1:])
        first = fit_shelves(
            t, y, float(opt.x[0]), float(widths[0]),
            None if symmetric else float(widths[1]),
        )
        fitted = first.lower_shelf + 0.5 * (first.upper_shelf - first.lower_shelf) * (
            1.0 + tanh_basis(t, float(opt.x[0]), float(widths[0]),
                             None if symmetric else float(widths[1])).reshape(-1)
        )
        weights = 1.0 / estimate_noise_std(fitted, y - fitted)
        opt = _search_shape(t, y, symmetric, weights, dbtt_grid, width_grid)

    dbtt = float(opt.x[0])
    widths = np.exp(opt.x[1:])
    width_lower = float(widths[0])
    width_upper = None if symmetric else float(widths[1])
    shelves = fit_shelves(t, y, dbtt, width_lower, width_upper, weights=weights)

    interval = None
    if profile_dbtt:
        interval = _profile_interval(t, y, dbtt, opt.x, symmetric, -opt.fun, weights)

    return RecoveryResult(
        dbtt=dbtt,
        width_lower=width_lower,
        width_upper=width_upper,
        shelves=shelves,
        log_evidence=-float(opt.fun),
        n_shape_params=len(opt.x),
        n_observations=len(t),
        dbtt_interval=interval,
        trace={"converged": bool(opt.success), "n_iter": int(opt.nit),
               "heteroscedastic": weights is not None,
               # Kept so `dbtt_posterior` can reuse the *same* fixed weights:
               # re-estimating them per grid point would make the evidences
               # incomparable, which is the failure this module already guards
               # against in the shape search.
               "weights": None if weights is None else np.asarray(weights)},
    )


def _profile_interval(t, y, dbtt_hat, theta_hat, symmetric, best_log_evidence, weights=None):
    """Approximate 95% profile-likelihood interval for the DBTT.

    Walks the DBTT away from its optimum, re-optimising the widths at each step,
    until the profile log evidence drops by ``chi2(0.95, 1) / 2``.
    """
    threshold = chi2.ppf(0.95, df=1) / 2.0
    span = max(t.max() - t.min(), 1.0)
    bounds = []

    for direction in (-1, 1):
        edge = None
        for step in np.linspace(0, span, 120)[1:]:
            candidate = dbtt_hat + direction * step

            def _obj(log_widths):
                theta = np.concatenate([[candidate], log_widths])
                return _neg_log_evidence(theta, t, y, symmetric, weights)

            res = minimize(_obj, theta_hat[1:], method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 800})
            if best_log_evidence - (-res.fun) > threshold:
                edge = candidate
                break
        bounds.append(edge)

    return tuple(bounds)


def compare_width_models(
    temperature: Sequence[float], energy: Sequence[float]
) -> dict:
    """Is ``C`` really a function of temperature, or just a constant?

    Fits the symmetric (constant ``C``) and asymmetric (``C_lower``,
    ``C_upper``) models and compares them three ways: profile log evidence
    (higher is better, but blind to the extra profiled parameter), BIC (lower is
    better, penalising it), and leave-one-out prediction error.
    """
    constant = recover(temperature, energy, symmetric=True, profile_dbtt=False)
    asymmetric = recover(temperature, energy, symmetric=False, profile_dbtt=False)

    return {
        "constant": constant,
        "asymmetric": asymmetric,
        "delta_log_evidence": asymmetric.log_evidence - constant.log_evidence,
        "delta_bic": asymmetric.bic - constant.bic,
        "loo_rmse": {
            "constant": loo_rmse(temperature, energy, symmetric=True),
            "asymmetric": loo_rmse(temperature, energy, symmetric=False),
        },
    }


def loo_rmse(temperature: Sequence[float], energy: Sequence[float], symmetric: bool) -> float:
    """Leave-one-out prediction RMSE [J] — an honest out-of-sample comparison."""
    t = np.asarray(temperature, dtype=float)
    y = np.asarray(energy, dtype=float)
    errors = []
    for i in range(len(t)):
        mask = np.ones(len(t), dtype=bool)
        mask[i] = False
        fit = recover(t[mask], y[mask], symmetric=symmetric, profile_dbtt=False)
        errors.append(y[i] - fit.predict([t[i]])[0])
    return float(np.sqrt(np.mean(np.square(errors))))


# ---------------------------------------------------------------------------
# the DBTT posterior
# ---------------------------------------------------------------------------
@dataclass
class DBTTPosterior:
    """``p(DBTT | data)`` on a grid, with summaries.

    The shelves and the noise precision are marginalised analytically by
    :func:`fit_shelves`, and the transition width is marginalised numerically
    here, so this is the full marginal posterior over the DBTT rather than a
    profile or a curvature approximation.
    """

    grid: np.ndarray
    """Temperatures the density is evaluated on [C]."""

    density: np.ndarray
    """Posterior density at each grid point, normalised to integrate to 1."""

    mean: float
    """Posterior mean [C]."""

    std: float
    """Posterior standard deviation [C]."""

    mode: float
    """Grid point of highest posterior density [C]."""

    interval: tuple
    """Equal-tailed credible interval [C]."""

    credible_mass: float
    """Mass the interval carries, e.g. ``0.95``."""

    width_grid: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Transition widths marginalised over [C]."""

    log_evidence: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Log evidence on the ``(DBTT, width)`` grid, before normalising."""

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        low, high = self.interval
        return (
            f"DBTTPosterior(mean={self.mean:.2f}, std={self.std:.2f}, "
            f"{self.credible_mass:.0%} CI=[{low:.2f}, {high:.2f}])"
        )


def dbtt_posterior(
    temperature: Sequence[float],
    energy: Sequence[float],
    result: Optional[RecoveryResult] = None,
    dbtt_grid: Optional[Sequence[float]] = None,
    width_grid: Optional[Sequence[float]] = None,
    credible_mass: float = 0.95,
    heteroscedastic: bool = True,
) -> DBTTPosterior:
    """The posterior over the DBTT, by marginalising the evidence over the width.

    Conditional on ``(DBTT, C)`` the shelves and the noise precision are
    integrated out in closed form, so the marginal likelihood is a function of
    two parameters only. Evaluating it on a grid and normalising gives the joint
    posterior; summing over the width gives ``p(DBTT | data)``.

    **Priors.** Flat in the DBTT over ``dbtt_grid``, and flat in ``log C`` over
    ``width_grid`` — the scale-invariant choice for a width. Both are proper
    only because the grids are bounded, so a grid that clips the posterior will
    distort it; :attr:`DBTTPosterior.density` at the grid edges is worth a look.

    Only the symmetric model is supported: with two widths the grid becomes
    three-dimensional, and `compare_width_models` exists to decide whether the
    second width is warranted at all.

    Parameters
    ----------
    result:
        A previous :func:`recover` result, reused for the fitted noise weights
        and to centre the default grids. Computed if not given.
    """
    t = np.asarray(temperature, dtype=float).reshape(-1)
    y = np.asarray(energy, dtype=float).reshape(-1)

    if result is None:
        result = recover(
            t, y, symmetric=True, profile_dbtt=False, heteroscedastic=heteroscedastic
        )
    weights = result.trace.get("weights")

    if dbtt_grid is None:
        # Wide enough to contain the posterior for the campaigns this is used
        # on, and clipped to the tested range, beyond which there is no data to
        # locate a transition with.
        half = max(60.0, 4.0 * result.width_lower)
        dbtt_grid = np.linspace(result.dbtt - half, result.dbtt + half, 121)
    if width_grid is None:
        width_grid = np.exp(
            np.linspace(
                np.log(max(result.width_lower / 8.0, MIN_WIDTH)),
                np.log(result.width_lower * 8.0),
                31,
            )
        )

    dbtt_grid = np.asarray(dbtt_grid, dtype=float)
    width_grid = np.asarray(width_grid, dtype=float)

    log_evidence = np.full((dbtt_grid.size, width_grid.size), -np.inf)
    for i, dbtt in enumerate(dbtt_grid):
        for j, width in enumerate(width_grid):
            log_evidence[i, j] = -_neg_log_evidence(
                np.array([dbtt, np.log(width)]), t, y, True, weights
            )

    # Normalise in a shifted exponential, so the largest term is exp(0).
    finite = np.isfinite(log_evidence)
    if not finite.any():
        raise RuntimeError("The evidence was infinite everywhere on the grid.")
    joint = np.where(finite, np.exp(log_evidence - log_evidence[finite].max()), 0.0)

    # Flat in log C: integrate over log-width, which is what the grid is uniform
    # in when `width_grid` is the default geometric one.
    log_widths = np.log(width_grid)
    density = (
        np.trapezoid(joint, log_widths, axis=1)
        if log_widths.size > 1
        else joint[:, 0]
    )

    area = np.trapezoid(density, dbtt_grid)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("The DBTT posterior did not normalise.")
    density = density / area

    mean = float(np.trapezoid(density * dbtt_grid, dbtt_grid))
    variance = float(np.trapezoid(density * (dbtt_grid - mean) ** 2, dbtt_grid))
    mode = float(dbtt_grid[int(np.argmax(density))])

    # Equal-tailed interval from the CDF.
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(dbtt_grid) * (density[:-1] + density[1:]) / 2)])
    cdf = cdf / cdf[-1]
    tail = (1.0 - credible_mass) / 2.0
    low = float(np.interp(tail, cdf, dbtt_grid))
    high = float(np.interp(1.0 - tail, cdf, dbtt_grid))

    return DBTTPosterior(
        grid=dbtt_grid,
        density=density,
        mean=mean,
        std=float(np.sqrt(max(variance, 0.0))),
        mode=mode,
        interval=(low, high),
        credible_mass=credible_mass,
        width_grid=width_grid,
        log_evidence=log_evidence,
    )


# ---------------------------------------------------------------------------
# reading the DBTT off a fitted model's curve
# ---------------------------------------------------------------------------
@dataclass
class CurveDBTT:
    """The DBTT read off a fitted model's posterior curve."""

    dbtt: float
    """Temperature where the curve crosses the mid-shelf energy [C]."""

    std: Optional[float]
    """First-order uncertainty [C], or ``None`` without a model uncertainty."""

    level: float
    """The mid-shelf energy the crossing is taken at [J]."""

    gradient: float
    """Curve slope at the crossing [J/C]; the delta method divides by it."""

    lower_shelf: float
    upper_shelf: float
    n_crossings: int
    """How many times the curve crosses the level. More than one means the fit
    is not monotonic through the transition, and the steepest was taken."""

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        spread = "" if self.std is None else f" +/- {self.std:.2f}"
        return f"CurveDBTT(dbtt={self.dbtt:.2f}{spread})"


def dbtt_from_curve(
    temperature: Sequence[float],
    mean: Sequence[float],
    sd: Optional[Sequence[float]] = None,
) -> CurveDBTT:
    """Read the DBTT off a fitted model's posterior, with its uncertainty.

    Where :func:`dbtt_posterior` fits the AHT to the *measurements*, this works
    on a model's own posterior curve — the mean and standard deviation a GP
    predicts across a temperature slice, as the Engine's prediction workflow
    returns them. The DBTT is the crossing of the mid-shelf energy,
    ``(LS + US) / 2``, with the shelves taken as the curve's own extremes.

    The uncertainty is propagated to first order: a vertical uncertainty
    ``sd(T*)`` at the crossing maps to a horizontal one through the slope,

        sd_DBTT  ~=  sd(T*) / |dY/dT|(T*)

    which is the delta method. It is a linearisation, so it is only as good as
    the curve being locally straight over ``+/- sd``; on a shallow transition,
    or where the slice is too coarse to resolve the gradient, it will be
    optimistic. It also ignores the correlation between neighbouring
    temperatures, which a draw-based estimate would capture — the Engine
    exposes no posterior draws, so that is not available here.

    Prefer :func:`dbtt_posterior` when the measurements are to hand; this is for
    when what you have is a fitted model.
    """
    t = np.asarray(temperature, dtype=float).reshape(-1)
    y = np.asarray(mean, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError(f"temperature and mean differ in length: {t.size} vs {y.size}")
    if t.size < 3:
        raise ValueError("need at least three points to locate a crossing")

    s = None
    if sd is not None:
        s = np.asarray(sd, dtype=float).reshape(-1)
        # Validate before reordering: indexing a short `sd` with the full
        # ordering raises IndexError rather than saying what is wrong.
        if s.size != t.size:
            raise ValueError(f"sd differs in length: {s.size} vs {t.size}")

    order = np.argsort(t)
    t, y = t[order], y[order]
    if s is not None:
        s = s[order]

    lower_shelf, upper_shelf = float(y.min()), float(y.max())
    level = 0.5 * (lower_shelf + upper_shelf)

    above = y >= level
    crossings = np.flatnonzero(np.diff(above.astype(int)) != 0)
    if crossings.size == 0:
        raise ValueError(
            "the curve never crosses its own mid-shelf energy; it is probably "
            "not a transition over this range"
        )

    # Linear interpolation within each crossing interval, then keep the
    # steepest: that is the transition, and any others are GP wiggle.
    def interpolate(i: int) -> tuple:
        t0, t1, y0, y1 = t[i], t[i + 1], y[i], y[i + 1]
        slope = (y1 - y0) / (t1 - t0)
        return (t0 + (level - y0) / slope if slope != 0 else t0), slope

    candidates = [interpolate(int(i)) for i in crossings]
    dbtt, gradient = max(candidates, key=lambda c: abs(c[1]))

    std = None
    if s is not None:
        sd_at_crossing = float(np.interp(dbtt, t, s))
        std = (
            float(sd_at_crossing / abs(gradient))
            if gradient != 0 and np.isfinite(gradient)
            else None
        )

    return CurveDBTT(
        dbtt=float(dbtt),
        std=std,
        level=level,
        gradient=float(gradient),
        lower_shelf=lower_shelf,
        upper_shelf=upper_shelf,
        n_crossings=int(crossings.size),
    )
