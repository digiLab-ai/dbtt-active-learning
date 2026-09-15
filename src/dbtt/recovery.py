"""Recover the AHT parameters (LS, US, C, DBTT) from measured Charpy data.

The fit exploits a structural fact about the AHT. Write it as

    Y(T) = a + b * h(T),    h(T) = tanh((T - DBTT) / C(T))

with ``a = (LS + US) / 2`` and ``b = (US - LS) / 2``. Conditional on the *shape*
parameters ``(DBTT, C)``, the model is **linear** in ``(a, b)``. So the shelves
have a closed-form Gaussian posterior — supplied here by the engine's
`BayesianLinearRegression`, which also reports the log marginal likelihood — and
only the two (or three) nonlinear shape parameters need an outer search.

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
import torch
from scipy.optimize import minimize
from scipy.stats import chi2

from uncertainty_engine.models.linear.bayesian_linear import (
    BayesianLinearRegression,
    BayesianLinearRegressionConfig,
)


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


def _blr(X: np.ndarray, y: np.ndarray):
    """Fit BLR on an explicit design (constant column included)."""
    model = BayesianLinearRegression(
        BayesianLinearRegressionConfig(
            X=torch.as_tensor(X, dtype=torch.float64),
            y=torch.as_tensor(y, dtype=torch.float64),
            fit_intercept=False,
        )
    )
    model.fit()
    return model.config.solution


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

    Uses the engine's `BayesianLinearRegression`, whose Type-II marginal
    likelihood both fits the noise precision and scores the shape.

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

    diagnostics = solution.diagnostics
    coef = torch.as_tensor(solution.coef).reshape(-1).numpy()
    a, b = float(coef[0]), float(coef[1])

    # posterior_covariance is already the parameter posterior covariance: the
    # engine documents the predictive variance as x^T C x + 1/alpha.
    cov = torch.as_tensor(solution.posterior_covariance).reshape(2, 2).numpy()
    var_ls = cov[0, 0] - 2 * cov[0, 1] + cov[1, 1]
    var_us = cov[0, 0] + 2 * cov[0, 1] + cov[1, 1]

    residual = y - design @ coef
    log_evidence = float(np.asarray(torch.as_tensor(diagnostics.scores).cpu())[-1])
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
               "heteroscedastic": weights is not None},
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
