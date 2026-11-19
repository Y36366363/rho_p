"""Focused public interface for median/MAD/Huber profile correlation.

The numerical kernel stays in ``cdelta`` to preserve historical simulations.
This interface defaults to the residual-checked estimator. Existing archived
runners importing ``cdelta`` retain their original legacy solver and results.
"""
from __future__ import annotations

from statistics import NormalDist
import numpy as np

from .cdelta import huber_profile_correlation_inference as _inference

__all__ = ["infer", "fisher_interval", "bootstrap_t_interval"]


def infer(x, y, *, reference_solver="score_checked", **kwargs):
    """Estimate Corr(|X-T_X|, |Y-T_Y|) from at least 12 finite IID pairs.

    The default checks each represented-input Huber score against 1/(10^8 n)
    and returns solver diagnostics; solver failure raises. All density,
    confidence-level and studentizer options are forwarded to the kernel.
    ``reference_solver='legacy'`` is available for historical comparisons.
    Regular-law Wald inference is pointwise; it is not a coverage guarantee
    for a given sample size or a regularity diagnostic.
    """
    return _inference(x, y, reference_solver=reference_solver, **kwargs)


def fisher_interval(estimate, standard_error, *, alpha=0.05):
    """Delta-method Fisher transform using the supplied complete-IF SE.

    This is tanh(atanh(r) +/- z * SE/(1-r^2)), not the normal-model
    Pearson interval with SE=1/sqrt(n-3). Endpoint estimates are rejected.
    """
    if not np.isfinite(estimate) or not -1 < estimate < 1:
        raise ValueError("Fisher transform requires -1 < estimate < 1")
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("standard_error must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    center = np.arctanh(estimate)
    half_width = z * standard_error / (1 - estimate**2)
    return {"lower": float(np.tanh(center - half_width)),
            "upper": float(np.tanh(center + half_width))}


def bootstrap_t_interval(x, y, *, n_resamples=999, seed=20261118,
                         alpha=0.05, **kwargs):
    """Experimental paired bootstrap-t with the entire reference chain refit.

    The bootstrap pivot is (rho* - rho_hat)/SE*. Its empirical quantiles
    are inverted using the original complete-IF SE. One index vector
    resamples both margins. Every resample refits median, MAD, Huber roots,
    density values and paired IF. A failed resample aborts; none is replaced
    or silently discarded. This candidate requires coverage validation.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least 2")
    original = infer(x, y, alpha=alpha, **kwargs)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    pivots = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, len(x), size=len(x))
        try:
            fit = infer(x[indices], y[indices], alpha=alpha, **kwargs)
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError) as exc:
            raise RuntimeError(f"bootstrap resample {i} failed; no replacement") from exc
        pivots[i] = (fit["estimate"] - original["estimate"]) / fit["standard_error"]
        if not np.isfinite(pivots[i]):
            raise RuntimeError(f"bootstrap resample {i} produced a nonfinite pivot")
    low, high = np.quantile(pivots, [alpha / 2, 1 - alpha / 2], method="linear")
    return {"estimate": original["estimate"],
            "standard_error": original["standard_error"],
            "lower": float(original["estimate"] - high * original["standard_error"]),
            "upper": float(original["estimate"] - low * original["standard_error"]),
            "pivots": pivots, "n_resamples": n_resamples, "seed": seed,
            "quantile_method": "linear", "experimental": True}
