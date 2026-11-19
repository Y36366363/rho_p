"""Scientific and failure-mode checks for the focused inference interface."""
from fractions import Fraction
import numpy as np
import pytest
from src import rho_profile
from src.cdelta import _huber_exact_score


def test_default_enforces_score_on_previously_flagged_sample():
    rng = np.random.default_rng(np.random.SeedSequence([2026090901, 10, 640, 70]))
    signs = rng.choice((-1., 1.), size=640)
    x, y = signs * np.exp(.1*rng.normal(size=640)), signs * np.exp(.1*rng.normal(size=640))
    fit = rho_profile.infer(x, y)
    for name, w in (("x", x), ("y", y)):
        t, s = fit["reference_location_" + name], fit["reference_scale_" + name]
        assert abs(_huber_exact_score(w, t, s)) <= Fraction(1, 10**8 * len(w))
    a = np.abs(x - fit["reference_location_x"])
    b = np.abs(y - fit["reference_location_y"])
    assert fit["estimate"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-13)
    assert fit["standard_error"] == pytest.approx(np.std(fit["influence"], ddof=1) / np.sqrt(len(x)))


def test_default_solver_failure_propagates():
    with pytest.raises(RuntimeError, match="budget"):
        rho_profile.infer(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60)**2, solver_max_iterations=1)


def test_bootstrap_resamples_intact_pairs_and_inverts_studentized_quantiles(monkeypatch):
    calls = []
    estimates, errors = iter([.2, .1, .4, .25]), iter([.05, .1, .05, .025])
    def fake(x, y, **kwargs):
        np.testing.assert_array_equal(y, 3*x + 7)
        calls.append((x.copy(), kwargs))
        return {"estimate": next(estimates), "standard_error": next(errors)}
    monkeypatch.setattr(rho_profile, "infer", fake)
    x = np.arange(12.)
    result = rho_profile.bootstrap_t_interval(x, 3*x+7, n_resamples=3, seed=17)
    np.testing.assert_allclose(result["pivots"], [-1, 4, 2])
    assert result["lower"] == pytest.approx(.005)
    assert result["upper"] == pytest.approx(.2425)
    rng = np.random.default_rng(17)
    for actual, _ in calls[1:]:
        np.testing.assert_array_equal(actual, x[rng.integers(0, 12, 12)])


def test_bootstrap_failure_is_not_silently_dropped(monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("zero MAD")
        return {"estimate": .2, "standard_error": .1}
    monkeypatch.setattr(rho_profile, "infer", fake)
    with pytest.raises(RuntimeError, match="resample 0 failed; no replacement"):
        rho_profile.bootstrap_t_interval(np.arange(12.), np.arange(12.), n_resamples=3)
    assert len(calls) == 2


def test_fisher_interval_uses_full_if_delta_se():
    ci = rho_profile.fisher_interval(.4, .1)
    assert np.arctanh(ci["upper"]) - np.arctanh(.4) == pytest.approx(1.959963984540054*.1/.84)
    assert -1 < ci["lower"] < .4 < ci["upper"] < 1
    with pytest.raises(ValueError):
        rho_profile.fisher_interval(1., .1)
