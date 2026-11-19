"""Independent finite-dimensional checks; these do not prove asymptotic limits."""
import numpy as np
import pytest
from scipy.optimize import brentq

from src.cdelta import _huber_location_influence, _gaussian_kde_density
from src.rho_profile import infer

CLIP, MAD_FACTOR = 1.345, 1.4826


def _exp_cdf(x):
    return -np.expm1(-max(x, 0.))


def _exp_score(t, s):
    """Exact integral of a clipped score against the unit exponential law."""
    lo, hi = max(0., t-CLIP*s), max(0., t+CLIP*s)
    mass = np.exp(-lo)-np.exp(-hi)
    first = (lo+1)*np.exp(-lo)-(hi+1)*np.exp(-hi)
    return -CLIP*_exp_cdf(t-CLIP*s)+(first-t*mass)/s+CLIP*np.exp(-hi)


@pytest.mark.parametrize("point", [.1, .5, .9, 3.])
def test_nuisance_chain_matches_contamination_of_exact_continuous_law(point):
    # Nonzero asymmetric MAD and Huber scale derivatives are essential here.
    m, d = np.log(2), np.arcsinh(.5)
    s = MAD_FACTOR*d
    t = brentq(lambda q: _exp_score(q, s), 0, 2, xtol=1e-14)
    lo, hi = max(0., t-CLIP*s), t+CLIP*s
    a = np.exp(-lo)-np.exp(-hi)
    b = ((lo+1)*np.exp(-lo)-(hi+1)*np.exp(-hi)-t*a)/s
    f0, fp, fm = np.exp(-np.array([m, m+d, m-d]))
    jacobian = np.array([[f0, 0, 0], [fp-fm, fp+fm, 0],
                         [0, -b/d, -a/s]])
    score = np.array([float(point <= m)-.5, float(abs(point-m) <= d)-.5,
                      np.clip((point-t)/s, -CLIP, CLIP)])
    expected = -np.linalg.solve(jacobian, score)
    assert abs(b) > .05
    assert abs(fp-fm) > .1
    errors = []
    for eps in (1e-4, 1e-5, 1e-6):
        cdf = lambda x: (1-eps)*_exp_cdf(x)+eps*float(point <= x)
        me = brentq(lambda x:cdf(x)-.5, .3, 1.2, xtol=1e-14)
        de = brentq(lambda q:(1-eps)*(_exp_cdf(me+q)-_exp_cdf(me-q))
                    +eps*float(abs(point-me) <= q)-.5, .2, .65, xtol=1e-14)
        se = MAD_FACTOR*de
        te = brentq(lambda q:(1-eps)*_exp_score(q, se)
                    +eps*np.clip((point-q)/se, -CLIP, CLIP), 0, 2, xtol=1e-14)
        numerical = (np.array([me, de, te])-np.array([m, d, t]))/eps
        errors.append(np.max(np.abs(numerical-expected)))
    assert errors[2] < errors[1] < errors[0]
    np.testing.assert_allclose(numerical, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("n", [127, 128])
def test_implemented_reference_influence_solves_raw_mad_score_system(n):
    w = np.random.default_rng(n).exponential(size=n)
    t, s, actual = _huber_location_influence(
        w, huber_c=CLIP, density_method="kde", analytic_density=None,
        density_folds=5, density_seed=0, reference_solver="score_checked")
    m = np.median(w); d = np.median(abs(w-m))
    # Independent scalar-density formula, evaluated at each fitted boundary.
    h = 1.06*np.std(w, ddof=1)*n**(-.2)
    density = lambda q:np.mean(np.exp(-.5*((w-q)/h)**2))/(np.sqrt(2*np.pi)*h)
    f0, fp, fm = density(m), density(m+d), density(m-d)
    u = (w-t)/s; active = abs(u) < CLIP
    a, b = active.mean(), np.mean(u*active)
    jacobian = np.array([[f0, 0, 0], [fp-fm, fp+fm, 0], [0, -b/d, -a/s]])
    scores = np.vstack(((w <= m).astype(float)-.5,
                        (abs(w-m) <= d).astype(float)-.5, np.clip(u, -CLIP, CLIP)))
    expected = -np.linalg.solve(jacobian, scores)
    np.testing.assert_allclose(actual, expected[2], atol=2e-14, rtol=2e-14)
    for q in (m, m+d, m-d):
        assert _gaussian_kde_density(w, q) == pytest.approx(density(q), rel=1e-14)


@pytest.mark.parametrize("n", [127, 128])
def test_complete_if_equals_reduced_part_plus_correlation_weighted_quadratics(n):
    rng = np.random.default_rng(10000+n)
    x = rng.standard_t(3, n)
    y = .35*x+rng.standard_t(3, n)
    result = infer(x, y)
    references = []
    for w in (x, y):
        references.append(_huber_location_influence(
            w, huber_c=CLIP, density_method="kde", analytic_density=None,
            density_folds=5, density_seed=0, reference_solver="score_checked"))
    (tx, _, ux), (ty, _, uy) = references
    a, b = abs(x-tx), abs(y-ty)
    ma, mb, nu = a.mean(), b.mean(), np.mean(a*b)
    qa, qb = np.mean(a*a), np.mean(b*b)
    va, vb = qa-ma*ma, qb-mb*mb
    denominator = np.sqrt(va*vb)
    sa, sb = np.sign(x-tx), np.sign(y-ty)
    if_ma = a-ma-sa.mean()*ux
    if_mb = b-mb-sb.mean()*uy
    if_nu = a*b-nu-np.mean(sa*b)*ux-np.mean(sb*a)*uy
    if_qa = a*a-qa-2*np.mean(x-tx)*ux
    if_qb = b*b-qb-2*np.mean(y-ty)*uy
    reduced = (if_nu-mb*if_ma-ma*if_mb)/denominator
    quadratic = ma/va*if_ma+mb/vb*if_mb-if_qa/(2*va)-if_qb/(2*vb)
    rho = (nu-ma*mb)/denominator
    full = reduced+rho*quadratic
    np.testing.assert_allclose(result["influence"], full-full.mean(), atol=2e-13, rtol=2e-13)
    assert result["standard_error"] == pytest.approx(np.std(full, ddof=1)/np.sqrt(n))
    # A paired cross-term bound requires the rho multiplier, even if raw
    # fourth moments are large. No asymptotic conclusion is drawn from this.
    lhs = abs(rho*np.mean((reduced-reduced.mean())*(quadratic-quadratic.mean())))
    rhs = np.std(reduced)*abs(rho)*np.std(quadratic)
    assert lhs <= rhs+1e-13


def test_mad_about_estimated_center_obeys_order_statistic_lipschitz_bound():
    # Includes midpoint ties, odd/even samples and remote atoms.
    for w in (np.array([-2., -1., 1., 2.]), np.array([-2., 0., 2.]),
              np.array([-100., -100., -.3, .1, .6, 100., 100.])):
        for old, new in ((0., .2), (-.7, .9), (.3, .3)):
            assert abs(np.median(abs(w-new))-np.median(abs(w-old))) <= abs(new-old)+1e-14
