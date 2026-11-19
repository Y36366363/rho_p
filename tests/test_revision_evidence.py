"""Public archived-data checks, independent of any author manuscript."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.summarize_revision_evidence import summarize, unique
from scripts.run_gaussian_positive_control import gaussian_target
from scripts.reproduce_arxiv_tables import gaussian_calibration, read, SOURCES, tables

ROOT = Path(__file__).resolve().parents[1]


def assert_diagnostics_match(actual, expected):
    """Allow roundoff in continuous summaries; keep structure and counts exact."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_diagnostics_match(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_diagnostics_match(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-15)
    else:
        assert actual == expected


def test_archived_mechanism_and_coverage_recomputed_from_every_record():
    report = summarize()
    stored = json.loads((ROOT / "results/revision_diagnostics_20261118.json").read_text())
    assert_diagnostics_match(report, stored)
    assert report["new_simulation_cells"] == 0
    tight = [r for r in report["mechanism"] if r["tau"] == .1]
    assert all(r["stage23_min_active_fraction"] == 1 for r in tight)
    assert all(r["stage23_max_gap"] < 3e-15 for r in tight)
    assert all(r["mean_scale"] < .7 for r in tight)
    for row in report["intervals"]:
        assert row["wald_coverage"] + row["below_truth"] + row["above_truth"] == pytest.approx(1.)


def test_duplicate_and_failed_archival_records_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        unique([{"id": 1}, {"id": 1}], ["id"])
    with pytest.raises(ValueError, match="failed"):
        unique([{"id": 1, "error": "fit failed"}], ["id"])


def test_gaussian_truth_matches_independent_conditional_integration():
    from scipy.integrate import quad
    from scipy.stats import norm
    for r in (.5, .8):
        s = np.sqrt(1-r*r)
        def integrand(x):
            conditional_abs_mean = 2*s*norm.pdf(r*x/s) + r*x*(2*norm.cdf(r*x/s)-1)
            return 2*x*conditional_abs_mean*norm.pdf(x)
        cross = quad(integrand, 0, np.inf, epsabs=1e-12)[0]
        assert gaussian_target(r) == pytest.approx((cross-2/np.pi)/(1-2/np.pi), abs=1e-11)


def test_gaussian_control_reports_all_cells_and_pooled_shortfall():
    report = gaussian_calibration()
    assert report == json.loads((ROOT / "results/gaussian_calibration_20261118.json").read_text())
    assert report["covered"] == 5648
    assert report["valid"] == 6000
    assert report["coverage"] == pytest.approx(5648/6000)
    assert report["all_cells_below_nominal"]
    assert .935 < report["descriptive_wilson_low"] < report["descriptive_wilson_high"] < .948
    gaussian = [row for row in tables()["regular"] if row[0].startswith("Bivariate normal")]
    assert len(gaussian) == 6
    assert [row[1] for row in gaussian].count("160") == 2


def test_gaussian_calibration_rejects_changed_flags_and_missing_records():
    rows = read(SOURCES[4])
    with pytest.raises(ValueError, match="incomplete"):
        gaussian_calibration(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        gaussian_calibration(rows + rows[:1])
    rows[0] = {**rows[0], "covered": str(1-int(rows[0]["covered"]))}
    with pytest.raises(ValueError, match="coverage flags"):
        gaussian_calibration(rows)
