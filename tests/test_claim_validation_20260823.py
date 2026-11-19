import csv

import numpy as np

from scripts.run_claim_validation_20260823 import (
    PROJECT_ROOT,
    fixed_center_profile_test,
)


def _read(name: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / "results" / name).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_fixed_center_profile_uses_only_independent_radii():
    rng = np.random.default_rng(2026082302)
    signs = rng.choice((-1.0, 1.0), 500)
    radius_x = np.exp(0.2 * rng.normal(size=500))
    radius_y = np.exp(0.2 * rng.normal(size=500))
    result = fixed_center_profile_test(signs * radius_x, signs * radius_y)
    assert np.isclose(result["estimate"], np.corrcoef(radius_x, radius_y)[0, 1])


def test_claim1_regular_confirmation_and_skew_boundary_are_frozen():
    rows = _read("claim1_wald_validation_20260823.tsv")
    assert len(rows) == 6
    regular_dependent = [row for row in rows if row["design_role"] == "regular_dependent_weak_null"]
    assert all(
        float(row["wilson_95_low"]) <= 0.05 <= float(row["wilson_95_high"])
        for row in regular_dependent
    )
    skew = sorted(
        (row for row in rows if row["design_role"] == "regular_slow_convergence_stress"),
        key=lambda row: int(row["n"]),
    )
    assert float(skew[0]["rejection_rate"]) > 0.10
    assert float(skew[1]["rejection_rate"]) < float(skew[0]["rejection_rate"])
