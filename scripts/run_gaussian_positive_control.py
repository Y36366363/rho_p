"""Reproducible nonzero Gaussian control with an exact population target.

All six cells are reported. This symmetric control has zero population
reference coefficients; it isolates performance away from skew margins.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rho_profile import infer


def gaussian_target(r):
    if not -1 <= r <= 1:
        raise ValueError("correlation must be in [-1, 1]")
    return float((np.sqrt(1 - r*r) + r*np.arcsin(r) - 1) / (np.pi/2 - 1))


def run(reps=1000, seed=2026111801):
    rows = []
    for r in (.5, .8):
        for n in (160, 640, 2560):
            truth = gaussian_target(r)
            for rep in range(reps):
                rng = np.random.default_rng(np.random.SeedSequence([seed, round(100*r), n, rep]))
                z = rng.normal(size=(n, 2))
                base = dict(r=r, n=n, replication=rep, root_seed=seed, truth=truth)
                try:
                    fit = infer(z[:, 0], r*z[:, 0] + np.sqrt(1-r*r)*z[:, 1])
                    low, high = fit["normal"]["lower"], fit["normal"]["upper"]
                    rows.append(dict(**base, estimate=fit["estimate"], se=fit["standard_error"],
                        covered=int(low <= truth <= high), below=int(high < truth), above=int(low > truth),
                        residual_max=max(abs(fit["solver_diagnostics"][w]["score_exact_float"]) for w in ("x", "y")), error=""))
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    rows.append(dict(**base, estimate="", se="", covered="", below="", above="", residual_max="", error=str(exc)))
            print(f"completed r={r}, n={n}, scheduled={reps}", flush=True)
    return rows


def summary(rows):
    result = []
    for r in (.5, .8):
        for n in (160, 640, 2560):
            cell = [row for row in rows if float(row["r"]) == r and int(row["n"]) == n]
            valid = [row for row in cell if not row["error"]]
            if not valid:
                raise ValueError("cell has no valid fits")
            estimate = np.array([float(row["estimate"]) for row in valid])
            se = np.array([float(row["se"]) for row in valid])
            p = np.mean([int(row["covered"]) for row in valid]); m = len(valid)
            z = 1.959963984540054; den = 1 + z*z/m
            center = (p + z*z/(2*m))/den
            half = z*np.sqrt(p*(1-p)/m + z*z/(4*m*m))/den
            result.append(dict(r=r, n=n, truth=gaussian_target(r), scheduled=len(cell), valid=m,
                failures=len(cell)-m, coverage=float(p), mcse=float(np.sqrt(p*(1-p)/m)),
                wilson_low=float(center-half), wilson_high=float(center+half),
                bias=float(estimate.mean()-gaussian_target(r)), sd=float(estimate.std(ddof=1)),
                mean_se=float(se.mean()), se_over_sd=float(se.mean()/estimate.std(ddof=1)),
                below=float(np.mean([int(row["below"]) for row in valid])),
                above=float(np.mean([int(row["above"]) for row in valid]))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026111801)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = run(args.reps, args.seed)
    for name, data in (("replications", rows), ("summary", summary(rows))):
        with (args.output_dir / f"{name}.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(data)
    manifest = dict(seed=args.seed, replications_per_cell=args.reps,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        solver="score_checked", generator="PCG64, SeedSequence([seed, round(100*r), n, replication]); n-by-2 normals",
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                      for name in ("src/cdelta.py", "src/rho_profile.py", "scripts/run_gaussian_positive_control.py")})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
