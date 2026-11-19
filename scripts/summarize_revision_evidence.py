"""Recompute mechanism and interval diagnostics from the replication records.

No reference fitting or new simulation is performed."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rho_profile import fisher_interval
from scripts.freeze_canonical_evidence_20260819 import normalized_text_sha256

SOURCES = ["reference_pathway_replications_20260909.tsv",
           "active_nuisance_replications_20260907.tsv",
           "moderate_benchmark_replications_20260920.tsv"]


def read(name):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def unique(rows, keys):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in index:
            raise ValueError(f"duplicate record: {key}")
        if row.get("error"):
            raise ValueError(f"failed archived record: {key}")
        index[key] = row
    return index


def summarize():
    records = read(SOURCES[0])
    index = unique(records, ["tau", "n", "replication", "stage"])
    if len(index) != 32000:
        raise ValueError("expected all 32000 pathway records")
    mechanism = []
    for tau in (.1, .4):
        for n in (80, 640):
            cells = {stage: sorted([r for r in records if float(r["tau"]) == tau
                      and int(r["n"]) == n and r["stage"] == stage],
                      key=lambda r: int(r["replication"]))
                     for stage in ("population_scale", "population_median", "full_fit")}
            for rows in cells.values():
                if [int(r["replication"]) for r in rows] != list(range(2000)):
                    raise ValueError("incomplete replication schedule")
            full = cells["full_fit"]
            arr = lambda field: np.array([float(r[field]) for r in full])
            gap = max(abs(float(a[f]) - float(b[f]))
                      for a, b in zip(cells["population_scale"], cells["population_median"])
                      for f in ("estimate", "se_stage_aware", "reference_x", "reference_y"))
            mechanism.append(dict(tau=tau, n=n, replications=len(full),
                mean_abs_median=float(np.mean([np.abs(arr("sample_median_x")), np.abs(arr("sample_median_y"))])),
                mean_scale=float(np.mean([arr("scale_x"), arr("scale_y")])),
                reference_correlation=float(np.corrcoef(arr("reference_x"), arr("reference_y"))[0, 1]),
                stage23_max_gap=gap,
                stage23_min_active_fraction=min(float(r[f]) for stage in ("population_scale", "population_median")
                    for r in cells[stage] for f in ("active_fraction_x", "active_fraction_y")),
                full_mean_active_fraction=float(np.mean([arr("active_fraction_x"), arr("active_fraction_y")]))))
    intervals = []
    for name, log_sd in zip(SOURCES[1:], (.6, .3)):
        records = read(name)
        unique(records, ["n", "replication", "method"])
        for n in (160, 640, 2560):
            rows = sorted([r for r in records if int(r["n"]) == n and r["method"] == "complete_kde"],
                          key=lambda r: int(r["replication"]))
            if [int(r["replication"]) for r in rows] != list(range(2000)):
                raise ValueError("incomplete interval replication schedule")
            r, se, truth = (np.array([float(row[key]) for row in rows]) for key in ("estimate", "se", "truth"))
            zcrit = 1.959963984540054
            covered = (r - zcrit * se <= truth) & (truth <= r + zcrit * se)
            if not np.array_equal(covered, [int(row["covered"]) for row in rows]):
                raise ValueError("stored coverage flags disagree with intervals")
            fisher = [fisher_interval(a, b) for a, b in zip(r, se)]
            fcovered = np.array([ci["lower"] <= t <= ci["upper"] for ci, t in zip(fisher, truth)])
            difference = fcovered.astype(float) - covered.astype(float)
            intervals.append(dict(log_sd=log_sd, n=n, replications=len(rows),
                wald_coverage=float(covered.mean()), fisher_coverage=float(fcovered.mean()),
                paired_difference=float(difference.mean()),
                paired_difference_mcse=float(difference.std(ddof=1) / np.sqrt(len(rows))),
                below_truth=float(np.mean(r + zcrit * se < truth)),
                above_truth=float(np.mean(r - zcrit * se > truth)),
                mean_se_over_sd=float(se.mean() / r.std(ddof=1))))
    population = read("active_nuisance_population_20260907.tsv")
    full_variance = float(max(population, key=lambda row: int(row["order_per_interval"]))["full_if_variance"])
    records = read(SOURCES[1])
    population_se_coverage = []
    for n in (160, 640, 2560):
        rows = [row for row in records if int(row["n"]) == n and row["method"] == "complete_kde"]
        estimate = np.array([float(row["estimate"]) for row in rows])
        truth = float(rows[0]["truth"])
        half = 1.959963984540054 * np.sqrt(full_variance / n)
        population_se_coverage.append(dict(log_sd=.6, n=n,
            coverage=float(np.mean(np.abs(estimate - truth) <= half))))
    return dict(mechanism=mechanism, intervals=intervals,
                population_se_coverage=population_se_coverage,
                population_source_sha256=normalized_text_sha256(ROOT / "results" / "active_nuisance_population_20260907.tsv"),
                source_sha256={name: normalized_text_sha256(ROOT / "results" / name) for name in SOURCES},
                new_simulation_cells=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize()
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("choose a fresh output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output)
