"""Export the numerical table bodies of the paper from results/.

The output directory must not already exist; input files are only read.
See REPRODUCE.md for simulation reruns."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_revision_evidence import summarize
from scripts.freeze_canonical_evidence_20260819 import normalized_text_sha256
from scripts.run_gaussian_positive_control import summary as gaussian_summary

SOURCES = ["reference_pathway_summary_20260909.tsv", "claim1_wald_validation_20260823.tsv",
           "active_nuisance_summary_20260907.tsv", "moderate_benchmark_summary_20260920.tsv",
           "gaussian_positive_control_20261118/replications.tsv",
           "gaussian_positive_control_20261118/summary.tsv"]


def read(name):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(value, decimals=4):
    return f"{float(value):.{decimals}f}"


def interval(row, low="wilson_low", high="wilson_high"):
    return f"[{f(row[low])}, {f(row[high])}]"


def gaussian_calibration(rows=None):
    """Describe the fixed six-cell design, preserving every replication.

    The pooled Wilson interval is descriptive: the six cells have different
    targets and potentially different coverage probabilities. The stratified
    MCSE estimates uncertainty of their equally weighted average coverage.
    """
    rows = read(SOURCES[4]) if rows is None else rows
    expected = {(r, n, rep) for r in (.5, .8) for n in (160, 640, 2560)
                for rep in range(1000)}
    seen, counts = set(), {(r, n): 0 for r in (.5, .8) for n in (160, 640, 2560)}
    z = NormalDist().inv_cdf(.975)
    for row in rows:
        key = (float(row["r"]), int(row["n"]), int(row["replication"]))
        if key not in expected or key in seen or row["error"]:
            raise ValueError("invalid, duplicate or failed Gaussian replication")
        seen.add(key)
        estimate, se, truth = (float(row[k]) for k in ("estimate", "se", "truth"))
        if not all(map(math.isfinite, (estimate, se, truth))) or se <= 0:
            raise ValueError("invalid Gaussian estimate or standard error")
        actual = (int(estimate-z*se <= truth <= estimate+z*se),
                  int(estimate+z*se < truth), int(estimate-z*se > truth))
        if actual != tuple(int(row[k]) for k in ("covered", "below", "above")):
            raise ValueError("Gaussian coverage flags disagree with intervals")
        counts[key[:2]] += actual[0]
    if seen != expected:
        raise ValueError("incomplete Gaussian replication schedule")
    hits, total = sum(counts.values()), len(rows)
    p = hits/total
    den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    cell_rates = [count/1000 for count in counts.values()]
    return dict(cells=6, valid=total, covered=hits, coverage=p,
                descriptive_wilson_low=center-half,
                descriptive_wilson_high=center+half,
                equal_weight_stratified_mcse=math.sqrt(sum(q*(1-q)/999 for q in cell_rates)/36),
                all_cells_below_nominal=all(q < .95 for q in cell_rates),
                minimum_cell_coverage=min(cell_rates), maximum_cell_coverage=max(cell_rates))


def tables():
    diagnostics = summarize()
    stages = read(SOURCES[0])
    labels = {"population_reference": "Fixed reference", "population_scale": "Fixed scale",
              "population_median": "Fixed median", "full_fit": "Full fit"}
    staged = [[r["n"], labels[r["stage"]], f(r["mean_rho"]), f(r["stage_aware_rate"]),
               interval(r, "stage_aware_wilson_low", "stage_aware_wilson_high"), f(r["direct_only_rate"])]
              for r in stages if float(r["tau"]) == .1 and r["stage"] != "population_median"]
    stages_full = [[f(r["tau"], 2), r["n"], labels[r["stage"]], f(r["mean_rho"]),
                    f(r["stage_aware_rate"]), interval(r, "stage_aware_wilson_low", "stage_aware_wilson_high"),
                    f(r["direct_only_rate"])] for r in stages]
    laws = {"profile_null_normal_sign_link": "Normal shared sign", "independent_t5": r"Independent $t_5$",
            "independent_strong_skew": "Independent strong skew"}
    nulls = read(SOURCES[1])
    null_full = [[laws[r["scenario"]], r["n"], r["root_cell_seed"], f(r["rejection_rate"]),
                  interval(r, "wilson_95_low", "wilson_95_high")] for r in nulls]
    regular = [[laws[r["scenario"]], r["n"], f(r["rejection_rate"]),
                interval(r, "wilson_95_low", "wilson_95_high")] for r in nulls if int(r["n"]) >= 640]
    skew_tables = {}
    method_labels = {"complete_kde": "F", "direct_only_ablation": "D", "oracle_fixed_reference": "O"}
    for file, sd in zip(SOURCES[2:4], (".6", ".3")):
        rows = read(file)
        for row in rows:
            if row["method"] == "complete_kde":
                regular.append([f"Lognormal log-SD $0{sd}$", row["n"], f(row["coverage"]), interval(row)])
        skew_tables["skew_"+sd[1]] = [[r["n"], method_labels[r["method"]], f(r["coverage"]),
            interval(r), f(r["bias"], 5), f(r["empirical_sd"], 5), f(r["mean_se"], 5), f(r["se_sd_ratio"])] for r in rows]
    gaussian = gaussian_summary(read(SOURCES[4]))
    gaussian_calibration()
    saved = read(SOURCES[5])
    for actual, expected in zip(gaussian, saved, strict=True):
        for key, value in actual.items():
            if abs(float(value) - float(expected[key])) > 1e-12:
                raise ValueError(f"Gaussian summary mismatch: {key}")
    for row in gaussian:
        regular.append([f"Bivariate normal $r={row['r']}$", str(row["n"]), f(row["coverage"]), interval(row)])
    gaussian_rows = [[f(r["r"], 1), str(r["n"]), f(r["truth"], 6), f(r["coverage"]), interval(r),
                      f(r["bias"], 5), f(r["se_over_sd"])] for r in gaussian]
    mechanism = [[f(r["tau"], 2), str(r["n"]), f(r["mean_abs_median"], 3), f(r["mean_scale"], 3),
                  f(r["reference_correlation"], 3), f(r["full_mean_active_fraction"], 3)] for r in diagnostics["mechanism"]]
    tails = [[f(r["log_sd"], 1), str(r["n"]), f(r["wald_coverage"]), f(r["fisher_coverage"]),
              f(r["paired_difference_mcse"], 5), f(r["below_truth"]), f(r["above_truth"])] for r in diagnostics["intervals"]]
    return dict(staged=staged, regular=regular, stages_full=stages_full, null_full=null_full,
                mechanism=mechanism, gaussian=gaussian_rows, tails=tails, **skew_tables)


def latex_rows(rows):
    return "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n"


def export(output):
    bodies = tables()
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in bodies.items():
        (output / (name + ".tex")).write_text(latex_rows(rows))
    sources = SOURCES + ["reference_pathway_replications_20260909.tsv",
                         "active_nuisance_replications_20260907.tsv", "moderate_benchmark_replications_20260920.tsv"]
    manifest = dict(table_rows={name: len(rows) for name, rows in bodies.items()},
        gaussian_control=gaussian_calibration(),
        source_sha256={name: normalized_text_sha256(ROOT / "results" / name) for name in sources})
    (output / "tables.json").write_text(json.dumps(bodies, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir), indent=2))
