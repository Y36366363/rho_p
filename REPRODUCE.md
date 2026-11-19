# Reproducing the profile-correlation results

Use CPython 3.12.3, NumPy 1.26.4 and SciPy 1.16.3 for the simulation environment, with pytest 7.4.4 for tests. Install `requirements.txt` in a Python 3.12 virtual environment as described in README.md. All commands below run from the repository root. No manuscript is required for the default tests.

## Tables, diagnostics and tests

```sh
python scripts/reproduce_arxiv_tables.py --output-dir outputs/tables
python scripts/summarize_revision_evidence.py --output outputs/diagnostics.json
python -m pytest -q
```

The table export produces nine `.tex` bodies, `tables.json`, and `manifest.json`. Output paths must not already exist; choose a different name on subsequent runs. The current table set has 78 rows, including all six Gaussian control cells. The diagnostic report also recomputes population-SE coverage for log-SD 0.6: .9675, .9590, .9580 at n=160,640,2560. Source hashes use LF-normalized text, so equivalent CRLF files give the same hashes.

An optional LaTeX comparison accepts a manuscript supplied separately:

```sh
python scripts/audit_revision_manuscript.py /path/to/main.tex --output outputs/manuscript-check.json
RHOP_REVISION_TEX=/path/to/main.tex python -m pytest -q tests/test_revision_manuscript.py
```

Without that environment variable, one manuscript comparison test is skipped. The comparison checks numerical table bodies and cross-references, not mathematical proofs.

## Result mapping

All filenames in the middle column are under `results/`.

| Paper location | Records | Generator |
|---|---|---|
| Table 2, S1 | `reference_pathway_{summary,replications}_20260909.tsv` | `run_reference_pathway_20260909.py` |
| S2, S6 | `revision_diagnostics_20261118.json` | `summarize_revision_evidence.py` |
| Table 3 null rows, S3 | `claim1_wald_validation_20260823.tsv` | `run_claim_validation_20260823.py` |
| Table 3 log-SD 0.6, S4 | `active_nuisance_{summary,replications}_20260907.tsv` | `validate_active_nuisance_20260907.py` |
| Table 3 log-SD 0.3, S5 | `moderate_benchmark_{summary,replications}_20260920.tsv` | `run_moderate_benchmark_20260920.py` |
| Table 3 Gaussian rows, S7 | `gaussian_positive_control_20261118/` | `run_gaussian_positive_control.py` |
| S2.2 reference root checks | `reference_pathway_root_sensitivity_20260909.tsv` | pathway generator |
| S2.2 solver comparison | `moderate_benchmark_solver_20260920.tsv` | moderate generator |
| S2.3 targets, variance, derivatives | `active_nuisance_{population,derivative}_20260907.tsv`, `moderate_benchmark_population_20260920.tsv` | active and moderate generators |
| S2.3 paired MCSE | `active_nuisance_paired_20260907.tsv`, `moderate_benchmark_paired_20260920.tsv` | active and moderate generators |
| S2.3 population-SE coverage | `population_se_coverage` in diagnostics JSON | diagnostic script |

`gaussian_calibration_20261118.json` summarizes all six Gaussian cells; its pooled Wilson interval is descriptive of the mixture of design cells.

## Rerun simulations

The Gaussian and moderate studies use the current numerical kernel and the residual-checked solver:

```sh
python scripts/run_gaussian_positive_control.py --output-dir outputs/gaussian
python scripts/run_moderate_benchmark_20260920.py --out-dir outputs/moderate
```

For the shared-sign and log-SD 0.6 studies, use the supplied implementation snapshot. This command constructs a temporary source tree, runs both full studies, and compares replication records and derived summaries with the supplied results:

```sh
mkdir -p outputs
python scripts/reproduce_core_studies_20260916.py --report outputs/core.json
```

Comparisons use exact discrete outcomes and a 1e-10 scaled tolerance for floating-point values. The snapshot is `archive/source_snapshots/cdelta_20260914.py`. Stages 2–3 of the pathway use brentq; full fitting and the log-SD 0.6 study use IRLS (`legacy`). Running `validate_active_nuisance_20260907.py --out-dir outputs/active` directly uses the current kernel instead; use the snapshot command above to reproduce the supplied study.

Run the null study in a separate clone because its script writes to `results/`:

```sh
python scripts/run_claim_validation_20260823.py --phase validation
```

That script also computes two related checks from the included input tables; only its claim1 output is needed for this paper. The null study uses IRLS. Likewise, run the pathway script only in a separate clone; it writes to `results/`. Compare results before replacing any supplied record.

Simulation reports may include machine-specific paths and package lists. Keep generated outputs under `outputs/`, which Git ignores. Runtimes depend on the machine.

## Source integrity

The September 7–9 manifests record LF-normalized SHA-256 values for the two runners, their protocols, implementation snapshot and results. Tests use functions in `scripts/verify_frozen_sources.py` to verify these records; that module has no CLI. Commit identifiers in those manifests identify the original development repository, not commits in this repository. The snapshot, the two runners and their protocols match the files recorded for the September 7–9 studies after LF normalization, as the manifest hashes confirm.

The moderate protocol includes a dated publication note for one editorially reworded sentence. Its design, seeds and reporting commitments are unchanged; the moderate runner records the protocol hash used by each new run. Existing result records are preserved.
