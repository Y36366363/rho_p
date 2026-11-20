# Robust-reference distance-profile correlation

Companion code and simulation records for *Inference for robust-reference distance-profile correlation: regular theory and finite-sample behavior*, by Jialiang Yao and Johan F. Hoorn. The target is rho_P = Corr(|X - T_X|, |Y - T_Y|), where each T is a Huber location fitted at a median/MAD scale. An arXiv identifier and citation will be added when available.

## Installation

Use Python 3.12 (the pinned dependencies support Python 3.11–3.12).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Estimate a profile correlation

```python
import numpy as np
from src.rho_profile import infer

rng = np.random.default_rng(20261118)
x = rng.normal(size=640)
y = .5*x + np.sqrt(.75)*rng.normal(size=640)
fit = infer(x, y)
print(fit["estimate"], fit["standard_error"])
print(fit["normal"])
print(fit["solver_diagnostics"])
```

Supply at least 12 finite, paired observations. The default `score_checked` solver checks the Huber score residual and reports failures. The standard error includes the median, MAD, Huber reference and paired profile moments. The `normal` result contains the untransformed Wald interval; finite-sample coverage is not guaranteed. `fisher_interval` transforms the estimate using the complete influence-function standard error. `bootstrap_t_interval` is experimental; its coverage improvement has not been validated.

## Reproduce the results

```sh
python scripts/reproduce_arxiv_tables.py --output-dir outputs/tables
python -m pytest -q
```

The table command creates nine LaTeX table bodies, `tables.json` and an input-hash manifest. Choose a new output directory for each run. See [REPRODUCE.md](REPRODUCE.md) for diagnostics, simulation commands and the table-to-data mapping.

- `src/`: numerical implementation and profile-correlation interface.
- `scripts/`: simulation, table generation and verification.
- `results/`: replication records and summaries.
- `docs/` and `archive/source_snapshots/`: study designs and the implementation used for the September 7–9 studies.
- `tests/`: numerical, solver and reproducibility checks.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
