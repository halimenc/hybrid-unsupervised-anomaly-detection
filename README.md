# Hybrid Unsupervised Transaction Anomaly Detection

An experimental, **fully unsupervised** transaction-anomaly detection project built with Python and PyTorch. It combines two independently trained masked-tabular Transformer models: one for **transaction context**, and another for **temporal / velocity behavior**. Both produce anomaly rankings; the hybrid score combines their signals.

> **Portfolio / research prototype — not a production fraud detection system.** Anomaly priorities are **not** probabilities of fraud, and a `CRITICAL` flag does not establish that fraud occurred. No particular detection performance is claimed without measured test results.

## Approach

**Model 1 — Static / contextual anomaly detection:** Builds numeric and categorical transaction context and learns to reconstruct masked numeric features. Larger reconstruction errors indicate unexpected combinations relative to the training data.

**Model 2 — Temporal / velocity anomaly detection:** Builds transaction-history features (time since the previous event, spans over recent events, merchant changes and merchant-region changes). It combines masked reconstruction with **directional rarity** of unusually short intervals learned from training data. The model operates on engineered temporal features; **it is not a sequence-to-sequence Transformer**.

**Hybrid prioritization:** Produces independent static and temporal scores, calibrates them against a separate calibration subset and combines them into an anomaly-priority ranking. **Isolation Forest** and **robust z-score** are *evaluation baselines only*; neither contributes to training or hybrid scoring.

**Training is unsupervised.** The `is_fraud` label is excluded from feature engineering, fitting, scoring and calibration. When present, it is joined by transaction ID **only for evaluation**, to compute metrics such as average precision (PR-AUC), ROC-AUC, precision@k and recall@k.

## Quick start — Windows / VS Code

Requirements: Windows with Python installed (`py --version`), VS Code with the Python extension, and a terminal opened **in the folder containing `run.py`**.

```powershell
# 1. Create a local Python environment (once)
py -m venv .venv

# 2. Install dependencies into that environment (once)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. A small synthetic smoke test — no dataset needed
.\.venv\Scripts\python.exe run.py --demo --accounts 12 --per-account 14 --epochs 1 --width 16 --layers 1
```

The **Run Python File** button launches `run.py` with its *default* arguments; it is **not equivalent to the commands above**. Prefer running the explicit terminal commands, or use one of the F5 configurations in `.vscode/launch.json`.

### Run on Kaggle data

Download the publicly available **[Credit Card Transactions Fraud Detection (Sparkov)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)** dataset and extract the following files:

```text
data/
└── raw/
    ├── fraudTrain.csv
    └── fraudTest.csv
```

Run a bounded pilot with a separate out-of-time test file:

```powershell
# Optional: keep PyTorch / numerical-library CPU threading modest
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"

.\.venv\Scripts\python.exe run.py --data data/raw/fraudTrain.csv --test-data data/raw/fraudTest.csv --accounts 200 --per-account 0 --test-accounts 300 --test-per-account 100 --epochs 8 --width 32 --layers 2 --output outputs/oot_test
```

- `--accounts`: number of sampled training accounts; `--per-account 0`: all available history for those training accounts.
- `--test-accounts`: number of sampled test accounts; `--test-per-account`: maximum test events per selected account. `0` means all, **which can substantially increase memory/time**.
- `--epochs`: training passes; `--width` and `--layers`: model size. A small pilot is a functionality check, **not evidence of model quality**.
- The training model never reads `is_fraud`; the label from `fraudTest.csv` is read separately for evaluation after scores are computed.
- **Important:** Account selection is random and does not use fraud labels. A small evaluation sample may contain **zero fraud events**, making ROC-AUC / PR-AUC undefined. If so, increase test coverage; do not report an invented score.

### Hardware / resources

**CPU training is supported; a dedicated GPU is not required.** The project was designed to permit bounded experiments on an **8 GB RAM laptop**, but **no reproducible peak-RAM, CPU-utilization or runtime benchmark is available**. Therefore 8 GB is a *practical target for small sampled runs*, **not a guaranteed maximum** or verified minimum. Available memory depends on the OS and other programs.

| Usage | Suggested starting point | Resource expectation |
|---|---|---|
| Smoke test | `--demo --accounts 12 --per-account 14 --epochs 1 --width 16 --layers 1` | Smallest workload; suitable for checking installation |
| Bounded Kaggle pilot | Command above, with `--width 32 --layers 2` | CPU-only, potentially lengthy; monitor RAM |
| Larger experiments | Increase `--accounts`, history length, epochs or model dimensions gradually | More RAM, CPU work and/or runtime; **not benchmarked** |

The CSV loader reads input in chunks, but selected account histories are concatenated in memory, and downstream feature/score arrays also occupy RAM. **Do not assume** that `--accounts` alone places a hard RAM bound. On Windows, monitor **Task Manager → Performance → Memory / CPU**. For lower CPU contention use the two environment variables shown above. If memory pressure occurs, lower `--accounts`, `--test-accounts`, `--test-per-account`, `--width` or `--layers`; keep enough temporal history for a meaningful experiment. The full Kaggle dataset on 8 GB RAM has **not** been validated.

## Outputs

Files are written under the selected `--output` directory (default `outputs/`):

| File | Purpose |
|---|---|
| `top_static.csv` | Top-ranked static anomalies |
| `top_temporal.csv` | Top-ranked temporal anomalies |
| `top_hybrid.csv` | Top-ranked hybrid priorities |
| `summary.json` | Run summary, training MSE, split and feature lists |
| `evaluation.json` | Fraud-label-based metrics and baseline comparisons, **when evaluation labels are supplied** |
| `static_model.pt`, `temporal_model.pt` | Transformer weights and feature names (not a complete deployable inference pipeline) |

Results are intentionally **not committed** to Git; supply anonymized summary tables or plots separately if publishing findings.

## Repository structure

```text
run.py                 CLI entry point
src/dataset.py         CSV ingestion and synthetic demo
src/features.py        Static and temporal feature engineering
src/transformer.py     Masked-tabular PyTorch models
src/pipeline.py        Training, calibration, ranking, evaluation
.vscode/launch.json    VS Code F5 launch profiles
tests/test_project.py  Automated checks
data/raw/               Local dataset files — excluded from Git
outputs/                Generated model artifacts/reports — excluded from Git
```

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Current limitations and future work

- The separate `fraudTest.csv` path provides a later-file evaluation, but **test-account subsampling may be unrepresentative**; a full external benchmark is not reported here.
- Temporal reconstruction uses engineered lag features, not a learned sequence encoder. Correlated features may still allow reconstruction shortcuts.
- Missing-value handling and score calibration are experimental; held-out early stopping, stronger categorical unknown handling and group masking remain future improvements.
- Time-distance / implied travel speed and per-account behavior profiles are not yet fully implemented.
- Stored `.pt` files do **not** yet contain all vocabularies, preprocessing state and calibration references needed for a standalone `predict` command.
- Account identifiers are pseudonymized with **unsalted SHA-256**, not securely anonymized. Never publish identifiers derived from real card data, raw transaction files, passwords, or secrets.

## Data and project scope

The models are an independent learning/portfolio implementation using public synthetic transaction data. No private organizational data or credentials are needed to run the project. Follow the Kaggle dataset's terms when obtaining and using its files. **The dataset is not bundled in this repository.**

## License

No license is granted by default. A public repository does not automatically grant others permission to reuse the code; licensing can be decided separately.
