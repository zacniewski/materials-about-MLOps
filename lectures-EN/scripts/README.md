# Demo scripts for lectures 1–8

Scripts to run live during the lectures. Each one corresponds to one lecture section
and runs standalone on synthetic data, so no datasets need to be downloaded.

## Environment setup

```bash
# in the repository root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install optuna                 # optional, lecture 3 (without it the script uses random search)
```

Run all commands **from the repository root**. Outputs (JSON, PNG files,
models, data) go to `..`, and MLflow runs go to `./mlruns`.
Both directories are in `.gitignore`.

## Lecture 1 – Introduction to MLOps

| Script | Section | What it shows | Time |
|---|---|---|---|
| `lecture_01/01_reproducibility.py` | 4.1 | Results without and with a seed, saving the experiment configuration to JSON | ~5 s |
| `lecture_01/02_data_drift.py [--plot]` | 4.2–4.3 | KS test for 3 drift scenarios, concept drift (X unchanged, accuracy drops to ~0.5) | ~2 s |
| `lecture_01/03_simple_mlops_system.py` | 8 | Model registry: training → registration → promotion of the better model only → prediction | ~5 s |

```bash
python lectures-EN/scripts/lecture_01/01_reproducibility.py
python lectures-EN/scripts/lecture_01/02_data_drift.py --plot
python lectures-EN/scripts/lecture_01/03_simple_mlops_system.py
```

## Lecture 2 – Data engineering

| Script | Section | What it shows | Time |
|---|---|---|---|
| `lecture_02/01_csv_vs_parquet.py [--rows N]` | 3 | Size, speed, column pruning and type preservation: CSV vs Parquet | ~10 s (1M rows) |
| `lecture_02/02_data_pipeline.py` | 5 | Cleaning → feature engineering → `ColumnTransformer` → model → saving the pipeline | ~5 s |
| `lecture_02/03_data_validation.py [--corrupt]` | 6 | Data validation as a quality gate (exit code 1 on errors) | ~1 s |
| `lecture_02/04_data_leakage.py` | 10 | Data leakage: feature selection before CV (0.85 on noise!) and target leakage | ~5 s |
| `lecture_02/05_dvc_demo.sh` | 8 | DVC: two data versions, `dvc push`, back to v1 via `git checkout` + `dvc checkout` | ~10 s |

```bash
python lectures-EN/scripts/lecture_02/01_csv_vs_parquet.py --rows 1000000
python lectures-EN/scripts/lecture_02/02_data_pipeline.py
python lectures-EN/scripts/lecture_02/03_data_validation.py
python lectures-EN/scripts/lecture_02/03_data_validation.py --corrupt; echo "exit code: $?"
python lectures-EN/scripts/lecture_02/04_data_leakage.py
bash   lectures-EN/scripts/lecture_02/05_dvc_demo.sh     # requires git and dvc (Linux/macOS/WSL/Git Bash)
```

The DVC demo runs in a temporary directory (`mktemp -d`), so it does not modify this repository.
At the end it prints the command to remove the temporary directories.

## Lecture 3 – Experimentation and tracking (MLflow)

| Script | Section | What it shows | Time |
|---|---|---|---|
| `lecture_03/01_mlflow_experiments.py` | 3 | 5 runs: parameters, metrics, CV, step metric, artifacts (confusion matrix), model with signature | ~40 s |
| `lecture_03/02_compare_runs.py` | 4 | `search_runs`, filtering, loading the best model, Model Registry + `champion` alias | ~5 s |
| `lecture_03/03_hyperparameter_tuning.py [--trials N]` | 6 | HPO with nested runs (Optuna TPE or random search) | ~3 s/trial |
| `lecture_03/04_reproducibility.py` | 8, 10 | Seed, git commit, data hash, library versions in tags; a run with FAILED status | ~5 s |

```bash
python lectures-EN/scripts/lecture_03/01_mlflow_experiments.py
python lectures-EN/scripts/lecture_03/02_compare_runs.py           # after 01
python lectures-EN/scripts/lecture_03/03_hyperparameter_tuning.py --trials 15
python lectures-EN/scripts/lecture_03/04_reproducibility.py
mlflow ui --port 5000                                            # → http://localhost:5000
```

By default the scripts store runs locally in `./mlruns`. To use an MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000      # separate terminal
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Reset before the next lecture: `rm -rf mlruns`.

## Lecture 4 – ML Pipelines

| Script | What it shows |
|---|---|
| `lecture_04/01_mini_orchestrator.py` | Mini orchestrator: DAG, cycle detection, parallel "waves", retry with backoff, hash-based cache |
| `lecture_04/02_quality_gate_pipeline.py` | Local pipeline ingest → validate → train → evaluate → quality gate with a lineage log |
| `lecture_04/03_kfp_compile.py` | Compiling a KFP v2 pipeline (`dsl.If/Else`, retry, resources) to YAML without a cluster |

## Lecture 5 – Model serving

| Script | What it shows |
|---|---|
| `lecture_05/01_fastapi_service.py` | FastAPI service tested with `TestClient` (200 / 422 / 400) |
| `lecture_05/02_latency_batching.py` | p50/p95/p99 and throughput vs batch size, impact of the tail on p99 |
| `lecture_05/03_deployment_strategies.py` | Shadow and canary simulation with sticky routing and auto-rollback |
| `lecture_05/04_batch_scoring.py` | Idempotent batch scoring into Parquet partitions |

## Lecture 6 – Monitoring

| Script | What it shows |
|---|---|
| `lecture_06/01_drift_metrics.py` | PSI, KL, JS, Wasserstein, KS for increasing drift; the p-value trap at large n |
| `lecture_06/02_concept_drift_stream.py` | 12 weeks: data drift vs concept drift, champion/challenger |
| `lecture_06/03_prometheus_metrics.py [--serve]` | Counter/Histogram/Gauge metrics, `/metrics` format, percentiles from buckets |

## Lecture 7 – CI/CD for ML

| Script | What it shows |
|---|---|
| `lecture_07/01_ml_unit_tests.py` | pytest suite: unit, contract, determinism, INV/DIR/MFT behavioral tests |
| `lecture_07/02_quality_gate.py [--bad]` | Champion vs challenger quality gate (exit code 1 on failure) |
| `lecture_07/03_ci_pipeline_sim.py [--break-data]` | Fail-fast CI pipeline simulation |

## Lecture 8 – Scalability and security

| Script | What it shows |
|---|---|
| `lecture_08/01_batching_latency_throughput.py` | Latency vs throughput, dynamic batching simulation |
| `lecture_08/02_adversarial_and_validation.py` | FGSM attack, input validation (pydantic v2), model extraction |
| `lecture_08/03_pii_masking.py` | PII masking, HMAC pseudonymization, k-anonymity |

## Tips for the instructor

- Run every script once before class. The first import of MLflow or sklearn takes longer.
- The scripts are deterministic (`random_state=42`), so results on the slides match the live ones.
  Exceptions are execution times and the intentionally random part of `01_reproducibility.py`.
- On slow hardware reduce the parameters: `--rows 200000` in CSV vs Parquet or `--trials 5` in HPO.
