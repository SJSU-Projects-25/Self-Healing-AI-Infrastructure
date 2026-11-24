# Label 2 model

This folder contains reproducible binary classifiers for the Label 2
experiment described in [`label2_ml_handoff.pdf`](../label2_ml_handoff.pdf).

## Training objects

- **Clean rows (`target=0`)**: Label 0 healthy telemetry plus rows outside a
  verified `stress_active_*.csv` interval in a Label 2 run.
- **Issue rows (`target=1`)**: Label 2 telemetry rows whose timestamps fall
  inside a verified RDMA `ib_write_bw` stress interval. The source label remains
  `2`.
- **Binary-model features**: GPU context, InfiniBand traffic/wait deltas, RDMA
  error deltas, and aligned DDP step time.
- **Excluded from features**: stress markers, stress tool names, IDs,
  timestamps, hostnames, and cumulative counters.
- **Evaluation**: leave-one-complete-run-out. Random row splitting is avoided
  because neighboring telemetry samples are correlated.

Runs without stress markers are reported and skipped rather than guessed.

## Run

From the repository root:

```powershell
python model\train_label2.py
```

Outputs:

- `label2_training_table.csv` — normalized, labeled training table
- `label2_logistic_regression.joblib` — interpretable baseline fit on all verified rows
- `label2_random_forest.joblib` — nonlinear comparison model fit on all verified rows
- `metrics.json` — run-level comparison metrics and data audit

Both models classify `clean` versus `label2_contention`; they are not yet
production detectors. Add independent marker-complete jobs before making
generalization claims.

## Three-class health experiment

`train_health.py` trains a separate random-forest classifier with:

- `0` — healthy training from Label 0, Label 1 healthy rows, and non-stress
  Label 2 rows
- `1` — annotated Label 1 AllReduce pause
- `2` — Label 2 rows inside verified RDMA stress intervals

Label 5 idle data is excluded until its timestamps and schema are validated.
Run it with:

```powershell
python model\train_health.py
```

Outputs are `health_training_table.csv`, `health_random_forest.joblib`, and
`health_metrics.json`. The evaluation holds out complete experiment runs.

The strengthened three-class evaluation uses 47,983 rows from thirty-nine complete runs:

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Healthy | 43,057 | 99.49% | 99.52% | 99.50% |
| AllReduce pause | 2,950 | 93.14% | 93.46% | 93.30% |
| IB contention | 1,976 | 99.74% | 98.63% | 99.19% |

Overall accuracy is 99.11%, macro F1 is 97.33%, and weighted F1 is 99.11%.
The AllReduce probability threshold was tuned from leave-one-run-out
out-of-fold predictions and is 0.4724. The tuned threshold is saved with the
model and loaded by `health_monitor.py`.

These are offline, leave-one-run-out results. The AllReduce-pause class has
2,950 rows from thirty-two experiments, and five Label 2 runs remain excluded
because their stress markers are missing.

## Conservative event inference

`health_monitor.py` separates row predictions from operational events:

```python
from pathlib import Path

from model.health_monitor import aggregate_events, classify_rows, load_health_model

model = load_health_model(Path("model/health_random_forest.joblib"))
rows = classify_rows(model, telemetry_frame)
events = aggregate_events(rows, min_consecutive=3)
```

An AllReduce pause or contention prediction is an `observation` until the
same class appears in at least three consecutive, time-contiguous samples on
the same node. This prevents one noisy row from triggering remediation.

The health model's change features are calculated from the preceding sample
within the same run and node. They do not use future rows or label metadata.
The health model also adds 5-second and 30-second rolling history features.
These use only preceding samples within the same run and node, including
step-time mean/median/p95/max/spike statistics and supporting GPU/IB averages.

Label 5 remains excluded: its current labeled files contain null timestamps
and duplicate rows and must be repaired before training.
