"""Train the first binary Label 2 contention detector.

Labels are created from stress_active_*.csv markers:
    0 = clean/non-contention
    2 = verified ib_write_bw contention

The split is by complete experiment run, never by individual telemetry row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "ib_port_xmit_data_delta",
    "ib_port_rcv_data_delta",
    "ib_port_xmit_packets_delta",
    "ib_port_rcv_packets_delta",
    "ib_port_xmit_wait_delta",
    "ib_port_xmit_discards_delta",
    "ib_sq_num_rnr_delta",
    "ib_sq_num_to_delta",
    "ddp_step_time_ms",
]

LOG_STEP_PATTERN = re.compile(
    r"step_time_ms=(?P<step>[0-9.]+).*?wall=(?P<timestamp>[0-9.]+)"
)


def _read_markers(run_dir: Path) -> tuple[float, float] | None:
    markers: list[tuple[float, float]] = []
    for path in run_dir.glob("stress_active_*.csv"):
        frame = pd.read_csv(path)
        if len(frame) < 2:
            continue
        values = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
        if len(values) >= 2:
            markers.append((float(values.iloc[0]), float(values.iloc[1])))
    if not markers:
        return None
    starts, ends = zip(*markers)
    return min(starts), max(ends)


def _read_step_times(run_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(run_dir.glob("step_time_*.csv"))
    if csv_paths:
        frame = pd.read_csv(csv_paths[0], usecols=["timestamp", "step_time_ms"])
        return frame.rename(columns={"step_time_ms": "ddp_step_time_ms"}).dropna(
            subset=["timestamp"]
        )

    observations: list[dict[str, float]] = []
    for path in sorted(run_dir.glob("node_*.log")):
        text = path.read_text(errors="replace")
        observations.extend(
            {
                "timestamp": float(match.group("timestamp")),
                "ddp_step_time_ms": float(match.group("step")),
            }
            for match in LOG_STEP_PATTERN.finditer(text)
        )
    return pd.DataFrame(observations, columns=["timestamp", "ddp_step_time_ms"])


def _join_step_times(telemetry: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    steps = _read_step_times(run_dir)
    if steps.empty:
        telemetry["ddp_step_time_ms"] = np.nan
        return telemetry
    left = telemetry.sort_values("timestamp")
    right = steps.sort_values("timestamp")
    return pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="nearest",
        tolerance=0.5,
    )


def _load_label2(root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for run_dir in sorted((root / "data" / "Label2").iterdir()):
        if not run_dir.is_dir():
            continue
        interval = _read_markers(run_dir)
        telemetry_paths = sorted(run_dir.glob("telemetry_*.csv"))
        if interval is None:
            skipped.append(run_dir.name)
            continue
        for path in telemetry_paths:
            frame = pd.read_csv(path)
            frame = _join_step_times(frame, run_dir)
            frame["label"] = np.where(
                frame["timestamp"].between(interval[0], interval[1]),
                2,
                0,
            )
            frame["run_id"] = run_dir.name
            frame["source"] = "label2"
            frame["node_id"] = frame.get("node_id", path.stem)
            frames.append(frame)
    if not frames:
        raise RuntimeError("No marker-complete Label 2 runs were found.")
    return pd.concat(frames, ignore_index=True), skipped


def _load_label0(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "data" / "Label0").glob("*_label0.csv")):
        frame = pd.read_csv(path)
        frame["label"] = 0
        frame["run_id"] = f"label0_{path.stem}"
        frame["source"] = "label0"
        frames.append(frame)
    if not frames:
        raise RuntimeError("No Label 0 *_label0.csv files were found.")
    return pd.concat(frames, ignore_index=True)


def _make_model() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    transform = ColumnTransformer([("numeric", numeric, FEATURES)])
    return Pipeline(
        [
            ("features", transform),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_tree_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                            FEATURES,
                        )
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _evaluate(
    model_factory: Callable[[], Pipeline],
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[dict[str, object], np.ndarray]:
    logo = LeaveOneGroupOut()
    predictions = np.full(len(y), -1, dtype=int)
    fold_count = 0
    for train_idx, test_idx in logo.split(x, y, groups):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            continue
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(x.iloc[test_idx])
        fold_count += 1

    evaluated = predictions >= 0
    if not evaluated.any():
        raise RuntimeError("No complete-run fold contained both target classes.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y[evaluated], predictions[evaluated], average="binary", zero_division=0
    )
    return (
        {
            "folds_evaluated": fold_count,
            "rows_evaluated": int(evaluated.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix(
                y[evaluated], predictions[evaluated]
            ).tolist(),
            "classification_report": classification_report(
                y[evaluated],
                predictions[evaluated],
                target_names=["clean", "label2_contention"],
                output_dict=True,
                zero_division=0,
            ),
        },
        predictions,
    )


def train(root: Path, output_dir: Path) -> dict[str, object]:
    label2, skipped = _load_label2(root)
    label0 = _load_label0(root)
    data = pd.concat([label2, label0], ignore_index=True)
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values(["run_id", "node_id", "timestamp"])

    # The first delta row has no preceding counter sample and is not evidence.
    delta_features = [name for name in FEATURES if name.endswith("_delta")]
    data = data.dropna(subset=delta_features, how="all").reset_index(drop=True)
    data["target"] = (data["label"] == 2).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "label2_training_table.csv", index=False)

    x = data[FEATURES]
    y = data["target"]
    groups = data["run_id"]
    logistic_metrics, _ = _evaluate(_make_model, x, y, groups)
    forest_metrics, _ = _evaluate(_make_tree_model, x, y, groups)
    metrics = {
        "evaluation": (
            "leave-one-complete-run-out; single-class test folds omitted"
        ),
        "rows_total": int(len(data)),
        "clean_rows": int((y == 0).sum()),
        "issue_rows": int((y == 1).sum()),
        "marker_complete_label2_runs": sorted(label2["run_id"].unique()),
        "label2_runs_skipped_missing_markers": skipped,
        "features": FEATURES,
        "models": {
            "logistic_regression": logistic_metrics,
            "random_forest": forest_metrics,
        },
    }

    final_model = _make_model()
    final_model.fit(x, y)
    final_forest = _make_tree_model()
    final_forest.fit(x, y)
    import joblib

    joblib.dump(final_model, output_dir / "label2_logistic_regression.joblib")
    joblib.dump(final_forest, output_dir / "label2_random_forest.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    metrics = train(args.root, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
