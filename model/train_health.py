"""Train a three-class training-health classifier.

Classes:
    0 = healthy training
    1 = normal AllReduce pause (Label 1)
    2 = verified InfiniBand contention (Label 2)

Label 5 idle data is intentionally excluded until its timestamps and schema
are validated for a separate operating-state experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

try:
    from .train_label2 import FEATURES, _join_step_times, _load_label0, _load_label2
except ImportError:
    from train_label2 import FEATURES, _join_step_times, _load_label0, _load_label2

CLASS_NAMES = ["healthy", "allreduce_pause", "ib_contention"]
LABEL1_TO_TARGET = {"healthy": 0, "allreduce_pause": 1}
DERIVED_FEATURES = [
    "gpu_util_pct_change",
    "ib_port_xmit_data_delta_change",
    "ib_port_xmit_packets_delta_change",
    "ib_port_xmit_wait_delta_change",
    "ddp_step_time_ms_change",
]
ROLLING_WINDOWS = ("5s", "30s")
HISTORY_FEATURES = [
    name
    for window in ROLLING_WINDOWS
    for name in (
        f"step_time_{window}_mean_ms",
        f"step_time_{window}_median_ms",
        f"step_time_{window}_std_ms",
        f"step_time_{window}_p95_ms",
        f"step_time_{window}_max_ms",
        f"step_time_{window}_spike_count",
        f"step_time_{window}_spike_fraction",
        f"gpu_util_{window}_min",
        f"gpu_util_{window}_max",
        f"gpu_util_{window}_range",
        f"gpu_util_{window}_mean",
        f"gpu_util_{window}_std",
        f"ib_xmit_wait_{window}_mean",
        f"ib_xmit_data_{window}_mean",
        f"ib_xmit_packets_{window}_mean",
        f"ib_rcv_data_{window}_mean",
        f"ib_rcv_packets_{window}_mean",
    )
]
RATE_FEATURES = [
    "ib_xmit_wait_rate",
    "ib_xmit_data_rate",
    "ib_xmit_packets_rate",
]
HEALTH_FEATURES = FEATURES + DERIVED_FEATURES + RATE_FEATURES + HISTORY_FEATURES


def _add_past_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add causal changes using only the preceding row for each node/run."""
    result = data.sort_values(["run_id", "node_id", "timestamp"]).copy()
    group = result.groupby(["run_id", "node_id"], sort=False)
    for column in DERIVED_FEATURES:
        base = column.removesuffix("_change")
        result[column] = pd.to_numeric(result[base], errors="coerce") - group[
            base
        ].shift(1)
    return result


def _add_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics from prior samples only, within each run and node."""
    result = data.sort_values(["run_id", "node_id", "timestamp"]).copy()

    for run_id, run_frame in result.groupby("run_id", sort=False):
        for node_id, node_indices in run_frame.groupby("node_id", sort=False).groups.items():
            indices = list(node_indices)
            frame = result.loc[indices].sort_values("timestamp")
            time_index = pd.to_datetime(
                pd.to_numeric(frame["timestamp"], errors="coerce"),
                unit="s",
                utc=True,
            )

            step = pd.Series(
                pd.to_numeric(frame["ddp_step_time_ms"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            gpu = pd.Series(
                pd.to_numeric(frame["gpu_util_pct"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            ib_wait = pd.Series(
                pd.to_numeric(frame["ib_port_xmit_wait_delta"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            ib_xmit = pd.Series(
                pd.to_numeric(frame["ib_port_xmit_data_delta"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            ib_packets = pd.Series(
                pd.to_numeric(frame["ib_port_xmit_packets_delta"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            ib_rcv = pd.Series(
                pd.to_numeric(frame["ib_port_rcv_data_delta"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)
            ib_rcv_packets = pd.Series(
                pd.to_numeric(frame["ib_port_rcv_packets_delta"], errors="coerce").to_numpy(),
                index=time_index,
            ).shift(1)

            for window in ROLLING_WINDOWS:
                step_window = step.rolling(window=window, min_periods=1)
                gpu_window = gpu.rolling(window=window, min_periods=1)
                ib_wait_window = ib_wait.rolling(window=window, min_periods=1)
                ib_xmit_window = ib_xmit.rolling(window=window, min_periods=1)
                ib_packets_window = ib_packets.rolling(window=window, min_periods=1)
                ib_rcv_window = ib_rcv.rolling(window=window, min_periods=1)
                ib_rcv_packets_window = ib_rcv_packets.rolling(
                    window=window, min_periods=1
                )

                median = step_window.median()
                values = {
                    f"step_time_{window}_mean_ms": step_window.mean(),
                    f"step_time_{window}_median_ms": median,
                    f"step_time_{window}_std_ms": step_window.std(),
                    f"step_time_{window}_p95_ms": step_window.quantile(0.95),
                    f"step_time_{window}_max_ms": step_window.max(),
                    f"gpu_util_{window}_min": gpu_window.min(),
                    f"gpu_util_{window}_max": gpu_window.max(),
                    f"gpu_util_{window}_range": gpu_window.max() - gpu_window.min(),
                    f"gpu_util_{window}_mean": gpu_window.mean(),
                    f"gpu_util_{window}_std": gpu_window.std(),
                    f"ib_xmit_wait_{window}_mean": ib_wait_window.mean(),
                    f"ib_xmit_data_{window}_mean": ib_xmit_window.mean(),
                    f"ib_xmit_packets_{window}_mean": ib_packets_window.mean(),
                    f"ib_rcv_data_{window}_mean": ib_rcv_window.mean(),
                    f"ib_rcv_packets_{window}_mean": ib_rcv_packets_window.mean(),
                }

                def spike_count(values_in_window: np.ndarray) -> float:
                    valid = values_in_window[~pd.isna(values_in_window)]
                    if valid.size == 0:
                        return 0.0
                    baseline = float(np.median(valid))
                    threshold = max(baseline * 1.5, baseline + 100.0)
                    return float(np.sum(valid > threshold))

                spike_counts = step_window.apply(spike_count, raw=True)
                values[f"step_time_{window}_spike_count"] = spike_counts
                values[f"step_time_{window}_spike_fraction"] = (
                    spike_counts
                    / step_window.count()
                )

                for column, series in values.items():
                    result.loc[frame.index, column] = series.to_numpy()

    return result


def _load_label1(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in sorted((root / "data" / "Label1").iterdir()):
        if not run_dir.is_dir():
            continue
        telemetry_paths = [
            path
            for path in sorted(run_dir.glob("telemetry_*.csv"))
            if not path.name.endswith("_label1.csv")
        ]
        marker_paths = sorted(run_dir.glob("label1_active_*.csv"))
        if not telemetry_paths or not marker_paths:
            continue

        run_frames: list[pd.DataFrame] = []
        for telemetry_path in telemetry_paths:
            frame = pd.read_csv(telemetry_path)
            if "timestamp" not in frame.columns:
                continue
            frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
            frame = frame.dropna(subset=["timestamp"])
            frame["label"] = "healthy"

            hostname = (
                frame["node_id"].astype(str).iloc[0].split(".")[0]
                if "node_id" in frame.columns and not frame.empty
                else ""
            )
            marker_matches = [
                path
                for path in marker_paths
                if hostname and hostname.lower() in path.stem.lower()
            ]
            for marker_path in marker_matches or marker_paths:
                markers = pd.read_csv(marker_path)
                if not {"timestamp", "label1_active"} .issubset(markers.columns):
                    continue
                markers["timestamp"] = pd.to_numeric(
                    markers["timestamp"], errors="coerce"
                )
                markers["label1_active"] = pd.to_numeric(
                    markers["label1_active"], errors="coerce"
                )
                markers = markers.dropna(
                    subset=["timestamp", "label1_active"]
                ).sort_values("timestamp")

                starts = markers.loc[
                    markers["label1_active"] == 1, "timestamp"
                ].tolist()
                ends = markers.loc[
                    markers["label1_active"] == 0, "timestamp"
                ].tolist()
                for start in starts:
                    end = next(
                        (candidate for candidate in ends if candidate >= start),
                        None,
                    )
                    if end is None:
                        continue
                    active = (
                        (frame["timestamp"] >= start)
                        & (frame["timestamp"] <= end)
                    )
                    frame.loc[active, "label"] = "allreduce_pause"

            run_frames.append(frame)

        if not run_frames:
            continue
        frame = pd.concat(run_frames, ignore_index=True)
        frame = _join_step_times(frame, run_dir)
        frame["target"] = frame["label"].map(LABEL1_TO_TARGET)
        if frame["target"].isna().any():
            unknown = sorted(frame.loc[frame["target"].isna(), "label"].unique())
            raise ValueError(f"Unknown Label 1 labels in {path}: {unknown}")
        frame["target"] = frame["target"].astype(int)
        frame["run_id"] = run_dir.name
        frame["source"] = "label1"
        frames.append(frame)
    if not frames:
        raise RuntimeError("No annotated Label 1 *_label1.csv files were found.")
    return pd.concat(frames, ignore_index=True)


def _make_model() -> Pipeline:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                HEALTH_FEATURES,
            )
        ]
    )
    return Pipeline(
        [
            ("features", transform),
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
    data: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray]:
    x = data[HEALTH_FEATURES]
    y = data["target"]
    groups = data["run_id"]
    predictions = np.full(len(data), -1, dtype=int)
    probabilities = np.full((len(data), len(CLASS_NAMES)), np.nan)
    logo = LeaveOneGroupOut()
    folds = 0
    for train_idx, test_idx in logo.split(x, y, groups):
        model = _make_model()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(x.iloc[test_idx])
        fold_probabilities = model.predict_proba(x.iloc[test_idx])
        probabilities[test_idx] = fold_probabilities
        folds += 1

    evaluated = predictions >= 0
    report = classification_report(
        y[evaluated],
        predictions[evaluated],
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    if set(model.classes_) >= {0, 1, 2}:
        model_class_indices = {
            int(class_name): index
            for index, class_name in enumerate(model.classes_)
        }
        pause_probability_index = model_class_indices[1]
        pause_probabilities = probabilities[:, pause_probability_index]

        thresholds = np.unique(
            np.clip(
                np.concatenate(
                    (
                        np.linspace(0.05, 0.95, 181),
                        pause_probabilities[evaluated],
                    )
                ),
                0.05,
                0.95,
            )
        )
        tuned_predictions = predictions.copy()
        best_threshold = 0.5
        best_f1 = f1_score(
            y[evaluated],
            predictions[evaluated],
            labels=[1],
            average="macro",
            zero_division=0,
        )

        for threshold in thresholds:
            candidate = predictions.copy()
            candidate[pause_probabilities >= threshold] = 1
            candidate_f1 = f1_score(
                y[evaluated],
                candidate[evaluated],
                labels=[1],
                average="macro",
                zero_division=0,
            )
            if candidate_f1 > best_f1:
                best_f1 = float(candidate_f1)
                best_threshold = float(threshold)
                tuned_predictions = candidate

        tuned_report = classification_report(
            y[evaluated],
            tuned_predictions[evaluated],
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        tuned_confusion = confusion_matrix(
            y[evaluated],
            tuned_predictions[evaluated],
            labels=[0, 1, 2],
        ).tolist()
        per_run = {}
        for run_id in sorted(groups.unique()):
            run_mask = evaluated & (groups == run_id)
            if not run_mask.any():
                continue
            run_report = classification_report(
                y[run_mask],
                tuned_predictions[run_mask],
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
            per_run[str(run_id)] = {
                "rows": int(run_mask.sum()),
                "allreduce_pause_f1": run_report["allreduce_pause"]["f1-score"],
                "allreduce_pause_precision": run_report["allreduce_pause"]["precision"],
                "allreduce_pause_recall": run_report["allreduce_pause"]["recall"],
            }
    else:
        best_threshold = 0.5
        tuned_report = report
        tuned_confusion = confusion_matrix(
            y[evaluated],
            predictions[evaluated],
            labels=[0, 1, 2],
        ).tolist()
        per_run = {}

    return (
        {
            "folds_evaluated": folds,
            "rows_evaluated": int(evaluated.sum()),
            "confusion_matrix": confusion_matrix(
                y[evaluated], predictions[evaluated], labels=[0, 1, 2]
            ).tolist(),
            "classification_report": report,
            "allreduce_pause_threshold": {
                "threshold": best_threshold,
                "selection": "leave-one-run-out out-of-fold AllReduce F1 maximum",
                "baseline_f1": report["allreduce_pause"]["f1-score"],
                "tuned_f1": tuned_report["allreduce_pause"]["f1-score"],
                "tuned_confusion_matrix": tuned_confusion,
                "tuned_classification_report": tuned_report,
                "per_run_allreduce_pause": per_run,
            },
        },
        tuned_predictions,
    )


def train(root: Path, output_dir: Path) -> dict[str, object]:
    label2, skipped_label2 = _load_label2(root)
    label2["target"] = (label2["label"] == 2).astype(int) * 2
    label0 = _load_label0(root)
    label0["target"] = 0
    label1 = _load_label1(root)

    data = pd.concat([label0, label1, label2], ignore_index=True)
    data[HEALTH_FEATURES] = data.reindex(
        columns=HEALTH_FEATURES, fill_value=np.nan
    ).apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values(["run_id", "node_id", "timestamp"])
    delta_features = [name for name in FEATURES if name.endswith("_delta")]
    data = data.dropna(subset=delta_features, how="all").reset_index(drop=True)
    data = _add_past_features(data)
    if "polling_interval_actual" in data.columns:
        interval = pd.to_numeric(
            data["polling_interval_actual"], errors="coerce"
        )
        for target, source in (
            ("ib_xmit_wait_rate", "ib_port_xmit_wait_delta"),
            ("ib_xmit_data_rate", "ib_port_xmit_data_delta"),
            ("ib_xmit_packets_rate", "ib_port_xmit_packets_delta"),
        ):
            data[target] = pd.to_numeric(data[source], errors="coerce").div(
                interval.replace(0, np.nan)
            )
    else:
        data[RATE_FEATURES] = np.nan
    data = _add_history_features(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "health_training_table.csv", index=False)
    evaluation, _ = _evaluate(data)

    final_model = _make_model()
    final_model.fit(data[HEALTH_FEATURES], data["target"])
    allreduce_threshold = evaluation["allreduce_pause_threshold"]["threshold"]
    setattr(final_model, "allreduce_pause_threshold", float(allreduce_threshold))
    joblib.dump(final_model, output_dir / "health_random_forest.joblib")

    forest = final_model.named_steps["classifier"]
    feature_importance = sorted(
        zip(HEALTH_FEATURES, forest.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    metrics: dict[str, object] = {
        "evaluation": "leave-one-complete-run-out",
        "runtime": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "rows_total": int(len(data)),
        "class_counts": {
            name: int((data["target"] == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "runs_by_source": {
            source: sorted(data.loc[data["source"] == source, "run_id"].unique())
            for source in sorted(data["source"].unique())
        },
        "label2_runs_skipped_missing_markers": skipped_label2,
        "features": HEALTH_FEATURES,
        "allreduce_pause_threshold": float(
            evaluation["allreduce_pause_threshold"]["threshold"]
        ),
        "top_feature_importances": [
            {"feature": name, "importance": float(importance)}
            for name, importance in feature_importance[:20]
        ],
        "evaluation_results": evaluation,
    }
    (output_dir / "health_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(train(args.root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
