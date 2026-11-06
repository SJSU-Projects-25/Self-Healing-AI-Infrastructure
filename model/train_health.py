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
from sklearn.metrics import classification_report, confusion_matrix
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
HEALTH_FEATURES = FEATURES + DERIVED_FEATURES


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


def _load_label1(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "data" / "Label1").glob("*/*_label1.csv")):
        run_dir = path.parent
        frame = pd.read_csv(path)
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
                SimpleImputer(strategy="median", add_indicator=True),
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
    logo = LeaveOneGroupOut()
    folds = 0
    for train_idx, test_idx in logo.split(x, y, groups):
        model = _make_model()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(x.iloc[test_idx])
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
    return (
        {
            "folds_evaluated": folds,
            "rows_evaluated": int(evaluated.sum()),
            "confusion_matrix": confusion_matrix(
                y[evaluated], predictions[evaluated], labels=[0, 1, 2]
            ).tolist(),
            "classification_report": report,
        },
        predictions,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "health_training_table.csv", index=False)
    evaluation, _ = _evaluate(data)

    final_model = _make_model()
    final_model.fit(data[HEALTH_FEATURES], data["target"])
    joblib.dump(final_model, output_dir / "health_random_forest.joblib")

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
