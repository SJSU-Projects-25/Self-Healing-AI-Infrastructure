"""Conservative row-to-event inference for the training-health model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .train_health import (
        CLASS_NAMES,
        DERIVED_FEATURES,
        HEALTH_FEATURES,
        HISTORY_FEATURES,
        RATE_FEATURES,
        ROLLING_WINDOWS,
        _add_history_features,
    )
    from .train_label2 import FEATURES
except ImportError:
    from train_health import (
        CLASS_NAMES,
        DERIVED_FEATURES,
        HEALTH_FEATURES,
        HISTORY_FEATURES,
        RATE_FEATURES,
        ROLLING_WINDOWS,
        _add_history_features,
    )
    from train_label2 import FEATURES

ISSUE_CLASSES = {1, 2}


def load_health_model(path: Path):
    """Load the fitted health pipeline from a trusted local path."""
    model = joblib.load(path)
    threshold = 0.5
    metrics_path = path.parent / "health_metrics.json"
    if metrics_path.exists():
        import json

        try:
            metrics = json.loads(metrics_path.read_text())
            top_level_threshold = metrics.get("allreduce_pause_threshold")
            if isinstance(top_level_threshold, (int, float)):
                threshold = float(top_level_threshold)
            else:
                nested_threshold = (
                    metrics.get("evaluation_results", {})
                    .get("allreduce_pause_threshold", {})
                    .get("threshold", 0.5)
                )
                threshold = float(nested_threshold)
        except (json.JSONDecodeError, TypeError, ValueError):
            threshold = 0.5
    setattr(model, "allreduce_pause_threshold", threshold)
    return model


def classify_rows(
    model,
    telemetry: pd.DataFrame,
    allreduce_pause_threshold: float | None = None,
) -> pd.DataFrame:
    """Add row-level class and probability columns to telemetry."""
    if allreduce_pause_threshold is None:
        allreduce_pause_threshold = float(
            getattr(model, "allreduce_pause_threshold", 0.5)
        )
    if not 0.0 <= allreduce_pause_threshold <= 1.0:
        raise ValueError("allreduce_pause_threshold must be between 0 and 1.")
    missing = sorted(set(FEATURES) - set(telemetry.columns))
    if missing:
        raise ValueError(f"Telemetry is missing required features: {missing}")
    if "timestamp" not in telemetry.columns:
        raise ValueError("Telemetry must contain timestamp.")

    result = telemetry.copy()
    result[FEATURES] = result[FEATURES].apply(pd.to_numeric, errors="coerce")
    sort_columns = ["timestamp"]
    if "node_id" in result.columns:
        sort_columns.insert(0, "node_id")
        result = result.sort_values(sort_columns)
        groups = result.groupby("node_id", sort=False)
    else:
        result = result.sort_values(sort_columns)
        groups = result.groupby(lambda _: 0, sort=False)
    for column in DERIVED_FEATURES:
        base = column.removesuffix("_change")
        result[column] = result[base] - groups[base].shift(1)
    if "polling_interval_actual" in result.columns:
        interval = pd.to_numeric(
            result["polling_interval_actual"], errors="coerce"
        )
        for target, source in (
            ("ib_xmit_wait_rate", "ib_port_xmit_wait_delta"),
            ("ib_xmit_data_rate", "ib_port_xmit_data_delta"),
            ("ib_xmit_packets_rate", "ib_port_xmit_packets_delta"),
        ):
            result[target] = result[source].div(interval.replace(0, np.nan))
    else:
        result[RATE_FEATURES] = np.nan
    result = _add_history_features(result)
    probabilities = model.predict_proba(result[HEALTH_FEATURES])
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    if 1 in model.classes_:
        pause_index = int(np.flatnonzero(model.classes_ == 1)[0])
        predictions = predictions.copy()
        predictions[
            probabilities[:, pause_index] >= allreduce_pause_threshold
        ] = 1
    result["predicted_class"] = predictions.astype(int)
    result["predicted_state"] = [
        CLASS_NAMES[int(value)] for value in predictions
    ]
    for index, name in enumerate(CLASS_NAMES):
        result[f"probability_{name}"] = probabilities[:, index]
    return result


def aggregate_events(
    classified: pd.DataFrame,
    min_consecutive: int = 3,
    max_gap_seconds: float = 1.5,
) -> pd.DataFrame:
    """Mark sustained issue predictions as actionable events.

    A single positive row is retained as an observation. It is not marked
    actionable unless the same issue class appears in consecutive samples
    on the same node with no excessive timestamp gap.
    """
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    required = {"timestamp", "predicted_class"}
    missing = sorted(required - set(classified.columns))
    if missing:
        raise ValueError(f"Classified data is missing columns: {missing}")

    result = classified.copy()
    node_column = "node_id" if "node_id" in result.columns else None
    if node_column is None:
        result["_event_node"] = "global"
        node_column = "_event_node"
    result["_event_timestamp"] = pd.to_numeric(
        result["timestamp"], errors="coerce"
    )
    if result["_event_timestamp"].isna().any():
        raise ValueError("Cannot aggregate rows with invalid timestamps.")
    result = result.sort_values([node_column, "_event_timestamp"]).copy()

    group_key = (
        result[node_column].astype(str)
        + "|"
        + result["predicted_class"].astype(str)
    )
    time_gap = result.groupby(group_key, sort=False)["_event_timestamp"].diff()
    result["_sequence_break"] = (
        time_gap.isna()
        | (time_gap > max_gap_seconds)
        | ~result["predicted_class"].isin(ISSUE_CLASSES)
    )
    result["_sequence_id"] = result["_sequence_break"].cumsum()
    sequence_sizes = result.groupby("_sequence_id")["predicted_class"].transform(
        "size"
    )
    result["event_id"] = np.where(
        result["predicted_class"].isin(ISSUE_CLASSES),
        "event-" + result["_sequence_id"].astype(str),
        "",
    )
    result["event_state"] = np.where(
        result["predicted_class"].isin(ISSUE_CLASSES)
        & (sequence_sizes >= min_consecutive),
        "actionable",
        "observation",
    )
    result["event_node"] = result[node_column].astype(str)
    return result.drop(
        columns=["_event_node", "_event_timestamp", "_sequence_break", "_sequence_id"],
        errors="ignore",
    )
