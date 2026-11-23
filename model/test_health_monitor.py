"""Focused tests for temporal health-event aggregation."""

import pandas as pd
import numpy as np

from health_monitor import aggregate_events, classify_rows
from train_health import _add_history_features


def test_requires_consecutive_same_class_rows() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": [1.0, 1.5, 2.0, 2.5, 3.0],
            "node_id": ["g8"] * 5,
            "predicted_class": [0, 2, 2, 2, 0],
        }
    )
    result = aggregate_events(rows, min_consecutive=3)
    assert result["event_state"].tolist() == [
        "observation",
        "actionable",
        "actionable",
        "actionable",
        "observation",
    ]


def test_timestamp_gap_breaks_event() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": [1.0, 1.5, 4.0, 4.5],
            "node_id": ["g8"] * 4,
            "predicted_class": [2, 2, 2, 2],
        }
    )
    result = aggregate_events(rows, min_consecutive=3)
    assert result["event_state"].tolist() == ["observation"] * 4


def test_history_features_do_not_use_current_or_future_rows() -> None:
    rows = pd.DataFrame(
        {
            "run_id": ["run-a"] * 4,
            "node_id": ["g8"] * 4,
            "timestamp": [1.0, 1.5, 2.0, 5.0],
            "ddp_step_time_ms": [100.0, 200.0, 300.0, 1000.0],
            "gpu_util_pct": [90.0, 80.0, 70.0, 10.0],
            "ib_port_xmit_wait_delta": [1.0, 2.0, 3.0, 40.0],
            "ib_port_xmit_data_delta": [10.0, 20.0, 30.0, 400.0],
            "ib_port_xmit_packets_delta": [1.0, 2.0, 3.0, 40.0],
            "ib_port_rcv_data_delta": [10.0, 20.0, 30.0, 400.0],
            "ib_port_rcv_packets_delta": [1.0, 2.0, 3.0, 40.0],
        }
    )
    result = _add_history_features(rows)

    assert np.isnan(result.iloc[0]["step_time_5s_mean_ms"])
    assert result.iloc[1]["step_time_5s_mean_ms"] == 100.0
    assert result.iloc[2]["step_time_5s_mean_ms"] == 150.0
    assert result.iloc[3]["step_time_5s_mean_ms"] == 200.0


def test_classify_rows_rejects_invalid_pause_threshold() -> None:
    model = object()
    telemetry = pd.DataFrame(
        {
            "timestamp": [1.0],
            "gpu_util_pct": [90.0],
        }
    )

    try:
        classify_rows(model, telemetry, allreduce_pause_threshold=1.5)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("Expected invalid threshold to raise ValueError")
