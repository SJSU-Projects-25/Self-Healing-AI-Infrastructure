"""Focused tests for temporal health-event aggregation."""

import pandas as pd

from health_monitor import aggregate_events


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
