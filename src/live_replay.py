#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import pandas as pd

from model.health_monitor import (
    aggregate_events,
    classify_rows,
    load_health_model,
)
from model.train_label2 import _join_step_times


RAW_FEATURES = [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-seconds", type=float, default=5.0)
    parser.add_argument("--history-seconds", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_health_model(
        Path("model/health_random_forest.joblib")
    )

    frames = []

    for telemetry_path in sorted(run_dir.glob("telemetry_*.csv")):
        frame = pd.read_csv(telemetry_path)

        # Align DDP step time with telemetry timestamps.
        frame = _join_step_times(frame, run_dir)

        # Required by the rolling-history implementation.
        frame["run_id"] = run_dir.name

        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No telemetry files found in {run_dir}")

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_numeric(
        data["timestamp"],
        errors="coerce",
    )
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values(
        ["timestamp", "node_id"]
    ).reset_index(drop=True)

    first_timestamp = float(data["timestamp"].min())

    data["_batch_id"] = (
        (data["timestamp"] - first_timestamp)
        // args.batch_seconds
    ).astype(int)

    all_predictions = []

    for batch_id in sorted(data["_batch_id"].unique()):
        batch_end = (
            first_timestamp
            + (int(batch_id) + 1) * args.batch_seconds
        )

        # Data available up to this point in the simulated stream.
        seen = data[
            data["timestamp"] <= batch_end
        ].copy()

        # Retain enough history for the 30-second features.
        history_start = (
            batch_end
            - args.history_seconds
            - args.batch_seconds
            - 1.0
        )

        history = seen[
            seen["timestamp"] >= history_start
        ].copy()

        classified_history = classify_rows(
            model,
            history,
        )

        current = classified_history[
            classified_history["_batch_id"] == batch_id
        ].copy()

        if current.empty:
            continue

        all_predictions.append(current)

        prediction_frame = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        events = aggregate_events(
            prediction_frame,
            min_consecutive=3,
            max_gap_seconds=1.5,
        )

        current_events = events[
            events["_batch_id"] == batch_id
        ].copy()

        sample_columns = [
            "timestamp",
            "node_id",
            "polling_interval_actual",
        ] + RAW_FEATURES

        sample_columns = [
            column
            for column in sample_columns
            if column in current.columns
        ]

        payload = {
            "schema_version": "health-v1",
            "run_id": run_dir.name,
            "batch_sequence": int(batch_id) + 1,
            "batch_start_timestamp": float(
                current["timestamp"].min()
            ),
            "batch_end_timestamp": float(
                current["timestamp"].max()
            ),
            "samples": json.loads(
                current[sample_columns].to_json(
                    orient="records"
                )
            ),
            "predictions": json.loads(
                current_events[
                    [
                        "timestamp",
                        "node_id",
                        "predicted_class",
                        "predicted_state",
                        "probability_healthy",
                        "probability_allreduce_pause",
                        "probability_ib_contention",
                        "event_state",
                        "event_id",
                    ]
                ].to_json(
                    orient="records"
                )
            ),
        }

        batch_path = output_dir / (
            f"batch_{int(batch_id):06d}.json"
        )

        batch_path.write_text(
            json.dumps(payload, indent=2)
        )

        states = (
            current_events["predicted_state"]
            .value_counts()
            .to_dict()
        )

        print(
            f"batch={int(batch_id):04d} "
            f"samples={len(current)} "
            f"states={states}"
        )

    if not all_predictions:
        raise RuntimeError("No predictions were generated")

    final_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    final_events = aggregate_events(
        final_predictions,
        min_consecutive=3,
        max_gap_seconds=1.5,
    )

    final_path = output_dir / "live_predictions.csv"
    final_events.to_csv(final_path, index=False)

    print()
    print("Streaming replay completed")
    print("Input rows:", len(data))
    print("Prediction rows:", len(final_events))
    print("Output:", final_path)
    print()
    print(final_events["predicted_state"].value_counts())


if __name__ == "__main__":
    main()
