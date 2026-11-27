#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import time

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


def read_current_telemetry(run_dir: Path) -> pd.DataFrame:
    frames = []

    for path in sorted(run_dir.glob("telemetry_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        if frame.empty:
            continue

        frame = _join_step_times(frame, run_dir)
        frame["run_id"] = run_dir.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_numeric(
        data["timestamp"],
        errors="coerce",
    )

    data = data.dropna(subset=["timestamp"])
    data = data.drop_duplicates(
        subset=["node_id", "timestamp"],
        keep="last",
    )

    return data.sort_values(
        ["timestamp", "node_id"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-seconds", type=float, default=5.0)
    parser.add_argument("--history-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_health_model(
        Path("model/health_random_forest.joblib")
    )

    first_timestamp = None
    processed_batches = set()
    all_predictions = []

    print("Live watcher started")
    print("Run directory:", run_dir)
    print("Output directory:", output_dir)

    while True:
        data = read_current_telemetry(run_dir)

        if data.empty:
            time.sleep(args.poll_seconds)
            continue

        if first_timestamp is None:
            first_timestamp = float(data["timestamp"].min())

        data["_batch_id"] = (
            (data["timestamp"] - first_timestamp)
            // args.batch_seconds
        ).astype(int)

        latest_timestamp = float(data["timestamp"].max())

        # A batch is complete only after its end timestamp arrives.
        last_complete_batch = int(
            (latest_timestamp - first_timestamp)
            // args.batch_seconds
        ) - 1

        for batch_id in range(last_complete_batch + 1):
            if batch_id in processed_batches:
                continue

            batch_end = (
                first_timestamp
                + (batch_id + 1) * args.batch_seconds
            )

            history_start = (
                batch_end
                - args.history_seconds
                - args.batch_seconds
                - 1.0
            )

            history = data[
                (data["timestamp"] <= batch_end)
                & (data["timestamp"] >= history_start)
            ].copy()

            if history.empty:
                continue

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
                "batch_sequence": batch_id + 1,
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
            }

            batch_path = output_dir / (
                f"batch_{batch_id:06d}.json"
            )

            batch_path.write_text(
                json.dumps(payload, indent=2)
            )

            prediction_path = (
                output_dir / "live_predictions.csv"
            )

            events.to_csv(
                prediction_path,
                index=False,
            )

            states = (
            current_events["predicted_state"]
            .value_counts()
            .to_dict()
            )

            print(
                f"processed batch={batch_id:04d} "
                f"samples={len(current)} "
                f"states={states}",
                flush=True,
            )
            processed_batches.add(batch_id)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
