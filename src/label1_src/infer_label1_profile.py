#!/usr/bin/env python3
"""Infer Label 1 pause behavior and create model-ready window features.

This script is intentionally dependency-free: it uses only Python's standard
library so it can run on an HPC login/compute node without pandas.

Usage:
    python3 infer_label1_profile.py results/label1_165397

Outputs are written inside the results directory unless --output-dir is used:
    label1_profile_summary.json
    label1_window_features.csv

Important:
  * node_0_0.log provides observed DDP step times.
  * label1_active_*.csv files are used only to validate/label training data.
    They must not be used as inference features in a live detector.
  * pause_every and pause_seconds printed by the workload are metadata, not
    model inputs. The inferred values below come from observed step-time data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STEP_RE = re.compile(
    r"\[rank\s+0\]\s+step=\s*(?P<step>\d+)"
    r".*?step_time_ms=\s*(?P<ms>[0-9.]+)"
    r".*?wall=\s*(?P<wall>[0-9.]+)"
)

OLD_STEP_RE = re.compile(
    r"\[rank\s+0\]\s+epoch=\s*(?P<step>\d+)"
    r"\s+step=\s*(?P<step2>\d+)"
    r".*?step_time=\s*(?P<ms>[0-9.]+)ms"
)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def std(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def numeric(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def parse_steps(log_path: Path) -> List[dict]:
    steps: List[dict] = []
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if match:
                steps.append(
                    {
                        "step": int(match.group("step")),
                        "step_time_ms": float(match.group("ms")),
                        "wall": float(match.group("wall")),
                    }
                )
                continue

            # Older workloads may not include a wall timestamp. They can still
            # be summarized, but cannot be aligned precisely to marker files.
            old = OLD_STEP_RE.search(line)
            if old:
                steps.append(
                    {
                        "step": int(old.group("step")),
                        "step_time_ms": float(old.group("ms")),
                        "wall": None,
                    }
                )
    return steps


def find_rank0_log(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("node_0_0*.log"))
    if not candidates:
        candidates = sorted(results_dir.glob("node_0*.log"))
    return candidates[0] if candidates else None


def find_marker_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("label1_active_*.csv"))


def marker_intervals(path: Path) -> List[dict]:
    rows = read_csv_rows(path)
    active = [r for r in rows if str(r.get("label1_active", "")) == "1"]
    inactive = [r for r in rows if str(r.get("label1_active", "")) == "0"]
    intervals = []
    for row in active:
        start = numeric(row.get("timestamp"))
        if start is None:
            continue
        following = [numeric(r.get("timestamp")) for r in inactive]
        following = [x for x in following if x is not None and x > start]
        if not following:
            continue
        end = min(following)
        intervals.append(
            {
                "start": start,
                "end": end,
                "duration_s": end - start,
                "pause_step": numeric(row.get("pause_step")),
                "pause_seconds": numeric(row.get("pause_seconds")),
                "source": path.name,
            }
        )
    return intervals


def in_interval(timestamp: Optional[float], intervals: Sequence[dict]) -> bool:
    if timestamp is None:
        return False
    return any(x["start"] <= timestamp <= x["end"] for x in intervals)


def overlaps(start: float, end: float, interval: dict) -> bool:
    return start <= interval["end"] and end >= interval["start"]


def all_intervals(marker_files: Sequence[Path]) -> List[dict]:
    result: List[dict] = []
    for path in marker_files:
        result.extend(marker_intervals(path))
    # Both nodes normally write the same pause schedule. Deduplicate those
    # mirrored intervals so the job is counted once rather than once per node.
    result.sort(key=lambda x: x["start"])
    unique: List[dict] = []
    seen = set()
    for interval in result:
        pause_step = interval.get("pause_step")
        if pause_step is not None:
            # The same logical event can have slightly different timestamps
            # on different nodes, but pause_step is the stable event key.
            key = ("pause_step", int(pause_step))
        else:
            key = ("timestamp", round(interval["start"], 2))
        if key not in seen:
            unique.append(interval)
            seen.add(key)
    return unique


def infer_profile(steps: Sequence[dict], warmup_steps: int, min_spike_ms: float) -> dict:
    usable = [x for x in steps if x["step"] > warmup_steps]
    times = [x["step_time_ms"] for x in usable]
    baseline = percentile(times, 0.50)
    spikes = [x for x in usable if x["step_time_ms"] >= min_spike_ms]
    if not spikes or baseline is None:
        return {
            "baseline_step_time_ms": baseline,
            "spike_count": len(spikes),
            "spike_steps": [x["step"] for x in spikes],
            "inferred_pause_every_steps": None,
            "inferred_spike_interval_seconds": None,
            "estimated_pause_duration_seconds": None,
        }

    step_gaps = [b["step"] - a["step"] for a, b in zip(spikes, spikes[1:])]
    wall_gaps = [
        b["wall"] - a["wall"]
        for a, b in zip(spikes, spikes[1:])
        if a["wall"] is not None and b["wall"] is not None
    ]
    spike_times = [x["step_time_ms"] for x in spikes]
    return {
        "baseline_step_time_ms": baseline,
        "spike_count": len(spikes),
        "spike_steps": [x["step"] for x in spikes],
        "spike_time_mean_ms": mean(spike_times),
        "spike_time_p95_ms": percentile(spike_times, 0.95),
        "spike_time_max_ms": max(spike_times),
        "inferred_pause_every_steps": percentile(step_gaps, 0.50),
        "spike_interval_steps_std": std(step_gaps),
        "inferred_spike_interval_seconds": percentile(wall_gaps, 0.50),
        "spike_interval_seconds_std": std(wall_gaps),
        "estimated_pause_duration_seconds": max(spike_times) / 1000.0 - baseline / 1000.0,
    }


def aggregate(values: Sequence[float], prefix: str) -> dict:
    if not values:
        return {}
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_std": std(values),
        f"{prefix}_p95": percentile(values, 0.95),
        f"{prefix}_max": max(values),
    }


TELEMETRY_COLUMNS = [
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "ib_port_xmit_wait_delta",
    "ib_port_xmit_data_delta",
    "ib_port_rcv_data_delta",
    "ib_port_xmit_packets_delta",
    "ib_port_rcv_packets_delta",
    "ib_port_xmit_discards_delta",
    "ib_port_rcv_errors_delta",
    "ib_sq_num_rnr_delta",
    "ib_sq_num_to_delta",
    "ib_rq_num_oos_delta",
]


def load_telemetry(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(results_dir.glob("telemetry_*.csv")):
        for row in read_csv_rows(path):
            timestamp = numeric(row.get("timestamp"))
            if timestamp is None:
                continue
            row["timestamp"] = timestamp
            row["source_file"] = path.name
            row["source_node"] = row.get("node_id", path.name)
            for column in TELEMETRY_COLUMNS:
                row[column] = numeric(row.get(column))
            rows.append(row)
    return rows


def window_features(
    steps: Sequence[dict],
    telemetry: Sequence[dict],
    intervals: Sequence[dict],
    window_seconds: float,
    stride_seconds: float,
    min_spike_ms: float,
    baseline_ms: Optional[float],
) -> List[dict]:
    timed_steps = [x for x in steps if x["wall"] is not None]
    if not timed_steps:
        return []
    start = timed_steps[0]["wall"]
    finish = timed_steps[-1]["wall"]
    result = []
    current = start
    while current <= finish:
        end = current + window_seconds
        ws = [x for x in timed_steps if current <= x["wall"] < end]
        ts = [x for x in telemetry if current <= x["timestamp"] < end]
        if ws:
            values = [x["step_time_ms"] for x in ws]
            spikes = [x for x in ws if x["step_time_ms"] >= min_spike_ms]
            spike_steps = [x["step"] for x in spikes]
            spike_gaps = [b - a for a, b in zip(spike_steps, spike_steps[1:])]
            spike_wall_gaps = [
                b["wall"] - a["wall"] for a, b in zip(spikes, spikes[1:])
            ]
            row = {
                "window_start": current,
                "window_end": end,
                "step_count": len(ws),
                "step_time_mean_ms": mean(values),
                "step_time_median_ms": percentile(values, 0.50),
                "step_time_std_ms": std(values),
                "step_time_p95_ms": percentile(values, 0.95),
                "step_time_max_ms": max(values),
                "step_time_spike_count": len(spikes),
                "step_time_spike_fraction": len(spikes) / len(ws),
                "spike_interval_steps_mean": mean(spike_gaps),
                "spike_interval_steps_std": std(spike_gaps),
                "spike_interval_seconds_mean": mean(spike_wall_gaps),
                "spike_interval_seconds_std": std(spike_wall_gaps),
                "estimated_pause_duration_ms": (
                    max(x["step_time_ms"] for x in spikes) - baseline_ms
                    if spikes and baseline_ms is not None
                    else None
                ),
                "marker_active": int(
                    any(overlaps(current, end, interval) for interval in intervals)
                ),
                "marker_pause_events": sum(
                    1 for interval in intervals if current <= interval["start"] < end
                ),
            }

            # Aggregate telemetry across nodes. Node names are not used as
            # predictive features; min/max/range capture rank asymmetry.
            by_column: Dict[str, List[float]] = {column: [] for column in TELEMETRY_COLUMNS}
            for sample in ts:
                for column in TELEMETRY_COLUMNS:
                    value = sample.get(column)
                    if value is not None:
                        by_column[column].append(value)
            for column, vals in by_column.items():
                row.update(aggregate(vals, column))

            gpu_by_node: Dict[str, List[float]] = {}
            for sample in ts:
                value = sample.get("gpu_util_pct")
                if value is not None:
                    gpu_by_node.setdefault(str(sample["source_node"]), []).append(value)
            gpu_means = [mean(values) for values in gpu_by_node.values() if values]
            if gpu_means:
                row["gpu_util_node_min"] = min(gpu_means)
                row["gpu_util_node_max"] = max(gpu_means)
                row["gpu_util_node_range"] = max(gpu_means) - min(gpu_means)
                row["gpu_util_node_mean"] = mean(gpu_means)

            result.append(row)
        current += stride_seconds
    return result


def clean_for_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    return value


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--profile-label",
        default="",
        help="Ground-truth Label 1 profile name, e.g. standard or very_frequent",
    )
    parser.add_argument(
        "--class-label",
        default="label1",
        help="Ground-truth class label stored as metadata; default: label1",
    )
    parser.add_argument(
        "--min-spike-ms",
        type=float,
        default=100.0,
        help="Minimum DDP step time considered a pause spike; default: 100 ms",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = find_rank0_log(results_dir)
    if log_path is None:
        raise SystemExit(f"No rank-0 log found in {results_dir}")

    steps = parse_steps(log_path)
    if not steps:
        raise SystemExit(f"No step-time records found in {log_path}")

    marker_files = find_marker_files(results_dir)
    intervals = all_intervals(marker_files)
    telemetry = load_telemetry(results_dir)
    profile = infer_profile(steps, args.warmup_steps, args.min_spike_ms)
    baseline_ms = profile.get("baseline_step_time_ms")

    windows = window_features(
        steps=steps,
        telemetry=telemetry,
        intervals=intervals,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_spike_ms=args.min_spike_ms,
        baseline_ms=baseline_ms,
    )

    # These are targets/metadata for training, not predictive features. Keeping
    # them in the generated file makes each job self-describing; downstream
    # training code must drop them before fitting the model.
    for row in windows:
        row["run_id"] = results_dir.name
        row["class_label"] = args.class_label
        row["profile_label"] = args.profile_label

    marker_pause_steps = [
        x["pause_step"] for x in intervals if x.get("pause_step") is not None
    ]
    marker_pause_seconds = [
        x["pause_seconds"] for x in intervals if x.get("pause_seconds") is not None
    ]
    marker_durations = [x["duration_s"] for x in intervals]
    marker_step_gaps = [
        b - a for a, b in zip(marker_pause_steps, marker_pause_steps[1:])
    ]

    summary = {
        "results_dir": str(results_dir),
        "rank0_log": log_path.name,
        "telemetry_files": [p.name for p in sorted(results_dir.glob("telemetry_*.csv"))],
        "marker_files": [p.name for p in marker_files],
        "step_count": len(steps),
        "first_step": steps[0]["step"],
        "last_step": steps[-1]["step"],
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "min_spike_ms": args.min_spike_ms,
        "class_label": args.class_label,
        "profile_label": args.profile_label,
        "inferred_from_step_times": profile,
        "ground_truth_marker_summary": {
            "pause_event_count": len(intervals),
            "pause_duration_mean_seconds": mean(marker_durations),
            "pause_duration_min_seconds": min(marker_durations) if marker_durations else None,
            "pause_duration_max_seconds": max(marker_durations) if marker_durations else None,
            "configured_pause_seconds_mean": mean(marker_pause_seconds),
            "configured_pause_every_steps": percentile(marker_step_gaps, 0.50),
        },
        "model_input_guidance": {
            "use_as_features": [
                "step_time_mean_ms",
                "step_time_std_ms",
                "step_time_p95_ms",
                "step_time_max_ms",
                "step_time_spike_count",
                "step_time_spike_fraction",
                "spike_interval_steps_mean",
                "spike_interval_seconds_mean",
                "gpu_util_node_min",
                "gpu_util_node_max",
                "gpu_util_node_range",
                "ib_port_xmit_wait_delta_mean",
                "ib_port_xmit_data_delta_mean",
                "ib_port_xmit_packets_delta_mean",
            ],
            "do_not_use_as_features": [
                "label1_active",
                "pause_seconds",
                "pause_step",
                "pause_every",
                "pause_rank",
                "hostname",
                "job_id",
                "timestamp",
            ],
        },
    }

    summary_path = output_dir / "label1_profile_summary.json"
    features_path = output_dir / "label1_window_features.csv"
    summary_path.write_text(
        json.dumps(clean_for_json(summary), indent=2), encoding="utf-8"
    )
    write_csv(features_path, windows)

    inferred = profile.get("inferred_pause_every_steps")
    estimated = profile.get("estimated_pause_duration_seconds")
    print(f"results_dir={results_dir}")
    print(f"step_count={len(steps)} spikes={profile.get('spike_count', 0)}")
    print(f"inferred_pause_every_steps={inferred}")
    print(f"estimated_pause_duration_seconds={estimated}")
    print(f"wrote={summary_path}")
    print(f"wrote={features_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
