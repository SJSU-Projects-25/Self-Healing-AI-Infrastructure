#!/usr/bin/env python3
"""Label telemetry and DDP step-time logs from a fresh Label 1 run."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


def host_from_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("telemetry_"):
        stem = stem[len("telemetry_"):]
    return stem.split("_", 1)[0]


def marker_for_host(input_dir: Path, host: str) -> Path | None:
    matches = sorted(input_dir.glob(f"label1_active_{host}*.csv"))
    return matches[0] if matches else None


def load_markers(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["timestamp", "label1_active", "pause_seconds", "pause_step"])
    markers = pd.read_csv(path)
    for col in ("timestamp", "label1_active", "pause_seconds", "pause_step"):
        if col in markers:
            markers[col] = pd.to_numeric(markers[col], errors="coerce")
    return markers.dropna(subset=["timestamp"]).sort_values("timestamp")


def label_telemetry(path: Path, input_dir: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    host = host_from_file(path)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    markers = load_markers(marker_for_host(input_dir, host))
    if markers.empty:
        frame["label1_active"] = 0
        frame["pause_seconds"] = 0.0
        frame["pause_step"] = 0
    else:
        frame = pd.merge_asof(
            frame,
            markers[["timestamp", "label1_active", "pause_seconds", "pause_step"]],
            on="timestamp",
            direction="backward",
        )
        frame["label1_active"] = frame["label1_active"].fillna(0).astype(int)
        frame["pause_seconds"] = frame["pause_seconds"].fillna(0.0)
        frame["pause_step"] = frame["pause_step"].fillna(0).astype(int)
    frame["phase"] = frame["label1_active"].map({1: "Label 1 pause", 0: "Normal"})
    frame["label"] = frame["label1_active"].map({1: "allreduce_pause", 0: "healthy"})

    output = path.with_name(f"{path.stem}_label1.csv")
    frame.to_csv(output, index=False)
    return frame, host


def collect_step_times(input_dir: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"\[rank 0\] step=\s*(\d+).*?step_time_ms=([0-9.]+).*?wall=([0-9.]+)"
    )
    rows: list[dict[str, object]] = []
    for log in sorted(input_dir.glob("node_*.log")):
        for line in log.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                rows.append(
                    {
                        "timestamp": float(match.group(3)),
                        "node": "rank0",
                        "step": int(match.group(1)),
                        "step_time_ms": float(match.group(2)),
                    }
                )
    result = pd.DataFrame(rows, columns=["timestamp", "node", "step", "step_time_ms"])
    if not result.empty:
        result.to_csv(input_dir / f"step_time_{input_dir.name.split('_')[-1]}.csv", index=False)
    return result


def mean_text(frame: pd.DataFrame, column: str, mask: pd.Series) -> str:
    if column not in frame or not mask.any():
        return "n/a"
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return f"{values.mean():.3f} (n={len(values)})" if not values.empty else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir
    telemetry = sorted(
        p for p in input_dir.glob("telemetry_*.csv") if not p.name.endswith("_label1.csv")
    )
    if not telemetry:
        raise SystemExit(f"No telemetry_*.csv files found in {input_dir}")

    for path in telemetry:
        frame, host = label_telemetry(path, input_dir)
        pause = frame["label1_active"] == 1
        normal = ~pause
        print(f"{host}: rows={len(frame)} pause_rows={int(pause.sum())} normal_rows={int(normal.sum())}")
        print(f"  gpu_util_pct: normal={mean_text(frame, 'gpu_util_pct', normal)} pause={mean_text(frame, 'gpu_util_pct', pause)}")
        print(f"  ib_port_xmit_wait_delta: normal={mean_text(frame, 'ib_port_xmit_wait_delta', normal)} pause={mean_text(frame, 'ib_port_xmit_wait_delta', pause)}")

    steps = collect_step_times(input_dir)
    if not steps.empty:
        print(f"step_time_ms: mean={steps['step_time_ms'].mean():.3f} n={len(steps)}")
        print(f"  slowest={steps['step_time_ms'].max():.3f} ms")
    else:
        print("No rank-0 step-time lines found")


if __name__ == "__main__":
    main()
