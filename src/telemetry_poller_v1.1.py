#!/usr/bin/env python3
"""
LENS POC — Telemetry Poller
===========================
Polls GPU metrics (via nvidia-smi) and InfiniBand counters
(via /sys/class/infiniband) at a configurable interval and
writes timestamped rows to a per-node CSV file.

No root/sudo required. Works with standard user permissions
on SJSU HPC3.

Output columns:
  timestamp, node_id, job_id,
  gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb,
  ib_port_xmit_data, ib_port_rcv_data,
  ib_port_xmit_wait, ib_port_xmit_data_delta,
  ib_port_rcv_data_delta, ib_port_xmit_wait_delta
"""

import argparse
import csv
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── Graceful shutdown ────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── GPU polling ──────────────────────────────────────────────────────────────

def poll_gpu():
    """
    Returns one dict per GPU on this node using nvidia-smi.
    Falls back gracefully if nvidia-smi is unavailable.
    """
    query = (
        "index,utilization.gpu,memory.used,memory.total"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, timeout=5, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[WARN] nvidia-smi failed: {e}", file=sys.stderr)
        return []

    results = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            results.append({
                "gpu_index":        int(parts[0]),
                "gpu_util_pct":     float(parts[1]),
                "gpu_mem_used_mb":  float(parts[2]),
                "gpu_mem_total_mb": float(parts[3]),
            })
        except ValueError:
            continue
    return results


# ── InfiniBand counter polling ───────────────────────────────────────────────

IB_COUNTER_NAMES = [
    "port_xmit_data",
    "port_rcv_data",
    "port_xmit_wait",
]

def _read_ib_counter(device: str, port: int, counter: str):
    """Read a single IB counter from sysfs. Returns None on failure."""
    path = Path(f"/sys/class/infiniband/{device}/ports/{port}/counters/{counter}")
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError) as e:
        print(f"[WARN] Cannot read IB counter {counter}: {e}", file=sys.stderr)
        return None

def poll_ib(device: str, port: int) -> dict:
    """Returns a dict of all IB counters for the given device/port."""
    return {
        name: _read_ib_counter(device, port, name)
        for name in IB_COUNTER_NAMES
    }

def detect_ib_device():
    """Auto-detect the first available IB device if mlx5_0 is not present."""
    base = Path("/sys/class/infiniband")
    if not base.exists():
        return None
    devices = sorted(base.iterdir())
    return devices[0].name if devices else None


# ── CSV writer ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp",
    "node_id",
    "job_id",
    "gpu_index",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "ib_port_xmit_data",
    "ib_port_rcv_data",
    "ib_port_xmit_wait",
    "ib_port_xmit_data_delta",
    "ib_port_rcv_data_delta",
    "ib_port_xmit_wait_delta",
]

def open_csv(output_dir: str, node_id: str, job_id: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(output_dir) / f"telemetry_{node_id}_{job_id}.csv"
    f = open(fname, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    print(f"[INFO] Writing telemetry to {fname}", file=sys.stderr)
    return f, writer


# ── Main polling loop ────────────────────────────────────────────────────────

def run_poller(args):
    node_id  = socket.gethostname()
    job_id   = os.environ.get("SLURM_JOB_ID", "local")

    # Resolve IB device
    ib_device = args.ib_device
    if not Path(f"/sys/class/infiniband/{ib_device}").exists():
        detected = detect_ib_device()
        if detected:
            print(f"[WARN] {ib_device} not found; using {detected}", file=sys.stderr)
            ib_device = detected
        else:
            print("[WARN] No InfiniBand device found. IB counters will be null.", file=sys.stderr)
            ib_device = None

    f, writer = open_csv(args.output_dir, node_id, job_id)

    prev_ib: dict = {}
    start_time = time.monotonic()
    deadline   = start_time + args.duration

    print(f"[INFO] Poller started on {node_id} | IB={ib_device}:{args.ib_port} | interval={args.interval}s", file=sys.stderr)

    try:
        while not _shutdown and time.monotonic() < deadline:
            ts = time.time()

            # -- GPU
            gpu_rows = poll_gpu()
            if not gpu_rows:
                # Insert a placeholder row so IB data is still captured
                gpu_rows = [{
                    "gpu_index": -1,
                    "gpu_util_pct": None,
                    "gpu_mem_used_mb": None,
                    "gpu_mem_total_mb": None,
                }]

            # -- IB
            ib_now = poll_ib(ib_device, args.ib_port) if ib_device else {}
            ib_delta = {}
            if prev_ib:
                for key in IB_COUNTER_NAMES:
                    cur  = ib_now.get(key)
                    prev = prev_ib.get(key)
                    ib_delta[key] = (cur - prev) if (cur is not None and prev is not None) else None
            prev_ib = ib_now

            # -- Write one row per GPU (IB data is per-node, same for all GPUs)
            for gpu in gpu_rows:
                row = {
                    "timestamp":               ts,
                    "node_id":                 node_id,
                    "job_id":                  job_id,
                    "gpu_index":               gpu["gpu_index"],
                    "gpu_util_pct":            gpu["gpu_util_pct"],
                    "gpu_mem_used_mb":         gpu["gpu_mem_used_mb"],
                    "gpu_mem_total_mb":        gpu["gpu_mem_total_mb"],
                    "ib_port_xmit_data":       ib_now.get("port_xmit_data"),
                    "ib_port_rcv_data":        ib_now.get("port_rcv_data"),
                    "ib_port_xmit_wait":       ib_now.get("port_xmit_wait"),
                    "ib_port_xmit_data_delta": ib_delta.get("port_xmit_data"),
                    "ib_port_rcv_data_delta":  ib_delta.get("port_rcv_data"),
                    "ib_port_xmit_wait_delta": ib_delta.get("port_xmit_wait"),
                }
                writer.writerow(row)

            f.flush()  # ensure data is written even if job is killed

            # -- Sleep for the remainder of the interval
            elapsed = time.time() - ts
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)

    finally:
        f.close()
        print(f"[INFO] Poller stopped on {node_id}. Rows written to CSV.", file=sys.stderr)


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LENS telemetry poller")
    p.add_argument("--output-dir",  default="./results",  help="Directory for CSV output")
    p.add_argument("--interval",    type=float, default=0.5, help="Polling interval in seconds")
    p.add_argument("--duration",    type=float, default=1700, help="Max run time in seconds")
    p.add_argument("--ib-device",   default="mlx5_0",     help="InfiniBand device name")
    p.add_argument("--ib-port",     type=int,   default=1, help="InfiniBand port number")
    return p.parse_args()


if __name__ == "__main__":
    run_poller(parse_args())
