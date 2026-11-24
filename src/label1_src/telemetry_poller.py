#!/usr/bin/env python3
"""
LENS POC — Telemetry Poller v2
================================
Polls GPU metrics (via nvidia-smi) and InfiniBand counters
(via /sys/class/infiniband) at a configurable interval and
writes timestamped rows to a per-node CSV file.

No root/sudo required. Works with standard user permissions
on SJSU HPC3.

Counter Groups:
  GPU:          utilization, memory
  IB Throughput:  xmit_data, rcv_data, xmit_packets, rcv_packets
  IB Congestion:  xmit_wait, xmit_discards, excessive_buffer_overrun
  IB Link Health: link_downed, link_error_recovery, symbol_error,
                  local_link_integrity_errors, port_rcv_remote_physical_errors
  IB Errors:      port_rcv_errors
  HW RDMA:        sq_num_rnr, sq_num_to, rq_num_oos, sq_num_oos
  Poller Health:  polling_interval_actual (jitter measurement)
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


# ── Counter definitions ──────────────────────────────────────────────────────

# Standard IB counters — /sys/class/infiniband/<dev>/ports/<port>/counters/
IB_STANDARD_COUNTERS = [
    # Throughput
    "port_xmit_data",
    "port_rcv_data",
    "port_xmit_packets",
    "port_rcv_packets",
    # Congestion
    "port_xmit_wait",
    "port_xmit_discards",
    "excessive_buffer_overrun_errors",
    # Link health
    "link_downed",
    "link_error_recovery",
    "symbol_error",
    "local_link_integrity_errors",
    "port_rcv_remote_physical_errors",
    # Errors
    "port_rcv_errors",
]

# Hardware RDMA counters — /sys/class/infiniband/<dev>/ports/<port>/hw_counters/
IB_HW_COUNTERS = [
    "sq_num_rnr",    # Receiver Not Ready retransmits — congestion signal
    "sq_num_to",     # Send queue timeouts — serious congestion/link issue
    "rq_num_oos",    # Receive queue out-of-sequence
    "sq_num_oos",    # Send queue out-of-sequence
]

# All counter names combined (for CSV headers)
ALL_IB_COUNTERS = IB_STANDARD_COUNTERS + IB_HW_COUNTERS

# Counters we compute deltas for (high-frequency changing counters)
DELTA_COUNTERS = [
    "port_xmit_data",
    "port_rcv_data",
    "port_xmit_packets",
    "port_rcv_packets",
    "port_xmit_wait",
    "port_xmit_discards",
    "excessive_buffer_overrun_errors",
    "symbol_error",
    "port_rcv_errors",
    "sq_num_rnr",
    "sq_num_to",
    "rq_num_oos",
    "sq_num_oos",
]

# Counters we report absolute values only (event counters — rare changes)
ABSOLUTE_ONLY_COUNTERS = [
    "link_downed",
    "link_error_recovery",
    "local_link_integrity_errors",
    "port_rcv_remote_physical_errors",
]


# ── GPU polling ──────────────────────────────────────────────────────────────

def poll_gpu():
    """
    Returns one dict per GPU on this node using nvidia-smi.
    Falls back gracefully if nvidia-smi is unavailable.
    """
    query = "index,utilization.gpu,memory.used,memory.total"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, timeout=5, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as e:
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

def _read_counter(path: Path):
    """Read a single counter from sysfs. Returns None on failure."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None

def poll_ib(device: str, port: int) -> dict:
    """
    Returns dict of all IB counters for the given device/port.
    Reads both standard counters/ and hw_counters/ directories.
    """
    base = Path(f"/sys/class/infiniband/{device}/ports/{port}")
    results = {}

    # Standard counters
    for name in IB_STANDARD_COUNTERS:
        results[name] = _read_counter(base / "counters" / name)

    # Hardware RDMA counters
    hw_base = base / "hw_counters"
    for name in IB_HW_COUNTERS:
        results[name] = _read_counter(hw_base / name)

    return results

def detect_ib_device():
    """Auto-detect the first available IB device."""
    base = Path("/sys/class/infiniband")
    if not base.exists():
        return None
    devices = sorted(base.iterdir())
    return devices[0].name if devices else None

def check_counter_overflow(current, previous, counter_name, bits=32):
    """
    Detect counter overflow and return corrected delta.
    IB counters are typically 32-bit (overflow at 2^32).
    port_xmit_data and port_rcv_data are 64-bit on mlx5,
    but 32-bit on mlx4.
    """
    if current is None or previous is None:
        return None

    max_val = (2 ** bits)

    # Detect saturated counter (e.g. port_xmit_wait = 4294967295)
    if previous == max_val - 1 and current == max_val - 1:
        return 0  # Counter is saturated, delta is meaningless

    # Normal delta
    if current >= previous:
        return current - previous
    else:
        # Overflow occurred
        print(f"[WARN] Counter overflow detected on {counter_name}: "
              f"{previous} -> {current}", file=sys.stderr)
        return (max_val - previous) + current


# ── CSV writer ───────────────────────────────────────────────────────────────

def build_fieldnames():
    """Build CSV column names dynamically from counter lists."""
    fields = [
        # Metadata
        "timestamp",
        "node_id",
        "job_id",
        "polling_interval_actual",  # jitter measurement — Task #6
        # GPU
        "gpu_index",
        "gpu_util_pct",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
    ]

    # IB absolute values
    for name in ALL_IB_COUNTERS:
        fields.append(f"ib_{name}")

    # IB deltas for high-frequency counters
    for name in DELTA_COUNTERS:
        fields.append(f"ib_{name}_delta")

    return fields

FIELDNAMES = build_fieldnames()

def open_csv(output_dir: str, node_id: str, job_id: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(output_dir) / f"telemetry_{node_id}_{job_id}.csv"
    f = open(fname, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    print(f"[INFO] Writing telemetry to {fname}", file=sys.stderr)
    return f, writer


# ── Main polling loop ────────────────────────────────────────────────────────

def run_poller(args):
    node_id = socket.gethostname()
    job_id  = os.environ.get("SLURM_JOB_ID", "local")

    # Resolve IB device
    ib_device = args.ib_device
    if not Path(f"/sys/class/infiniband/{ib_device}").exists():
        detected = detect_ib_device()
        if detected:
            print(f"[WARN] {ib_device} not found; using {detected}",
                  file=sys.stderr)
            ib_device = detected
        else:
            print("[WARN] No InfiniBand device found. IB counters will be null.",
                  file=sys.stderr)
            ib_device = None

    # Log which hw_counters are actually available
    if ib_device:
        hw_path = Path(f"/sys/class/infiniband/{ib_device}/ports/{args.ib_port}/hw_counters")
        if hw_path.exists():
            available_hw = [p.name for p in hw_path.iterdir()]
            print(f"[INFO] hw_counters available: {sorted(available_hw)}",
                  file=sys.stderr)
        else:
            print("[WARN] hw_counters directory not found — RDMA counters will be null",
                  file=sys.stderr)

    f, writer = open_csv(args.output_dir, node_id, job_id)

    prev_ib   = {}
    prev_time = None
    start_time = time.monotonic()
    deadline   = start_time + args.duration

    print(
        f"[INFO] Poller v2 started on {node_id} | "
        f"IB={ib_device}:{args.ib_port} | "
        f"interval={args.interval}s | "
        f"counters={len(ALL_IB_COUNTERS)} IB + {len(IB_HW_COUNTERS)} HW",
        file=sys.stderr
    )

    try:
        while not _shutdown and time.monotonic() < deadline:
            poll_start = time.time()

            # -- Measure actual polling interval (Task #6 — jitter)
            actual_interval = (poll_start - prev_time) if prev_time else None
            prev_time = poll_start

            # -- GPU
            gpu_rows = poll_gpu()
            if not gpu_rows:
                gpu_rows = [{
                    "gpu_index": -1,
                    "gpu_util_pct": None,
                    "gpu_mem_used_mb": None,
                    "gpu_mem_total_mb": None,
                }]

            # -- IB
            ib_now = poll_ib(ib_device, args.ib_port) if ib_device else {}

            # -- Compute deltas with overflow detection
            ib_delta = {}
            if prev_ib:
                for name in DELTA_COUNTERS:
                    # Use 64-bit for data counters, 32-bit for others
                    bits = 64 if "data" in name else 32
                    ib_delta[name] = check_counter_overflow(
                        ib_now.get(name),
                        prev_ib.get(name),
                        name,
                        bits=bits
                    )
            prev_ib = ib_now.copy()

            # -- Write one row per GPU
            for gpu in gpu_rows:
                row = {
                    # Metadata
                    "timestamp":               poll_start,
                    "node_id":                 node_id,
                    "job_id":                  job_id,
                    "polling_interval_actual": round(actual_interval, 4)
                                               if actual_interval else None,
                    # GPU
                    "gpu_index":               gpu["gpu_index"],
                    "gpu_util_pct":            gpu["gpu_util_pct"],
                    "gpu_mem_used_mb":         gpu["gpu_mem_used_mb"],
                    "gpu_mem_total_mb":        gpu["gpu_mem_total_mb"],
                }

                # IB absolute values
                for name in ALL_IB_COUNTERS:
                    row[f"ib_{name}"] = ib_now.get(name)

                # IB deltas
                for name in DELTA_COUNTERS:
                    row[f"ib_{name}_delta"] = ib_delta.get(name)

                writer.writerow(row)

            f.flush()

            # -- Sleep for remainder of interval
            elapsed   = time.time() - poll_start
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)

    finally:
        f.close()
        print(f"[INFO] Poller stopped on {node_id}.", file=sys.stderr)


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LENS telemetry poller v2")
    p.add_argument("--output-dir",  default="./results")
    p.add_argument("--interval",    type=float, default=0.5)
    p.add_argument("--duration",    type=float, default=1700)
    p.add_argument("--ib-device",   default="mlx4_0")
    p.add_argument("--ib-port",     type=int,   default=1)
    return p.parse_args()

if __name__ == "__main__":
    run_poller(parse_args())
