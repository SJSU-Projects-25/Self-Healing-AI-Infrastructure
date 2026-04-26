#!/usr/bin/env python3
"""
LENS POC — Pre-flight Sanity Check
====================================
Run this INTERACTIVELY on an HPC3 login node (or in an salloc session)
BEFORE submitting the full SLURM job.

Checks:
  1. Python version
  2. PyTorch + CUDA availability
  3. nvidia-smi accessibility and output
  4. InfiniBand sysfs counters (mlx5_0 or auto-detected)
  5. SLURM environment variables
  6. Output directory is writable

Usage:
  python3 preflight_check.py
  python3 preflight_check.py --ib-device mlx5_1   # if your device differs
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  !"

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

def check_python():
    section("1. Python version")
    v = sys.version_info
    if v >= (3, 10):
        print(f"{PASS} Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"{WARN} Python {v.major}.{v.minor}.{v.micro} — recommend 3.10+")

def check_torch():
    section("2. PyTorch + CUDA")
    try:
        import torch
        print(f"{PASS} PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory // (1024**2)
                print(f"{PASS} GPU {i}: {name} ({mem} MB)")
        else:
            print(f"{WARN} CUDA not available — GPU polling will still work via nvidia-smi")
    except ImportError:
        print(f"{FAIL} PyTorch not installed. Run: pip install torch")

def check_nvidia_smi():
    section("3. nvidia-smi")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            print(f"{PASS} GPU {parts[0]}: util={parts[1]}%  mem={parts[2]}/{parts[3]} MB")
    except FileNotFoundError:
        print(f"{FAIL} nvidia-smi not found — is CUDA module loaded?")
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} nvidia-smi error: {e}")

def check_ib(ib_device, ib_port):
    section(f"4. InfiniBand counters  ({ib_device} port {ib_port})")
    base = Path("/sys/class/infiniband")

    if not base.exists():
        print(f"{FAIL} /sys/class/infiniband not found")
        return

    devices = sorted(base.iterdir())
    if not devices:
        print(f"{FAIL} No IB devices found under /sys/class/infiniband")
        return

    available = [d.name for d in devices]
    print(f"{PASS} Available IB devices: {available}")

    # Fall back to auto-detect
    if not (base / ib_device).exists():
        ib_device = devices[0].name
        print(f"{WARN} Requested device not found — using {ib_device}")

    counters = ["port_xmit_data", "port_rcv_data", "port_xmit_wait"]
    for c in counters:
        path = base / ib_device / "ports" / str(ib_port) / "counters" / c
        try:
            val = int(path.read_text().strip())
            print(f"{PASS} {c} = {val}")
        except FileNotFoundError:
            print(f"{FAIL} {c} not found at {path}")
        except ValueError:
            print(f"{WARN} {c} — could not parse value")

def check_slurm():
    section("5. SLURM environment")
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_NTASKS", "SLURM_PROCID"]
    found_any = False
    for var in slurm_vars:
        val = os.environ.get(var)
        if val:
            print(f"{PASS} {var} = {val}")
            found_any = True
    if not found_any:
        print(f"{WARN} No SLURM vars found — you're on a login node (expected for pre-flight)")
        print(f"      These will be set automatically when the job runs via sbatch.")

def check_output_dir(output_dir):
    section("6. Output directory writability")
    p = Path(output_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        print(f"{PASS} {output_dir} is writable")
    except Exception as e:
        print(f"{FAIL} Cannot write to {output_dir}: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ib-device",   default="mlx5_0")
    p.add_argument("--ib-port",     type=int, default=1)
    p.add_argument("--output-dir",  default="./results")
    args = p.parse_args()

    print("=" * 50)
    print("  LENS POC — Pre-flight Sanity Check")
    print("=" * 50)

    check_python()
    check_torch()
    check_nvidia_smi()
    check_ib(args.ib_device, args.ib_port)
    check_slurm()
    check_output_dir(args.output_dir)

    print(f"\n{'='*50}")
    print("  Pre-flight complete. Review any ✗ or ! above.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
