#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class CommunicationModel(nn.Module):
    def __init__(self, parameter_mb: int) -> None:
        super().__init__()
        count = max(1, parameter_mb * 1024 * 1024 // 4)
        self.weight = nn.Parameter(torch.randn(count, dtype=torch.float32))

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        return self.weight.square().mean() + dummy.sum() * 0.0


def int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(
        os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))
    )
    local_rank = int(
        os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0))
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workload")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size, local_rank


def marker_writer(output_dir: Path, host: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"label1_active_{host}.csv"
    is_new = not path.exists() or path.stat().st_size == 0

    handle = path.open("a", newline="", buffering=1)
    writer = csv.writer(handle)

    if is_new:
        writer.writerow(
            [
                "timestamp",
                "hostname",
                "label1_active",
                "pause_seconds",
                "pause_step",
            ]
        )
        handle.flush()

    def write(active: int, pause_seconds: float, step: int) -> None:
        writer.writerow(
            [
                f"{time.time():.9f}",
                host,
                active,
                f"{pause_seconds:.6f}",
                step,
            ]
        )
        handle.flush()

    return handle, write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=float_env("RUN_SECONDS", 300.0),
    )
    parser.add_argument(
        "--pause-rank",
        type=int,
        default=int_env("PAUSE_RANK", 1),
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=int_env("PAUSE_EVERY", 200),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=float_env("PAUSE_SECONDS", 0.25),
    )
    parser.add_argument(
        "--parameter-mb",
        type=int,
        default=int_env("DDP_PARAMETER_MB", 64),
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR"),
    )
    parser.add_argument(
        "--master-port",
        default=os.environ.get("MASTER_PORT"),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.master_addr:
        os.environ["MASTER_ADDR"] = args.master_addr

    if args.master_port:
        os.environ["MASTER_PORT"] = str(args.master_port)

    if args.pause_every < 1:
        raise ValueError("--pause-every must be at least 1")

    if args.pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative")

    rank = None
    handle = None

    try:
        rank, world_size, local_rank = setup_distributed()

        device = torch.device("cuda", local_rank)
        host = socket.gethostname()

        handle, write_marker = marker_writer(
            Path(args.output_dir),
            host,
        )

        if rank == 0:
            print(
                f"[rank {rank}/{world_size}] "
                f"Label 1 synchronization workload "
                f"parameter_mb={args.parameter_mb} "
                f"run_seconds={args.run_seconds:.1f} "
                f"pause_rank={args.pause_rank} "
                f"pause_every={args.pause_every} "
                f"pause_seconds={args.pause_seconds:.3f}",
                flush=True,
            )
        else:
            print(
                f"[rank {rank}/{world_size}] Using device: {device}",
                flush=True,
            )

        model = DDP(
            CommunicationModel(args.parameter_mb).to(device),
            device_ids=[local_rank],
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-4,
        )

        dummy = torch.zeros(1, device=device)
        started = time.monotonic()
        step = 0

        while True:
            # Rank 0 controls termination for every rank.
            run_flag = torch.tensor(
                    [
                        1
                        if rank == 0
                        and time.monotonic() - started < args.run_seconds
                        else 0
                        ],
                    device=device,
                    dtype=torch.int32,
                    )

            dist.broadcast(run_flag, src=0)

            if int(run_flag.item()) == 0:
               break

            step += 1
            step_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            loss = model(dummy)

            # Rank 0 controls the pause schedule for every rank.
            # This prevents ranks from entering different collectives.
            pause_flag = torch.tensor(
                [
                    1
                    if rank == 0 and step % args.pause_every == 0
                    else 0
                ],
                device=device,
                dtype=torch.int32,
            )

            dist.broadcast(pause_flag, src=0)

            if int(pause_flag.item()) == 1:
                write_marker(
                    1,
                    args.pause_seconds,
                    step,
                )

                if rank == args.pause_rank:
                    time.sleep(args.pause_seconds)

                loss.backward()

                dist.barrier()

                write_marker(
                    0,
                    args.pause_seconds,
                    step,
                )
            else:
                loss.backward()

            optimizer.step()
            torch.cuda.synchronize(device)

            step_time_ms = (
                time.perf_counter() - step_start
            ) * 1000.0

            if rank == 0:
                print(
                    f"[rank 0] step={step:6d} "
                    f"loss={loss.item():.6f} "
                    f"step_time_ms={step_time_ms:.3f} "
                    f"wall={time.time():.9f}",
                    flush=True,
                )

        dist.barrier()

        if rank == 0:
            print(
                f"[rank 0] Label 1 workload complete steps={step}",
                flush=True,
            )

    finally:
        if handle is not None:
            handle.close()

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
