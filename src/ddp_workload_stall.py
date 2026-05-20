#!/usr/bin/env python3
"""
LENS POC — DDP Workload with Application Layer Stall Injection
==============================================================
Produces Label 2 (Network-Induced Stall) telemetry data by
injecting artificial delays after AllReduce on a designated
stall rank. All other ranks stall waiting for the delayed rank,
producing the same telemetry fingerprint as a real network stall:

  - gpu_util drops to 0% on ALL nodes during stall
  - xmit_wait_delta spikes on stall node (back-pressured)
  - xmit_data_delta drops across all nodes
  - sq_num_rnr_delta may spike on stall node

Stall parameters are controlled via CLI arguments so the same
script can simulate different fault severities without code changes.

Usage (via run_node_stall.sh — do not run directly):
  python3 ddp_workload_stall.py \\
      --master-addr <host> --master-port 29500 \\
      --stall-rank 1 --stall-every 5 --stall-duration 4.0
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ── Model definition (same as ddp_workload.py) ───────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ConvBlock(64, 128)
        self.layer2 = ConvBlock(128, 256)
        self.layer3 = ConvBlock(256, 512)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── DDP setup / teardown ─────────────────────────────────────────────────────

def setup_ddp(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())


def teardown_ddp():
    dist.destroy_process_group()


# ── Training loop with stall injection ───────────────────────────────────────

def train(args):
    rank       = int(os.environ.get("SLURM_PROCID",  0))
    world_size = int(os.environ.get("SLURM_NTASKS",  1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    print(f"[rank {rank}/{world_size}] Initializing DDP on "
          f"{args.master_addr}:{args.master_port}")
    print(f"[rank {rank}] Stall config: rank={args.stall_rank} "
          f"every={args.stall_every} steps duration={args.stall_duration}s")

    setup_ddp(rank, world_size, args.master_addr, str(args.master_port))

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[rank {rank}] Using device: {device}")

    model     = SmallResNet(num_classes=1000).to(device)
    model     = DDP(model, device_ids=[local_rank]
                    if torch.cuda.is_available() else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    batch_size  = args.batch_size
    image_size  = 224
    num_classes = 1000

    step  = 0
    epoch = 0
    start = time.time()

    # Track stall events for logging
    stall_count = 0

    print(f"[rank {rank}] Starting training loop with stall injection "
          f"(batch={batch_size}, epochs={args.epochs})")

    while epoch < args.epochs:
        epoch += 1

        inputs  = torch.randn(batch_size, 3, image_size, image_size,
                              device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        # ── AllReduce happens here (backward pass) ─────────────────────────
        loss.backward()
        # ── AllReduce complete — inject stall on designated rank ───────────

        if rank == args.stall_rank and step % args.stall_every == 0:
            stall_count += 1
            print(
                f"[rank {rank}] *** STALL #{stall_count} at step {step} — "
                f"sleeping {args.stall_duration}s to simulate network delay ***"
            )
            # This sleep AFTER AllReduce makes other ranks wait at the
            # next barrier — producing the network stall telemetry signature
            time.sleep(args.stall_duration)
            print(f"[rank {rank}] *** STALL #{stall_count} complete ***")

        optimizer.step()
        step += 1
        elapsed = time.time() - start

        if rank == 0 and step % 10 == 0:
            step_time_ms = (elapsed / step) * 1000
            print(
                f"[rank 0] epoch={epoch:4d}  step={step:5d}  "
                f"loss={loss.item():.4f}  "
                f"step_time={step_time_ms:.1f}ms  "
                f"stalls_injected={stall_count}  "
                f"elapsed={elapsed:.1f}s"
            )

    if rank == 0:
        print(
            f"[rank 0] Training complete. "
            f"Steps: {step}  Stalls: {stall_count}  "
            f"Time: {time.time()-start:.1f}s"
        )

    teardown_ddp()


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LENS DDP workload with application-layer stall injection"
    )
    # DDP args
    p.add_argument("--epochs",        type=int,   default=99999)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--master-addr",   type=str,   default="localhost")
    p.add_argument("--master-port",   type=int,   default=29500)
    # Stall injection args
    p.add_argument("--stall-rank",     type=int,   default=1,
                   help="Which rank injects the stall (0, 1, or 2)")
    p.add_argument("--stall-every",    type=int,   default=5,
                   help="Inject stall every N steps")
    p.add_argument("--stall-duration", type=float, default=4.0,
                   help="Duration of each stall in seconds")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
