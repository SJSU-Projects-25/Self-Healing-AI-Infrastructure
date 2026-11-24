#!/usr/bin/env python3
"""
LENS POC — DDP Workload
=======================
A minimal PyTorch Distributed Data Parallel (DDP) training job
designed to drive sustained GPU utilization across all nodes so
the telemetry poller can observe meaningful GPU and IB counter changes.

Model: Small ResNet-style CNN (no pretrained weights needed)
Data:  Synthetic random tensors (no dataset download required)

Usage (launched via srun in submit_poc.sh — do not run directly):
  srun python3 ddp_workload.py --master-addr <host> --master-port 29500
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ── A simple CNN block to produce realistic GPU load ────────────────────────

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
    """
    ~6M parameter CNN — large enough to drive meaningful GPU utilization
    and produce non-trivial NCCL all-reduce traffic between nodes.
    """
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

def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="nccl",          # uses InfiniBand / RoCE for inter-node
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())


def teardown_ddp():
    dist.destroy_process_group()


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    # SLURM sets these when using srun
    rank       = int(os.environ.get("SLURM_PROCID",    0))
    world_size = int(os.environ.get("SLURM_NTASKS",    1))
    local_rank = int(os.environ.get("SLURM_LOCALID",   0))

    print(f"[rank {rank}/{world_size}] Initializing DDP on {args.master_addr}:{args.master_port}")
    setup_ddp(rank, world_size, args.master_addr, str(args.master_port))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[rank {rank}] Using device: {device}")

    # Model
    model = SmallResNet(num_classes=1000).to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Optimizer + loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data — avoids any dataset dependency
    batch_size  = args.batch_size
    image_size  = 224          # standard ImageNet resolution
    num_classes = 1000

    step  = 0
    epoch = 0
    start = time.time()

    print(f"[rank {rank}] Starting training loop (batch={batch_size}, epochs={args.epochs})")

    while epoch < args.epochs:
        epoch += 1

        # Generate a random batch on GPU directly — maximizes GPU compute
        inputs  = torch.randn(batch_size, 3, image_size, image_size, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()       # triggers NCCL all-reduce across nodes
        optimizer.step()

        step += 1
        elapsed = time.time() - start

        if rank == 0 and step % 10 == 0:
            step_time_ms = (elapsed / step) * 1000
            print(
                f"[rank 0] epoch={epoch:4d}  step={step:5d}  "
                f"loss={loss.item():.4f}  "
                f"step_time={step_time_ms:.1f}ms  "
                f"elapsed={elapsed:.1f}s"
            )

    if rank == 0:
        print(f"[rank 0] Training complete. Total steps: {step}  Time: {time.time()-start:.1f}s")

    teardown_ddp()


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LENS DDP workload")
    p.add_argument("--epochs",      type=int,   default=999,      help="Number of training epochs")
    p.add_argument("--batch-size",  type=int,   default=64,       help="Per-GPU batch size")
    p.add_argument("--master-addr", type=str,   default="localhost", help="DDP master node hostname")
    p.add_argument("--master-port", type=int,   default=29500,    help="DDP master port")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
