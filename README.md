**Self-Healing AI Infrastructure utilizing GPU and Network Telemetry**

CMPE 295A/B Master's Project — San Jose State University
 
> **Team:** Ashish Bhusal · Madhunica Balasubramanian · Nishan Paudel · Saim Sheikh
> 
> **Advisor:** Gopinath Vinodh
 
---
 
## Overview
 
This is a closed-loop infrastructure monitoring system designed for distributed AI training clusters. It implements the MAPE-K autonomic control framework — Monitor, Analyze, Plan, Execute — to detect network-induced GPU stalls, attribute their root cause, and trigger bounded corrective actions without requiring administrator access.
 
This repository contains the **Monitor layer** — the telemetry collection pipeline that continuously polls GPU performance metrics and InfiniBand hardware counters from all nodes in a distributed training job, producing a labeled time-series dataset for downstream ML classification.

**Analyze layer** TBD
**Plan layer** TBD
**Execute layer** TBD
 
---
## Repository Structure
 
```
src/
├── submit_poc.sh          # SLURM batch script — job orchestration
├── run_node.sh            # Per-node wrapper — process lifecycle management
├── telemetry_poller.py    # Core telemetry agent — GPU + IB counter collection
├── ddp_workload.py        # Synthetic DDP training workload — GPU load generator
├── preflight_check.py     # Pre-submission sanity checker
├── results/               # Output directory — created at runtime
│   └── <JOBID>/
│       ├── telemetry_<node>_<JOBID>.csv
│       ├── poller_<node>_<task>.log
│       └── ddp_<node>_<task>.log
└── README.md
```
 
---
## What the Pipeline Collects
 
Each poll cycle (default 0.5 seconds) produces one CSV row per GPU per node containing:
 
### GPU Metrics (via nvidia-smi)
| Field | Description |
|---|---|
| `gpu_util_pct` | GPU utilization percentage (0–100) |
| `gpu_mem_used_mb` | GPU memory used in MB |
| `gpu_mem_total_mb` | GPU total memory in MB |
 
### InfiniBand Counters (via /sys/class/infiniband)
 
**Throughput — Data Sent/Received Rate**
| Counter | Delta | Description |
|---|---|---|
| `ib_port_xmit_data` | ✓ | Bytes transmitted per interval |
| `ib_port_rcv_data` | ✓ | Bytes received per interval |
| `ib_port_xmit_packets` | ✓ | Packets transmitted per interval |
| `ib_port_rcv_packets` | ✓ | Packets received per interval |
 
**Congestion — Backpressure Signals**
| Counter | Delta | Description |
|---|---|---|
| `ib_port_xmit_wait` | ✓ | Ticks port stalled waiting for flow control credits |
| `ib_port_xmit_discards` | ✓ | Packets discarded on transmit side |
| `ib_excessive_buffer_overrun_errors` | ✓ | Buffer overrun events |
 
**Physical Link Health**
| Counter | Description |
|---|---|
| `ib_link_downed` | Number of times link went down |
| `ib_link_error_recovery` | Number of link error recovery events |
| `ib_symbol_error` | Physical layer bit errors |
| `ib_local_link_integrity_errors` | Local HCA integrity errors |
| `ib_port_rcv_remote_physical_errors` | Physical errors reported by remote end |
| `ib_port_rcv_errors` | Total received packets with errors |
 
**RDMA Transport (hw_counters/)**
| Counter | Delta | Description |
|---|---|---|
| `ib_sq_num_rnr` | ✓ | Receiver Not Ready retransmissions — congestion signal |
| `ib_sq_num_to` | ✓ | Send queue timeouts — serious fault signal |
| `ib_rq_num_oos` | ✓ | Receive queue out-of-sequence packets |
| `ib_sq_num_oos` | ✓ | Send queue out-of-sequence packets |
 
**Poller Health**
| Field | Description |
|---|---|
| `polling_interval_actual` | Measured elapsed time since previous poll (jitter diagnostic) |
 
---
 
## Cluster Environment
 
Validated on **SJSU HPC3** (`coe-hpc3.sjsu.edu`):
 
| Parameter | Value |
|---|---|
| Scheduler | SLURM |
| Partition | `gpuqs` |
| GPU Nodes | g4, g7, g11 (NVIDIA P100, 12 GB VRAM) |
| IB Device | `mlx4_0` (Mellanox ConnectX-3/4 HCA) |
| Python | `python3/3.11.7` via virtualenv |
| PyTorch | `2.6.0+cu124` |
 
---
 
## Prerequisites
 
Before submitting the job, verify the following on your cluster:
 
```bash
# 1. Check available IB devices
ls /sys/class/infiniband/
 
# 2. Check GPU is visible
nvidia-smi
 
# 3. Check available partitions
sinfo -o "%P %a %l %D %t %N" | grep gpu
 
# 4. Check node memory and status
for node in g8 g10 g11; do
    echo "=== $node ==="
    scontrol show node $node | grep -E "Gres|FreeMem|Reason"
done
```
 
---
 
## Quick Start
 
### Step 1 — Run preflight check
 
Run this on the login node or inside an `salloc` session before submitting:
 
```bash
python3 preflight_check.py --ib-device mlx4_0
```
 
Expected output:
```
✓ Python 3.11.7
✓ PyTorch 2.6.0+cu124
✓ GPU 0: util=0%  mem=1/12288 MB
✓ Available IB devices: ['mlx4_0']
✓ port_xmit_data = <value>
✓ port_rcv_data = <value>
✓ port_xmit_wait = <value>
✓ ./results is writable
```
 
Fix any `✗` failures before proceeding.
 
---
 
### Step 2 — Configure submit_poc.sh
 
Edit the following in `submit_poc.sh` to match your cluster:
 
```bash
#SBATCH --partition=gpuqs        # your GPU partition
#SBATCH --nodelist=g8,g10,g11   # nodes with confirmed IB and free memory
#SBATCH --mem=32G                # memory per node
#SBATCH --time=00:30:00          # walltime
 
ACTIVATE="/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate"  # your Python env
```
 
Also update `run_node.sh` if your IB device differs from `mlx4_0`:
```bash
--ib-device mlx4_0   # update if your nodes use mlx5_0 or other
```
 
---
 
### Step 3 — Submit the job
 
```bash
cd ~/lens_poc
sbatch submit_poc.sh
```
 
Expected output:
```
Submitted batch job <JOBID>
```
 
---
 
### Step 4 — Monitor the job
 
```bash
# Check job status
squeue -u $USER
 
# Watch live log
tail -f lens_poc_<JOBID>.log
 
# Check row counts growing (run in second terminal)
watch -n 5 "wc -l ~/lens_poc/results/<JOBID>/*.csv"
```
 
---
 
### Step 5 — Validate results
 
```bash
# Check CSV files were produced
ls results/<JOBID>/
 
# Verify headers
head -1 results/<JOBID>/telemetry_<node>_<JOBID>.csv
 
# Spot check GPU util and IB delta (columns 6 and 30)
awk -F',' 'NR>1 {print $6, $30}' results/<JOBID>/telemetry_<node>_<JOBID>.csv | tail -20
 
# Validate polling jitter
awk -F',' 'NR>1 && $4!="" {print $4}' results/<JOBID>/telemetry_<node>_<JOBID>.csv | \
awk '{sum+=$1; count++; if($1>max) max=$1; if(min=="" || $1<min) min=$1}
     END {printf "count=%d avg=%.4f min=%.4f max=%.4f jitter=%.4fms\n",
          count, sum/count, min, max, (max-min)*1000}'
```
 
---
 
## Validated Results (SJSU HPC3 — Job 30052)
 
| Metric | g4 | g7 | g11 |
|---|---|---|---|
| gpu_util_pct | 99–100% | 99–100% | 99–100% |
| xmit_data_delta (typical) | 35–41M units | 35–41M units | 35–41M units |
| xmit_wait_delta (avg) | ~40 ticks | ~200 ticks | ~3000 ticks |
| symbol_error_delta | 0 | 0 | 0 |
| sq_num_rnr_delta | 0 | 0 | 0 |
| Rows collected | 899+ | 899+ | 899+ |
| Polling jitter | 0.1ms | — | — |
 
> **Note:** g11 shows consistently higher `xmit_wait_delta` than g4 and g7 even during healthy operation. This is natural node-level variation — the ML classifier must learn node-relative baselines rather than absolute thresholds.
 
---
 
## Workbook Task Exit Criteria Status
 
| Task | Description | Exit Criteria | Status |
|---|---|---|---|
| Task #2 | GPU + IB counter polling | Non-zero values, continuous polling | ✅ DONE |
| Task #3 | IB counter stability | port_xmit_data monotonically increasing | ✅ DONE |
| Task #4 | GPU-network correlation | xmit_data_delta non-zero during GPU activity | ✅ DONE |
| Task #6 | Clock sync + jitter | Polling jitter < 50ms | ✅ DONE (0.1ms) |
| Task #7 | Multi-node symmetry | Correlated counter changes across nodes | ✅ DONE |
| Task #5 | Idle noise floor | Background Net_Xmit_Delta documented | ⏳ PENDING |
| Task #8 | Poller robustness | No missing rows at job boundary | ⏳ PENDING |
| Task #11 | Fault injection | Labeled dataset covering all 6 labels | ⏳ PENDING |
 
---
 
## Fault Injection (Coming Soon)
 
The following fault scenarios are planned to generate labeled training data for the ML classifier:
 
| Label | Scenario | Method | Status |
|---|---|---|---|
| 0 | Healthy training | Normal DDP run | ✅ Done |
| 1 | Normal AllReduce pause | Observed during training | ✅ Done |
| 2 | Network-induced stall | Apptainer tc/netem delay | ⏳ Planned |
| 3 | Physical link error | Apptainer tc/netem loss | ⏳ Planned |
| 4 | Compute fault | CUDA sync barrier delay | ⏳ Planned |
| 5 | Idle | Poller with no DDP job | ⏳ Planned |
 
---
 
## Debugging
 
### Job completes in under 10 seconds
```bash
cat lens_poc_<JOBID>.err
# Look for: module load failures, Python import errors, srun errors
```
 
### CSV files missing or only 10 rows
```bash
cat results/<JOBID>/poller_*.log
# Look for: IB device not found, permission denied, path errors
```
 
### Job stuck in PD state
```bash
squeue -u $USER
# Check REASON column:
# Resources       → nodes busy, wait or change nodelist
# ReqNodeNotAvail → requested nodes are down
sinfo -o "%P %a %l %D %t %N" | grep gpuqs
```
 
### GPU utilization stuck at 0%
```bash
cat results/<JOBID>/ddp_*.log
# No DDP logs     → srun step failed, check .err file
# DDP logs exist  → NCCL initialization hung, check master node resolution
```
 
### Wrong IB device name
```bash
# Check available devices on a node via salloc
salloc --nodes=1 --gres=gpu:1 --partition=gpuqs --nodelist=g8 --time=00:10:00
ls /sys/class/infiniband/
exit
```
 
---
 
## Design Notes
 
**Why single srun architecture?**
The SLURM version on HPC3 does not support concurrent srun steps on the same allocation. Running the poller and DDP workload as two separate srun calls results in the second step being blocked. Both processes are co-located inside a single srun via `run_node.sh`, which starts the poller as a background process and DDP as the foreground process on each node.
 
**Why sysfs over RDMA APIs?**
IB counters are read directly from sysfs rather than via libibverbs or any RDMA management API. This eliminates all privilege requirements and removes dependency on the IB software stack beyond what the kernel driver exposes by default.
 
**Why CSV over time-series database?**
Output is written to flat CSV files on the shared filesystem rather than pushed to Prometheus or InfluxDB. This avoids additional infrastructure dependencies and keeps the pipeline fully self-contained within the SLURM job.
 
**Why delta computation at collection time?**
Per-interval deltas are computed inside the poller rather than in the downstream classifier. This reduces feature engineering burden on the Analyze layer and ensures overflow handling is applied consistently at the source.
 
---
 
## No Elevated Privileges Required
 
All telemetry sources are accessible with standard user permissions:
 
```
nvidia-smi                                    → standard user
/sys/class/infiniband/*/counters/             → readable without root
/sys/class/infiniband/*/hw_counters/          → readable without root
SLURM job submission (sbatch/srun)            → standard user
```
 
No `sudo`, `root`, `CAP_BPF`, or `CAP_NET_ADMIN` is required for baseline operation.
 
---
 
## References
 
[1] Y. Deng et al., "Minder: Faulty Machine Detection for Large-scale Distributed Model Training," arXiv:2411.01791v2, Apr. 2025.
 
[2] E. Darzi, A. Pareja, and S. Bharadwaj, "Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure," arXiv:2510.16946v1, Oct. 2025.
 
[3] D. Landau, J. Barbosa, and N. Saurabh, "eBPF-Based Instrumentation for Generalisable Diagnosis of Performance Degradation," arXiv:2505.13160v1, May 2025.
 
[4] J. Sommers, N. Rudolph, and R. Durairajan, "Schooling NOOBs with eBPF," eBPF '23, Sep. 2023.
 
[5] Z. Hu et al., "Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms," arXiv:2507.04786v2, Jul. 2025.
 
[6] J. O. Kephart and D. M. Chess, "The vision of autonomic computing," Computer, vol. 36, no. 1, pp. 41–50, Jan. 2003.
