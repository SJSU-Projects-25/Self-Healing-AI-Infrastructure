#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# LENS — Node Health Check Script
# Checks GPU health, IB presence, memory, and node state
# before selecting nodes for job submission.
#
# Usage:
#   bash check_nodes.sh                    # checks default candidates
#   bash check_nodes.sh g5 g7 g11         # checks specific nodes
#   bash check_nodes.sh --auto 3           # auto-selects 3 best nodes
#
# Output:
#   Prints per-node health summary
#   Exits 0 if enough healthy nodes found
#   Exits 1 if not enough healthy nodes
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
PARTITION="gpuqs"
MIN_FREE_MEM_MB=20000       # minimum free memory in MB
REQUIRED_NODES=3            # number of nodes needed
IB_REQUIRED=true            # require InfiniBand device
EXCLUDE_NODES="g9,g6"      # known bad nodes — comma separated

# ── Colors ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

PASS="${GREEN}  ✓${NC}"
FAIL="${RED}  ✗${NC}"
WARN="${YELLOW}  !${NC}"

# ── Parse arguments ───────────────────────────────────────────────
AUTO_MODE=false
AUTO_COUNT=3
CANDIDATE_NODES=()

if [[ $# -eq 0 ]]; then
    # No args — get idle nodes from sinfo
    AUTO_MODE=true
elif [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
    AUTO_COUNT="${2:-3}"
else
    CANDIDATE_NODES=("$@")
fi

# ── Get candidate nodes ───────────────────────────────────────────
if $AUTO_MODE; then
    echo "Fetching idle nodes from partition: $PARTITION..."
    # Get idle nodes, exclude known bad ones
    IDLE_NODES=$(sinfo -o "%t %N" -p $PARTITION --noheader | \
        grep "^idle" | \
        awk '{print $2}' | \
        tr ',' '\n' | \
        sed 's/\[//g; s/\]//g' | \
        awk -F'[' '{print $1}')

    # Expand nodelist format (e.g. g[1-3,5] → g1 g2 g3 g5)
    IDLE_EXPANDED=$(sinfo -o "%N" -p $PARTITION --state=idle --noheader 2>/dev/null | \
        head -1 | \
        xargs -I{} scontrol show hostnames {} 2>/dev/null || true)

    if [[ -z "$IDLE_EXPANDED" ]]; then
        echo -e "${FAIL} No idle nodes found in partition $PARTITION"
        exit 1
    fi

    # Convert to array, exclude bad nodes
    while IFS= read -r node; do
        skip=false
        IFS=',' read -ra EXCL <<< "$EXCLUDE_NODES"
        for excl in "${EXCL[@]}"; do
            if [[ "$node" == "$excl" ]]; then
                skip=true
                break
            fi
        done
        if ! $skip && [[ -n "$node" ]]; then
            CANDIDATE_NODES+=("$node")
        fi
    done <<< "$IDLE_EXPANDED"

    echo "Found ${#CANDIDATE_NODES[@]} idle candidate nodes"
fi

# ── Per-node health check function ────────────────────────────────
check_node() {
    local node=$1
    local issues=0
    local warnings=0

    echo ""
    echo "─────────────────────────────────────"
    echo " Node: $node"
    echo "─────────────────────────────────────"

    # Get node info from SLURM
    NODE_INFO=$(scontrol show node "$node" 2>/dev/null)
    if [[ -z "$NODE_INFO" ]]; then
        echo -e "${FAIL} Cannot get node info from SLURM"
        return 1
    fi

    # 1. Check node state
    STATE=$(echo "$NODE_INFO" | grep -oP 'State=\K\S+')
    if echo "$STATE" | grep -qiE "down|drain|fail|error|not_responding"; then
        echo -e "${FAIL} Node state: $STATE"
        ((issues++))
    elif echo "$STATE" | grep -qi "idle"; then
        echo -e "${PASS} Node state: $STATE"
    elif echo "$STATE" | grep -qi "mix\|alloc"; then
        echo -e "${WARN} Node state: $STATE (partially allocated)"
        ((warnings++))
    else
        echo -e "${WARN} Node state: $STATE"
        ((warnings++))
    fi

    # 2. Check Reason field
    REASON=$(echo "$NODE_INFO" | grep -oP 'Reason=\K[^\s]+' || true)
    if [[ -n "$REASON" && "$REASON" != "none" && "$REASON" != "Not" ]]; then
        # Minor memory accounting warnings are OK
        if echo "$REASON" | grep -qi "LowRealMemory\|osp"; then
            echo -e "${WARN} Reason flag: $REASON (minor — acceptable)"
            ((warnings++))
        else
            echo -e "${FAIL} Reason flag: $REASON"
            ((issues++))
        fi
    else
        echo -e "${PASS} No critical reason flags"
    fi

    # 3. Check free memory
    FREE_MEM=$(echo "$NODE_INFO" | grep -oP 'FreeMem=\K\d+')
    if [[ -z "$FREE_MEM" ]]; then
        echo -e "${WARN} Cannot determine free memory"
        ((warnings++))
    elif [[ "$FREE_MEM" -lt "$MIN_FREE_MEM_MB" ]]; then
        echo -e "${FAIL} Free memory: ${FREE_MEM}MB (need ${MIN_FREE_MEM_MB}MB)"
        ((issues++))
    else
        FREE_GB=$(echo "scale=1; $FREE_MEM/1024" | bc)
        echo -e "${PASS} Free memory: ${FREE_MEM}MB (~${FREE_GB}GB)"
    fi

    # 4. Check GPU presence
    GRES=$(echo "$NODE_INFO" | grep -oP 'Gres=\K\S+')
    if [[ -z "$GRES" || "$GRES" == "(null)" ]]; then
        echo -e "${FAIL} No GPU found (Gres is empty)"
        ((issues++))
    else
        echo -e "${PASS} GPU: $GRES"
    fi

    # 5. Check IB presence via sysfs (requires SSH or job — use IP pattern instead)
    # IB nodes follow 172.16.3.x pattern — check NodeAddr
    NODE_ADDR=$(echo "$NODE_INFO" | grep -oP 'NodeAddr=\K\S+')
    NODE_NUM=$(echo "$node" | grep -oP '\d+')

    if [[ -n "$NODE_NUM" ]]; then
        EXPECTED_IB_IP="172.16.3.$((NODE_NUM + 200))"
        # We can't SSH to verify, but known no-IB nodes are g3, g12
        if [[ "$node" == "g3" || "$node" == "g12" ]]; then
            echo -e "${FAIL} Known node with no InfiniBand: $node"
            ((issues++))
        else
            echo -e "${PASS} IB expected at: $EXPECTED_IB_IP (mlx4_0)"
        fi
    fi

    # 6. Check node is reachable (ping)
    if ping -c 1 -W 2 "$node" &>/dev/null; then
        echo -e "${PASS} Node is reachable (ping)"
    else
        echo -e "${WARN} Node ping failed (may be normal on this cluster)"
        ((warnings++))
    fi

    # Summary
    echo ""
    if [[ $issues -eq 0 && $warnings -eq 0 ]]; then
        echo -e "${GREEN}  → HEALTHY — recommended for use${NC}"
        return 0
    elif [[ $issues -eq 0 ]]; then
        echo -e "${YELLOW}  → ACCEPTABLE — $warnings warning(s), usable${NC}"
        return 0
    else
        echo -e "${RED}  → UNHEALTHY — $issues critical issue(s), skip this node${NC}"
        return 1
    fi
}

# ── Run checks ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " LENS Node Health Check"
echo " Partition: $PARTITION"
echo " Required:  $REQUIRED_NODES healthy nodes"
echo "============================================"

HEALTHY_NODES=()
FAILED_NODES=()

for node in "${CANDIDATE_NODES[@]}"; do
    if check_node "$node"; then
        HEALTHY_NODES+=("$node")
    else
        FAILED_NODES+=("$node")
    fi

    # Stop checking once we have enough
    if [[ ${#HEALTHY_NODES[@]} -ge $REQUIRED_NODES ]]; then
        break
    fi
done

# ── Final summary ─────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Health Check Summary"
echo "============================================"
echo " Healthy nodes: ${HEALTHY_NODES[*]:-none}"
echo " Failed nodes:  ${FAILED_NODES[*]:-none}"
echo ""

if [[ ${#HEALTHY_NODES[@]} -ge $REQUIRED_NODES ]]; then
    # Build nodelist string
    NODELIST=$(IFS=,; echo "${HEALTHY_NODES[*]:0:$REQUIRED_NODES}")
    echo -e "${GREEN} ✓ Sufficient healthy nodes found!${NC}"
    echo ""
    echo " Recommended nodelist: $NODELIST"
    echo ""
    echo " To update submit_poc.sh automatically:"
    echo "   sed -i 's/--nodelist=.*/--nodelist=$NODELIST/' ~/lens_poc/submit_poc.sh"
    echo ""
    echo " Or submit directly:"
    echo "   sbatch --nodelist=$NODELIST ~/lens_poc/submit_poc.sh"
    echo ""
    # Export for use in other scripts
    echo "$NODELIST" > /tmp/lens_healthy_nodes.txt
    exit 0
else
    HAVE=${#HEALTHY_NODES[@]}
    echo -e "${RED} ✗ Not enough healthy nodes (have $HAVE, need $REQUIRED_NODES)${NC}"
    echo " Try again later or check cluster status:"
    echo "   sinfo -o '%P %a %l %D %t %N' | grep gpuqs"
    exit 1
fi
