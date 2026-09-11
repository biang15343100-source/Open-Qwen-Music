#!/usr/bin/env bash
set -euo pipefail

# Run the same command on every node. The scheduler handles node discovery and process placement.
# This launcher only uses standard torchrun rendezvous arguments.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-open-qwen-music}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--module PYTHON_MODULE | PYTHON_SCRIPT] [ARGS...]" >&2
  exit 2
fi
if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer" >&2
  exit 2
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, NNODES)" >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi

if (( NNODES == 1 )); then
  launcher=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone
    --nnodes=1
    "--nproc-per-node=$NPROC_PER_NODE"
  )
else
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "Multi-node runs require MASTER_ADDR" >&2
    exit 2
  fi
  launcher=(
    "$PYTHON_BIN" -m torch.distributed.run
    "--nnodes=$NNODES"
    "--nproc-per-node=$NPROC_PER_NODE"
    "--node-rank=$NODE_RANK"
    "--rdzv-backend=$RDZV_BACKEND"
    "--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"
    "--rdzv-id=$RDZV_ID"
  )
fi

if [[ "$1" == "--module" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--module requires a Python module" >&2
    exit 2
  fi
  launcher+=(--module "$2")
  shift 2
else
  launcher+=("$1")
  shift
fi
launcher+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${launcher[@]}"
  printf '\n'
  exit 0
fi
exec "${launcher[@]}"
