#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Supplementary run on a larger model: is the wrong answer also a fast one?
# Same harness, Qwen3-8B-Base, three cells per tree, more trials, then
# timing.py tables r2's time to first token next to its correctness.
#
#   ./run_timing.sh <tree> [trials]      # tree = main | fix
#
# At 0.6B the 1.9k-token prefill is too cheap for a cache hit to show; at 8B
# it costs roughly 13x more. Needs GPU 0 to itself (the 8B weights are 16 GB).
set -euo pipefail

TREE_NAME=${1:?usage: ./run_timing.sh <tree> [trials]}
case "$TREE_NAME" in
  main|fix) ;;
  *) echo "unknown tree '$TREE_NAME' (main | fix)" >&2; exit 2 ;;
esac
TRIALS=${2:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
TREE="$HERE/trees/$TREE_NAME"
CLONE=${VLLM_CLONE:-$HOME/Projects/vllm}
PY=${PY:-$CLONE/.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B-Base}
GPU_UTIL=${GPU_UTIL:-0.75}
OUT="$HERE/results-8b/$TREE_NAME"
STORAGE="$HERE/storage-8b/$TREE_NAME"
CELL_TIMEOUT=${CELL_TIMEOUT:-1800}

[ -d "$TREE" ] || { echo "no tree at $TREE; run ./setup_trees.sh first" >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits || true)
if ! [[ "$used" =~ ^[0-9]+$ ]] || [ "$used" -gt 4000 ]; then
  echo "GPU 0 not free (memory used: '$used' MiB)" >&2
  exit 1
fi
export PYTHONPATH="$TREE"

rm -rf "$STORAGE" "$OUT"
mkdir -p "$STORAGE" "$OUT"
{
  date -Is
  echo "tree: $TREE_NAME ($(git -C "$TREE" rev-parse HEAD))"
  echo "harness: $(git -C "$HERE" rev-parse HEAD)$(
    [ -n "$(git -C "$HERE" status --porcelain -- . 2>/dev/null)" ] && echo ' (dirty)')"
  echo "model: $MODEL"
  "$PY" -c 'import torch, vllm; print("torch", torch.__version__); print("vllm", vllm.__version__)'
  nvidia-smi --id=0 --query-gpu=name,driver_version,memory.total --format=csv
} > "$OUT/ENV.txt"

FAILED=()
run() {
  echo ">> harness.py $*"
  local rc=0
  timeout -k 30 "$CELL_TIMEOUT" "$PY" "$HERE/harness.py" "$@" --results "$OUT" \
    --model "$MODEL" --gpu-memory-utilization "$GPU_UTIL" >> "$OUT/run.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! exit $rc: harness.py $*" >&2
    FAILED+=("$*")
  fi
}

run store --storage "$STORAGE"
run baseline
run trial --storage "$STORAGE" --policy fail --abort none --trials "$TRIALS"
run trial --storage "$STORAGE" --policy fail --abort queue --trials "$TRIALS"
run trial --storage "$STORAGE" --policy fail --abort queue --trials "$TRIALS" --no-load-failure

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '!! failed: %s\n' "${FAILED[@]}" >&2
  exit 1
fi
