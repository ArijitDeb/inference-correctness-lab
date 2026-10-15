#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run store, baseline and every trial cell for one tree.
#
#   ./run_matrix.sh <tree> [trials]      # tree = main | pr-53298 | fix
#
# Needs GPU 0 to itself: refuses to start while more than MAX_USED_MIB
# (default 4000) is in use on it. Re-running a tree deletes that tree's
# previous results and storage. Each cell is capped at CELL_TIMEOUT seconds
# (default 1800) so a hang can't hold the GPU.
set -euo pipefail

TREE_NAME=${1:?usage: ./run_matrix.sh <tree> [trials]}
case "$TREE_NAME" in
  main|pr-53298|fix) ;;
  *) echo "unknown tree '$TREE_NAME' (main | pr-53298 | fix)" >&2; exit 2 ;;
esac
TRIALS=${2:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
TREE="$HERE/trees/$TREE_NAME"
CLONE=${VLLM_CLONE:-$HOME/Projects/vllm}
PY=${PY:-$CLONE/.venv/bin/python}
OUT="$HERE/results/$TREE_NAME"
STORAGE="$HERE/storage/$TREE_NAME"
MAX_USED_MIB=${MAX_USED_MIB:-4000}
CELL_TIMEOUT=${CELL_TIMEOUT:-1800}

[ -d "$TREE" ] || { echo "no tree at $TREE; run ./setup_trees.sh first" >&2; exit 1; }

# Number GPUs the same way for nvidia-smi and CUDA, so the check below looks
# at the device the run will use.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits || true)
if ! [[ "$used" =~ ^[0-9]+$ ]]; then
  echo "could not read GPU 0 memory from nvidia-smi (got '$used')" >&2
  exit 1
fi
if [ "$used" -gt "$MAX_USED_MIB" ]; then
  echo "GPU 0 has ${used} MiB in use (limit ${MAX_USED_MIB}). Free the GPU first." >&2
  exit 1
fi

export PYTHONPATH="$TREE"
resolved=$("$PY" -c 'import vllm; print(vllm.__file__)')
case "$resolved" in
  "$TREE"/*) ;;
  *) echo "vllm resolves to $resolved, not $TREE" >&2; exit 1 ;;
esac

rm -rf "$STORAGE" "$OUT"
mkdir -p "$STORAGE" "$OUT"
{
  date -Is
  echo "tree: $TREE_NAME ($(git -C "$TREE" rev-parse HEAD))"
  echo "harness: $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo uncommitted)$(
    [ -n "$(git -C "$HERE" status --porcelain -- . 2>/dev/null)" ] && echo ' (dirty)')"
  "$PY" -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda); print("vllm", vllm.__version__)'
  nvidia-smi --id=0 --query-gpu=name,driver_version,memory.total --format=csv
} > "$OUT/ENV.txt"

# Every cell runs even if one fails, but a failure is never swallowed: the
# failed cells are listed at the end and the script exits nonzero.
FAILED=()
run() {
  echo ">> harness.py $*"
  local rc=0 cell_log
  cell_log=$(mktemp)
  timeout -k 30 "$CELL_TIMEOUT" "$PY" "$HERE/harness.py" "$@" --results "$OUT" > "$cell_log" 2>&1 || rc=$?
  cat "$cell_log" >> "$OUT/run.log"
  if [ "$rc" -ne 0 ]; then
    echo "!! exit $rc: harness.py $*" >&2
    tail -n 5 "$cell_log" | sed 's/^/   | /' >&2
    FAILED+=("$*")
  else
    grep '^wrote' "$cell_log" || true
  fi
  rm -f "$cell_log"
}

run store --storage "$STORAGE"
run baseline
for policy in fail recompute; do
  for abort in none queue; do
    run trial --storage "$STORAGE" --policy "$policy" --abort "$abort" --trials "$TRIALS"
    run trial --storage "$STORAGE" --policy "$policy" --abort "$abort" --trials "$TRIALS" --canary
  done
  # Cross-check: the monkeypatch fallback should agree with the real queue path.
  run trial --storage "$STORAGE" --policy "$policy" --abort patch --trials "$TRIALS"
  # vLLM's default scheduling: the batch queue, several batches in flight.
  run trial --storage "$STORAGE" --policy "$policy" --abort none --trials "$TRIALS" --async-scheduling
  run trial --storage "$STORAGE" --policy "$policy" --abort queue --trials "$TRIALS" --async-scheduling
  # Positive control: the load succeeds, only the abort remains. Valid cached
  # KV must survive on every tree, the fix included.
  run trial --storage "$STORAGE" --policy "$policy" --abort queue --trials "$TRIALS" --no-load-failure
done

# Build the table once every tree has run: compare.py (see README).

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "!! ${#FAILED[@]} cell(s) failed on $TREE_NAME (full output in $OUT/run.log):" >&2
  printf '   %s\n' "${FAILED[@]}" >&2
  exit 1
fi
