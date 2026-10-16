#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the #61912 follow-up probes against one vLLM source tree, on CPU.
# The tree must import (compiled extensions available) and contain tests/.
#
# Usage: run_all.sh <vllm-tree>     (PYTHON defaults to `python`)
set -euo pipefail
TREE=${1:?usage: run_all.sh <vllm-tree>}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$TREE" || exit 1
echo "vllm $(git rev-parse --short=8 HEAD)$(git diff --quiet HEAD -- vllm || echo ' +local changes to vllm/')"
run() {
  echo "== $*"
  PYTHONPATH="$HERE:$TREE" CUDA_VISIBLE_DEVICES="" VLLM_TARGET_DEVICE=cpu \
    VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$HERE/$1" "${@:2}" 2>&1 | grep -E "^(policy=|async-|pp3-|recompute |Traceback|[A-Za-z]+Error:)"
}
for p in fail recompute; do for rep in 0 1; do run probe_reset_order.py "$p" "$rep"; done; done
for p in fail recompute; do for f in 1 0; do run probe_gate_scope.py "$p" "$f"; done; done
run probe_rewound_preempt.py
