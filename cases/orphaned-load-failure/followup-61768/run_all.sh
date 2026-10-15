#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the three follow-up probes against one vLLM source tree, on CPU.
# The tree must import (compiled extensions available) and contain tests/.
#
# Usage: run_all.sh <vllm-tree>     (PYTHON defaults to `python`)
set -u
TREE=${1:?usage: run_all.sh <vllm-tree>}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$TREE" || exit 1
echo "vllm $(git rev-parse --short=8 HEAD)"
run() {
  echo "== $*"
  PYTHONPATH="$TREE" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/$1" "${@:2}" 2>&1 |
    grep -E "^(role=|AsyncScheduler |Scheduler )"
}
for p in fail recompute; do run probe_reset_preempt.py kv_consumer "$p"; done
for r in kv_consumer kv_producer; do run probe_pressure_preempt.py "$r" fail; done
for r in kv_consumer kv_producer; do for p in fail recompute; do run probe_abort_reuse.py "$r" "$p"; done; done
