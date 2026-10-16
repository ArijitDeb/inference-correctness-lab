# SPDX-License-Identifier: Apache-2.0
"""Probe: when does r1 get readmitted after preemption? Run it on 89a93299 and
on 89a93299 + gate-ablation.patch; any difference is the admission gate's effect.

Driver: D batches in flight. After schedule(k), the output of step k-D+1 is
processed, as the engine's batch queue does. Report (if failed) rejects one of
r1's loaded blocks in the output of step 0.

Scenarios (all top-level kv_producer unless noted, so pressure can preempt r1):
  async-deliver-pressure  async (D=2); a best-effort connector, emulated by
                          setting the scheduler's requires_kv_delivery to False
                          (stale output is delivered, not dropped)
  async-drop-pressure     async (D=2); NixlConnector default for a producer
                          (requires_kv_delivery True: stale output dropped)
  async-reset-producer    async; reset_prefix_cache(reset_running_requests=True)
                          right after schedule(0)
  async-reset-consumer    same, kv_consumer (the reset raises; scheduling continues)
  pp3-deliver-pressure    synchronous PP=3 (D=3), chunked r1, deliver emulation
  pp3-drop-pressure       synchronous PP=3 (D=3), Nixl producer default

Output per run: step at which r1 was preempted, (step, start) of its readmission,
and the first scheduled-token entries for r1 per step ("-" = not scheduled).

Usage: python probe_gate_scope.py <policy> <failed:0|1>     (from the vllm tree)
"""

import sys
from collections import deque

from probe_common import BS, add, block_ids, make_scheduler, output, start

policy, failed = sys.argv[1], sys.argv[2] == "1"


def run(name, rid, *, role, deliver, mode, reset_at=None):
    if mode == "async":
        s = make_scheduler(policy, role, num_blocks=11)  # 10 usable blocks
        depth = 2
        add(s, rid + 1, num_tokens=BS, same_prefix=False, max_tokens=2)
        add(s, rid + 2, num_tokens=BS, same_prefix=False, max_tokens=1)
        r1 = add(s, rid + 3, external=7 * BS)
        bad = 3
    else:  # synchronous PP=3, r1 chunked so two of its steps are in flight
        s = make_scheduler(
            policy, role, num_blocks=10, async_scheduling=False, pp=3, chunk=BS
        )
        depth = 3
        add(s, rid + 1, num_tokens=3 * BS, same_prefix=False, max_tokens=4)
        add(s, rid + 2, num_tokens=BS, same_prefix=False, max_tokens=1)
        add(s, rid + 4, num_tokens=BS, same_prefix=False, max_tokens=1)
        r1 = add(s, rid + 3, num_tokens=5 * BS, external=3 * BS)
        bad = 1
    if deliver:
        s.requires_kv_delivery = False  # emulate a best-effort connector
    queue, b1, preempted_at, readmitted, trace = deque(), None, None, None, []
    for k in range(14):
        st = s.schedule()
        if k == 0:
            b1 = block_ids(s, r1)
        if preempted_at is None and r1.request_id in (st.preempted_req_ids or set()):
            preempted_at = k
        if (
            preempted_at is not None
            and readmitted is None
            and k > preempted_at
            and r1.request_id in st.num_scheduled_tokens
        ):
            readmitted = (k, start(st, r1))
        trace.append(f"{k}:{st.num_scheduled_tokens.get(r1.request_id, '-')}")
        if reset_at == k:
            try:
                s.reset_prefix_cache(reset_running_requests=True)
            except RuntimeError as exc:
                assert "Failed to reset KV cache" in str(exc), exc
            preempted_at = k
        queue.append((k, st))
        if len(queue) >= depth:
            j, sj = queue.popleft()
            invalid = {b1[bad]} if (failed and j == 0) else None
            s.update_from_output(sj, output(sj, 900 + j, invalid))
    print(
        f"{name:22s} policy={policy} failed={failed} | preempted_at={preempted_at} "
        f"readmitted(step,start)={readmitted} r1={r1.status.name} "
        f"requires_kv_delivery={s.requires_kv_delivery} r1_steps={' '.join(trace[:8])}"
    )


base = 1000 + 100 * failed + 50 * (policy == "recompute")
run("async-deliver-pressure", base + 0, role="kv_producer", deliver=True, mode="async")
run("async-drop-pressure", base + 10, role="kv_producer", deliver=False, mode="async")
run("async-reset-producer", base + 20, role="kv_producer", deliver=False, mode="async", reset_at=0)
run("async-reset-consumer", base + 30, role="kv_consumer", deliver=False, mode="async", reset_at=0)
run("pp3-deliver-pressure", base + 40, role="kv_producer", deliver=True, mode="pp3")
run("pp3-drop-pressure", base + 50, role="kv_producer", deliver=False, mode="pp3")
