# SPDX-License-Identifier: Apache-2.0
"""Probe: a RUNNING owner that the report rewinds, then pressure preempts.

Consumer role, `recompute`, async scheduling, 9 usable blocks. A small request A
(1 block, decodes) and r1 (8-block prompt, 7 blocks offered as a synchronous
external load) are scheduled in step N; step N+1 is scheduled before report N.
Report N rejects block 3 (failed=True) or reports nothing (load-ok control).
Under `recompute`, the report finds r1 running and rewinds it to 48 computed
tokens. Once the deferred-free fence has passed, A's next block forces a
preemption of r1. The probe prints r1's start when it is readmitted: 48 if its
blocks were repaired or evicted, 112 if it hits its own unrepaired blocks again.

Usage: python probe_rewound_preempt.py     (from the vllm tree)
"""

from probe_common import BS, add, block_ids, hashed, make_scheduler, output, start

for failed in (True, False):
    s = make_scheduler("recompute", "kv_consumer", num_blocks=10)
    rid = 300 + 10 * failed
    add(s, rid, num_tokens=BS, same_prefix=False, max_tokens=2)
    r1 = add(s, rid + 1, external=7 * BS)
    step_n = s.schedule()
    b1 = block_ids(s, r1)
    step_n1 = s.schedule()
    s.update_from_output(step_n, output(step_n, 999, {b1[3]} if failed else None))
    after = (
        f"after report N: r1={r1.status.name} computed={r1.num_computed_tokens} "
        f"hashed(b1)={hashed(s, b1)}"
    )
    s.update_from_output(step_n1, output(step_n1, 998))
    preempted_at = resumed = None
    for k in range(2, 10):
        st = s.schedule()
        if preempted_at is None and r1.request_id in (st.preempted_req_ids or set()):
            preempted_at = f"N+{k}"
        elif preempted_at is not None and resumed is None and r1.request_id in st.num_scheduled_tokens:
            resumed = (f"N+{k}", start(st, r1))
        s.update_from_output(st, output(st, 5))
        if resumed:
            break
    print(
        f"recompute kv_consumer failed={failed} | {after} | "
        f"preempted_at={preempted_at} resumed(step,start)={resumed} (correct start after failure: {3 * BS})"
    )
