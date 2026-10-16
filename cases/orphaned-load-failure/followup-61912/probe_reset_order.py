# SPDX-License-Identifier: Apache-2.0
"""Probe: reset in the load step, then schedule(N+1) BEFORE report N.

Consumer role (the default), async scheduling. r1 is an 8-block prompt whose
first 7 blocks are offered as a synchronous external load; step N computes its
last block locally. Between schedule(N) and update_from_output(N):
`reset_prefix_cache(reset_running_requests=True)` preempts r1 and then raises
(r1's blocks are still held by the deferred free); scheduling continues, as the
engine does. schedule(N+1) runs before the step-N output arrives, which is the
async engine's order. Report N then rejects block 3 (failed=True), or reports
nothing (failed=False, the load-ok control). Step N's output carries token 999,
step N+1's token 998: synthetic markers of which step produced a token.

After r1 finishes, an optional `repair` request (the identical 128-token prompt)
runs, then a longer 144-token prompt with the same prefix. The longer prompt
shows whether r1's original block 7, computed in step N over the failed blocks,
is still served from the prefix cache.

Usage: python probe_reset_order.py <policy> <repair:0|1>     (from the vllm tree)
"""

import sys

from probe_common import BS, add, block_ids, delivered, hashed, make_scheduler, output, start

policy, repair = sys.argv[1], sys.argv[2] == "1"

for failed in (True, False):
    s = make_scheduler(policy, "kv_consumer")
    rid = 100 + 10 * failed + repair
    r1 = add(s, rid, external=7 * BS, max_tokens=2)
    step_n = s.schedule()
    b1 = block_ids(s, r1)
    assert len(b1) == 8 and step_n.num_scheduled_tokens[r1.request_id] == BS

    try:
        s.reset_prefix_cache(reset_running_requests=True)
        reset = "ok"
    except RuntimeError as exc:
        assert "Failed to reset KV cache" in str(exc), exc
        reset = "raised"
    status_after_reset = r1.status.name

    step_n1 = s.schedule()  # before update_from_output(step_n)
    r1_in_n1 = r1.request_id in step_n1.num_scheduled_tokens
    r1_start_n1 = start(step_n1, r1)

    s.update_from_output(step_n, output(step_n, 999, {b1[3]} if failed else None))
    after_report = f"hashed(b1)={hashed(s, b1)} r1={r1.status.name}"
    res = s.update_from_output(step_n1, output(step_n1, 998))
    got_998 = 998 in delivered(res, r1)

    for _ in range(12):  # drain r1
        if r1.is_finished():
            break
        st = s.schedule()
        s.update_from_output(st, output(st, 5))
    assert r1.is_finished()

    r2_start = None
    if repair:
        r2 = add(s, rid + 1000, max_tokens=1)  # identical 128-token prompt
        for _ in range(12):
            if r2.is_finished():
                break
            st = s.schedule()
            if r2_start is None:
                r2_start = start(st, r2)
            s.update_from_output(st, output(st, 5))
        assert r2.is_finished()

    r3 = add(s, rid + 2000, num_tokens=9 * BS, max_tokens=1)  # longer, same prefix
    st = s.schedule()
    r3_gets_b1_7 = block_ids(s, r3)[7] == b1[7]

    print(
        f"policy={policy} repair={int(repair)} failed={failed} | reset={reset} "
        f"r1={status_after_reset} | N+1: r1 scheduled={r1_in_n1} start={r1_start_n1} "
        f"| after report N: {after_report} | token 998 delivered={got_998} "
        f"r1_final={r1.status.name} r1_out={list(r1.output_token_ids)} "
        f"| identical r2 start={r2_start} | longer r3 start={start(st, r3)} "
        f"r3 gets r1's block 7={r3_gets_b1_7}"
    )
