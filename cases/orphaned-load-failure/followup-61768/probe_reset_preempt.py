# SPDX-License-Identifier: Apache-2.0
"""Probe: reset_prefix_cache(reset_running_requests=True)
called between schedule(N) and update_from_output(N) while r1's sync load
(which fails) is in flight. Async scheduling.

Usage: python probe_reset_preempt.py <role> <policy>
"""

import sys

sys.path.insert(0, ".")

from tests.v1.kv_connector.unit.utils import (  # noqa: E402
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

role, policy = sys.argv[1], sys.argv[2]
BS = 16

cfg = create_vllm_config(kv_role=role, kv_load_failure_policy=policy)
cfg.scheduler_config.async_scheduling = True
sched = create_scheduler(cfg, num_blocks=100)
conn = create_mock_connector()
ext = {}
conn.get_num_new_matched_tokens.side_effect = lambda r, n: (ext.get(r.request_id, 0), False)
conn.request_finished.return_value = (False, None)
conn.take_events.return_value = ()
conn.has_pending_block_frees.return_value = False
conn.has_pending_push_work.return_value = False
sched.connector = conn
pool = sched.kv_cache_manager.block_pool

r1 = create_request(request_id=601, num_tokens=8 * BS, common_prefix_len=8 * BS)
ext[r1.request_id] = 7 * BS
sched.add_request(r1)
sN = sched.schedule()
(b,) = sched.kv_cache_manager.get_block_ids(r1.request_id)
b = list(b)
ext[r1.request_id] = 0

try:
    ok = sched.reset_prefix_cache(reset_running_requests=True)
    reset = f"returned {ok}"
except RuntimeError as e:
    reset = f"RuntimeError({str(e)[:40]}...)"
status_after_reset = r1.status.name
stale = (r1.num_stale_output_tokens, r1.drop_stale_output)

sched.update_from_output(sN, create_model_runner_output([r1], invalid_block_ids={b[3]}))
hashed = "".join("H" if pool.blocks[x].block_hash is not None else "-" for x in b)
r1_after = r1.status.name

hit = None
for _ in range(3):
    if r1.is_finished():
        break
    s = sched.schedule()
    if r1.request_id in s.num_scheduled_tokens:
        hit = r1.num_computed_tokens - s.num_scheduled_tokens[r1.request_id]
        break
    sched.update_from_output(s, create_model_runner_output([]))

print(
    f"role={role} policy={policy} defer={sched.defer_block_free} | reset: {reset} "
    f"r1 after reset={status_after_reset} stale(tokens,drop)={stale} | after report: "
    f"hashed={hashed} r1={r1_after} | r1 resume local hit={hit} (correct=48)"
)
