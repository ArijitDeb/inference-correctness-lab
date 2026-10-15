# SPDX-License-Identifier: Apache-2.0
"""Probe: owner with a failing sync load is preempted
by KV pressure inside a real schedule() call under async scheduling.

Usage: python probe_pressure_preempt.py <role> <policy>
Prints one result line per run. CPU only.
"""

import sys

sys.path.insert(0, ".")  # run from the vllm tree root

from tests.v1.kv_connector.unit.utils import (  # noqa: E402
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.v1.request import RequestStatus  # noqa: E402

role, policy = sys.argv[1], sys.argv[2]
BS = 16

cfg = create_vllm_config(kv_role=role, kv_load_failure_policy=policy)
cfg.scheduler_config.async_scheduling = True
sched = create_scheduler(cfg, num_blocks=11)  # null + 10 usable
conn = create_mock_connector()
ext = {}  # req_id -> external tokens offered (sync)
conn.get_num_new_matched_tokens.side_effect = lambda r, n: (ext.get(r.request_id, 0), False)
conn.request_finished.return_value = (False, None)
conn.take_events.return_value = ()
conn.has_pending_block_frees.return_value = False
conn.has_pending_push_work.return_value = False
sched.connector = conn
pool = sched.kv_cache_manager.block_pool

tag = (
    f"{type(sched).__name__} role={role} policy={policy} "
    f"defer_block_free={sched.defer_block_free} "
    f"max_concurrent_batches={cfg.max_concurrent_batches}"
)

# r0: 31-token prompt, decodes; crosses into its 3rd block at position 32.
r0 = create_request(request_id=900, num_tokens=31, max_tokens=4)
sched.add_request(r0)
s1 = sched.schedule()
sched.update_from_output(s1, create_model_runner_output([r0]))

# r1: 8-block prompt, 7 blocks offered as a synchronous external load.
r1 = create_request(request_id=901, num_tokens=8 * BS, common_prefix_len=8 * BS)
ext[r1.request_id] = 7 * BS
sched.add_request(r1)
s2 = sched.schedule()  # step N: r0 decode + r1 (16 local tokens, 112 loaded)
assert r1.request_id in s2.num_scheduled_tokens, s2.num_scheduled_tokens
(b,) = sched.kv_cache_manager.get_block_ids(r1.request_id)
b = list(b)
ext[r1.request_id] = 0  # nothing external on resume: isolate the local hit
free_after_N = pool.get_num_free_blocks()

# step N+1, BEFORE update_from_output(N): r0 needs a new block.
s3 = sched.schedule()
preempted = r1.status == RequestStatus.PREEMPTED
r0_in_s3 = r0.request_id in s3.num_scheduled_tokens

# Late report for step N: block 3 of r1 failed to load.
out2 = create_model_runner_output([r0, r1], invalid_block_ids={b[3]})
sched.update_from_output(s2, out2)
hashed = [pool.blocks[x].block_hash is not None for x in b]
r1_status_after_report = r1.status.name

# Drive until r1 is scheduled again (or finishes); record its local hit.
pending = [s3]
hit = None
for _ in range(12):
    if r1.is_finished():
        break
    s = sched.schedule()
    if r1.request_id in s.num_scheduled_tokens and hit is None:
        hit = r1.num_computed_tokens - s.num_scheduled_tokens[r1.request_id]
    pending.append(s)
    so = pending.pop(0)
    reqs = [sched.requests[i] for i in so.num_scheduled_tokens if i in sched.requests]
    sched.update_from_output(so, create_model_runner_output(reqs))
    if hit is not None:
        break

print(
    f"{tag} | free_after_N={free_after_N} preempted_in_N+1={preempted} "
    f"r0_sched_N+1={r0_in_s3} | after report: hashed={''.join('H' if h else '-' for h in hashed)} "
    f"r1_status={r1_status_after_report} | r1 resume local hit={hit} "
    f"(correct={3 * BS}) r1_final={r1.status.name}"
)
