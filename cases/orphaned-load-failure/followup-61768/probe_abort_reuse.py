# SPDX-License-Identifier: Apache-2.0
"""Probe: owner with a failing sync load is ABORTED
after schedule(N); an unrelated request C is scheduled in N+1 (before
update_from_output(N)) under a tight pool; then the late report arrives.

Usage: python probe_abort_reuse.py <role> <policy>
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
from vllm.v1.request import RequestStatus  # noqa: E402

role, policy = sys.argv[1], sys.argv[2]
BS = 16

cfg = create_vllm_config(
    kv_role=role, kv_load_failure_policy=policy, max_num_batched_tokens=256
)
cfg.scheduler_config.async_scheduling = True
sched = create_scheduler(cfg, num_blocks=9)  # null + 8 usable
conn = create_mock_connector()
ext = {}
conn.get_num_new_matched_tokens.side_effect = lambda r, n: (ext.get(r.request_id, 0), False)
conn.request_finished.return_value = (False, None)
conn.take_events.return_value = ()
conn.has_pending_block_frees.return_value = False
conn.has_pending_push_work.return_value = False
sched.connector = conn
pool = sched.kv_cache_manager.block_pool

r1 = create_request(request_id=801, num_tokens=8 * BS, common_prefix_len=8 * BS)
ext[r1.request_id] = 7 * BS
sched.add_request(r1)
sN = sched.schedule()
(b,) = sched.kv_cache_manager.get_block_ids(r1.request_id)
b = list(b)
bad = b[3]

sched.finish_requests(r1.request_id, RequestStatus.FINISHED_ABORTED)
rec = getattr(sched, "_finished_inflight_kv", None)
rec_ids = list(rec[r1.request_id].block_ids) if rec and r1.request_id in rec else None

# Unrelated request C: different tokens, no external hit.
c = create_request(request_id=802, num_tokens=8 * BS)
sched.add_request(c)
sN1 = sched.schedule()
c_sched = c.request_id in sN1.num_scheduled_tokens
(cb,) = sched.kv_cache_manager.get_block_ids(c.request_id) or ([],)
cb = list(cb)
c_has_bad = bad in cb
c_hashed_before = [pool.blocks[x].block_hash is not None for x in cb]

outN = sched.update_from_output(
    sN, create_model_runner_output([r1], invalid_block_ids={bad})
)
c_status = c.status.name
c_ncomp = c.num_computed_tokens
c_hashed_after = [pool.blocks[x].block_hash is not None for x in cb]
owner_bad_hashed = pool.blocks[bad].block_hash is not None if not c_has_bad else "n/a"

print(
    f"role={role} policy={policy} defer={sched.defer_block_free} | "
    f"record={'yes' if rec_ids else 'no'} | C scheduled N+1={c_sched} "
    f"C got bad block={c_has_bad} (C idx {cb.index(bad) if c_has_bad else '-'}) | "
    f"after report: C status={c_status} C.num_computed={c_ncomp} "
    f"C hashes before={''.join('H' if h else '-' for h in c_hashed_before)} "
    f"after={''.join('H' if h else '-' for h in c_hashed_after)}"
)
