# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the #61912 follow-up probes. Run from the vLLM tree root.

The scheduler is real (`create_scheduler` from vLLM's own test utils). Only the
KV connector is replaced: a `Mock(spec=...)` in which every method the scheduler
consults returns the `KVConnectorBase_V1` default, except
`get_num_new_matched_tokens`, which offers a synchronous external load once per
request. Any method not listed raises AttributeError, so an unexpected call fails
the probe instead of returning a truthy Mock.

Events reach the scheduler only through public entry points: `add_request`,
`schedule`, `update_from_output`, `finish_requests` and
`reset_prefix_cache(reset_running_requests=True)`. A load failure is reported the
way a connector reports it: `invalid_block_ids` on the step's output.
"""

import sys
from unittest.mock import Mock

sys.path.insert(0, ".")  # run from the vllm tree root

from tests.v1.kv_connector.unit.utils import (  # noqa: E402
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BS = 16  # block size used by create_vllm_config
_REQUESTS = {}  # req_id -> Request, kept after the scheduler forgets it

_CONNECTOR_METHODS = [
    "get_num_new_matched_tokens",
    "get_external_cache_hit_sources",
    "get_loaded_kv_cache_group_ids",
    "has_pending_block_frees",
    "has_pending_push_work",
    "on_new_request",
    "update_connector_output",
    "update_state_after_alloc",
    "request_finished",
    "register_finished_partial_tail",
    "take_events",
    "build_connector_meta",
    "get_kv_connector_stats",
    "bind_kv_cache_manager",
    "reset_cache",
    "shutdown",
    "requires_kv_delivery",
    "get_kv_connector_kv_cache_events",
    "supports_divergent_local_hybrid_hits",
]


def make_scheduler(
    policy, role, *, num_blocks=10000, async_scheduling=True, pp=1, chunk=None
):
    """Scheduler with kv_load_failure_policy=`policy` and top-level kv_role=`role`.

    `requires_kv_delivery` comes from the real connector the config builds
    (NixlConnector: True for kv_producer, False for kv_consumer).
    """
    cfg = create_vllm_config(
        kv_load_failure_policy=policy, kv_role=role, max_num_batched_tokens=256
    )
    cfg.scheduler_config.async_scheduling = async_scheduling
    cfg.parallel_config.pipeline_parallel_size = pp
    if chunk is not None:
        cfg.scheduler_config.long_prefill_token_threshold = chunk
    sched = create_scheduler(cfg, num_blocks=num_blocks)
    real = sched.connector
    offers = {}  # req_id -> external tokens offered, consumed on first lookup

    conn = Mock(spec=_CONNECTOR_METHODS)
    conn.get_num_new_matched_tokens.side_effect = lambda req, n: (
        offers.pop(req.request_id, 0),
        False,  # synchronous load
    )
    conn.get_external_cache_hit_sources.side_effect = lambda req, n: (
        {"external": n} if n else {}
    )
    conn.get_loaded_kv_cache_group_ids.side_effect = (
        real.get_loaded_kv_cache_group_ids
    )
    conn.has_pending_block_frees.return_value = False
    conn.has_pending_push_work.return_value = False
    conn.on_new_request.return_value = None
    conn.update_connector_output.return_value = None
    conn.update_state_after_alloc.return_value = None
    conn.request_finished.return_value = (False, None)
    conn.register_finished_partial_tail.return_value = False
    conn.take_events.return_value = ()
    conn.build_connector_meta.return_value = None
    conn.get_kv_connector_stats.return_value = None
    conn.bind_kv_cache_manager.return_value = None
    conn.reset_cache.return_value = None
    conn.shutdown.return_value = None
    conn.get_kv_connector_kv_cache_events.return_value = None
    conn.requires_kv_delivery = real.requires_kv_delivery
    conn.supports_divergent_local_hybrid_hits = False
    sched.connector = conn
    sched._probe_offers = offers
    return sched


def add(sched, rid, *, num_tokens=8 * BS, same_prefix=True, external=0, max_tokens=16):
    """Add a request. same_prefix=True: all prompts share one token sequence."""
    req = create_request(
        request_id=rid,
        num_tokens=num_tokens,
        common_prefix_len=num_tokens if same_prefix else 0,
        max_tokens=max_tokens,
    )
    if external:
        sched._probe_offers[req.request_id] = external
    _REQUESTS[req.request_id] = req
    sched.add_request(req)
    return req


def output(step, token, invalid=None):
    """Model output for `step`: every scheduled request samples `token`."""
    reqs = [_REQUESTS[rid] for rid in step.num_scheduled_tokens]
    return create_model_runner_output(
        reqs, invalid_block_ids=invalid, token_id=token
    )


def start(step, req):
    """Tokens `req` already had when `step` began (its cache hit on admission)."""
    if req.request_id not in step.num_scheduled_tokens:
        return None
    return req.num_computed_tokens - step.num_scheduled_tokens[req.request_id]


def block_ids(sched, req):
    (ids,) = sched.kv_cache_manager.get_block_ids(req.request_id)
    return list(ids)


def hashed(sched, ids):
    """'H' if the block is still in the prefix cache, '-' if not."""
    pool = sched.kv_cache_manager.block_pool
    return "".join("H" if pool.blocks[b].block_hash is not None else "-" for b in ids)


def delivered(result, req):
    """Token ids delivered to the client for `req` by one update_from_output."""
    return [
        t
        for outs in result.values()
        for o in outs.outputs
        if o.request_id == req.request_id
        for t in o.new_token_ids
    ]
