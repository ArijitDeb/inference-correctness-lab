# SPDX-License-Identifier: Apache-2.0
"""Fault-injecting KV connector for the orphaned-load-failure repro.

Builds on the upstream example connector in
``examples/disaggregated/kv_load_failure_recovery_offline/``
(``LoadRecoveryExampleConnector``), which, in sync mode, drops the first load
request in each step's connector metadata and reports every block that load
covered through ``get_block_ids_with_load_errors()``. The load covers the
block-aligned prompt minus its last token, so block 0 is the first bad block
and the last, partial block (computed locally) is not reported.

This subclass adds four things the harness needs:

1. A scheduler-side grant: only the request whose internal id is ``GRANT_TO``
   is offered external tokens. Everyone else gets ``(0, False)``. Without
   this, a correct fix (which evicts r1's poisoned blocks) gives r2 a zero
   local hit, r2 then loads from storage, the injected failure fires again
   and the result is unreadable.
2. A worker-side hook: in the step that carries r1's load (inside
   ``execute_model``), ``ON_LOAD_STEP`` is called. The harness uses it to put
   r1's id on ``EngineCore.aborts_queue``, so the abort goes through the real
   ``_process_aborts_queue`` path before ``update_from_output`` runs.
3. ``FAIL_LOADS = False`` turns the injected failure off: the load succeeds
   and only the abort remains, the positive control (a correct fix must keep
   valid cached KV of an aborted request).
4. A record of what happened (``STATE``) for the result JSON, including the
   external tokens offered to every request.

Everything relies on ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` and the uniproc
executor: the scheduler-side and worker-side connector instances are separate
objects, but they live in the harness's process and share this module's
globals. The module must be imported under the same name the
``kv_connector_module_path`` uses (``olf_connector``).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from load_recovery_example_connector import LoadRecoveryExampleConnector

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

# Set by the harness between add_request() and the first step().
GRANT_TO: str | None = None
# Called on the worker side, inside execute_model, in the step with r1's load.
ON_LOAD_STEP: Callable[[], None] | None = None
# False: let the load succeed (positive control).
FAIL_LOADS: bool = True

STATE: dict[str, Any] = {}


def reset_state() -> None:
    STATE.clear()
    STATE.update(
        granted_tokens=0,
        grants={},
        r1_block_ids=None,
        load_steps=0,
        dropped_loads=0,
        dropped_block_ids=None,
    )


reset_state()


class OrphanedLoadFailureConnector(LoadRecoveryExampleConnector):
    # ---- scheduler side ----

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if GRANT_TO is None or request.request_id != GRANT_TO:
            STATE["grants"].setdefault(request.request_id, 0)
            return 0, False
        num_tokens, load_async = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if num_tokens:
            STATE["granted_tokens"] = num_tokens
            STATE["grants"][request.request_id] = num_tokens
        return num_tokens, load_async

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens > 0 and request.request_id == GRANT_TO:
            STATE["r1_block_ids"] = list(blocks.get_block_ids()[0])

    # ---- worker side ----

    def bind_connector_metadata(self, connector_metadata: "KVConnectorMetadata") -> None:
        has_load = any(not r.is_store for r in connector_metadata.requests)
        if FAIL_LOADS:
            # The parent drops the first load request and sets
            # self._invalid_block_ids; clear_connector_metadata() resets it to
            # None at the end of every step.
            super().bind_connector_metadata(connector_metadata)
        else:
            # Skip the parent's injection: plain ExampleConnector loading.
            super(LoadRecoveryExampleConnector, self).bind_connector_metadata(connector_metadata)
        if self._invalid_block_ids:
            STATE["dropped_loads"] += 1
            STATE["dropped_block_ids"] = sorted(self._invalid_block_ids)
        if has_load:
            STATE["load_steps"] += 1
            if ON_LOAD_STEP is not None:
                ON_LOAD_STEP()
