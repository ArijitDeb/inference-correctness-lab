# Follow-up probes for vllm-project/vllm#61912

CPU-only scheduler probes for Lewis-panda's patch
[`89a93299`](https://github.com/Lewis-panda/vllm/commit/89a93299dd19219cb1e507a61a299a7aeb79659c)
for load-failure reports that arrive after the owner was **preempted**
(#61912), on top of #61769 (`7238ccdf`).

They drive the real `Scheduler`/`AsyncScheduler` through public entry points
only: `add_request`, `schedule`, `update_from_output`, `finish_requests` and
`reset_prefix_cache(reset_running_requests=True)`. Only the KV connector is a
mock (`probe_common.py`): a `Mock(spec=...)` whose methods return the
`KVConnectorBase_V1` defaults, except that it offers a synchronous external load
once per request. A load failure is reported as a connector reports it,
`invalid_block_ids` on the step's output. Model outputs are synthetic: token 999
marks step N's output, 998 step N+1's, so a delivered token shows which step
produced it. No model runs, and nothing here measures latency.

Earlier evidence (pressure preemption and abort-then-reuse on main and #61769)
is in [`../followup-61768/`](../followup-61768/README.md).

| Probe | Case |
|---|---|
| `probe_reset_order.py <policy> <repair>` | consumer role; `reset_prefix_cache(reset_running_requests=True)` in the load step, then `schedule(N+1)` before report N; afterwards an optional identical prompt (`repair=1`) and a longer prompt with the same prefix |
| `probe_gate_scope.py <policy> <failed>` | step at which a preempted owner is readmitted, under KV pressure or after a reset, with two (async) or three (synchronous PP=3) batches in flight |
| `probe_rewound_preempt.py` | consumer role, `recompute`: the report rewinds a running owner, then KV pressure preempts it |

Every scenario runs with `failed=True` (report N rejects one of r1's loaded
blocks) and with a load-ok control (`failed=False`, same steps, no report).

## Trees

| Output file | Tree |
|---|---|
| `outputs/pr-7238ccdf.txt` | #61769 head `7238ccdf` |
| `outputs/patch-89a93299.txt` | Lewis-panda/vllm `89a93299` (parent `7238ccdf`) |
| `outputs/patch-89a93299-gate-ablated.txt` | `89a93299` + `gate-ablation.patch` (removes only the new admission check in `schedule()`) |
| `outputs/combined-53298-1eca68f4-89a93299.txt` | local merge: #53298 head `1eca68f4`, then `git merge 89a93299`; two conflicts resolved by keeping both sides: in `tests/v1/kv_connector/unit/utils.py` both sets of `assert_scheduler_empty` checks, in `_preempt_request` `self._remove_request_from_kv_recovery(request)` followed by `self._track_inflight_kv(request)` |

## Running

```bash
PYTHON=/path/to/vllm/.venv/bin/python ./run_all.sh /path/to/vllm-tree
```

The tree must be importable (compiled extensions available) and include
`tests/` (the probes use `tests/v1/kv_connector/unit/utils.py`, identical in
`7238ccdf` and `89a93299`). vLLM's test utils build the config from
`facebook/opt-125m`, which must be in the local Hugging Face cache;
`run_all.sh` sets `HF_HUB_OFFLINE=1`. CPU (`CUDA_VISIBLE_DEVICES=""`), V1 model
runner. In the outputs, `H` = block still hashed in the prefix cache, `-` = not.

## Results

### Reset in the load step, consumer role (`probe_reset_order.py`)

The reset preempts r1, then raises because r1's blocks are still held; the
probe continues scheduling, as the engine does. r1's block 7 is computed in step
N, over the blocks that then fail.

| `failed=True` | #61769 `7238ccdf` | patch `89a93299` | patch, gate ablated | + #53298 |
|---|---|---|---|---|
| r1 scheduled in N+1 (before report N) | yes, 112-token hit | no | yes, 112-token hit | no |
| `fail`: r1 after report N | `FINISHED_ERROR`, `HHH----H` | `FINISHED_ERROR`, `HHH-----` | `FINISHED_ERROR`, `HHH-----` | `FINISHED_ERROR`, `HHH-----` |
| `fail`: longer same-prefix prompt, straight after | starts at 48 | starts at 48 | starts at 48 | starts at 48 |
| `fail`: longer prompt after an identical prompt recomputed blocks 3-6 | **hits r1's block 7** (starts at 128) | starts at 128, fresh block 7 | starts at 128, fresh block 7 | starts at 128, fresh block 7 |
| `recompute`: r1 after report N | `RUNNING`, `HHHHHHHH` | `PREEMPTED`, `HHH-----` | `RUNNING`, `HHH-----` | `PREEMPTED`, `HHH-----` |
| `recompute`: token 998 (step N+1, over the unloaded blocks) delivered | **yes** | no | **yes** | no |
| `recompute`: longer same-prefix prompt, straight after r1 | **hits r1's block 7** (starts at 128) | starts at 128, fresh block 7 | starts at 48 | starts at 128, fresh block 7 |

Under `recompute`, r1 recomputes blocks 3-6 itself, so on `7238ccdf` no second
request is needed before a longer prompt reaches the retained block 7. With the
gate ablated, the longer prompt's hit stops at 48 and does not reach r1's
block 7 (a lost cache hit, not wrong KV).

An identical 128-token prompt never reaches block 7: the cache hit is capped at
`num_tokens - 1` (`get_computed_blocks`, `vllm/v1/core/kv_cache_manager.py`).
It takes a longer prompt with the same prefix.

Load-ok controls (all trees): r1's blocks stay `HHHHHHHH` and the longer prompt
reuses r1's block 7, which is valid there. Token 998 is delivered wherever r1 was
scheduled in N+1, which is correct when the load succeeded.

### Admission gate: patch vs. patch with the gate removed (`probe_gate_scope.py`)

Readmission step of the preempted owner, `(step, start)`; `-` = not readmitted
(failed under `fail`). Depth = batches in flight. "Deliver" emulates a
best-effort connector by setting the scheduler's `requires_kv_delivery` to
False; "drop" is the NixlConnector producer default.

| Scenario | Run | patch | gate ablated |
|---|---|---|---|
| async (depth 2), producer, KV pressure, deliver | all | same | same |
| async (depth 2), producer, KV pressure, drop | all | same | same |
| async, consumer, reset (with two competing requests) | all | same | same |
| async, producer, reset | `fail`, failed | `-` | (1, 0), then failed |
| | `recompute`, failed / load-ok controls | (2, 0) | (1, 0) |
| sync PP=3 (depth 3), producer, KV pressure, deliver | `recompute`, failed | (4, 16) | (3, 16) |
| | load-ok | (6, 48) | (6, 48) |
| sync PP=3 (depth 3), producer, KV pressure, drop | `recompute`, failed | (4, 16) | (3, 16) |
| | load-ok | (4, 48) | (3, 48) |

The PP=3 rows omit `fail` with a failure: there both trees fail r1 without
readmitting it.

With two batches in flight, the step that preempts admits no waiting requests,
and the owner's record has drained by the next `schedule()`, so removing the
gate changes nothing under pressure. After a reset it can hold readmission back
one step, and with three batches in flight it can also delay pressure-preempted
owners. In the reset-order probe above, removing it lets r1 into step N+1 and
token 998 is delivered under `recompute`.

### Running owner rewound, then preempted (`probe_rewound_preempt.py`)

Consumer role, `recompute`. The report finds r1 running and rewinds it to 48
computed tokens without evicting its blocks; KV pressure preempts it at N+2.

| | #61769 `7238ccdf` | patch `89a93299` | patch, gate ablated | + #53298 |
|---|---|---|---|---|
| r1 resumes at N+3 with start (correct: 48) | 112 | 112 | 112 | 48 |
| load-ok control | 112 | 112 | 112 | 112 |

The 112-token hit includes blocks 3-6, which were never repaired. This case is
live-owner recovery, outside the patch; #53298 handles it.

## Limits

CPU scheduler probes with a mock connector and synthetic outputs. They show
which blocks stay in the prefix cache and which step's output is delivered, not
model output quality. Whether a real connector loads synchronously under a
top-level `kv_producer` role is not established here (see `../followup-61768/`).
PP=3 was probed with synchronous scheduling; other configurations with three or
more batches in flight were not run. No latency was measured.
