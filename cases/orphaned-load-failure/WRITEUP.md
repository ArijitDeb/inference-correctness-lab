# vLLM published KV before it was written: the orphaned load failure

**Summary.** With prefix caching on, vLLM v1 adds a request's KV blocks to its prefix cache when it schedules the request, before the KV exists. When a synchronous connector load for those blocks fails and is reported through `invalid_block_ids`, the scheduler's failure handler is expected to evict them again. If the request is aborted in the same step, it has left the queues the handler scans by the time the report arrives; with no affected request left, the handler returns before evicting, and the never-loaded blocks stay in the cache. A later request with the same prompt reuses them and its output is wrong. Issue [vllm-project/vllm#61768](https://github.com/vllm-project/vllm/issues/61768), fix [#61769](https://github.com/vllm-project/vllm/pull/61769), reproduction and full results in this directory.

## Publishing before the work is done

Inference engines publish state early for speed and rely on a promise that the work will land. vLLM's prefix cache is one example. `KVCacheManager.allocate_slots` hashes a request's full blocks into the cache at schedule time, before the model runs. For locally computed KV, this relies on execution ordering making the writes visible before any dependent read. For KV loaded synchronously from an external store (disaggregated prefill, KV offload), the blocks are published before the load has happened, and the promise is kept by `Scheduler._handle_invalid_blocks`: when the worker reports blocks whose load failed, the handler finds the request that owns them and, under the default `kv_load_failure_policy="fail"`, evicts them.

Three questions find bugs of this shape in any subsystem:

1. What does it publish before the work is done?
2. Who holds the promise that the work will land or be undone?
3. What happens if that holder leaves (abort, preemption, failure) before the promise is settled, or a reader arrives first?

Here the holder is the request, and the question is what happens if it leaves.

## How the promise breaks

`EngineCore.step` runs `schedule → execute_model → _process_aborts_queue → update_from_output`; `step_with_batch_queue`, used for async scheduling, keeps the same order of aborts before output processing for each batch. An abort that arrives while the model runs, for example a client disconnecting, is applied before that step's output is processed. So:

1. `schedule()` admits request R with a synchronous external load and hashes its blocks.
2. `execute_model()`: R's load fails; the worker reports the blocks in `invalid_block_ids`.
3. `_process_aborts_queue()` applies R's abort. `finish_requests` frees R; freeing keeps the blocks' hashes.
4. `update_from_output()` calls `_handle_invalid_blocks`, which looks for affected requests in `running` and `kv_holding_waiting`. R has left both. With no affected request left, it returns before evicting.
5. A later request with the same prompt gets a prefix-cache hit on R's never-loaded blocks.

The handler's early return is reasonable on its own, and so is applying aborts before processing output. The bug lives in the interleaving: the report is orphaned, because its owner left between scheduling the load and the processing of its failure report.

## Making the window deterministic

The window runs from scheduling the load to processing its report, so racing a real client disconnect against a real store failure would give a flaky reproduction. Instead the harness places the abort exactly. A fault-injecting connector (a subclass of vLLM's own `LoadRecoveryExampleConnector` example) drops the load inside `execute_model`, and in that same call puts the request's id on `EngineCore.aborts_queue`. The engine then applies the abort through its production path, `_process_aborts_queue`, before the output is processed. Nothing in the scheduler is patched.

Two records make the result checkable, not just plausible:

- A probe on `update_from_output` records, for each failure report, whether the owner was still in `running` or `kv_holding_waiting` when the report arrived. "Owner gone at report" is the window, observed rather than assumed.
- A probe on `_process_aborts_queue` records which path actually applied each abort. An earlier version of the harness told the client side about the abort too early under async scheduling, and that client call became the abort itself. The table looked fine. Recording the path, and failing the table when it doesn't match, closed that hole.

## Is the model wrong, or just different?

Comparing greedy text is a weak test. Two runs can agree on every argmax while their probabilities drift, or differ on one token for numerical reasons. So every request also asks for the top 20 log-probabilities, and the comparison looks at the first generated token: the probability r2 assigns to the token the baseline produced, its rank, and the overlap of the two top-20 sets. The baseline is the same prompt on the same model with no connector and no cache: a full recompute.

In each trial, r1 is the request whose load fails and which is aborted mid-step; r2 then repeats r1's prompt with no external load, so any cache hit it gets is local. "Synchronous" describes the connector load, not the scheduler: both scheduling modes are tested. The prompt is the numbers 1 to 400 ending "The next number after 400 is". GPU results were produced on 2026-10-14 on an RTX 5090 with Qwen3-0.6B, on upstream main `04462732`, #53298 at `a7dd070b` merged onto that main, and the fix at `e23ecfa8` on the same main. The fix was then rebased onto main `81e1c3f1` and re-validated there with the CPU tests.

`fail` policy, sync scheduling, three trials per row (probabilities are the range across trials; the text is representative):

| | r2's prefix-cache hit | r2's output | P(baseline's first token) |
|---|---|---|---|
| baseline (full recompute) | 0 | `401. So, the answer is 401.` | 0.962 |
| main, r1 aborted mid-step | 1904 of 1909 tokens | `7 7 7 7 7 7 7 7 …` | 0.0005-0.005 (rank 8-11) |
| with the fix | 0 | `401. So, the answer is 401.` | 0.962 |
| control: r1's load succeeds, r1 aborted mid-step | 1904 | `401. So, the answer is 401.` | 0.963 |

The abort rows are the same under the `recompute` policy and under async scheduling, and the same on #53298. That PR handles recompute recovery, including handing repair ownership over when a request is cancelled after a failure was processed; here the request is gone before the failure report can establish repair ownership. The two are complementary: the "recompute, no abort" control fails on main and on the fix and passes with #53298.

The successful-load control matters as much as the bug row. A fix that flushed the cache on every abort would pass the bug row and fail the control.

## Wrong, and fast

On a 0.6B model the 1.9k-token prefill costs a few milliseconds, so a cache hit doesn't show up in time to first token. On Qwen3-8B-Base it does. Same harness, same prompt, `fail` policy, sync scheduling, five trials per row, on main `04462732` and the fix `e23ecfa8` (time to first token measured in-process from adding r2 to its first token, in eager mode; a comparison between rows, not a serving benchmark; the slowest trial in each row is the first, warming up):

| r2 on | r2's prefix-cache hit | Matches baseline | Time to first token, median |
|---|---|---|---|
| main, no abort (r1 failed and was evicted, so r2 recomputes) | 0 | 5/5 | 136 ms |
| **main, r1 aborted mid-step (the bug)** | **1904** | **0/5** | **22 ms** |
| main, r1's load succeeded and r1 was aborted | 1904 | 5/5 | 22 ms |
| **with the fix, r1 aborted mid-step** | **0** | **5/5** | **135 ms** |

In the bug row, r2 gave the baseline's first token a probability of 0.0009 to 0.51 across trials (baseline 0.991) and diverged from the baseline in every trial. Full table: [RESULTS-8B.md](RESULTS-8B.md).

The wrong answer arrives about six times faster than the correct recompute and exactly as fast as a legitimate cache hit. Neither its time to first token nor its cache hit distinguishes it from a healthy request; only a comparison of the output does. The fix gives up that speed for those requests, and only for them: with it, the control's valid cache hit stays at 21 ms.

## The fix

The request's place in the scheduler's queues is the wrong lifetime for this state. Once the request has left the queues the handler scans, outputs of the steps that scheduled it can still arrive, and only those carry the failure report. So the fix keeps a small record of a departing request's block ids for as long as it has scheduled tokens whose output hasn't been processed (`num_in_flight_tokens > 0`). Each output counts the record down, and `_handle_invalid_blocks` evicts from the first invalid block onward for any recorded request, under both policies, before its early return. The blocks after the invalid one are evicted too: they were computed on top of it, and keeping their hashes would let a later request that recomputes the bad block rebuild the chain and reach them again.

This covers the existing single-KV-cache-group path; `_handle_invalid_blocks` already rejects block-level reports with several groups. Keeping physical block ids rather than hashes is safe because a recorded id can't be handed to another request while the record lives. With one batch in flight, nothing allocates between the abort and the output that retires the record. With several batches in flight and a connector in the `kv_consumer` or `kv_both` role (the only roles that load KV), vLLM defers freeing the request's blocks until the steps that might still write them have been processed; those deferred frees can drain at the start of `update_from_output`, but no allocation happens before the invalid blocks are evicted and the record is retired in the same call. The change is 51 lines in the scheduler.

## Cost

The fix adds no GPU kernels and no host synchronization. Its costs are on the CPU: copying a request's block ids when it is freed with work in flight (with sync scheduling only for requests aborted mid-step, with async scheduling also for requests that finish while their next batch is in flight), counting each record down as outputs arrive (nothing when there are no records), and scanning the recorded block ids when a load failure is reported. That overhead has not been benchmarked. Separately, requests that would have reused poisoned blocks now recompute them: more GPU work, by design. The bug itself costs correctness rather than compute: a deployment with a connector that reports block-level load failures can return wrong output for later requests that reuse the invalid blocks, while those entries remain in the cache, and the request's prefix-cache hit counts as a success (on the 8B model, the wrong answer arrived about six times faster than a correct recompute). Reproducing everything here takes about 25 minutes on one RTX 5090.
