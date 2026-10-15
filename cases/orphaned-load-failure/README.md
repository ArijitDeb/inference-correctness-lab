# vLLM orphaned load failure: GPU reproduction

**The bug.** A request's synchronous external KV load fails, and the request is aborted in the same engine step. `EngineCore.step` runs `schedule → execute_model → _process_aborts_queue → update_from_output` (`vllm/v1/engine/core.py:644-668` at `04462732`). So the abort frees the request *before* the load-failure report is processed. `Scheduler._handle_invalid_blocks` scans only `running` and `kv_holding_waiting`, finds nobody, and returns early (`if not total_failed_requests: return set()`) without evicting. The request's blocks were hashed into the prefix cache at schedule time, and freeing keeps the hash. A later identical prompt gets a prefix-cache hit on KV that was never written. This happens under the default `kv_load_failure_policy="fail"` as well as `recompute`, with sync and with async scheduling.

**Results:** [RESULTS.md](RESULTS.md) (table, environment, commits). **Fix:** the commit measured here is `e23ecfa8` on branch [`orphaned-load-failure-04462732`](https://github.com/ArijitDeb/vllm/tree/orphaned-load-failure-04462732), on the same base as `main`. The PR branch [`orphaned-load-failure`](https://github.com/ArijitDeb/vllm/tree/orphaned-load-failure) carries the same patch rebased onto current upstream main. Both carry the CPU regression tests.

**The name.** The worker's failure report arrives after its owner is gone: an orphaned report, so nothing cleans up after it.

This repo turns the CPU-level finding (scheduler state is wrong) into a token-level one (the model's output is wrong), on three vLLM trees:

| Tree | What it is | Expected |
|---|---|---|
| `main` | `04462732` | abort rows: r2 hits r1's never-written blocks, r2 ≠ baseline |
| `pr-53298` | PR [#53298](https://github.com/vllm-project/vllm/pull/53298) at `a7dd070b`, merged onto the same main (branch `olf/pr-53298-on-main`) | same as main (the PR registers repair owners only from `running`, only under `recompute`). The PR's head as of 2026-10-14, `1eca68f4`, sits on a newer main whose kernels differ from this build (221 files under `csrc/`), so it is checked on the CPU regression tests instead: same failure, r2 hits 112 tokens where 48 are valid |
| `fix` | `e23ecfa8` (branch `orphaned-load-failure-04462732`), based on the same main | abort rows: r2 cached = 0, r2 = baseline |

## How the abort lands in the window, deterministically

There's no racing the clock. The fault-injecting connector (`olf_connector.py`) drops r1's load on the worker side, inside `execute_model`. At that moment it puts r1's id on `EngineCore.aborts_queue`. `step()` drains that queue before `update_from_output`, which is exactly where a real client abort arriving mid-step lands. The abort goes through the production path (`_process_aborts_queue → abort_requests → finish_requests`), and the `abort via` column records, per trial, that this path is what finished r1. The client side is told about the abort only after the engine has applied it, as cleanup, so it can never be the abort itself. The scheduler's own logic is untouched: `UpdateProbe` wraps `update_from_output` only to observe, then delegates.

`--abort patch` is a fallback that monkeypatches `update_from_output` to finish the owners first. The matrix runs it once per policy as a cross-check: it should agree with the queue path. `UpdateProbe` records, for each report, whether r1 was still in `running` when the report arrived. It also records `kv_holding_waiting` membership, the other collection the handler scans, and which `update_from_output` call carried the report. In the queue rows, the `owner gone at report` column is the proof that the window was hit: the report for a step that scheduled r1 arrived after r1 had left both collections. In the patch rows it is not independent evidence, because the probe performed the abort itself.

## Trial flow (per trial, one engine per cell)

1. **canary** (with `--canary`): an unrelated prompt with a distinctive word (`PINEAPPLE`), run in a KV pool sized so r1 has to reuse its freed blocks. Without it, r1's blocks come from never-used pool memory, whose contents we did not measure; that could mask the corruption. With it, the `canary reuse` column shows how many of r1's blocks were the canary's freed blocks. The hypothesis was that r2's output would be pulled toward the canary. It was not borne out: the canary word never appeared in r2 (0/3 in every canary row), although r2's garbage differs from the fresh-pool case, so whatever the unloaded blocks hold does shape the output. No content transfer is claimed.
2. **r1**: the decode prompt. Its external load is granted and then dropped, so the blocks are never written. With `--abort queue|patch` it is aborted in the same step.
3. **r2**: the same prompt as a new request. The connector offers r2 nothing, so any hit is a local prefix-cache hit on r1's blocks.

Trials are **repeated trials within one engine per cell, not cold runs**, and they measure determinism, not a rate: the injection is deterministic. A natural occurrence rate would need a separate measurement of how wide the window gets, for example under a slow store. Trials share one engine per cell, with `reset_prefix_cache()` between them; the KV pool's memory is not cleared, so token-level output can in principle differ between trials of a cell (the `diverge@` column would show several values). The state columns (`hashed`, `r2 cached`) don't depend on memory contents.

The storage directory is shared by every cell of a tree. The `kv_both` connector also stores KV for requests it has no folder for (the canary, for example). That never affects results: loads are always dropped, so only whether r1's folder exists matters, and it must exist for r1 to be granted.

## Two more variants

- **`--async-scheduling`** runs vLLM's default scheduling: `AsyncScheduler` with the batch queue (`step_with_batch_queue`, two batches in flight). The engine drains aborts before each batch's `update_from_output` on this path too (`core.py:768-769`), and `abort via` confirms it applied r1's abort there. Each result records the scheduler class, the batch-queue size and the step function, so the JSON shows the path really ran. Under the batch queue a request can look finished to the client while a batch that scheduled it is still in flight, so the harness drains the queue after every request (`drain_batch_queue`) before measuring and before the next request starts. Without that, r2 runs concurrently with r1's failure report and takes the shared-request path instead: the handler finds r2 running with the invalid blocks and fails it by policy. The baseline always uses plain sync scheduling: it is the oracle, full recompute with no cache.
- **`--no-load-failure`** is the positive control: the load succeeds and only the mid-step abort remains. A correct fix must keep the aborted request's valid cached KV (r2 cached = 1904, r2 = baseline) on every tree, including the fix. A fix that flushed the cache on every abort would fail here. `abort via` shows the abort still landed mid-step through the queue.

Every trial also records the external tokens the connector offered r2 (`r2 ext`, must be 0) and, on trees with the fix, how many of the fix's records are left after r2 (`records left`, must be 0).

## Reading the output: is the model wrong, or just different?

Greedy token equality is a weak oracle: two runs can agree on every argmax while their distributions drift, or differ on a token for numerical reasons. So every request asks vLLM for the top 20 log-probabilities, and the table reads the distribution at r2's first generated token, right after the prompt:

- The probability r2 gives the token the baseline produced. The baseline's own value is printed under the table. A healthy r2 matches it; an r2 that attended over unpopulated KV gives it a small fraction of that.
- Its rank in r2's top 20, and how many of the baseline's top-20 candidates r2 still considers (∩).

The `next token, per trial` column shows all three for each trial, in trial order, as `P (rank, ∩)`.

These turn "the text looks wrong" into "the model no longer conditions on its prompt", and they would catch a partial corruption that greedy decoding happens to survive.

## Running it

Setup, once: a vLLM clone at `$VLLM_CLONE` (default `~/Projects/vllm`) installed in its own venv at `04462732` with precompiled kernels (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`, as in vLLM's contributor setup), plus the two refs the trees are built from:

```bash
cd "$VLLM_CLONE"
git branch main-04462732 04462732a40a8329ad60593d754b576c54c5fe5d
git fetch origin pull/53298/head && git branch pr-53298 a7dd070bace3178117e9ec13bb7f70403cddc15f
git fetch https://github.com/ArijitDeb/vllm orphaned-load-failure-04462732:orphaned-load-failure-04462732
```

Then, from this directory (`cases/orphaned-load-failure/`):

```bash
./setup_trees.sh main
./setup_trees.sh pr
./setup_trees.sh fix orphaned-load-failure-04462732

# the GPU must be free: nothing else may hold more than 4 GB
PY=$VLLM_CLONE/.venv/bin/python ./run_matrix.sh main
PY=$VLLM_CLONE/.venv/bin/python ./run_matrix.sh pr-53298
PY=$VLLM_CLONE/.venv/bin/python ./run_matrix.sh fix

python3 compare.py      # rebuilds results/table.md from the JSON; any Python 3.10+
python3 -m pytest tests # checks the table logic, no GPU needed
```

**Tested on:** one RTX 5090 (32 GB, driver 610.57.04), torch 2.13.0+cu130, vLLM at `04462732` (`0.31.1rc1.dev222`), Qwen/Qwen3-0.6B in bf16. Each engine takes 40% of GPU memory by default (`--gpu-memory-utilization`); about 25 minutes for all three trees.

`run_matrix.sh` refuses to start while the GPU has more than 4 GB in use, and checks that `vllm` resolves to the tree under test. It runs every cell even if one fails, lists the failures and exits nonzero. Run `compare.py` once every tree is done; it marks any missing cell. Re-running a tree deletes that tree's previous results and storage. Each tree runs 18 engines on Qwen3-0.6B (store, baseline, and per policy: 4 sync cells, the patch cross-check, 2 async cells and the load-ok control), which has full attention only (one KV cache group; with several, `_handle_invalid_blocks` raises).

`setup_trees.sh` makes git worktrees of the clone under `trees/` and symlinks the clone's precompiled artifacts into each. The kernels were built for `04462732`, so every tree must sit on that base. The PR tree is #53298 at `a7dd070b` merged onto it (branch `olf/pr-53298-on-main` in the clone).

## Reading the table

Check the controls first: the `none` rows.

- `fail` with no abort: r1 errors (failed by policy), its blocks are evicted, so `hashed` is 0 and r2 cached = 0, r2 = baseline.
- `recompute` with no abort: r1 recomputes, so r2 cached is about the full prompt (valid blocks) and r2 = baseline.

**Expected exception on `main` and `fix`:** the `recompute` control fails on both, with r1 itself different from the baseline. This is consistent with the public issue [#49250](https://github.com/vllm-project/vllm/issues/49250) (`recompute` gives wrong output when a connector rejects a synchronous load). The only code difference between `main` and `pr-53298` is PR #53298 (8 files under `vllm/`, in the scheduler and the GPU model runner), which says it fixes #49250, and the `pr-53298` tree's `recompute` control matches the baseline. We did not isolate the mechanism. We treat it as separate from this bug because the `fail` rows and the abort rows are identical between `main` and `pr-53298`. `fix` sits on the same main without #53298, so it inherits this control failure; what the fix changes is its abort rows. Any other control failure means stop: the environment, drift in the example, or a further bug in the #49250 family.

The `recompute` control compares tokens exactly, although r2 reuses cached KV while the baseline does one full prefill. Different kernel shapes in bf16 could in principle flip a late greedy token. Treat a mismatch as numeric noise only if all of these hold: r1 matches the baseline, the divergence is not at token 0, and a rerun reproduces the same `diverge@` index. Otherwise it is a control failure.

In an abort row, the primary signal is **r2 cached > 0**: r1's loaded blocks were never written, so any hit on them is the bug. Token divergence is secondary: if cached > 0 but tokens still match, the unwritten memory happened to be harmless, and the state bug is still proven. Run with `--canary` or a larger `--max-tokens`.

The decode prompt is 1909 tokens. The connector loads the block-aligned prompt minus its last token, align(1908, 16) = 1904 tokens = 119 blocks, and reports all of them; the 120th, partial block is computed locally. So block 0 is the first bad block, and a correct fix gives r2 cached = 0 in failure rows. The property that "blocks before the bad one stay cached" is covered by the CPU regression tests on the fix branch (`tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py`), not here.

## Pitfall found while building this

Running a vLLM unit test with `PYTHONPATH=<another tree>` from inside the clone does **not** test that tree. pytest's rootdir handling puts the clone first on `sys.path`, so the clone's `vllm` wins. Run tests from inside the tree's own worktree, or check `vllm.__file__` in the session.

## Files

| File | What |
|---|---|
| `olf_connector.py` | the fault-injecting connector (subclass of the upstream example's `LoadRecoveryExampleConnector`) |
| `harness.py` | `store` / `baseline` / `trial` modes; one JSON per invocation, with env and tree commit |
| `compare.py` | the table, regenerable from the JSON |
| `setup_trees.sh`, `run_matrix.sh` | tree setup and the full matrix for one tree |
| `tests/test_compare.py` | tests for the table logic, including that `compare.py` expects exactly the cells `run_matrix.sh` runs |
| `RESULTS.md` | the published table and the environment it came from |

Public sources only: the vLLM repo (Apache-2.0), public issues and PRs, public weights. Licensed under Apache-2.0, see [LICENSE](../../LICENSE).
