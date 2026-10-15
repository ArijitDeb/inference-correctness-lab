# SPDX-License-Identifier: Apache-2.0
"""GPU repro harness for the vLLM orphaned-load-failure bug.

The bug: a request's synchronous external KV load fails, and the request is
aborted in the same engine step. ``EngineCore.step`` runs
``schedule -> execute_model -> _process_aborts_queue -> update_from_output``,
so the abort frees the request before the load-failure report is processed.
``Scheduler._handle_invalid_blocks`` then finds no affected request, returns
early, and never evicts. The never-written blocks keep the prefix-cache hashes
they got at schedule time, and a later identical prompt hits them.

Modes (one engine per invocation, results written as JSON):

  store     ExampleConnector (kv_both) prefill: generate 1 token for the base
            prompt, save base + that token as the decode prompt.
  baseline  No connector: the decode prompt, greedy, full recompute. The
            oracle every trial is compared against.
  trial     Fault-injected connector. Per trial: [canary] -> r1 (load fails,
            optionally aborted mid-step) -> r2 (same prompt, local hits only).

Must run with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (set below, before vLLM is
imported) so EngineCore, the scheduler and the worker live in this process.
Which vLLM tree is used is decided by ``PYTHONPATH``; the JSON records
``vllm.__file__`` and the tree's commit so a result can't be misattributed.

Every abort records which path applied it (``abort_via``): ``queue`` is
``EngineCore._process_aborts_queue`` inside the step, the production path for
an abort that arrives while a step runs; ``patch`` is the fallback that
finishes the owner inside ``update_from_output``.
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

HERE = Path(__file__).resolve().parent
TREE = Path(vllm.__file__).resolve().parents[1]
EXAMPLE_DIR = TREE / "examples/disaggregated/kv_load_failure_recovery_offline"
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(HERE))

import olf_connector  # noqa: E402  (needs EXAMPLE_DIR on sys.path)
from compare import cell_name  # noqa: E402

BLOCK_SIZE = 16
CANARY_WORD = "PINEAPPLE"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def base_prompt_text(n: int) -> str:
    """A prompt whose correct continuation is obvious to a human reader."""
    numbers = ", ".join(str(i) for i in range(1, n + 1))
    return f"Here is a list of numbers: {numbers}. The next number after {n} is"


def canary_token_ids(tokenizer: Any, length: int) -> list[int]:
    """An unrelated prompt of exactly ``length`` tokens with a distinctive word.

    If r1's never-written blocks still hold the canary's KV, r2 attends over
    it; seeing the canary word in r2's output is the crude signal for that.
    """
    unit = tokenizer.encode(
        f"Remember this: the password is {CANARY_WORD}. ", add_special_tokens=False
    )
    ids = (unit * (length // len(unit) + 1))[:length]
    return ids


# --------------------------------------------------------------------------
# Environment record
# --------------------------------------------------------------------------


def git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def env_record() -> dict[str, Any]:
    return {
        "vllm_file": vllm.__file__,
        "vllm_version": vllm.__version__,
        "tree": str(TREE),
        "tree_commit": git(["rev-parse", "HEAD"], TREE),
        "tree_branch": git(["rev-parse", "--abbrev-ref", "HEAD"], TREE),
        # The harness imports vllm/ and the example connector under examples/.
        "tree_dirty": bool(
            git(["status", "--porcelain", "--untracked-files=no", "--", "vllm", "examples"], TREE)
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "harness_commit": git(["rev-parse", "HEAD"], HERE),
        "harness_dirty": bool(git(["status", "--porcelain", "--", "."], HERE)),
    }


# --------------------------------------------------------------------------
# Engine driving
# --------------------------------------------------------------------------


class LogCapture(logging.Handler):
    """Collects the scheduler's KV-load-failure log lines."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "KV load failure" in msg:
            self.lines.append(msg)


def output_record(out: Any, aborted: bool = False) -> dict[str, Any]:
    if out is None:
        return {"status": "aborted" if aborted else "missing"}
    comp = out.outputs[0]
    logprobs = None
    step0_topk = None
    if comp.logprobs:
        logprobs = [
            step[tok].logprob for step, tok in zip(comp.logprobs, comp.token_ids, strict=False)
        ]
        # The full top-k at the first generated token: the model's next-token
        # distribution right after the prompt, the most direct read of
        # whether it attended to that prompt.
        step0_topk = {str(tok): lp.logprob for tok, lp in comp.logprobs[0].items()}
    return {
        "status": "finished",
        "finish_reason": comp.finish_reason,
        "token_ids": list(comp.token_ids),
        "text": comp.text,
        "logprobs": logprobs,
        "step0_topk": step0_topk,
        "num_cached_tokens": out.num_cached_tokens,
        "prompt_len": len(out.prompt_token_ids or []),
    }


def run_request(
    llm: LLM,
    ext_id: str,
    prompt_ids: list[int],
    params: SamplingParams,
    *,
    core: Any = None,
    grant_external: bool = False,
    client_abort_after_drop: bool = False,
) -> tuple[dict[str, Any], str, list[float]]:
    """Add one request and step the engine until nothing is unfinished.

    ``grant_external`` makes this the one request the connector offers
    external tokens to (``olf_connector.GRANT_TO``). With
    ``client_abort_after_drop``, once the engine has applied the abort (the
    scheduler no longer holds the request as unfinished), the client side is
    told too, otherwise ``has_unfinished_requests()`` never clears: an abort
    applied inside the engine produces no output. Waiting for the engine
    first keeps this a cleanup: under the batch queue an earlier client abort
    would itself be the abort, bypassing the path under test.
    """
    engine = llm.llm_engine
    t_add = time.perf_counter()
    internal_id = engine.add_request(ext_id, TokensPrompt(prompt_token_ids=prompt_ids), params)
    if grant_external:
        olf_connector.GRANT_TO = internal_id
    final = None
    first_token_s: float | None = None
    step_times: list[float] = []
    client_aborted = False
    # Prefill takes a few chunked steps, then one step per output token. A
    # loop past this means an abort path left the request stuck; fail rather
    # than hold the GPU lane.
    max_steps = (params.max_tokens or 16) + 64
    while engine.has_unfinished_requests():
        if len(step_times) >= max_steps:
            raise RuntimeError(f"{ext_id}: still unfinished after {max_steps} steps")
        t0 = time.perf_counter()
        outs = engine.step()
        step_times.append(time.perf_counter() - t0)
        for out in outs:
            if out.request_id != ext_id:
                continue
            if first_token_s is None and out.outputs and out.outputs[0].token_ids:
                first_token_s = time.perf_counter() - t_add
            if out.finished:
                final = out
        if (
            client_abort_after_drop
            and not client_aborted
            and final is None
            and olf_connector.STATE["load_steps"] > 0
            and not request_unfinished(core, internal_id)
        ):
            # internal=True: the scheduler-side id. Idempotent on the engine
            # side, which skips already-finished ids.
            engine.abort_request([internal_id], internal=True)
            client_aborted = True
    record = output_record(final, aborted=client_aborted)
    record["ttft_s"] = first_token_s
    return record, internal_id, step_times


def request_unfinished(core: Any, request_id: str) -> bool:
    request = core.scheduler.requests.get(request_id)
    return request is not None and not request.is_finished()


class AbortProbe:
    """Records which path finished the target request as aborted.

    Wraps this engine's ``_process_aborts_queue``: if the target was
    unfinished before the call and finished after it, the queue applied it.
    ``UpdateProbe`` records the ``patch`` path the same way.
    """

    def __init__(self, core: Any) -> None:
        self.via: str | None = None
        orig = core._process_aborts_queue

        def wrapped() -> None:
            target = olf_connector.GRANT_TO
            before = target is not None and request_unfinished(core, target)
            orig()
            if before and not request_unfinished(core, target) and self.via is None:
                self.via = "queue"

        core._process_aborts_queue = wrapped


def drain_batch_queue(llm: LLM, core: Any, limit: int = 16) -> int:
    """Step until no batch is still in flight; return how many steps it took.

    Under async scheduling a request can look finished to the client while a
    batch that scheduled it is still in the batch queue, with its output (and
    any load-failure report) not yet processed. Each measurement must start
    after those outputs, or the next request runs concurrently with them.
    No-op under sync scheduling (no batch queue).
    """
    steps = 0
    while core.batch_queue:
        if steps >= limit:
            raise RuntimeError(f"batch queue still busy after {limit} steps")
        llm.llm_engine.step()
        steps += 1
    return steps


# --------------------------------------------------------------------------
# Scheduler instrumentation (and the fallback abort)
# --------------------------------------------------------------------------


class UpdateProbe:
    """Wraps Scheduler.update_from_output to see the abort-before-report window.

    For every output of a step that scheduled the granted request, records
    how many invalid block ids it reports and whether the request was still
    in ``running`` or ``kv_holding_waiting`` (the collections the handler
    scans) when it arrived.

    In ``queue`` and ``none`` modes it only observes and delegates. In
    ``patch`` mode, when a report carries invalid block ids, it first finishes
    their owners in ``running`` as aborted: the state ``EngineCore.step``
    produces when an abort lands during ``execute_model``. Its window record
    is then not independent evidence; the queue rows are.
    """

    def __init__(self, patch_abort: bool) -> None:
        self.patch_abort = patch_abort
        self.armed = patch_abort
        self.events: list[dict[str, Any]] = []
        self.num_updates = 0
        self.patched_target = False
        self._orig = Scheduler.update_from_output
        probe = self

        def wrapped(sched: Scheduler, scheduler_output: Any, model_runner_output: Any):
            probe.num_updates += 1
            kvo = getattr(model_runner_output, "kv_connector_output", None)
            invalid = set(getattr(kvo, "invalid_block_ids", None) or ())
            target = olf_connector.GRANT_TO
            if invalid and probe.armed:
                owners = [
                    r.request_id
                    for r in sched.running
                    if invalid & set(sched.kv_cache_manager.get_block_ids(r.request_id)[0])
                ]
                sched.finish_requests(owners, RequestStatus.FINISHED_ABORTED)
                probe.armed = False
                probe.patched_target = target in owners
            if target is not None and target in scheduler_output.num_scheduled_tokens:
                probe.events.append(
                    {
                        "update_index": probe.num_updates,
                        "num_invalid_block_ids": len(invalid),
                        "target_in_running": any(r.request_id == target for r in sched.running),
                        "target_in_kv_holding_waiting": any(
                            r.request_id == target for r in sched.kv_holding_waiting
                        ),
                        "target_in_requests": target in sched.requests,
                        "target_scheduled_this_step": target
                        in scheduler_output.num_scheduled_tokens,
                    }
                )
            return probe._orig(sched, scheduler_output, model_runner_output)

        Scheduler.update_from_output = wrapped  # type: ignore[method-assign]

    def rearm(self) -> None:
        self.armed = self.patch_abort
        self.patched_target = False


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def common_llm_kwargs(args: argparse.Namespace, max_model_len: int) -> dict[str, Any]:
    return dict(
        model=args.model,
        enforce_eager=True,
        seed=0,
        enable_prefix_caching=True,
        # False: one batch in flight (EngineCore.step). True, vLLM's default:
        # the batch queue (step_with_batch_queue), several batches in flight.
        async_scheduling=getattr(args, "async_scheduling", False),
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=BLOCK_SIZE,
    )


# vLLM's default max_logprobs is 20.
TOP_K_LOGPROBS = 20


def sampling(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=TOP_K_LOGPROBS)


def mode_store(args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        **common_llm_kwargs(args, max_model_len=4096),
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(args.storage)},
        ),
    )
    tokenizer = llm.get_tokenizer()
    base_ids = tokenizer.encode(base_prompt_text(args.list_len), add_special_tokens=False)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=base_ids)],
        SamplingParams(temperature=0.0, max_tokens=1),
    )[0]
    gen = out.outputs[0].token_ids[0]
    decode_ids = base_ids + [gen]
    return {
        "base_prompt_len": len(base_ids),
        "generated_token": gen,
        "generated_text": out.outputs[0].text,
        "decode_prompt_ids": decode_ids,
        "canary_prompt_ids": canary_token_ids(tokenizer, len(decode_ids)),
    }


def load_store(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads((args.results / "store.json").read_text())["result"]


def trial_max_model_len(store: dict[str, Any], args: argparse.Namespace) -> int:
    return len(store["decode_prompt_ids"]) + args.max_tokens + BLOCK_SIZE


def mode_baseline(args: argparse.Namespace) -> dict[str, Any]:
    store = load_store(args)
    kwargs = common_llm_kwargs(args, trial_max_model_len(store, args))
    kwargs["async_scheduling"] = False  # the oracle: plain full recompute
    llm = LLM(**kwargs)
    out = llm.generate([TokensPrompt(prompt_token_ids=store["decode_prompt_ids"])], sampling(args))[
        0
    ]
    if out.num_cached_tokens:
        raise RuntimeError(f"baseline hit {out.num_cached_tokens} cached tokens; must be 0")
    return {"baseline": output_record(out)}


def mode_trial(args: argparse.Namespace) -> dict[str, Any]:
    store = load_store(args)
    decode_ids = store["decode_prompt_ids"]
    max_model_len = trial_max_model_len(store, args)
    kwargs = common_llm_kwargs(args, max_model_len)
    if args.canary:
        # Size the pool so it barely holds one request: after the canary is
        # freed, r1's allocation has to reuse the canary's blocks, which still
        # hold its KV. +2 leaves room for the null block and rounding.
        kwargs["num_gpu_blocks_override"] = math.ceil(max_model_len / BLOCK_SIZE) + 2

    llm = LLM(
        **kwargs,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OrphanedLoadFailureConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": str(args.storage),
                "async_load": False,
            },
            kv_connector_module_path="olf_connector",
            kv_load_failure_policy=args.policy,
        ),
    )
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is None or not hasattr(core, "aborts_queue"):
        raise RuntimeError(
            "EngineCore is not in-process; VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "must be set before vllm is imported."
        )
    executor = type(core.model_executor).__name__
    if executor != "UniProcExecutor":
        raise RuntimeError(
            f"executor is {executor}; the connector's shared module state needs "
            "UniProcExecutor (one process, one worker)."
        )
    block_pool = core.scheduler.kv_cache_manager.block_pool
    engine_info = {
        "executor": executor,
        "scheduler_class": type(core.scheduler).__name__,
        "batch_queue_size": core.batch_queue_size,
        "step_fn": core.step_fn.__name__,
        "fail_loads": not args.no_load_failure,
    }
    olf_connector.FAIL_LOADS = not args.no_load_failure

    logs = LogCapture()
    logging.getLogger("vllm").addHandler(logs)
    probe = UpdateProbe(patch_abort=args.abort == "patch")
    aborts = AbortProbe(core)
    params = sampling(args)

    trials = []
    for t in range(args.trials):
        if t > 0:
            # Clears every hash, the poisoned ones included, so trials don't
            # see each other's cache. The pool's memory is not cleared.
            if not llm.reset_prefix_cache():
                raise RuntimeError("reset_prefix_cache refused")
        olf_connector.reset_state()
        olf_connector.GRANT_TO = None
        olf_connector.ON_LOAD_STEP = None
        probe.rearm()
        probe.events.clear()
        aborts.via = None
        logs.lines.clear()
        rec: dict[str, Any] = {"trial": t}

        canary_blocks: set[int] = set()
        if args.canary:
            rec["canary"], _, _ = run_request(
                llm,
                f"canary-{t}",
                store["canary_prompt_ids"],
                SamplingParams(temperature=0.0, max_tokens=1),
                core=core,
            )
            drain_batch_queue(llm, core)
            # Only the canary has run since the reset, so every hashed block
            # is one of its blocks (freed, KV still in memory).
            canary_blocks = {b.block_id for b in block_pool.blocks if b.block_hash is not None}

        if args.abort == "queue":
            # Fires on the worker side inside execute_model, in the step with
            # r1's load. The engine drains the queue before that step's
            # update_from_output (core.py: step and step_with_batch_queue).
            def on_load_step() -> None:
                core.aborts_queue.put_nowait([olf_connector.GRANT_TO])

            olf_connector.ON_LOAD_STEP = on_load_step

        rec["r1"], _, rec["r1_step_times_s"] = run_request(
            llm,
            f"r1-{t}",
            decode_ids,
            params,
            core=core,
            grant_external=True,
            client_abort_after_drop=args.abort != "none",
        )
        olf_connector.ON_LOAD_STEP = None
        rec["r1_drain_steps"] = drain_batch_queue(llm, core)
        if olf_connector.STATE["granted_tokens"] == 0:
            raise RuntimeError(
                "r1 got no external tokens: the store did not match. Re-run "
                "`store` with the same model and storage dir."
            )
        rec["abort_via"] = "patch" if probe.patched_target else aborts.via
        rec["connector"] = dict(olf_connector.STATE)
        rec["update_probe"] = list(probe.events)
        r1_blocks = olf_connector.STATE["r1_block_ids"] or []
        rec["r1_blocks_reusing_canary"] = len(set(r1_blocks) & canary_blocks)
        rec["r1_num_blocks"] = len(r1_blocks)
        # Snapshot after r1, before r2: which of r1's blocks are still hashed.
        rec["r1_blocks_still_hashed"] = sum(
            block_pool.blocks[b].block_hash is not None for b in r1_blocks
        )

        # r2: same prompt, new request. GRANT_TO still names r1, so the
        # connector offers r2 nothing; any hit is a local prefix-cache hit.
        rec["r2"], r2_internal, _ = run_request(llm, f"r2-{t}", decode_ids, params, core=core)
        drain_batch_queue(llm, core)
        rec["r2_external_tokens"] = olf_connector.STATE["grants"].get(r2_internal)
        rec["kv_failure_log_lines"] = list(logs.lines)
        # The fix's bookkeeping must be empty once nothing is in flight; None
        # on trees without the fix.
        records = getattr(core.scheduler, "_finished_inflight_kv", None)
        rec["finished_inflight_records_left"] = None if records is None else len(records)
        trials.append(rec)
    return {"engine": engine_info, "canary_word": CANARY_WORD, "trials": trials}


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["store", "baseline", "trial"])
    p.add_argument("--results", type=Path, required=True, help="results dir for this tree")
    p.add_argument("--storage", type=Path, help="ExampleConnector storage dir (store, trial)")
    p.add_argument("--policy", choices=["fail", "recompute"], default="fail")
    p.add_argument(
        "--abort",
        choices=["none", "queue", "patch"],
        default="queue",
        help="queue: real aborts_queue path, triggered inside execute_model; "
        "patch: monkeypatch of update_from_output; none: control",
    )
    p.add_argument("--canary", action="store_true", help="prime the pool with an unrelated prompt")
    p.add_argument(
        "--async-scheduling",
        action="store_true",
        help="vLLM's default batch-queue scheduling (trial only)",
    )
    p.add_argument(
        "--no-load-failure",
        action="store_true",
        help="positive control: the load succeeds, only the abort remains",
    )
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--list-len", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--name", help="output file stem (default: derived from mode/flags)")
    args = p.parse_args()

    if args.mode in ("store", "trial") and args.storage is None:
        p.error("--storage is required for store and trial")
    args.results.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = {"store": mode_store, "baseline": mode_baseline, "trial": mode_trial}[args.mode](args)
    name = args.name or (
        args.mode
        if args.mode != "trial"
        else "trial-{}-{}".format(
            args.policy,
            cell_name(args.abort, args.canary, args.async_scheduling, args.no_load_failure),
        )
    )
    record = {
        "mode": args.mode,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "env": env_record(),
        "wall_s": round(time.time() - started, 1),
        "result": result,
    }
    out = args.results / f"{name}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
