# SPDX-License-Identifier: Apache-2.0
"""Build the comparison table from harness JSON.

Reads ``results/<tree>/{baseline,trial-*}.json`` for every tree directory and
prints one row per (tree, policy, abort, variant) cell, aggregated over the
repeated trials of that cell (one engine per cell; trials are not cold runs).
Writes the same table to ``results/table.md``. Regenerable: the table is a pure
function of the JSON files. Standard library only; any Python 3.10+ runs it.

Columns:
  variant     sync (one batch in flight) or async (vLLM's default batch queue);
              canary: pool primed with an unrelated prompt; loadok: the load
              succeeds (positive control, only the abort remains)
  abort via   which path finished r1 as aborted: queue (EngineCore's
              _process_aborts_queue inside the step, the production path) or
              patch (the fallback inside update_from_output)
  r1          r1's terminal state; "aborted" means the engine aborted it mid-step
  owner gone at report
              trials where a load-failure report for a step that scheduled r1
              reached update_from_output after r1 had left both `running` and
              `kv_holding_waiting`, the two collections the handler scans
  hashed      r1 blocks still hashed after r1, out of r1's blocks
  r2 ext      external tokens the connector offered r2 (must be 0: any r2 hit
              is a local prefix-cache hit)
  r2 cached   r2's prefix-cache hit (num_cached_tokens); in failure rows r1's
              loaded blocks were never populated for this prompt, so any hit
              on them is the bug
  r2==base    trials whose r2 token ids equal the full-recompute baseline
  diverge@    first token index where r2 differs from the baseline
  next token, per trial
              at r2's first generated token: the probability r2 gives the
              baseline's first token, its rank in r2's top 20, and how many of
              the baseline's top-20 candidates r2 shares (∩), one entry per
              trial in trial order. The baseline's own value is printed under
              the table. Greedy match is a weak oracle; this reads the
              distribution itself
  canary reuse
              r1 blocks that were the canary's freed blocks, out of r1's
  canary word trials whose r2 text contains the canary word
  records left
              the fix's bookkeeping after r2 (must be 0; "-" without the fix)

r2's time to first token is recorded in the JSON (``ttft_s``) but not tabled:
at this model size a 1.9k-token prefill costs a few milliseconds and Python
overhead dominates, so the cache hit shows no measurable speedup here.

A tree directory missing its baseline or any of the expected trial files is
named at the top of the table and on stderr, and the exit code is 1, so a
partial matrix can't pass for a complete one. The same holds for a row whose
abort was not applied by the path it claims (``abort via`` must equal the
``--abort`` mode for queue and patch rows): such a row proves nothing about
that path.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CANARY_WORD = "PINEAPPLE"  # older JSON did not record it
TREE_ORDER = ("main", "pr-53298", "fix")
# What run_matrix.sh produces per tree, per policy, in reading order: the
# sync cells (with and without canary), the patch cross-check, the async
# cells and the load-ok positive control.
CELL_NAMES = (
    "none",
    "none-canary",
    "queue",
    "queue-canary",
    "patch",
    "none-async",
    "queue-async",
    "queue-loadok",
)
POLICIES = ("fail", "recompute")


def cell_name(abort: str, canary: bool, async_scheduling: bool, no_load_failure: bool) -> str:
    """The cell part of a trial file name; harness.py names its output with it."""
    return (
        f"{abort}{'-canary' if canary else ''}"
        f"{'-async' if async_scheduling else ''}"
        f"{'-loadok' if no_load_failure else ''}"
    )


EXPECTED_TRIALS = [f"trial-{policy}-{name}.json" for policy in POLICIES for name in CELL_NAMES]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def diverge_cell(r2: dict, base_ids: list[int]) -> str | int | None:
    ids = r2.get("token_ids") or []
    if not ids:
        return "no tokens"
    return first_divergence(ids, base_ids)


def owner_gone_at_report(trial: dict) -> bool:
    return any(
        e["num_invalid_block_ids"] > 0
        and e["target_scheduled_this_step"]
        and not e["target_in_running"]
        and not e.get("target_in_kv_holding_waiting", False)
        for e in trial["update_probe"]
    )


def variant(a: dict) -> str:
    parts = ["async" if a.get("async_scheduling") else "sync"]
    if a.get("canary"):
        parts.append("canary")
    if a.get("no_load_failure"):
        parts.append("loadok")
    return "+".join(parts)


def next_token_stats(r2: dict, base: dict) -> tuple[str, str, str]:
    """P, rank and top-20 overlap for the baseline's first token in r2."""
    topk, base_topk = r2.get("step0_topk"), base.get("step0_topk")
    if not topk or not base_topk or not base.get("token_ids"):
        return "-", "-", "-"
    target = str(base["token_ids"][0])
    ranked = sorted(topk, key=topk.get, reverse=True)
    overlap = len(set(topk) & set(base_topk))
    if target not in topk:
        return f">top{len(topk)}", "-", str(overlap)
    return f"{math.exp(topk[target]):.3g}", str(ranked.index(target) + 1), str(overlap)


def next_token_cell(trials: list[dict], base: dict) -> str:
    """One "P (rank, ∩)" entry per trial, in trial order, so values stay paired."""
    entries = []
    for t in trials:
        p, rank, overlap = next_token_stats(t["r2"], base)
        entries.append(f"{p} (r{rank}, ∩{overlap})" if p != "-" else "-")
    return " / ".join(entries)


def r1_cell(trial: dict, base_ids: list[int]) -> str:
    r1 = trial["r1"]
    if r1["status"] != "finished":
        return r1["status"]
    if r1["finish_reason"] in ("error", "abort"):
        return r1["finish_reason"]
    return "== base" if r1["token_ids"] == base_ids else "!= base"


def _set_order(v: Any) -> tuple[bool, bool, Any]:
    """Numbers first in numeric order, then strings, then None."""
    is_number = isinstance(v, (int, float))
    return (v is None, not is_number, v if is_number else str(v))


def fmt_set(values: list) -> str:
    """Distinct values, numbers in numeric order, None shown as "-"."""
    return "/".join("-" if v is None else str(v) for v in sorted(set(values), key=_set_order))


def provenance_errors(abort: str, trials: list[dict]) -> list[str]:
    """Trials whose abort was not applied by the path the row claims."""
    if abort not in ("queue", "patch"):
        return []
    return [
        f"trial {t.get('trial', i)}: abort via {t.get('abort_via')!r}, expected {abort!r}"
        for i, t in enumerate(trials)
        if t.get("abort_via") != abort
    ]


def missing_files(tree_dir: Path) -> list[str]:
    expected = ["baseline.json", *EXPECTED_TRIALS]
    return [name for name in expected if not (tree_dir / name).exists()]


def cell_sort_key(path: Path) -> tuple[int, int]:
    name = path.stem.removeprefix("trial-")
    policy, _, cell = name.partition("-")
    return (
        POLICIES.index(policy) if policy in POLICIES else len(POLICIES),
        CELL_NAMES.index(cell) if cell in CELL_NAMES else len(CELL_NAMES),
    )


def rows_for_tree(tree_dir: Path, problems: list[str] | None = None) -> list[dict]:
    base_path = tree_dir / "baseline.json"
    if not base_path.exists():
        return []
    base = json.loads(base_path.read_text())["result"]["baseline"]
    base_ids = base["token_ids"]
    rows = []
    for path in sorted(tree_dir.glob("trial-*.json"), key=cell_sort_key):
        rec = json.loads(path.read_text())
        a, env, result = rec["args"], rec["env"], rec["result"]
        trials = result["trials"]
        if not trials:
            print(f"WARNING {tree_dir.name}/{path.name}: no trials", file=sys.stderr)
            continue
        if problems is not None:
            problems.extend(
                f"PROVENANCE {tree_dir.name}/{path.name}: {err}"
                for err in provenance_errors(a["abort"], trials)
            )
        n = len(trials)
        canary_word = result.get("canary_word", DEFAULT_CANARY_WORD).lower()
        r2_ids = [t["r2"].get("token_ids", []) for t in trials]
        rows.append(
            {
                "tree": tree_dir.name,
                "commit": env["tree_commit"][:8] + ("+" if env["tree_dirty"] else ""),
                "policy": a["policy"],
                "abort": a["abort"],
                "variant": variant(a),
                "abort via": fmt_set([t.get("abort_via") for t in trials]),
                "r1": fmt_set([r1_cell(t, base_ids) for t in trials]),
                "owner gone at report": (f"{sum(owner_gone_at_report(t) for t in trials)}/{n}"),
                "hashed": fmt_set(
                    [f"{t['r1_blocks_still_hashed']}/{t['r1_num_blocks']}" for t in trials]
                ),
                "r2 ext": fmt_set([t.get("r2_external_tokens") for t in trials]),
                "r2 cached": fmt_set([t["r2"].get("num_cached_tokens") for t in trials]),
                "r2==base": f"{sum(ids == base_ids for ids in r2_ids)}/{n}",
                "diverge@": fmt_set([diverge_cell(t["r2"], base_ids) for t in trials]),
                "next token, per trial": next_token_cell(trials, base),
                "canary reuse": (
                    fmt_set(
                        [f"{t['r1_blocks_reusing_canary']}/{t['r1_num_blocks']}" for t in trials]
                    )
                    if a["canary"]
                    else "-"
                ),
                "canary word": (
                    f"{sum(canary_word in t['r2'].get('text', '').lower() for t in trials)}/{n}"
                    if a["canary"]
                    else "-"
                ),
                "records left": fmt_set([t.get("finished_inflight_records_left") for t in trials]),
                "r2 text (trial 0)": repr(trials[0]["r2"].get("text", ""))[:48],
            }
        )
    return rows


def to_markdown(rows: list[dict], base_text: dict[str, str]) -> str:
    if not rows:
        return "(no results)\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    lines.append("")
    for tree, text in base_text.items():
        lines.append(f"Baseline ({tree}): {text}")
    return "\n".join(lines) + "\n"


def tree_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    return (TREE_ORDER.index(name) if name in TREE_ORDER else len(TREE_ORDER), name)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    if not results.is_dir():
        print(f"no results directory at {results}; run run_matrix.sh first", file=sys.stderr)
        sys.exit(2)
    rows, base_text, incomplete = [], {}, []
    for tree_dir in sorted((p for p in results.iterdir() if p.is_dir()), key=tree_sort_key):
        missing = missing_files(tree_dir)
        if missing:
            incomplete.append(f"INCOMPLETE {tree_dir.name}: missing {', '.join(missing)}")
        tree_rows = rows_for_tree(tree_dir, incomplete)
        if tree_rows:
            rows.extend(tree_rows)
            base = json.loads((tree_dir / "baseline.json").read_text())["result"]["baseline"]
            p_self = next_token_stats(base, base)[0]
            base_text[tree_dir.name] = f"{base['text'][:60]!r}, P(next) = {p_self}"
    table = to_markdown(rows, base_text)
    if incomplete:
        table = "\n".join(incomplete) + "\n\n" + table
        print("\n".join(incomplete), file=sys.stderr)
    (results / "table.md").write_text(table)
    print(table)
    if incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
