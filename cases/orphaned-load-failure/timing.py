# SPDX-License-Identifier: Apache-2.0
"""Table r2's time to first token next to its correctness (run_timing.sh output).

Reads ``results-8b/<tree>/{baseline,trial-*}.json`` and prints one row per
(tree, cell): r2's prefix-cache hit, whether its tokens match the baseline,
the probability it gives the baseline's next token, and its time to first
token (median, with min and max over the trials). Writes the same table to
``results-8b/table.md``. Standard library only.

Time to first token is measured in-process, from add_request to the first
step that returns a token, in eager mode. It compares rows of this table; it
is not a serving-latency benchmark.
"""

import json
import statistics
import sys
from pathlib import Path

from compare import cell_sort_key, next_token_stats, variant

HERE = Path(__file__).resolve().parent


def ms(x: float) -> str:
    return f"{1000 * x:.0f}"


def rows_for_tree(tree_dir: Path) -> list[dict]:
    base = json.loads((tree_dir / "baseline.json").read_text())["result"]["baseline"]
    rows = []
    for path in sorted(tree_dir.glob("trial-*.json"), key=cell_sort_key):
        rec = json.loads(path.read_text())
        a, trials = rec["args"], rec["result"]["trials"]
        ttfts = [t["r2"]["ttft_s"] for t in trials if t["r2"].get("ttft_s") is not None]
        if not ttfts:
            continue
        p_values = {next_token_stats(t["r2"], base)[0] for t in trials}
        rows.append(
            {
                "tree": tree_dir.name,
                "cell": f"{a['policy']}, abort {a['abort']}, {variant(a)}",
                "abort via": "/".join(sorted({str(t.get("abort_via") or "-") for t in trials})),
                "r2 cached": "/".join(
                    sorted({str(t["r2"].get("num_cached_tokens")) for t in trials})
                ),
                "r2==base": f"{sum(t['r2'].get('token_ids') == base['token_ids'] for t in trials)}"
                f"/{len(trials)}",
                "P(next)": "/".join(sorted(p_values)),
                "r2 TTFT ms, median (min-max)": (
                    f"{ms(statistics.median(ttfts))} ({ms(min(ttfts))}-{ms(max(ttfts))})"
                ),
                "n": str(len(ttfts)),
            }
        )
    return rows


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results-8b"
    if not results.is_dir():
        print(f"no results directory at {results}; run run_timing.sh first", file=sys.stderr)
        sys.exit(2)
    rows = []
    for tree in ("main", "fix"):
        if (results / tree / "baseline.json").exists():
            rows.extend(rows_for_tree(results / tree))
    if not rows:
        print("(no results)")
        sys.exit(1)
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    table = "\n".join(lines) + "\n"
    (results / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
