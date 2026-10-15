# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure functions behind the published table (no GPU, no vLLM)."""

import math
import re
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import compare  # noqa: E402


def test_first_divergence():
    assert compare.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert compare.first_divergence([1, 9, 3], [1, 2, 3]) == 1
    assert compare.first_divergence([1, 2], [1, 2, 3]) == 2


def test_diverge_cell_without_tokens():
    assert compare.diverge_cell({"token_ids": []}, [1, 2]) == "no tokens"
    assert compare.diverge_cell({}, [1, 2]) == "no tokens"


def _event(invalid=119, running=False, waiting=False, scheduled=True):
    return {
        "num_invalid_block_ids": invalid,
        "target_in_running": running,
        "target_in_kv_holding_waiting": waiting,
        "target_scheduled_this_step": scheduled,
    }


@pytest.mark.parametrize(
    "event,expected",
    [
        (_event(), True),
        (_event(running=True), False),
        (_event(waiting=True), False),
        (_event(invalid=0), False),
        (_event(scheduled=False), False),
    ],
)
def test_owner_gone_at_report(event, expected):
    assert compare.owner_gone_at_report({"update_probe": [event]}) is expected


def test_next_token_stats():
    base = {"token_ids": [7], "step0_topk": {"7": math.log(0.9), "8": math.log(0.1)}}
    in_top = {"step0_topk": {"8": math.log(0.6), "7": math.log(0.3), "5": math.log(0.1)}}
    assert compare.next_token_stats(in_top, base) == ("0.3", "2", "2")
    out_of_top = {"step0_topk": {"1": 0.0, "2": -1.0}}
    assert compare.next_token_stats(out_of_top, base) == (">top2", "-", "0")
    assert compare.next_token_stats({}, base) == ("-", "-", "-")


def test_next_token_cell_keeps_trials_paired_and_ordered():
    base = {"token_ids": [7], "step0_topk": {"7": 0.0}}
    trials = [
        {"r2": {"step0_topk": {"7": math.log(0.5)}}},
        {"r2": {"step0_topk": {"1": 0.0}}},
    ]
    assert compare.next_token_cell(trials, base) == "0.5 (r1, ∩1) / >top1 (r-, ∩0)"


def test_fmt_set_orders_numbers_numerically():
    assert compare.fmt_set([10, 8, 11, 8]) == "8/10/11"
    assert compare.fmt_set([None, 3]) == "3/-"
    assert compare.fmt_set(["b", "a"]) == "a/b"


def _matrix_cells() -> set[str]:
    """The cells run_matrix.sh runs, derived from its `run trial` lines."""
    text = (CASE_DIR / "run_matrix.sh").read_text()
    loop_aborts = re.search(r"for abort in ([\w ]+); do", text).group(1).split()
    cells = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("run trial"):
            continue
        abort = re.search(r'--abort "?(\$abort|\w+)"?', line).group(1)
        for a in loop_aborts if abort == "$abort" else [abort]:
            cells.add(
                compare.cell_name(
                    a,
                    "--canary" in line,
                    "--async-scheduling" in line,
                    "--no-load-failure" in line,
                )
            )
    return cells


def test_expected_trials_match_run_matrix():
    assert _matrix_cells() == set(compare.CELL_NAMES)
    assert len(compare.EXPECTED_TRIALS) == len(compare.POLICIES) * len(compare.CELL_NAMES)
