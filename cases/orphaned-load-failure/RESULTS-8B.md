# Results: time to first token on Qwen3-8B-Base

Supplementary to [RESULTS.md](RESULTS.md): is the wrong answer also a fast one? Produced on 2026-10-15 by this case's code at commit `4f11f5d` (`run_timing.sh` on each tree, then `timing.py`). The table is `results-8b/table.md` from that run, unedited.

| Tree | vLLM commit |
|---|---|
| `main` | `04462732a40a8329ad60593d754b576c54c5fe5d` |
| `fix` | `e23ecfa8964f792afe299c4359c9769888bedd77` (branch [`orphaned-load-failure-04462732`](https://github.com/ArijitDeb/vllm/tree/orphaned-load-failure-04462732)) |

**Environment:** NVIDIA GeForce RTX 5090 (32 GB, driver 610.57.04), torch 2.13.0+cu130, vLLM 0.31.1rc1.dev222+g04462732a, Qwen/Qwen3-8B-Base in bf16, 75% of GPU memory, greedy, 32 output tokens, five trials per cell, `fail` policy, sync scheduling.

Time to first token is measured in-process, from adding r2 to the first step that returns a token, in eager mode. It compares the rows of this table; it is not a serving-latency benchmark. In every row the slowest trial is the first one, warming up.

**In short:** on `main`, the request that reuses never-loaded KV is answered in 22 ms, the same as a legitimate cache hit (22 ms) and about six times faster than a correct recompute (136 ms), and it is wrong in all five trials. With the fix, that request recomputes (135 ms) and is correct, while the valid cache hit stays fast (21 ms).

| tree | cell | abort via | r2 cached | r2==base | P(next) | r2 TTFT ms, median (min-max) | n |
|---|---|---|---|---|---|---|---|
| main | fail, abort none, sync | - | 0 | 5/5 | 0.991 | 136 (136-298) | 5 |
| main | fail, abort queue, sync | queue | 1904 | 0/5 | 0.000947/0.0202/0.0465/0.17/0.507 | 22 (21-188) | 5 |
| main | fail, abort queue, sync+loadok | queue | 1904 | 5/5 | 0.991 | 22 (22-192) | 5 |
| fix | fail, abort none, sync | - | 0 | 5/5 | 0.991 | 135 (135-301) | 5 |
| fix | fail, abort queue, sync | queue | 0 | 5/5 | 0.991 | 135 (135-297) | 5 |
| fix | fail, abort queue, sync+loadok | queue | 1904 | 5/5 | 0.991 | 21 (21-186) | 5 |
