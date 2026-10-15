# Results: orphaned load failure

Produced on 2026-10-14 by this case's code at commit `751c8eb` (`run_matrix.sh` on each tree, then `compare.py`). The table below is `results/table.md` from that run, unedited; this file only adds the provenance above it. Later commits change only documentation, the table's consistency checks and script hardening, not what is measured.

| Tree | vLLM commit | What it is |
|---|---|---|
| `main` | `04462732a40a8329ad60593d754b576c54c5fe5d` | upstream main |
| `pr-53298` | `3df0beef0ebc11846a4dd805ac9ad864edf4dd58` | [#53298](https://github.com/vllm-project/vllm/pull/53298) at `a7dd070b` merged onto that main |
| `fix` | `e23ecfa8964f792afe299c4359c9769888bedd77` | the fix, branch [`orphaned-load-failure-04462732`](https://github.com/ArijitDeb/vllm/tree/orphaned-load-failure-04462732) |

**Environment:** NVIDIA GeForce RTX 5090 (32607 MiB, driver 610.57.04), torch 2.13.0+cu130, vLLM 0.31.1rc1.dev222+g04462732a, Qwen/Qwen3-0.6B, greedy, 32 output tokens, 3 trials per cell.

**In short:**

- On `main` and on `pr-53298`, under both policies and both scheduling modes, aborting r1 in the step its load fails leaves 119 of its 120 blocks hashed. r2 then gets a 1904-token prefix hit on them, diverges from the baseline at token 0, and gives the baseline's next token a probability of 0.0005-0.005 (baseline 0.962).
- With the fix, the same rows evict those blocks: r2 hits 0 cached tokens, matches the baseline in every trial, and gives 0.962, with no bookkeeping left over.
- In the positive control (the load succeeds, r1 is still aborted mid-step), the valid cache survives on every tree, the fix included: r2 hits 1904 tokens and matches the baseline.
- Every queue row's abort was applied by `EngineCore._process_aborts_queue` (`abort via` = queue); `compare.py` fails the table otherwise.
- The `recompute` no-abort control fails on `main` and `fix` and passes on `pr-53298`, consistent with [#49250](https://github.com/vllm-project/vllm/issues/49250) (see the README). It is not this bug.

Column definitions are in `compare.py`'s docstring and the README's "Reading the table".

| tree | commit | policy | abort | variant | abort via | r1 | owner gone at report | hashed | r2 ext | r2 cached | r2==base | diverge@ | next token, per trial | canary reuse | canary word | records left | r2 text (trial 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 04462732 | fail | none | sync | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| main | 04462732 | fail | none | sync+canary | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | 115/120 | 0/3 | - | '401. So, the answer is 401.\n**Final Answer**\n |
| main | 04462732 | fail | queue | sync | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | fail | queue | sync+canary | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.00493 (r10, ∩11) / 0.00142 (r8, ∩11) / 0.00142 (r8, ∩11) | 115/120 | 0/3 | - | '0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 i |
| main | 04462732 | fail | patch | sync | patch | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | fail | none | async | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| main | 04462732 | fail | queue | async | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | fail | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| main | 04462732 | recompute | none | sync | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | >top20 (r-, ∩0) / >top20 (r-, ∩0) / >top20 (r-, ∩0) | - | - | - | 'ắc작작작작작작작작작작작작작작작작작작작작작작작작작작작작작작작' |
| main | 04462732 | recompute | none | sync+canary | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | 0.007 (r6, ∩7) / 0.0122 (r7, ∩7) / 0.0105 (r5, ∩7) | 115/120 | 0/3 | - | '11111111111111111111111111111111' |
| main | 04462732 | recompute | queue | sync | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | recompute | queue | sync+canary | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.00493 (r10, ∩11) / 0.00142 (r8, ∩11) / 0.00142 (r8, ∩11) | 115/120 | 0/3 | - | '0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 i |
| main | 04462732 | recompute | patch | sync | patch | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | recompute | none | async | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | >top20 (r-, ∩0) / >top20 (r-, ∩0) / >top20 (r-, ∩0) | - | - | - | '작작endantsắc작ellaneousellaneousellaneousắc작endan |
| main | 04462732 | recompute | queue | async | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| main | 04462732 | recompute | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | fail | none | sync | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | fail | none | sync+canary | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | 115/120 | 0/3 | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | fail | queue | sync | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | fail | queue | sync+canary | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.00493 (r10, ∩11) / 0.00142 (r8, ∩11) / 0.00142 (r8, ∩11) | 115/120 | 0/3 | - | '0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 i |
| pr-53298 | 3df0beef | fail | patch | sync | patch | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | fail | none | async | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | fail | queue | async | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | fail | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | recompute | none | sync | - | == base | 0/3 | 120/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | recompute | none | sync+canary | - | == base | 0/3 | 120/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | 115/120 | 0/3 | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | recompute | queue | sync | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | recompute | queue | sync+canary | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.00493 (r10, ∩11) / 0.00142 (r8, ∩11) / 0.00142 (r8, ∩11) | 115/120 | 0/3 | - | '0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 is0 i |
| pr-53298 | 3df0beef | recompute | patch | sync | patch | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | recompute | none | async | - | == base | 0/3 | 120/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| pr-53298 | 3df0beef | recompute | queue | async | queue | aborted | 3/3 | 119/120 | 0 | 1904 | 0/3 | 0 | 0.000458 (r10, ∩11) / 0.00488 (r8, ∩11) / 0.00256 (r11, ∩10) | - | - | - | '7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ' |
| pr-53298 | 3df0beef | recompute | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | - | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | none | sync | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | none | sync+canary | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | 115/120 | 0/3 | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | queue | sync | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | queue | sync+canary | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | 115/120 | 0/3 | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | patch | sync | patch | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | none | async | - | error | 0/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | queue | async | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | fail | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | recompute | none | sync | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | >top20 (r-, ∩0) / >top20 (r-, ∩0) / >top20 (r-, ∩0) | - | - | 0 | '작작endantsắc작작작작작작작작작작작작작작작작작작작작작작작작작작작작' |
| fix | e23ecfa8 | recompute | none | sync+canary | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | 0.0111 (r8, ∩7) / 0.00418 (r13, ∩7) / >top20 (r-, ∩0) | 115/120 | 0/3 | 0 | '15. PPE.\nPE.\nPPE.\nPPE.\nPPE.\nPPE.\nPE.\nPPE |
| fix | e23ecfa8 | recompute | queue | sync | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | recompute | queue | sync+canary | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | 115/120 | 0/3 | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | recompute | patch | sync | patch | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | recompute | none | async | - | != base | 0/3 | 120/120 | 0 | 1904 | 0/3 | 0 | >top20 (r-, ∩0) / >top20 (r-, ∩0) / >top20 (r-, ∩0) | - | - | 0 | '작작성작작작작성작작작작작작작작작작작작작작작작작작작작작작작작' |
| fix | e23ecfa8 | recompute | queue | async | queue | aborted | 3/3 | 0/120 | 0 | 0 | 3/3 | - | 0.962 (r1, ∩20) / 0.962 (r1, ∩20) / 0.962 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |
| fix | e23ecfa8 | recompute | queue | sync+loadok | queue | aborted | 0/3 | 119/120 | 0 | 1904 | 3/3 | - | 0.963 (r1, ∩20) / 0.963 (r1, ∩20) / 0.963 (r1, ∩20) | - | - | 0 | '401. So, the answer is 401.\n**Final Answer**\n |

Baseline (main): '401. So, the answer is 401.\n**Final Answer**\nThe next number', P(next) = 0.962
Baseline (pr-53298): '401. So, the answer is 401.\n**Final Answer**\nThe next number', P(next) = 0.962
Baseline (fix): '401. So, the answer is 401.\n**Final Answer**\nThe next number', P(next) = 0.962
