# Follow-up probes for vllm-project/vllm#61768

CPU-only scheduler probes for the cases raised after #61769: an owner that is
**preempted** (by KV pressure, or by `reset_prefix_cache(reset_running_requests=True)`)
rather than aborted before its sync-load failure report arrives, and an aborted
owner whose blocks are **reallocated** before the report.

They drive the real `Scheduler`/`AsyncScheduler` through `schedule()` and
`update_from_output()`. Only the KV connector is a mock: it reports a synchronous
external load, and the step's output reports one block (block 3 of 8) as failed.
These are scheduler-level probes; real-connector reachability is discussed in
the issue, not shown here.

| Probe | Case |
|---|---|
| `probe_reset_preempt.py <role> <policy>` | `reset_prefix_cache(reset_running_requests=True)` between `schedule(N)` and `update_from_output(N)` |
| `probe_pressure_preempt.py <role> <policy>` | KV-pressure preemption of the owner inside `schedule(N+1)` |
| `probe_abort_reuse.py <role> <policy>` | owner aborted after `schedule(N)`; unrelated request C scheduled in N+1; then the late report |

## Running

```bash
PYTHON=/path/to/vllm/.venv/bin/python ./run_all.sh /path/to/vllm-tree
```

The tree must be importable (compiled extensions available) and include `tests/`.
Async scheduling, CPU (`CUDA_VISIBLE_DEVICES=""`).

## Results

Full output: `outputs/main-a97d19d7.txt` (upstream main) and
`outputs/pr-7238ccdf.txt` (#61769). `H` = block still hashed in the prefix cache.

| Case | main `a97d19d7` | #61769 `7238ccdf` |
|---|---|---|
| reset, `kv_consumer`, both policies | reset raises; 8/8 blocks hashed; owner resumes on a 112-token hit (correct: 48) | same |
| KV pressure, `kv_consumer`, `fail` | owner not preempted; failed correctly; `HHH-----` | same |
| KV pressure, `kv_producer`, `fail` | owner preempted; `HHHHHHH-`; resumes on 112; not failed | same |
| abort + reuse, `kv_consumer` | C cannot be scheduled until the report is handled | same |
| abort + reuse, `kv_producer`, `fail` | C (unrelated) fails; C's entries `HHHH----` | C fails; `--------` |
| abort + reuse, `kv_producer`, `recompute` | C rewound from 128 to 64 computed tokens; `HHHHHHHH` | same rewind; `-----HHH` |
