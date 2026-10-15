# inference-correctness-lab

Reproductions of correctness bugs in LLM serving engines, built to answer one question: did the model get worse, or only slower? Each case injects a fault deterministically at the point the engine is vulnerable, proves the interleaving actually happened, and compares the engine's output to a full-recompute baseline. It reads the next-token distribution (the probability and rank of the baseline's token, top-k overlap) rather than greedy text alone, so a partial corruption that greedy decoding happens to survive still shows.

| Case | Engine | Summary |
|---|---|---|
| [orphaned-load-failure](cases/orphaned-load-failure/) | vLLM v1 | A request aborted in the step its sync KV load fails leaves its never-loaded blocks in the prefix cache; a later identical prompt decodes from them. |

Each case directory is self-contained: its README states the bug, the method, how to run it, and how to read the table, and its `RESULTS.md` holds the published numbers. Links from upstream issues and PRs point at a tag (for example `orphaned-load-failure-v1`), so a case stays reproducible as the repository grows.

Licensed under [Apache-2.0](LICENSE).
