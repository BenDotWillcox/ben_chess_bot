# BenBot runtime reliability snapshot

Measured 2026-07-10 against a freshly started, local, single-worker Uvicorn process. The first readiness poll initiated loading of Maia2 and Stockfish; the client then polled until ready. One first-inference warmup was recorded and excluded before the warm latency sample.

## Aggregate results

| Metric | Result |
| --- | ---: |
| Server start to dependency-ready | 4.3 s |
| Client poll-to-ready duration | 3,887.77 ms |
| Excluded first-inference warmup | 847.44 ms |
| Warm inference mean / p50 / p95 (5 requests) | 31.91 / 31.78 / 33.23 ms |
| Concurrent successful throughput (20 requests, concurrency 8) | 30.37 requests/s |
| Concurrent p50 / p95 latency | 258.76 / 265.29 ms |
| Concurrent successes / rejects / timeouts / failures | 20 / 0 / 0 / 0 |
| API-process RSS after load / after workload | 601.05 / 660.46 MiB |

The RSS values are the Windows working set of the **API process only**. Stockfish is a child process and is excluded, so these numbers are not total service memory.

## Configuration and provenance

- Base revision: `e4161b46a381ca12f59a263583b5bcb6cc88794d`; the worktree was dirty with the goal's uncommitted implementation, so this is directional evidence and must be rerun from a clean release commit for a release baseline.
- Python 3.11.3 on Windows 10 AMD64; 16 logical CPUs; `cpu`, Maia2 `rapid`, FEN memory strategy, `alpha=0.7`, `min_count=1`.
- Stockfish enabled but optional, 400-centipawn veto, 0.05-second analysis budget.
- Inference executor: one worker plus four queued calls. Client concurrency was 8.
- The benchmark raised the per-minute request limit to 200 so the workload measured inference capacity rather than the default public rate limit of 60. The server still used the production inference admission and operation timeouts.

The 20-request workload did not reach backpressure rejection on this machine; the zero rejection/timeout/failure rates apply only to this offered load. Unit and API tests separately force queue saturation and timeout paths.

See the [machine-readable aggregate](summary.json) and [reproduction methodology](../../docs/runtime_reliability.md). No positions, predictions, request bodies, or per-request traces are published.
