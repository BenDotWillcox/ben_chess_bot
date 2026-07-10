# Runtime reliability measurements

`benchmark_runtime.py` records the previously missing cold/warm latency, process RSS, bounded-concurrency throughput, backpressure rejections, timeouts, and failure rates in one JSON result.

For a real measurement, start a fresh single-worker BenBot process with the intended production environment, then run:

```bash
python benchmark_runtime.py \
  --base-url http://127.0.0.1:7860 \
  --warmup-requests 1 \
  --warm-requests 10 \
  --requests 40 \
  --concurrency 8 \
  --output runtime-benchmark.json
```

The first `/ready` poll starts one background initialization of Maia2 and enabled Stockfish. Schema v2 records the complete client poll-to-ready duration and server uptime at readiness, because background preload may begin before the benchmark connects. Server uptime—not any individual HTTP request—is the cold-start metric, and it is valid only when the target process was freshly started. The harness then runs and records an excluded warmup inference before measuring warm and concurrent requests with the fixed post-`e4` prediction payload. Offered throughput and successful throughput are separate so fast backpressure rejections cannot inflate the useful-throughput claim; rejection, timeout, and hard-failure rates also remain separate.

RSS comes from the API process working set on Windows or `/proc/self/status` on Linux. It explicitly excludes the Stockfish child process, so it is not total service memory. The API returns `null` with source `unavailable` on unsupported platforms rather than inventing a value.

CI runs the same harness against dependency fakes to prove the measurement path and JSON artifact. Those CI numbers are not model performance evidence. The harness automatically records the commit (or `unknown`), Python/platform/machine/CPU provenance, and readiness-reported service settings. Publish a real result only with confirmation that the target started cold and enough surrounding container detail to interpret those fields.
