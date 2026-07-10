from __future__ import annotations

import httpx

from benchmark_runtime import RequestResult, classify_response, percentile, run_benchmark, summarize
from smoke_api import run_smoke
from smoke_api import wait_for_readiness
from test_browser_smoke import browser_server


def test_benchmark_classifies_backpressure_timeouts_and_failures() -> None:
    assert classify_response(200) == "success"
    assert classify_response(429) == "rejected"
    assert classify_response(503) == "rejected"
    assert classify_response(504) == "timeout"
    assert classify_response(500) == "failure"


def test_benchmark_summary_reports_rates_throughput_and_percentiles() -> None:
    results = [
        RequestResult("success", 200, 10.0),
        RequestResult("success", 200, 30.0),
        RequestResult("rejected", 503, 2.0, "busy"),
        RequestResult("timeout", 504, 100.0, "timed out"),
    ]
    summary = summarize(results, wall_seconds=0.2)

    assert summary["offered_throughput_requests_per_second"] == 20.0
    assert summary["successful_throughput_requests_per_second"] == 10.0
    assert summary["counts"] == {"success": 2, "rejected": 1, "timeout": 1, "failure": 0}
    assert summary["unsuccessful_request_rate"] == 0.5
    assert summary["hard_failure_rate"] == 0.25
    assert summary["success_latency_ms"]["p50"] == 10.0
    assert percentile([], 0.95) is None


def test_benchmark_harness_runs_end_to_end_against_dependency_fakes() -> None:
    with browser_server() as base_url:
        with httpx.Client(base_url=base_url, timeout=10) as client:
            smoke = run_smoke(client, require_stockfish=False)
        report = run_benchmark(
            base_url=base_url,
            timeout_seconds=10,
            warmup_requests=1,
            warm_requests=1,
            concurrent_requests=4,
            concurrency=2,
        )

    assert smoke["ok"] is True
    assert len(smoke["game"]["prefix"].split()) == 2
    assert report["cold_readiness"]["status"] == "ready"
    assert report["schema_version"] == 2
    assert report["cold_readiness"]["client_poll_to_ready_ms"] >= 0
    assert report["cold_readiness"]["server_start_to_ready_seconds"] is not None
    assert report["warm_inference"]["counts"]["success"] == 1
    assert report["bounded_concurrency"]["requests"] == 4
    assert report["process_after_benchmark"]["scope"] == "api_process_only_excludes_stockfish_child"
    assert report["provenance"]["git_revision"] != ""
    assert report["provenance"]["git_worktree_dirty"] in {True, False, None}
    assert report["service_configuration"]["strategy"] == "fen"


def test_readiness_polling_handles_nonblocking_startup() -> None:
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class Client:
        def __init__(self):
            self.responses = [
                Response(503, {"ready": False, "status": "starting", "initializing": True}),
                Response(
                    200,
                    {
                        "ready": True,
                        "status": "ready",
                        "initializing": False,
                        "dependencies": {"stockfish": {"enabled": False, "status": "disabled"}},
                    },
                ),
            ]

        def get(self, _path):
            return self.responses.pop(0)

    payload, elapsed_ms, attempts = wait_for_readiness(
        Client(),
        timeout_seconds=1,
        poll_interval_seconds=0,
    )
    assert payload["ready"] is True
    assert elapsed_ms >= 0
    assert attempts == 2
