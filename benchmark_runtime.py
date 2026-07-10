#!/usr/bin/env python3
"""Measure BenBot cold readiness and bounded warm-inference behavior."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from smoke_api import wait_for_readiness


PREDICT_PAYLOAD = {
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "you_color": "black",
    "uci_prefix_before": "e2e4",
    "elo_self": 1650,
    "elo_oppo": 1650,
    "top_n": 5,
    "temperature": 1.0,
}


@dataclass(frozen=True)
class RequestResult:
    outcome: str
    status_code: int | None
    latency_ms: float
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 2)


def classify_response(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "success"
    if status_code in {429, 503}:
        return "rejected"
    if status_code == 504:
        return "timeout"
    return "failure"


def predict_once(client: httpx.Client) -> RequestResult:
    started = time.perf_counter()
    try:
        response = client.post("/predict", json=PREDICT_PAYLOAD)
        latency_ms = (time.perf_counter() - started) * 1000
        outcome = classify_response(response.status_code)
        error = None if outcome == "success" else response.text[:240]
        return RequestResult(outcome, response.status_code, round(latency_ms, 2), error)
    except httpx.TimeoutException as exc:
        return RequestResult("timeout", None, round((time.perf_counter() - started) * 1000, 2), str(exc))
    except httpx.RequestError as exc:
        return RequestResult("failure", None, round((time.perf_counter() - started) * 1000, 2), str(exc))


def summarize(results: list[RequestResult], wall_seconds: float) -> dict[str, object]:
    counts = {outcome: sum(result.outcome == outcome for result in results) for outcome in (
        "success",
        "rejected",
        "timeout",
        "failure",
    )}
    successes = [result.latency_ms for result in results if result.outcome == "success"]
    total = len(results)
    unsuccessful = total - counts["success"]
    hard_failures = counts["timeout"] + counts["failure"]
    return {
        "requests": total,
        "counts": counts,
        "offered_throughput_requests_per_second": round(total / wall_seconds, 2) if wall_seconds > 0 else None,
        "successful_throughput_requests_per_second": (
            round(counts["success"] / wall_seconds, 2) if wall_seconds > 0 else None
        ),
        "success_latency_ms": {
            "mean": round(statistics.fmean(successes), 2) if successes else None,
            "p50": percentile(successes, 0.50),
            "p95": percentile(successes, 0.95),
            "max": round(max(successes), 2) if successes else None,
        },
        "unsuccessful_request_rate": round(unsuccessful / total, 4) if total else 0.0,
        "hard_failure_rate": round(hard_failures / total, 4) if total else 0.0,
        "errors": [asdict(result) for result in results if result.outcome != "success"][:20],
    }


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def revision() -> str:
    configured = os.getenv("GITHUB_SHA")
    if configured:
        return configured
    return _git("rev-parse", "HEAD") or "unknown"


def worktree_dirty() -> bool | None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status is None:
        return None
    ignored_runtime_files = {"browser-server.log", "browser-server.pid", "runtime-benchmark-ci-fake.json"}
    meaningful_lines = [
        line
        for line in status.splitlines()
        if line[3:].strip().replace("\\", "/") not in ignored_runtime_files
    ]
    return bool(meaningful_lines)


def provenance() -> dict[str, object]:
    return {
        "git_revision": revision(),
        "git_worktree_dirty": worktree_dirty(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }


def run_benchmark(
    *,
    base_url: str,
    timeout_seconds: float,
    warmup_requests: int,
    warm_requests: int,
    concurrent_requests: int,
    concurrency: int,
) -> dict[str, object]:
    limits = httpx.Limits(max_connections=max(10, concurrency + 2), max_keepalive_connections=max(5, concurrency))
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds, limits=limits) as client:
        readiness, cold_readiness_ms, readiness_attempts = wait_for_readiness(
            client,
            timeout_seconds=timeout_seconds,
            wait_for_enabled_stockfish=True,
        )

        warmup_results = [predict_once(client) for _ in range(warmup_requests)]
        if any(result.outcome != "success" for result in warmup_results):
            raise RuntimeError(f"warmup inference failed: {warmup_results}")
        warm_results = [predict_once(client) for _ in range(warm_requests)]
        concurrent_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            concurrent_results = list(executor.map(lambda _index: predict_once(client), range(concurrent_requests)))
        concurrent_wall_seconds = time.perf_counter() - concurrent_started

        final_ready = client.get("/ready")
        final_ready.raise_for_status()
        final_runtime = final_ready.json()

    return {
        "schema_version": 2,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "target": base_url,
        "provenance": provenance(),
        "methodology": {
            "cold_start_assumption": (
                "Cold readiness is valid only when the target process was freshly started before this command."
            ),
            "warm_requests": warm_requests,
            "warmup_requests_excluded": warmup_requests,
            "concurrent_requests": concurrent_requests,
            "concurrency": concurrency,
            "client_timeout_seconds": timeout_seconds,
        },
        "cold_readiness": {
            "client_poll_to_ready_ms": round(cold_readiness_ms, 2),
            "readiness_poll_attempts": readiness_attempts,
            "server_start_to_ready_seconds": readiness.get("uptime_seconds"),
            "status": readiness.get("status"),
            "dependencies": readiness.get("dependencies"),
            "process_after_load": readiness.get("process"),
        },
        "warmup_inference_excluded": [asdict(result) for result in warmup_results],
        "warm_inference": summarize(warm_results, sum(result.latency_ms for result in warm_results) / 1000),
        "bounded_concurrency": summarize(concurrent_results, concurrent_wall_seconds),
        "process_after_benchmark": final_runtime.get("process"),
        "runtime_limits": final_runtime.get("limits"),
        "inference_capacity": final_runtime.get("inference"),
        "service_configuration": final_runtime.get("configuration"),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--warmup-requests", type=positive_int, default=1)
    parser.add_argument("--warm-requests", type=positive_int, default=5)
    parser.add_argument("--requests", type=positive_int, default=20, dest="concurrent_requests")
    parser.add_argument("--concurrency", type=positive_int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        warmup_requests=args.warmup_requests,
        warm_requests=args.warm_requests,
        concurrent_requests=args.concurrent_requests,
        concurrency=args.concurrency,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
