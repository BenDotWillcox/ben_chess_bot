from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from api.runtime import (
    BoundedExecutor,
    JsonLogFormatter,
    SessionStore,
    SlidingWindowRateLimiter,
    WorkQueueFull,
    WorkTimedOut,
    process_memory_snapshot,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_session_store_expires_and_evicts_lru_sessions() -> None:
    clock = FakeClock()
    store = SessionStore[str](max_sessions=2, ttl_seconds=10, clock=clock)

    store.put("a", "A")
    clock.advance(1)
    store.put("b", "B")
    assert store.get("a") == "A"  # a is now most recently used

    removed = store.put("c", "C")
    assert removed == ["b"]
    assert store.get("b") is None
    assert store.get("a") == "A"

    clock.advance(10)
    assert store.get("a") is None
    assert store.get("c") is None
    assert store.stats()["active"] == 0


def test_sliding_window_rate_limiter_recovers_after_window() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=5, clock=clock)

    assert limiter.check("client").allowed
    assert limiter.check("client").allowed
    rejected = limiter.check("client")
    assert not rejected.allowed
    assert rejected.retry_after == pytest.approx(5)

    clock.advance(5)
    assert limiter.check("client").allowed


def test_bounded_executor_rejects_when_worker_and_queue_are_full() -> None:
    executor = BoundedExecutor(max_workers=1, max_queue=0, name="test-bounded")
    started = threading.Event()
    release = threading.Event()

    def blocking_work() -> str:
        started.set()
        release.wait(timeout=2)
        return "done"

    result: list[str] = []
    worker = threading.Thread(
        target=lambda: result.append(
            executor.run(blocking_work, timeout_seconds=2, admission_timeout_seconds=0)
        )
    )
    worker.start()
    assert started.wait(timeout=1)
    with pytest.raises(WorkQueueFull):
        executor.run(lambda: "overflow", timeout_seconds=1, admission_timeout_seconds=0)
    release.set()
    worker.join(timeout=2)
    assert result == ["done"]
    executor.shutdown()


def test_timed_out_running_work_retains_capacity_until_completion() -> None:
    executor = BoundedExecutor(max_workers=1, max_queue=0, name="test-timeout")
    release = threading.Event()

    with pytest.raises(WorkTimedOut):
        executor.run(lambda: release.wait(timeout=2), timeout_seconds=0.01, admission_timeout_seconds=0)
    assert executor.stats()["active"] == 1
    with pytest.raises(WorkQueueFull):
        executor.run(lambda: None, timeout_seconds=1, admission_timeout_seconds=0)

    release.set()
    deadline = time.monotonic() + 1
    while executor.stats()["active"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert executor.stats()["active"] == 0
    executor.shutdown()


def test_bounded_executor_shutdown_is_idempotent_and_rejects_new_work() -> None:
    executor = BoundedExecutor(max_workers=1, max_queue=0, name="test-shutdown")
    assert not executor.closed
    executor.shutdown(wait=False)
    executor.shutdown(wait=False)

    assert executor.closed
    with pytest.raises(WorkQueueFull, match="shut down"):
        executor.run(lambda: None, timeout_seconds=1, admission_timeout_seconds=0)


def test_json_formatter_emits_structured_event_fields() -> None:
    record = logging.LogRecord("benbot", logging.INFO, __file__, 1, "request_complete", (), None)
    record.event = "request_complete"
    record.fields = {"status_code": 200, "duration_ms": 12.5}
    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["event"] == "request_complete"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 12.5


def test_process_memory_snapshot_has_bounded_standard_schema() -> None:
    snapshot = process_memory_snapshot()
    assert set(snapshot) == {"rss_mb", "peak_rss_mb", "source", "scope"}
    assert snapshot["rss_mb"] is None or snapshot["rss_mb"] > 0
    assert snapshot["scope"] == "api_process_only_excludes_stockfish_child"
