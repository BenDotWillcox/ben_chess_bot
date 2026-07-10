from __future__ import annotations

import os

import pytest


os.environ["CHESS_BOT_MOVE_DELAY_SECONDS"] = "0"
os.environ["CHESS_BOT_PRELOAD_ON_STARTUP"] = "0"


@pytest.fixture
def api_module(monkeypatch):
    import api.main as main
    from api.runtime import SessionStore, SlidingWindowRateLimiter

    init_thread = main.dependency_init_thread
    if init_thread is not None and init_thread.is_alive():
        init_thread.join(timeout=2)
    if main.inference_executor.closed:
        main.inference_executor = main._new_inference_executor()
    if main.safety_executor.closed:
        main.safety_executor = main._new_safety_executor()

    monkeypatch.setattr(
        main,
        "sessions",
        SessionStore(max_sessions=16, ttl_seconds=60),
    )
    monkeypatch.setattr(
        main,
        "rate_limiter",
        SlidingWindowRateLimiter(max_requests=1000, window_seconds=60),
    )
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", False)
    monkeypatch.setattr(main, "REQUIRE_STOCKFISH", False)
    monkeypatch.setattr(main, "policy", None)
    monkeypatch.setattr(main, "policy_retry_after", 0.0)
    monkeypatch.setattr(main, "policy_runtime_blocked", False)
    monkeypatch.setattr(main, "safety", None)
    monkeypatch.setattr(main, "safety_retry_after", 0.0)
    monkeypatch.setattr(main, "safety_needs_reload", False)
    monkeypatch.setattr(main, "safety_generation", 0)
    monkeypatch.setattr(main, "runtime_shutting_down", False)
    with main.dependency_lock:
        main.dependency_state["policy"] = {
            "status": "not_loaded",
            "error": None,
            "last_attempt": None,
            "last_ready": None,
        }
        main.dependency_state["stockfish"] = {
            "status": "disabled",
            "error": None,
            "last_attempt": None,
            "last_ready": None,
        }
    yield main
    init_thread = main.dependency_init_thread
    if init_thread is not None and init_thread.is_alive():
        init_thread.join(timeout=2)
