from __future__ import annotations

import chess
import pytest
import time
import threading
from fastapi.testclient import TestClient

from api.runtime import SessionStore, SlidingWindowRateLimiter, WorkQueueFull, WorkTimedOut
from engine_safety import SafetyDecision
from personalized_policy import PolicyMove


class FakePolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, *, fen: str, **_kwargs) -> list[PolicyMove]:
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        board = chess.Board(fen)
        preferred = "e2e4" if board.turn == chess.WHITE else "e7e5"
        alternatives = [move.uci() for move in board.legal_moves if move.uci() != preferred]
        moves = [preferred, *alternatives[:4]]
        return [
            PolicyMove(move, 0.8 if index == 0 else 0.05, 0.8, 0.0, "maia2", 0)
            for index, move in enumerate(moves)
        ]


def configure_fake_policy(main, monkeypatch, *, fail: bool = False) -> FakePolicy:
    fake = FakePolicy(fail=fail)
    monkeypatch.setattr(main, "policy", fake)
    main._set_dependency("policy", "ready")
    return fake


def poll_readiness(client: TestClient, predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    response = client.get("/ready")
    while not predicate(response.json()) and time.monotonic() < deadline:
        time.sleep(0.01)
        response = client.get("/ready")
    return response


def test_health_is_liveness_only_and_status_badges_are_cached(api_module, monkeypatch) -> None:
    main = api_module
    monkeypatch.setattr(main, "get_policy", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    client = TestClient(main.app)

    health = client.get("/health")
    badge = client.get("/status-badge")
    svg = client.get("/status.svg")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["policy_loaded"] is False
    assert badge.json()["message"] == "starting"
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert "starting" in svg.text


def test_readiness_loads_policy_and_surfaces_optional_stockfish_degradation(api_module, monkeypatch) -> None:
    main = api_module
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)
    monkeypatch.setattr(main, "REQUIRE_STOCKFISH", False)

    def load_policy():
        main.policy = object()
        main._set_dependency("policy", "ready")
        return main.policy

    def fail_stockfish():
        main._set_dependency("stockfish", "degraded", error="Stockfish synthetic failure")
        return None

    monkeypatch.setattr(main, "get_policy", load_policy)
    monkeypatch.setattr(main, "get_safety", fail_stockfish)
    client = TestClient(main.app)
    response = poll_readiness(client, lambda payload: payload["ready"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["policy"]["status"] == "ready"
    assert payload["dependencies"]["stockfish"]["error"] == "Stockfish synthetic failure"
    badge = TestClient(main.app).get("/status-badge").json()
    assert badge["message"] == "degraded"
    assert badge["color"] == "orange"


def test_required_stockfish_degradation_fails_readiness(api_module, monkeypatch) -> None:
    main = api_module
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)
    monkeypatch.setattr(main, "REQUIRE_STOCKFISH", True)
    monkeypatch.setattr(main, "policy", object())
    main._set_dependency("policy", "ready")
    main._set_dependency("stockfish", "degraded", error="unavailable")
    monkeypatch.setattr(main, "get_safety", lambda: None)

    client = TestClient(main.app)
    response = poll_readiness(client, lambda payload: payload["status"] == "unavailable")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert TestClient(main.app).get("/status-badge").json()["isError"] is True


def test_api_smoke_game_plays_legal_human_and_bot_moves(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    client = TestClient(main.app)

    game = client.post("/new-game", json={"human_color": "white", "mode": "argmax"})
    assert game.status_code == 200
    game_id = game.json()["game_id"]
    assert game.json()["prefix"] == ""

    reply = client.post(f"/game/{game_id}/move", json={"move": "e4", "mode": "argmax"})
    assert reply.status_code == 200
    payload = reply.json()
    assert payload["prefix"] == "e2e4 e7e5"
    assert payload["bot_move"]["selected"]["move"] == "e7e5"
    assert payload["bot_move"]["safety"]["status"] == "disabled"
    assert payload["policy_config"]["strategy"] == "fen"
    assert payload["policy_config"]["alpha"] == 0.7
    assert payload["policy_config"]["min_count"] == 1


def test_failed_bot_inference_rolls_back_human_move(api_module, monkeypatch) -> None:
    main = api_module
    fake = configure_fake_policy(main, monkeypatch)
    client = TestClient(main.app, raise_server_exceptions=False)
    game_id = client.post("/new-game", json={"human_color": "white"}).json()["game_id"]
    fake.fail = True

    failed = client.post(f"/game/{game_id}/move", json={"move": "e4"})
    recovered = client.get(f"/game/{game_id}")

    assert failed.status_code == 500
    assert recovered.status_code == 200
    assert recovered.json()["prefix"] == ""
    assert recovered.json()["turn"] == "white"


def test_session_expiry_and_rate_limit_are_enforced(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    now = [0.0]
    monkeypatch.setattr(
        main,
        "sessions",
        SessionStore(max_sessions=2, ttl_seconds=5, clock=lambda: now[0]),
    )
    monkeypatch.setattr(
        main,
        "rate_limiter",
        SlidingWindowRateLimiter(max_requests=1, window_seconds=60),
    )
    client = TestClient(main.app)

    created = client.post("/new-game", json={"human_color": "white"})
    game_id = created.json()["game_id"]
    limited = client.post("/new-game", json={"human_color": "white"})
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"

    now[0] = 5.0
    expired = client.get(f"/game/{game_id}")
    assert expired.status_code == 429  # the limiter applies before session lookup


def test_expired_session_returns_not_found(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    now = [0.0]
    monkeypatch.setattr(main, "sessions", SessionStore(max_sessions=2, ttl_seconds=5, clock=lambda: now[0]))
    client = TestClient(main.app)
    game_id = client.post("/new-game", json={"human_color": "white"}).json()["game_id"]

    now[0] = 5.0
    response = client.get(f"/game/{game_id}")
    assert response.status_code == 404
    assert "expired" in response.json()["detail"]


def test_stockfish_failure_is_returned_as_degraded_move_state(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)

    class BrokenSafety:
        def choose_safe_move(self, *_args):
            raise RuntimeError("engine crashed")

        def close(self):
            return None

    broken = BrokenSafety()
    monkeypatch.setattr(main, "safety", broken)
    main._set_dependency("stockfish", "ready")
    client = TestClient(main.app)
    game_id = client.post("/new-game", json={"human_color": "white"}).json()["game_id"]

    response = client.post(f"/game/{game_id}/move", json={"move": "e4", "mode": "argmax"})
    assert response.status_code == 200
    safety = response.json()["bot_move"]["safety"]
    assert safety["status"] == "degraded"
    assert safety["available"] is False
    assert safety["error"] == "Stockfish is unavailable"
    assert "engine crashed" not in response.text
    assert response.json()["safety_status"]["status"] == "degraded"


def test_stale_safety_completion_cannot_clear_later_degradation(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)

    class FakeSafety:
        def choose_safe_move(self, _board, candidates):
            original = candidates[0]
            replacement = next(candidate for candidate in candidates if candidate.move != original.move)
            return SafetyDecision(replacement, original, True, "synthetic_veto", 25, -500, 25)

        def close(self):
            return None

    class InterleavingExecutor:
        closed = False

        def run(self, fn, **_kwargs):
            completed = fn()
            # Another admitted call times out after this work completed but
            # before its caller can publish the successful result.
            main._mark_safety_degraded(WorkTimedOut("other call timed out"), discard=False)
            return completed

        def stats(self):
            return {"active": 0, "capacity": 2, "workers": 1, "queue_capacity": 1}

    engine = FakeSafety()
    monkeypatch.setattr(main, "safety", engine)
    monkeypatch.setattr(main, "safety_generation", 7)
    monkeypatch.setattr(main, "safety_executor", InterleavingExecutor())
    main._set_dependency("stockfish", "ready")
    client = TestClient(main.app)
    game_id = client.post("/new-game", json={"human_color": "white"}).json()["game_id"]

    response = client.post(f"/game/{game_id}/move", json={"move": "e4", "mode": "argmax"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["bot_move"]["selected"]["move"] == "e7e5"
    assert payload["bot_move"]["safety"]["status"] == "degraded"
    assert payload["safety_status"]["available"] is False
    assert main._dependency_snapshot("stockfish")["status"] == "degraded"
    assert main.readiness_payload()["status"] == "degraded"
    assert main.get_safety() is None


def test_active_timed_out_stockfish_worker_is_not_reused(api_module, monkeypatch) -> None:
    main = api_module
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)
    class StaleEngine:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    engine = StaleEngine()
    replacement = object()
    monkeypatch.setattr(main, "safety", engine)
    monkeypatch.setattr(main, "safety_retry_after", 0.0)
    monkeypatch.setattr(main, "safety_needs_reload", True)
    main._set_dependency("stockfish", "degraded", error="timed out")
    monkeypatch.setattr(main.safety_executor, "stats", lambda: {"active": 1})

    assert main.get_safety() is None
    assert main._dependency_snapshot("stockfish")["status"] == "degraded"

    monkeypatch.setattr(main.safety_executor, "stats", lambda: {"active": 0})
    monkeypatch.setattr(main, "_resolve_executable", lambda _path: "stockfish")
    monkeypatch.setattr(main, "StockfishBlunderVeto", lambda **_kwargs: replacement)
    assert main.get_safety() is replacement
    assert engine.closed
    assert main._dependency_snapshot("stockfish")["status"] == "ready"


@pytest.mark.parametrize(
    ("failure", "status_code", "detail"),
    [
        (WorkQueueFull("full"), 503, "inference capacity"),
        (WorkTimedOut("slow"), 504, "inference timed out"),
    ],
)
def test_predict_maps_inference_backpressure_and_timeout_to_explicit_http_status(
    api_module, monkeypatch, failure, status_code, detail
) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)

    class FailingExecutor:
        closed = False

        def stats(self):
            return {"active": 0, "capacity": 1, "workers": 1, "queue_capacity": 0}

        def run(self, *_args, **_kwargs):
            raise failure

    monkeypatch.setattr(main, "inference_executor", FailingExecutor())
    response = TestClient(main.app).post(
        "/predict",
        json={"fen": chess.Board().fen(), "you_color": "white"},
    )

    assert response.status_code == status_code
    assert detail in response.json()["detail"]
    if status_code == 503:
        assert response.headers["retry-after"] == "1"


def test_malformed_dependency_env_is_safe_in_payloads_and_fails_readiness(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    monkeypatch.setenv("CHESS_BOT_ALPHA", "secret-not-a-number")
    monkeypatch.setenv("CHESS_BOT_MIN_COUNT", "invalid")
    monkeypatch.setenv("CHESS_BOT_BLUNDER_VETO_CP", "invalid")
    monkeypatch.setenv("CHESS_BOT_ENGINE_TIME", "invalid")
    client = TestClient(main.app)

    game = client.post("/new-game", json={"human_color": "white"})
    assert game.status_code == 200
    config = game.json()["policy_config"]
    assert config["config_valid"] is False
    assert config["alpha"] is None
    assert config["stockfish_veto_cp"] is None

    response = poll_readiness(client, lambda payload: payload["status"] == "unavailable")
    assert response.status_code == 503
    assert response.json()["configuration"]["policy_config_valid"] is False
    assert "secret-not-a-number" not in response.text


def test_malformed_stockfish_env_degrades_without_leaking_or_500(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    monkeypatch.setattr(main, "STOCKFISH_ENABLED", True)
    monkeypatch.setattr(main, "_resolve_executable", lambda _path: "stockfish")
    monkeypatch.setenv("CHESS_BOT_BLUNDER_VETO_CP", "private-invalid-value")
    client = TestClient(main.app)

    response = poll_readiness(
        client,
        lambda payload: payload["status"] == "degraded" and not payload["initializing"],
    )
    assert response.status_code == 200
    assert response.json()["configuration"]["stockfish_config_valid"] is False
    assert response.json()["dependencies"]["stockfish"]["error"] == "Stockfish is unavailable"
    assert "private-invalid-value" not in response.text


def test_forwarded_for_does_not_bypass_rate_limit_by_default(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    monkeypatch.setattr(main, "TRUST_PROXY_HEADERS", False)
    monkeypatch.setattr(main, "rate_limiter", SlidingWindowRateLimiter(max_requests=1, window_seconds=60))
    client = TestClient(main.app)

    first = client.post("/new-game", json={"human_color": "white"}, headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.post("/new-game", json={"human_color": "white"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.parametrize(
    "payload",
    [
        {"fen": "x" * 129, "you_color": "white"},
        {"fen": chess.Board().fen(), "you_color": "white", "uci_prefix_before": "x" * 4097},
        {"fen": chess.Board().fen(), "you_color": "white", "elo_self": -1},
        {"fen": chess.Board().fen(), "you_color": "white", "elo_oppo": 999999},
    ],
)
def test_predict_rejects_oversized_or_out_of_range_inputs(api_module, payload) -> None:
    response = TestClient(api_module.app).post("/predict", json=payload)
    assert response.status_code == 422


def test_move_rejects_oversized_notation(api_module, monkeypatch) -> None:
    main = api_module
    configure_fake_policy(main, monkeypatch)
    client = TestClient(main.app)
    game_id = client.post("/new-game", json={"human_color": "white"}).json()["game_id"]
    assert client.post(f"/game/{game_id}/move", json={"move": "x" * 17}).status_code == 422


def test_policy_timeout_stays_unavailable_until_real_inference_succeeds(api_module, monkeypatch) -> None:
    main = api_module
    fake = configure_fake_policy(main, monkeypatch)

    class TimedOutExecutor:
        closed = False

        def run(self, *_args, **_kwargs):
            raise WorkTimedOut("stuck")

        def stats(self):
            return {"active": 1, "capacity": 1, "workers": 1, "queue_capacity": 0}

    monkeypatch.setattr(main, "inference_executor", TimedOutExecutor())
    with pytest.raises(Exception):
        main._run_model_work(lambda: None)
    assert main.readiness_payload()["status"] == "unavailable"
    assert main.get_policy() is fake
    assert main.readiness_payload()["status"] == "unavailable"

    class RecoveredExecutor:
        closed = False

        def run(self, fn, **_kwargs):
            return fn()

        def stats(self):
            return {"active": 0, "capacity": 1, "workers": 1, "queue_capacity": 0}

    monkeypatch.setattr(main, "inference_executor", RecoveredExecutor())
    result = main._run_model_work(
        lambda: fake.predict(fen=chess.Board().fen(), you_color="white", elo_self=1650, elo_oppo=1650)
    )
    assert result
    assert main.readiness_payload()["status"] == "ready"


def test_ready_is_nonblocking_single_flight_during_cold_load(api_module, monkeypatch) -> None:
    main = api_module
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_load():
        calls.append(1)
        started.set()
        release.wait(timeout=2)
        main.policy = object()
        main._set_dependency("policy", "ready")
        return main.policy

    monkeypatch.setattr(main, "get_policy", slow_load)
    client = TestClient(main.app)
    before = time.monotonic()
    first = client.get("/ready")
    elapsed = time.monotonic() - before
    assert first.status_code == 503
    assert first.json()["status"] == "starting"
    assert elapsed < 0.5
    assert started.wait(timeout=1)
    client.get("/ready")
    assert len(calls) == 1
    release.set()
    response = poll_readiness(client, lambda payload: payload["ready"])
    assert response.status_code == 200


def test_shutdown_drains_safety_executor_before_closing_engine(api_module, monkeypatch) -> None:
    main = api_module
    events = []

    class FakeExecutor:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def shutdown(self, *, wait=False):
            self.closed = True
            events.append((self.name, wait))

    class FakeSafety:
        def close(self):
            events.append(("engine", True))

    monkeypatch.setattr(main, "inference_executor", FakeExecutor("inference"))
    monkeypatch.setattr(main, "safety_executor", FakeExecutor("safety"))
    monkeypatch.setattr(main, "safety", FakeSafety())
    main.shutdown()
    main.shutdown()

    assert events == [("inference", False), ("safety", True), ("engine", True)]
