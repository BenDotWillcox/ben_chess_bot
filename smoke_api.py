#!/usr/bin/env python3
"""Run a lightweight readiness + legal two-ply game smoke test.

The default targets a running local deployment. ``--in-process`` is useful for
release validation because it exercises the real Maia2 and Stockfish loaders
without first managing a separate Uvicorn process.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from typing import Iterator, Protocol

import httpx


class HttpClient(Protocol):
    def get(self, url: str, **kwargs): ...

    def post(self, url: str, **kwargs): ...


def _request(client: HttpClient, method: str, path: str, **kwargs) -> tuple[object, float]:
    started = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response, elapsed_ms


def wait_for_readiness(
    client: HttpClient,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.25,
    require_stockfish: bool = False,
    wait_for_enabled_stockfish: bool = False,
) -> tuple[dict[str, object], float, int]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    attempts = 0
    last_payload: dict[str, object] = {}
    while time.perf_counter() < deadline:
        attempts += 1
        response = client.get("/ready")
        try:
            last_payload = response.json()
        except ValueError as exc:
            raise RuntimeError("readiness returned invalid JSON") from exc
        stockfish = last_payload.get("dependencies", {}).get("stockfish", {})
        stockfish_enabled = bool(stockfish.get("enabled"))
        stockfish_ready = stockfish.get("status") == "ready"
        stockfish_satisfied = (
            (not require_stockfish or stockfish_ready)
            and (not wait_for_enabled_stockfish or not stockfish_enabled or stockfish_ready)
        )
        if response.status_code == 200 and last_payload.get("ready") and stockfish_satisfied:
            return last_payload, (time.perf_counter() - started) * 1000, attempts
        if require_stockfish and not stockfish_enabled and not last_payload.get("initializing"):
            raise RuntimeError("Stockfish is required by this smoke test but disabled")
        if (
            (require_stockfish or (wait_for_enabled_stockfish and stockfish_enabled))
            and last_payload.get("status") == "degraded"
            and not last_payload.get("initializing")
        ):
            raise RuntimeError(f"Stockfish did not become ready: {stockfish}")
        if last_payload.get("status") == "unavailable" and not last_payload.get("initializing"):
            raise RuntimeError(f"service is unavailable: {last_payload}")
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"service did not become ready within {timeout_seconds:g}s: {last_payload}")


def run_smoke(
    client: HttpClient,
    *,
    require_stockfish: bool,
    readiness_timeout_seconds: float = 240.0,
) -> dict[str, object]:
    readiness, ready_ms, readiness_attempts = wait_for_readiness(
        client,
        timeout_seconds=readiness_timeout_seconds,
        require_stockfish=require_stockfish,
    )
    stockfish = readiness.get("dependencies", {}).get("stockfish", {})
    if require_stockfish and stockfish.get("status") != "ready":
        raise RuntimeError(f"Stockfish is not ready: {stockfish}")

    game_response, new_game_ms = _request(
        client,
        "post",
        "/new-game",
        json={"human_color": "white", "top_k": 5, "mode": "argmax", "temperature": 1.0},
    )
    game = game_response.json()
    if game.get("prefix") != "" or game.get("turn") != "white":
        raise RuntimeError(f"unexpected initial game state: {game}")

    reply_response, move_ms = _request(
        client,
        "post",
        f"/game/{game['game_id']}/move",
        json={"move": "e4", "top_k": 5, "mode": "argmax", "temperature": 1.0},
    )
    reply = reply_response.json()
    selected = (reply.get("bot_move") or {}).get("selected") or {}
    prefix = str(reply.get("prefix", "")).split()
    if len(prefix) != 2 or prefix[0] != "e2e4" or selected.get("move") != prefix[1]:
        raise RuntimeError(f"bot did not complete a legal two-ply smoke game: {reply}")

    return {
        "ok": True,
        "readiness": readiness,
        "game": {
            "game_id": game["game_id"],
            "prefix": reply["prefix"],
            "fen": reply["fen"],
            "bot_move": selected,
            "safety": reply["bot_move"].get("safety"),
        },
        "latency_ms": {
            "readiness": round(ready_ms, 1),
            "new_game": round(new_game_ms, 1),
            "human_plus_bot_move": round(move_ms, 1),
        },
        "readiness_attempts": readiness_attempts,
    }


@contextmanager
def make_client(*, base_url: str, timeout: float, in_process: bool) -> Iterator[HttpClient]:
    if in_process:
        os.environ.setdefault("CHESS_BOT_PRELOAD_ON_STARTUP", "0")
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
        return

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        yield client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("CHESS_BOT_BASE_URL", "http://127.0.0.1:7860"),
        help="Running BenBot base URL (ignored with --in-process).",
    )
    parser.add_argument("--timeout", type=float, default=240.0, help="Per-request timeout in seconds.")
    parser.add_argument("--in-process", action="store_true", help="Exercise the imported FastAPI app directly.")
    parser.add_argument(
        "--allow-degraded-stockfish",
        action="store_true",
        help="Pass when the optional Stockfish safety layer is degraded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with make_client(base_url=args.base_url, timeout=args.timeout, in_process=args.in_process) as client:
        result = run_smoke(
            client,
            require_stockfish=not args.allow_degraded_stockfish,
            readiness_timeout_seconds=args.timeout,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
