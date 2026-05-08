#!/usr/bin/env python3
"""Smoke-test the FastAPI backend in process."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    health.raise_for_status()

    game = client.post("/new-game", json={"human_color": "white", "top_k": 5})
    game.raise_for_status()
    game_payload = game.json()

    reply = client.post(f"/game/{game_payload['game_id']}/move", json={"move": "e4", "top_k": 5})
    reply.raise_for_status()
    reply_payload = reply.json()

    print(
        json.dumps(
            {
                "health": health.json(),
                "new_game": {
                    "game_id": game_payload["game_id"],
                    "turn": game_payload["turn"],
                    "bot_move": game_payload["bot_move"],
                },
                "after_e4": {
                    "turn": reply_payload["turn"],
                    "prefix": reply_payload["prefix"],
                    "fen": reply_payload["fen"],
                    "bot_move": reply_payload["bot_move"]["selected"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
