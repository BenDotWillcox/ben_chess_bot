"""Test-only Uvicorn entrypoint with lightweight dependency fakes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import uvicorn


os.environ["CHESS_BOT_MOVE_DELAY_SECONDS"] = "0"
os.environ["CHESS_BOT_PRELOAD_ON_STARTUP"] = "0"
os.environ["CHESS_BOT_STOCKFISH_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as main  # noqa: E402
from personalized_policy import PolicyMove  # noqa: E402


class BrowserSmokePolicy:
    def predict(self, *, fen: str, **_kwargs) -> list[PolicyMove]:
        board = chess.Board(fen)
        preferred = "e2e4" if board.turn == chess.WHITE else "e7e5"
        legal = [move.uci() for move in board.legal_moves]
        ordered = [preferred, *[move for move in legal if move != preferred]][:5]
        return [
            PolicyMove(move, 0.8 if index == 0 else 0.05, 0.8, 0.0, "maia2", 0)
            for index, move in enumerate(ordered)
        ]


main.policy = BrowserSmokePolicy()
main._set_dependency("policy", "ready")
main._set_dependency("stockfish", "disabled")


if __name__ == "__main__":
    uvicorn.run(main.app, host="127.0.0.1", port=int(os.getenv("CHESS_BOT_TEST_PORT", "8765")))
