"""Helpers for tracking board state and policy move history."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class PolicyGameState:
    board: chess.Board = field(default_factory=chess.Board)
    uci_history: list[str] = field(default_factory=list)

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def prefix(self) -> str:
        return " ".join(self.uci_history)

    def push_uci(self, move_uci: str) -> chess.Move:
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move for current position: {move_uci}")
        self.board.push(move)
        self.uci_history.append(move.uci())
        return move

    def push_san(self, move_san: str) -> chess.Move:
        move = self.board.parse_san(move_san)
        self.board.push(move)
        self.uci_history.append(move.uci())
        return move

    def push_policy_move(self, move_uci: str) -> chess.Move:
        return self.push_uci(move_uci)

    def reset(self) -> None:
        self.board.reset()
        self.uci_history.clear()

    def result_or_none(self) -> str | None:
        if self.board.is_game_over(claim_draw=True):
            return self.board.result(claim_draw=True)
        return None
