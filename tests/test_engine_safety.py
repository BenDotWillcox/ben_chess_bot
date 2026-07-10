from __future__ import annotations

import chess

from engine_safety import StockfishBlunderVeto
from personalized_policy import PolicyMove


def candidate(move: str) -> PolicyMove:
    return PolicyMove(move, 0.5, 0.5, 0.0, "maia2", 0)


def fake_veto(evals: dict[str, int], *, threshold: int = 100) -> StockfishBlunderVeto:
    veto = object.__new__(StockfishBlunderVeto)
    veto.veto_cp = threshold
    veto.avoid_draw_when_winning_cp = threshold
    veto.avoid_shuffle_cp = 150
    veto.evaluate_board = lambda _board, _turn: 0
    veto.evaluate_after_move = lambda _board, move, _turn: evals[move]
    veto.drawish_reason = lambda _board: None
    veto.shuffle_reason = lambda _board, _move: None
    return veto


def test_stockfish_veto_replaces_move_beyond_threshold() -> None:
    board = chess.Board()
    original = candidate("e2e4")
    safer = candidate("d2d4")
    veto = fake_veto({"e2e4": -250, "d2d4": 20}, threshold=100)

    decision = veto.choose_safe_move(board, [original, safer])

    assert decision.vetoed
    assert decision.original == original
    assert decision.selected == safer
    assert "delta_270_cp" in decision.reason


def test_stockfish_veto_preserves_style_move_within_threshold() -> None:
    board = chess.Board()
    original = candidate("e2e4")
    alternative = candidate("d2d4")
    veto = fake_veto({"e2e4": 0, "d2d4": 50}, threshold=100)

    decision = veto.choose_safe_move(board, [original, alternative])

    assert not decision.vetoed
    assert decision.selected == original
    assert decision.reason == "within_threshold"
