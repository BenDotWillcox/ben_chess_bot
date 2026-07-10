from __future__ import annotations

import pytest

from game_state import PolicyGameState


def test_game_state_tracks_legal_move_prefix_and_reset() -> None:
    state = PolicyGameState()
    state.push_san("e4")
    state.push_uci("e7e5")

    assert state.prefix == "e2e4 e7e5"
    assert state.board.peek().uci() == "e7e5"
    state.reset()
    assert state.prefix == ""
    assert len(state.board.move_stack) == 0


def test_game_state_rejects_illegal_policy_move_without_mutation() -> None:
    state = PolicyGameState()
    with pytest.raises(ValueError, match="Illegal move"):
        state.push_policy_move("e7e5")
    assert state.prefix == ""
    assert len(state.board.move_stack) == 0
