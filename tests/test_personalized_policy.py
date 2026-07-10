from __future__ import annotations

import chess
import pytest

from personalized_policy import PersonalizedMaia2Policy, apply_temperature, normalize, position_key


def test_normalize_preserves_keys_and_sums_to_one() -> None:
    result = normalize({"e2e4": 2.0, "d2d4": 1.0})
    assert result == pytest.approx({"e2e4": 2 / 3, "d2d4": 1 / 3})
    assert sum(result.values()) == pytest.approx(1.0)


def test_temperature_sharpens_distribution_and_rejects_nonpositive_values() -> None:
    original = {"e2e4": 0.75, "d2d4": 0.25}
    sharpened = apply_temperature(original, 0.5)
    softened = apply_temperature(original, 2.0)

    assert sharpened["e2e4"] > original["e2e4"]
    assert softened["e2e4"] < original["e2e4"]
    assert sum(sharpened.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(original, 0)


def test_combined_source_prefers_qualified_prefix_then_fen_then_maia() -> None:
    fen = chess.Board().fen()
    maia_probs = {"e2e4": 0.6, "d2d4": 0.4}
    policy = PersonalizedMaia2Policy(
        maia_model=object(),
        prepared=[],
        fen_book={position_key(fen): {"d2d4": 3}},
        prefix_book={"white|": {"e2e4": 2}},
        strategy="combined",
        alpha=0.5,
        min_count=2,
        inference_each=lambda *_: (maia_probs, None),
    )

    prefix_moves = policy.predict(fen, 1650, 1650, "white")
    assert {move.source for move in prefix_moves} == {"prefix"}
    assert prefix_moves[0].move == "e2e4"

    policy.prefix_book = {"white|": {"e2e4": 1}}
    fen_moves = policy.predict(fen, 1650, 1650, "white")
    assert {move.source for move in fen_moves} == {"fen"}
    assert fen_moves[0].move == "d2d4"

    policy.fen_book = {}
    maia_moves = policy.predict(fen, 1650, 1650, "white")
    assert {move.source for move in maia_moves} == {"maia2"}


def test_constructor_rejects_invalid_personalization_configuration() -> None:
    with pytest.raises(ValueError, match="strategy"):
        PersonalizedMaia2Policy(object(), [], {}, {}, strategy="unknown")
    with pytest.raises(ValueError, match="alpha"):
        PersonalizedMaia2Policy(object(), [], {}, {}, alpha=1.1)
    with pytest.raises(ValueError, match="min_count"):
        PersonalizedMaia2Policy(object(), [], {}, {}, min_count=0)
