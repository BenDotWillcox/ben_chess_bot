"""Shared defaults and candidate selection for BenBot command-line tools."""
from __future__ import annotations

import random
from collections.abc import Sequence

from personalized_policy import PolicyMove


DEFAULT_STRATEGY = "fen"
DEFAULT_ALPHA = 0.7
DEFAULT_MIN_COUNT = 1


def select_candidate(
    candidates: Sequence[PolicyMove],
    *,
    mode: str,
    rng: random.Random | None = None,
) -> PolicyMove:
    """Select from an existing ranked distribution without rerunning Maia2."""
    moves = list(candidates)
    if not moves:
        raise RuntimeError("No candidate moves available")
    if mode == "argmax":
        return moves[0]
    if mode == "sample":
        chooser = rng if rng is not None else random
        return chooser.choices(moves, weights=[move.probability for move in moves], k=1)[0]
    raise ValueError("mode must be 'argmax' or 'sample'")
