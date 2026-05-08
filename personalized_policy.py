"""Runtime policy for a personalized Maia2 chess bot."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maia2 import inference, model


def position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def prefix_key(you_color: str, uci_prefix_before: str) -> str:
    return f"{you_color}|{uci_prefix_before.strip()}"


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {move: value / total for move, value in weights.items()}


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if math.isclose(temperature, 1.0):
        return probs
    adjusted = {move: value ** (1.0 / temperature) for move, value in probs.items()}
    return normalize(adjusted)


@dataclass(frozen=True)
class PolicyMove:
    move: str
    probability: float
    maia_probability: float
    personal_probability: float
    source: str
    count: int


class PersonalizedMaia2Policy:
    def __init__(
        self,
        maia_model: Any,
        prepared: list[Any],
        fen_book: dict[str, dict[str, int]],
        prefix_book: dict[str, dict[str, int]],
        strategy: str = "combined",
        alpha: float = 0.5,
        min_count: int = 1,
    ) -> None:
        self.maia_model = maia_model
        self.prepared = prepared
        self.fen_book = fen_book
        self.prefix_book = prefix_book
        self.strategy = strategy
        self.alpha = alpha
        self.min_count = min_count

    @classmethod
    def load(
        cls,
        books_path: str | Path = "artifacts/personal_books.json",
        model_type: str = "rapid",
        device: str = "cpu",
        strategy: str = "combined",
        alpha: float = 0.5,
        min_count: int = 1,
    ) -> "PersonalizedMaia2Policy":
        books = json.loads(Path(books_path).read_text())
        maia_model = model.from_pretrained(model_type, device)
        prepared = inference.prepare()
        return cls(
            maia_model=maia_model,
            prepared=prepared,
            fen_book=books["fen_book"],
            prefix_book=books["prefix_book"],
            strategy=strategy,
            alpha=alpha,
            min_count=min_count,
        )

    def _select_counts(self, fen: str, you_color: str, uci_prefix_before: str) -> tuple[dict[str, int] | None, str]:
        fen_counts = self.fen_book.get(position_key(fen))
        prefix_counts = self.prefix_book.get(prefix_key(you_color, uci_prefix_before))

        if self.strategy == "fen":
            return fen_counts, "fen" if fen_counts else "maia2"
        if self.strategy == "prefix":
            return prefix_counts, "prefix" if prefix_counts else "maia2"
        if self.strategy == "combined":
            if prefix_counts and sum(prefix_counts.values()) >= self.min_count:
                return prefix_counts, "prefix"
            if fen_counts and sum(fen_counts.values()) >= self.min_count:
                return fen_counts, "fen"
            return None, "maia2"
        raise ValueError(f"Unsupported strategy: {self.strategy}")

    def predict(
        self,
        fen: str,
        elo_self: int,
        elo_oppo: int,
        you_color: str,
        uci_prefix_before: str = "",
        top_n: int = 10,
        temperature: float = 1.0,
    ) -> list[PolicyMove]:
        maia_probs, _ = inference.inference_each(self.maia_model, self.prepared, fen, elo_self, elo_oppo)
        counts, source = self._select_counts(fen, you_color, uci_prefix_before)

        count_total = sum(counts.values()) if counts else 0
        if counts and count_total >= self.min_count:
            personal_probs = {move: count / count_total for move, count in counts.items()}
            blended = {}
            for move, maia_prob in maia_probs.items():
                blended[move] = ((1.0 - self.alpha) * maia_prob) + (
                    self.alpha * personal_probs.get(move, 0.0)
                )
            blended = normalize(blended)
        else:
            personal_probs = {}
            blended = maia_probs
            source = "maia2"

        blended = apply_temperature(blended, temperature)
        sorted_moves = sorted(blended.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return [
            PolicyMove(
                move=move,
                probability=round(probability, 6),
                maia_probability=round(float(maia_probs.get(move, 0.0)), 6),
                personal_probability=round(float(personal_probs.get(move, 0.0)), 6),
                source=source,
                count=int(counts.get(move, 0)) if counts else 0,
            )
            for move, probability in sorted_moves
        ]

    def choose_move(
        self,
        fen: str,
        elo_self: int,
        elo_oppo: int,
        you_color: str,
        uci_prefix_before: str = "",
        mode: str = "argmax",
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> PolicyMove:
        moves = self.predict(
            fen=fen,
            elo_self=elo_self,
            elo_oppo=elo_oppo,
            you_color=you_color,
            uci_prefix_before=uci_prefix_before,
            top_n=top_k,
            temperature=temperature,
        )
        if not moves:
            raise ValueError("Policy returned no legal moves")
        if mode == "argmax":
            return moves[0]
        if mode == "sample":
            weights = [move.probability for move in moves]
            return random.choices(moves, weights=weights, k=1)[0]
        raise ValueError("mode must be 'argmax' or 'sample'")
