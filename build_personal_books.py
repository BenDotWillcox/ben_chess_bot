#!/usr/bin/env python3
"""Build runtime personalization books from the training split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def training_fingerprint(train: pd.DataFrame) -> str:
    required = ["game_id", "date_utc", "ply_index", "fen", "move"]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"Training split is missing reproducibility columns: {missing}")
    ordered = train.sort_values(["date_utc", "game_id", "ply_index"], kind="stable")
    digest = hashlib.sha256()
    for row in ordered[required].itertuples(index=False, name=None):
        digest.update(json.dumps([str(value) for value in row], separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_books(train: pd.DataFrame, max_prefix_ply: int) -> dict[str, Any]:
    if max_prefix_ply < 0:
        raise ValueError("max_prefix_ply must be non-negative")
    required = {"fen", "move", "ply_prefix_before", "you_color", "uci_prefix_before"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training split is missing book columns: {missing}")
    train = train.sort_values(["date_utc", "game_id", "ply_index"], kind="stable").reset_index(drop=True)
    fen_book: dict[str, Counter[str]] = defaultdict(Counter)
    prefix_book: dict[str, Counter[str]] = defaultdict(Counter)

    for row in train.itertuples(index=False):
        fen_book[position_key(row.fen)][row.move] += 1

        if row.ply_prefix_before <= max_prefix_ply:
            prefix_key = f"{row.you_color}|{row.uci_prefix_before}"
            prefix_book[prefix_key][row.move] += 1

    return {
        "metadata": {
            "schema_version": 2,
            "source": "chronological train split",
            "samples": int(len(train)),
            "games": int(train["game_id"].nunique()),
            "date_min": str(train["date_utc"].min()),
            "date_max": str(train["date_utc"].max()),
            "training_sha256": training_fingerprint(train),
            "max_prefix_ply": max_prefix_ply,
            "fen_positions": len(fen_book),
            "prefix_positions": len(prefix_book),
        },
        "fen_book": {key: counter_to_dict(value) for key, value in sorted(fen_book.items())},
        "prefix_book": {key: counter_to_dict(value) for key, value in sorted(prefix_book.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", default="data/processed/splits/chron/train.parquet")
    parser.add_argument("--output", default="artifacts/personal_books.json")
    parser.add_argument("--max-prefix-ply", type=int, default=20)
    args = parser.parse_args()

    train = pd.read_parquet(args.train_split).reset_index(drop=True)
    books = build_books(train, max_prefix_ply=args.max_prefix_ply)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(books, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(books["metadata"], indent=2))


if __name__ == "__main__":
    main()
