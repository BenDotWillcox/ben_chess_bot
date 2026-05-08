#!/usr/bin/env python3
"""Evaluate a simple player-specific reranker on top of Maia2."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from maia2 import inference, model


def position_key(fen: str) -> str:
    """Use the game-state fields that determine legal moves, not clocks."""
    return " ".join(fen.split()[:4])


def build_position_book(train: pd.DataFrame, max_fullmove: int | None) -> dict[str, Counter[str]]:
    if max_fullmove is not None and "fullmove_number" in train.columns:
        train = train[train["fullmove_number"] <= max_fullmove]

    book: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        book[position_key(row.fen)][row.move] += 1
    return dict(book)


def blend_move_probs(
    maia_probs: dict[str, float],
    personal_counts: Counter[str] | None,
    alpha: float,
    min_count: int,
) -> tuple[dict[str, float], bool]:
    if not personal_counts or sum(personal_counts.values()) < min_count:
        return maia_probs, False

    total = sum(personal_counts.values())
    personal_probs = {move: count / total for move, count in personal_counts.items()}

    blended = {}
    for move, maia_prob in maia_probs.items():
        blended[move] = ((1.0 - alpha) * maia_prob) + (alpha * personal_probs.get(move, 0.0))

    blended = dict(sorted(blended.items(), key=lambda item: item[1], reverse=True))
    return blended, True


def top_k_hit(move_probs: dict[str, float], actual_move: str, k: int) -> bool:
    return actual_move in list(move_probs.keys())[:k]


def target_probability(move_probs: dict[str, float], actual_move: str) -> float:
    return float(move_probs.get(actual_move, 0.0))


def summarize(
    source: pd.DataFrame,
    maia_probs: list[dict[str, float]],
    personalized_probs: list[dict[str, float]],
    used_personal: list[bool],
    include_phases: bool = True,
) -> dict[str, Any]:
    actual_moves = source["move"].tolist()

    def metrics(probs: list[dict[str, float]]) -> dict[str, float]:
        return {
            "top1_accuracy": round(sum(top_k_hit(p, m, 1) for p, m in zip(probs, actual_moves)) / len(source), 4),
            "top3_accuracy": round(sum(top_k_hit(p, m, 3) for p, m in zip(probs, actual_moves)) / len(source), 4),
            "top5_accuracy": round(sum(top_k_hit(p, m, 5) for p, m in zip(probs, actual_moves)) / len(source), 4),
            "mean_target_probability": round(
                sum(target_probability(p, m) for p, m in zip(probs, actual_moves)) / len(source),
                4,
            ),
        }

    changed_top1 = 0
    improved_top1 = 0
    worsened_top1 = 0
    for maia, personalized, actual in zip(maia_probs, personalized_probs, actual_moves):
        maia_top = next(iter(maia))
        personal_top = next(iter(personalized))
        if maia_top != personal_top:
            changed_top1 += 1
        maia_hit = maia_top == actual
        personal_hit = personal_top == actual
        if personal_hit and not maia_hit:
            improved_top1 += 1
        if maia_hit and not personal_hit:
            worsened_top1 += 1

    summary: dict[str, Any] = {
        "samples": int(len(source)),
        "personal_book_coverage": round(sum(used_personal) / len(source), 4),
        "maia2": metrics(maia_probs),
        "personalized": metrics(personalized_probs),
        "top1_changed_samples": int(changed_top1),
        "top1_improved_samples": int(improved_top1),
        "top1_worsened_samples": int(worsened_top1),
    }

    if include_phases and "fullmove_number" in source.columns:
        summary["by_phase"] = {}
        phases = {
            "opening_1_10": source["fullmove_number"] <= 10,
            "middlegame_11_30": (source["fullmove_number"] > 10) & (source["fullmove_number"] <= 30),
            "endgame_31_plus": source["fullmove_number"] > 30,
        }
        for phase, mask in phases.items():
            idx = source[mask].index.tolist()
            if not idx:
                continue
            phase_source = source.loc[idx].reset_index(drop=True)
            phase_maia = [maia_probs[i] for i in idx]
            phase_personalized = [personalized_probs[i] for i in idx]
            phase_used = [used_personal[i] for i in idx]
            phase_summary = summarize(
                phase_source,
                phase_maia,
                phase_personalized,
                phase_used,
                include_phases=False,
            )
            summary["by_phase"][phase] = {
                "samples": phase_summary["samples"],
                "personal_book_coverage": phase_summary["personal_book_coverage"],
                "maia2": phase_summary["maia2"],
                "personalized": phase_summary["personalized"],
            }

    return summary


def evaluate(
    train_path: Path,
    eval_path: Path,
    model_type: str,
    device: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    alpha: float,
    min_count: int,
    max_book_fullmove: int | None,
) -> dict[str, Any]:
    train = pd.read_parquet(train_path).reset_index(drop=True)
    source = pd.read_parquet(eval_path).reset_index(drop=True)
    if limit is not None:
        source = source.head(limit).reset_index(drop=True)

    required = ["fen", "move", "elo_self", "elo_oppo"]
    missing = [col for col in required if col not in source.columns or col not in train.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    book = build_position_book(train, max_fullmove=max_book_fullmove)

    maia_model = model.from_pretrained(model_type, device)
    preds, _ = inference.inference_batch(
        source[required].copy(),
        maia_model,
        verbose=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    maia_probs = preds["move_probs"].tolist()
    personalized_probs = []
    used_personal = []
    for row, probs in zip(source.itertuples(index=False), maia_probs):
        blended, used = blend_move_probs(
            probs,
            book.get(position_key(row.fen)),
            alpha=alpha,
            min_count=min_count,
        )
        personalized_probs.append(blended)
        used_personal.append(used)

    summary = summarize(source, maia_probs, personalized_probs, used_personal)
    summary.update(
        {
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "model_type": model_type,
            "device": device,
            "batch_size": batch_size,
            "alpha": alpha,
            "min_count": min_count,
            "max_book_fullmove": max_book_fullmove,
            "book_positions": len(book),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", default="data/processed/splits/chron/train.parquet")
    parser.add_argument("--eval-split", default="data/processed/splits/chron/val.parquet")
    parser.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.35, help="Blend weight for personal move probabilities.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum book observations for a position.")
    parser.add_argument(
        "--max-book-fullmove",
        type=int,
        default=None,
        help="Only build the personal book from positions through this fullmove number.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = evaluate(
        train_path=Path(args.train_split),
        eval_path=Path(args.eval_split),
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        alpha=args.alpha,
        min_count=args.min_count,
        max_book_fullmove=args.max_book_fullmove,
    )

    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
