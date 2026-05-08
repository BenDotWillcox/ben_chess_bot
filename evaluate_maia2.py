#!/usr/bin/env python3
"""Evaluate a pretrained Maia2 model against player move splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from maia2 import inference, model


def top_k_hit(move_probs: dict[str, float], actual_move: str, k: int) -> bool:
    return actual_move in list(move_probs.keys())[:k]


def target_probability(move_probs: dict[str, float], actual_move: str) -> float:
    return float(move_probs.get(actual_move, 0.0))


def summarize_predictions(preds: pd.DataFrame, source: pd.DataFrame) -> dict[str, Any]:
    move_probs = preds["move_probs"].tolist()
    actual_moves = source["move"].tolist()

    summary: dict[str, Any] = {
        "samples": int(len(source)),
        "top1_accuracy": round(sum(top_k_hit(p, m, 1) for p, m in zip(move_probs, actual_moves)) / len(source), 4),
        "top3_accuracy": round(sum(top_k_hit(p, m, 3) for p, m in zip(move_probs, actual_moves)) / len(source), 4),
        "top5_accuracy": round(sum(top_k_hit(p, m, 5) for p, m in zip(move_probs, actual_moves)) / len(source), 4),
        "mean_target_probability": round(
            sum(target_probability(p, m) for p, m in zip(move_probs, actual_moves)) / len(source),
            4,
        ),
    }

    if "you_color" in source.columns:
        summary["by_color"] = {}
        for color, group in source.groupby("you_color", sort=True):
            idx = group.index.tolist()
            probs = [move_probs[i] for i in idx]
            moves = group["move"].tolist()
            summary["by_color"][color] = {
                "samples": int(len(group)),
                "top1_accuracy": round(sum(top_k_hit(p, m, 1) for p, m in zip(probs, moves)) / len(group), 4),
                "top3_accuracy": round(sum(top_k_hit(p, m, 3) for p, m in zip(probs, moves)) / len(group), 4),
                "top5_accuracy": round(sum(top_k_hit(p, m, 5) for p, m in zip(probs, moves)) / len(group), 4),
            }

    if "fullmove_number" in source.columns:
        summary["by_phase"] = {}
        phases = {
            "opening_1_10": source["fullmove_number"] <= 10,
            "middlegame_11_30": (source["fullmove_number"] > 10) & (source["fullmove_number"] <= 30),
            "endgame_31_plus": source["fullmove_number"] > 30,
        }
        for phase, mask in phases.items():
            group = source[mask]
            if len(group) == 0:
                continue
            idx = group.index.tolist()
            probs = [move_probs[i] for i in idx]
            moves = group["move"].tolist()
            summary["by_phase"][phase] = {
                "samples": int(len(group)),
                "top1_accuracy": round(sum(top_k_hit(p, m, 1) for p, m in zip(probs, moves)) / len(group), 4),
                "top3_accuracy": round(sum(top_k_hit(p, m, 3) for p, m in zip(probs, moves)) / len(group), 4),
                "top5_accuracy": round(sum(top_k_hit(p, m, 5) for p, m in zip(probs, moves)) / len(group), 4),
            }

    return summary


def evaluate(
    split_path: Path,
    model_type: str,
    device: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
) -> dict[str, Any]:
    source = pd.read_parquet(split_path).reset_index(drop=True)
    if limit is not None:
        source = source.head(limit).reset_index(drop=True)

    required = ["fen", "move", "elo_self", "elo_oppo"]
    missing = [col for col in required if col not in source.columns]
    if missing:
        raise ValueError(f"{split_path} missing required columns: {missing}")

    eval_input = source[required].copy()
    maia_model = model.from_pretrained(model_type, device)
    preds, maia_top1 = inference.inference_batch(
        eval_input,
        maia_model,
        verbose=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    summary = summarize_predictions(preds, source)
    summary.update(
        {
            "split_path": str(split_path),
            "model_type": model_type,
            "device": device,
            "batch_size": batch_size,
            "maia2_reported_top1_accuracy": maia_top1,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="data/processed/splits/chron/val.parquet",
        help="Path to a split parquet file.",
    )
    parser.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = evaluate(
        split_path=Path(args.split),
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )

    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
