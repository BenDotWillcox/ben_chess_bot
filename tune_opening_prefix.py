#!/usr/bin/env python3
"""Tune and compare exact-position and opening-prefix personalization books."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_personalized_maia2 import (
    blend_move_probs,
    build_position_book,
    position_key,
    summarize,
)
from tune_personalization import DEFAULT_ALPHAS, DEFAULT_MIN_COUNTS, run_maia


def prefix_key(row: Any) -> tuple[str, str]:
    return (row.you_color, row.uci_prefix_before)


def build_prefix_book(train: pd.DataFrame, max_prefix_ply: int) -> dict[tuple[str, str], Counter[str]]:
    train = train[train["ply_prefix_before"] <= max_prefix_ply]
    book: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        book[prefix_key(row)][row.move] += 1
    return dict(book)


def choose_counts(
    strategy: str,
    row: Any,
    fen_book: dict[str, Counter[str]],
    prefix_book: dict[tuple[str, str], Counter[str]],
    min_count: int,
) -> Counter[str] | None:
    fen_counts = fen_book.get(position_key(row.fen))
    prefix_counts = prefix_book.get(prefix_key(row))

    if strategy == "fen":
        return fen_counts
    if strategy == "prefix":
        return prefix_counts
    if strategy == "combined":
        if prefix_counts and sum(prefix_counts.values()) >= min_count:
            return prefix_counts
        return fen_counts
    raise ValueError(f"Unsupported strategy: {strategy}")


def evaluate_strategy(
    source: pd.DataFrame,
    maia_probs: list[dict[str, float]],
    fen_book: dict[str, Counter[str]],
    prefix_book: dict[tuple[str, str], Counter[str]],
    strategy: str,
    alpha: float,
    min_count: int,
) -> dict[str, Any]:
    personalized_probs = []
    used_personal = []
    for row, probs in zip(source.itertuples(index=False), maia_probs):
        counts = choose_counts(strategy, row, fen_book, prefix_book, min_count=min_count)
        blended, used = blend_move_probs(probs, counts, alpha=alpha, min_count=min_count)
        personalized_probs.append(blended)
        used_personal.append(used)

    summary = summarize(source, maia_probs, personalized_probs, used_personal)
    return {
        "strategy": strategy,
        "alpha": alpha,
        "min_count": min_count,
        "coverage": summary["personal_book_coverage"],
        "maia_top1": summary["maia2"]["top1_accuracy"],
        "personalized_top1": summary["personalized"]["top1_accuracy"],
        "top1_lift": round(summary["personalized"]["top1_accuracy"] - summary["maia2"]["top1_accuracy"], 4),
        "maia_top3": summary["maia2"]["top3_accuracy"],
        "personalized_top3": summary["personalized"]["top3_accuracy"],
        "maia_top5": summary["maia2"]["top5_accuracy"],
        "personalized_top5": summary["personalized"]["top5_accuracy"],
        "maia_mean_target_probability": summary["maia2"]["mean_target_probability"],
        "personalized_mean_target_probability": summary["personalized"]["mean_target_probability"],
        "top1_changed_samples": summary["top1_changed_samples"],
        "top1_improved_samples": summary["top1_improved_samples"],
        "top1_worsened_samples": summary["top1_worsened_samples"],
        "summary": summary,
    }


def row_for_csv(result: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "strategy": result["strategy"],
        "alpha": result["alpha"],
        "min_count": result["min_count"],
        "coverage": result["coverage"],
        "maia_top1": result["maia_top1"],
        "personalized_top1": result["personalized_top1"],
        "top1_lift": result["top1_lift"],
        "maia_top3": result["maia_top3"],
        "personalized_top3": result["personalized_top3"],
        "maia_top5": result["maia_top5"],
        "personalized_top5": result["personalized_top5"],
        "maia_mean_target_probability": result["maia_mean_target_probability"],
        "personalized_mean_target_probability": result["personalized_mean_target_probability"],
        "top1_changed_samples": result["top1_changed_samples"],
        "top1_improved_samples": result["top1_improved_samples"],
        "top1_worsened_samples": result["top1_worsened_samples"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    val_rows: list[dict[str, Any]],
    best_val: dict[str, Any],
    test_row: dict[str, Any],
) -> None:
    top_rows = sorted(val_rows, key=lambda row: row["personalized_top1"], reverse=True)[:8]
    lines = [
        "# Opening Prefix Personalization Grid Search",
        "",
        "## Method",
        "",
        "- Base model: Maia2 rapid.",
        "- Train source: chronological train split.",
        "- Tuning source: chronological validation split.",
        "- Final check: chronological test split using the best validation setting.",
        "- FEN strategy: exact legal-position key from FEN fields 1-4.",
        "- Prefix strategy: exact UCI move history before the player move, separated by color.",
        "- Combined strategy: use prefix book when available, otherwise fall back to FEN book.",
        "",
        "## Search Space",
        "",
        f"- Strategies: `{', '.join(args.strategies)}`",
        f"- Alpha values: `{', '.join(str(v) for v in args.alphas)}`",
        f"- Min-count values: `{', '.join(str(v) for v in args.min_counts)}`",
        f"- Max prefix ply: `{args.max_prefix_ply}`",
        "",
        "## Top Validation Settings",
        "",
        "| strategy | alpha | min_count | coverage | Maia2 top1 | personalized top1 | lift | top3 | top5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            "| "
            f"{row['strategy']} | {row['alpha']} | {row['min_count']} | {row['coverage']:.4f} | "
            f"{row['maia_top1']:.4f} | {row['personalized_top1']:.4f} | {row['top1_lift']:.4f} | "
            f"{row['personalized_top3']:.4f} | {row['personalized_top5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Setting",
            "",
            f"- strategy: `{best_val['strategy']}`",
            f"- alpha: `{best_val['alpha']}`",
            f"- min_count: `{best_val['min_count']}`",
            f"- validation top1: `{best_val['personalized_top1']}`",
            f"- validation lift: `{best_val['top1_lift']}`",
            "",
            "## Held-Out Test Result",
            "",
            f"- Maia2 top1: `{test_row['maia_top1']}`",
            f"- personalized top1: `{test_row['personalized_top1']}`",
            f"- top1 lift: `{test_row['top1_lift']}`",
            f"- coverage: `{test_row['coverage']}`",
            f"- top3: `{test_row['personalized_top3']}`",
            f"- top5: `{test_row['personalized_top5']}`",
            f"- improved top1 samples: `{test_row['top1_improved_samples']}`",
            f"- worsened top1 samples: `{test_row['top1_worsened_samples']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", default="data/processed/splits/chron/train.parquet")
    parser.add_argument("--val-split", default="data/processed/splits/chron/val.parquet")
    parser.add_argument("--test-split", default="data/processed/splits/chron/test.parquet")
    parser.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--min-counts", type=int, nargs="+", default=DEFAULT_MIN_COUNTS)
    parser.add_argument("--strategies", nargs="+", default=["fen", "prefix", "combined"])
    parser.add_argument("--max-prefix-ply", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/opening_prefix_grid")
    args = parser.parse_args()

    train = pd.read_parquet(args.train_split).reset_index(drop=True)
    val = pd.read_parquet(args.val_split).reset_index(drop=True)
    test = pd.read_parquet(args.test_split).reset_index(drop=True)

    fen_book = build_position_book(train, max_fullmove=None)
    prefix_book = build_prefix_book(train, max_prefix_ply=args.max_prefix_ply)

    print("Running Maia2 on validation split...")
    val_maia_probs = run_maia(val, args.model_type, args.device, args.batch_size, args.num_workers)

    val_results = []
    for strategy in args.strategies:
        for alpha in args.alphas:
            for min_count in args.min_counts:
                val_results.append(
                    evaluate_strategy(
                        val,
                        val_maia_probs,
                        fen_book,
                        prefix_book,
                        strategy=strategy,
                        alpha=alpha,
                        min_count=min_count,
                    )
                )

    best_val = max(
        val_results,
        key=lambda result: (
            result["personalized_top1"],
            result["personalized_mean_target_probability"],
            -result["top1_worsened_samples"],
        ),
    )

    print("Running Maia2 on test split for selected setting...")
    test_maia_probs = run_maia(test, args.model_type, args.device, args.batch_size, args.num_workers)
    test_result = evaluate_strategy(
        test,
        test_maia_probs,
        fen_book,
        prefix_book,
        strategy=best_val["strategy"],
        alpha=best_val["alpha"],
        min_count=best_val["min_count"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_rows = [row_for_csv(result, "val") for result in val_results]
    test_row = row_for_csv(test_result, "test")
    write_csv(output_dir / "grid_results.csv", val_rows + [test_row])

    summary = {
        "method": {
            "model_type": args.model_type,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "alphas": args.alphas,
            "min_counts": args.min_counts,
            "strategies": args.strategies,
            "max_prefix_ply": args.max_prefix_ply,
            "fen_book_positions": len(fen_book),
            "prefix_book_positions": len(prefix_book),
        },
        "best_validation": {k: v for k, v in best_val.items() if k != "summary"},
        "heldout_test": {k: v for k, v in test_result.items() if k != "summary"},
        "best_validation_detail": best_val["summary"],
        "heldout_test_detail": test_result["summary"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(output_dir / "report.md", args, val_rows, row_for_csv(best_val, "val"), test_row)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
