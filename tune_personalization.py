#!/usr/bin/env python3
"""Tune the simple Maia2 personalization blend and write report artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from maia2 import inference, model

from evaluate_personalized_maia2 import (
    blend_move_probs,
    build_position_book,
    position_key,
    summarize,
)


DEFAULT_ALPHAS = [0.1, 0.2, 0.35, 0.5, 0.7]
DEFAULT_MIN_COUNTS = [1, 2, 3]


def run_maia(
    source: pd.DataFrame,
    model_type: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, float]]:
    required = ["fen", "move", "elo_self", "elo_oppo"]
    maia_model = model.from_pretrained(model_type, device)
    preds, _ = inference.inference_batch(
        source[required].copy(),
        maia_model,
        verbose=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return preds["move_probs"].tolist()


def evaluate_combo(
    source: pd.DataFrame,
    maia_probs: list[dict[str, float]],
    book: dict,
    alpha: float,
    min_count: int,
) -> dict[str, Any]:
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
    return {
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
    test_result: dict[str, Any],
) -> None:
    top_rows = sorted(val_rows, key=lambda row: row["personalized_top1"], reverse=True)[:5]
    lines = [
        "# Maia2 Personalization Grid Search",
        "",
        "## Method",
        "",
        "- Base model: Maia2 rapid.",
        "- Train source: chronological train split.",
        "- Tuning source: chronological validation split.",
        "- Final check: chronological test split using the best validation setting.",
        "- Personalization: exact-position move-frequency book blended with Maia2 legal-move probabilities.",
        "",
        "## Search Space",
        "",
        f"- Alpha values: `{', '.join(str(v) for v in args.alphas)}`",
        f"- Min-count values: `{', '.join(str(v) for v in args.min_counts)}`",
        "",
        "## Top Validation Settings",
        "",
        "| alpha | min_count | coverage | Maia2 top1 | personalized top1 | lift | top3 | top5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            "| "
            f"{row['alpha']} | {row['min_count']} | {row['coverage']:.4f} | "
            f"{row['maia_top1']:.4f} | {row['personalized_top1']:.4f} | {row['top1_lift']:.4f} | "
            f"{row['personalized_top3']:.4f} | {row['personalized_top5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Setting",
            "",
            f"- alpha: `{best_val['alpha']}`",
            f"- min_count: `{best_val['min_count']}`",
            f"- validation top1: `{best_val['personalized_top1']}`",
            f"- validation lift: `{best_val['top1_lift']}`",
            "",
            "## Held-Out Test Result",
            "",
            f"- Maia2 top1: `{test_result['maia_top1']}`",
            f"- personalized top1: `{test_result['personalized_top1']}`",
            f"- top1 lift: `{test_result['top1_lift']}`",
            f"- coverage: `{test_result['coverage']}`",
            f"- top3: `{test_result['personalized_top3']}`",
            f"- top5: `{test_result['personalized_top5']}`",
            f"- improved top1 samples: `{test_result['top1_improved_samples']}`",
            f"- worsened top1 samples: `{test_result['top1_worsened_samples']}`",
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
    parser.add_argument("--max-book-fullmove", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/personalization_grid")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train = pd.read_parquet(args.train_split).reset_index(drop=True)
    val = pd.read_parquet(args.val_split).reset_index(drop=True)
    test = pd.read_parquet(args.test_split).reset_index(drop=True)

    book = build_position_book(train, max_fullmove=args.max_book_fullmove)

    print("Running Maia2 on validation split...")
    val_maia_probs = run_maia(val, args.model_type, args.device, args.batch_size, args.num_workers)

    val_results = []
    for alpha in args.alphas:
        for min_count in args.min_counts:
            val_results.append(evaluate_combo(val, val_maia_probs, book, alpha, min_count))

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
    test_result = evaluate_combo(test, test_maia_probs, book, best_val["alpha"], best_val["min_count"])

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
            "max_book_fullmove": args.max_book_fullmove,
            "book_positions": len(book),
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
