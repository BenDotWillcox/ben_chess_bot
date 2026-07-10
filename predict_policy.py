#!/usr/bin/env python3
"""Predict moves with the personalized Maia2 runtime policy."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict

from cli_policy import DEFAULT_ALPHA, DEFAULT_MIN_COUNT, DEFAULT_STRATEGY, select_candidate
from personalized_policy import PersonalizedMaia2Policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one BenBot distribution. The CLI defaults to deterministic argmax; "
            "use --mode sample --temperature 0.8 --top-n 5 to mirror live selection."
        )
    )
    parser.add_argument("--fen", required=True)
    parser.add_argument("--elo-self", type=int, default=1650)
    parser.add_argument("--elo-oppo", type=int, default=1650)
    parser.add_argument("--you-color", choices=["white", "black"], required=True)
    parser.add_argument("--prefix", default="", help="Space-separated UCI moves before this position.")
    parser.add_argument("--books", default="artifacts/personal_books.json")
    parser.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--strategy",
        choices=["fen", "prefix", "combined"],
        default=DEFAULT_STRATEGY,
        help=f"Personal-memory strategy (validated default: {DEFAULT_STRATEGY}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Personal-prior blend weight (validated default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help=f"Minimum matching observations (validated default: {DEFAULT_MIN_COUNT}).",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=["argmax", "sample"],
        default="argmax",
        help="Selection mode (CLI default: deterministic argmax; live app: sample).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible sampling seed; omit for live variation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    policy = PersonalizedMaia2Policy.load(
        books_path=args.books,
        model_type=args.model_type,
        device=args.device,
        strategy=args.strategy,
        alpha=args.alpha,
        min_count=args.min_count,
    )
    moves = policy.predict(
        fen=args.fen,
        elo_self=args.elo_self,
        elo_oppo=args.elo_oppo,
        you_color=args.you_color,
        uci_prefix_before=args.prefix,
        top_n=args.top_n,
        temperature=args.temperature,
    )
    selected = select_candidate(
        moves,
        mode=args.mode,
        rng=random.Random(args.seed) if args.seed is not None else None,
    )
    print(
        json.dumps(
            {
                "selected": asdict(selected),
                "moves": [asdict(move) for move in moves],
                "config": {
                    "model_type": args.model_type,
                    "strategy": args.strategy,
                    "alpha": args.alpha,
                    "min_count": args.min_count,
                    "mode": args.mode,
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
