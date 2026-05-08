#!/usr/bin/env python3
"""Predict moves with the personalized Maia2 runtime policy."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from personalized_policy import PersonalizedMaia2Policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", required=True)
    parser.add_argument("--elo-self", type=int, default=1650)
    parser.add_argument("--elo-oppo", type=int, default=1650)
    parser.add_argument("--you-color", choices=["white", "black"], required=True)
    parser.add_argument("--prefix", default="", help="Space-separated UCI moves before this position.")
    parser.add_argument("--books", default="artifacts/personal_books.json")
    parser.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--strategy", choices=["fen", "prefix", "combined"], default="combined")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    args = parser.parse_args()

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
    selected = policy.choose_move(
        fen=args.fen,
        elo_self=args.elo_self,
        elo_oppo=args.elo_oppo,
        you_color=args.you_color,
        uci_prefix_before=args.prefix,
        mode=args.mode,
        top_k=args.top_n,
        temperature=args.temperature,
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
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
