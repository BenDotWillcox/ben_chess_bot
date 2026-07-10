#!/usr/bin/env python3
"""Play a local terminal game against the personalized Maia2 policy."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess

from cli_policy import DEFAULT_ALPHA, DEFAULT_MIN_COUNT, DEFAULT_STRATEGY, select_candidate
from engine_safety import StockfishBlunderVeto
from game_state import PolicyGameState
from personalized_policy import PersonalizedMaia2Policy, PolicyMove


def parse_human_move(state: PolicyGameState, raw: str) -> chess.Move:
    raw = raw.strip()
    try:
        move = chess.Move.from_uci(raw)
        if move in state.board.legal_moves:
            return move
    except ValueError:
        pass
    return state.board.parse_san(raw)


def format_policy_move(move: PolicyMove) -> str:
    return (
        f"{move.move} "
        f"(p={move.probability:.3f}, maia={move.maia_probability:.3f}, "
        f"personal={move.personal_probability:.3f}, source={move.source}, count={move.count})"
    )


def print_position(state: PolicyGameState) -> None:
    print()
    print(state.board)
    print()
    print(f"FEN: {state.fen}")
    print(f"Prefix: {state.prefix}")
    print()


def push_move(state: PolicyGameState, move: chess.Move) -> None:
    state.board.push(move)
    state.uci_history.append(move.uci())


def maybe_policy_move(
    state: PolicyGameState,
    policy: PersonalizedMaia2Policy,
    bot_color: str,
    elo_self: int,
    elo_oppo: int,
    mode: str,
    top_k: int,
    temperature: float,
    safety: StockfishBlunderVeto | None,
    rng: random.Random | None = None,
) -> None:
    if state.result_or_none() is not None:
        return
    bot_turn = (state.board.turn == chess.WHITE and bot_color == "white") or (
        state.board.turn == chess.BLACK and bot_color == "black"
    )
    if not bot_turn:
        return

    candidates = policy.predict(
        fen=state.fen,
        elo_self=elo_self,
        elo_oppo=elo_oppo,
        you_color=bot_color,
        uci_prefix_before=state.prefix,
        top_n=top_k,
        temperature=temperature,
    )

    selected = select_candidate(candidates, mode=mode, rng=rng)
    if mode == "sample":
        candidates = [selected] + [candidate for candidate in candidates if candidate.move != selected.move]

    if safety is not None:
        decision = safety.choose_safe_move(state.board, candidates)
        selected = decision.selected
        if decision.vetoed:
            print(
                "Safety veto: "
                f"{decision.original.move} -> {decision.selected.move} "
                f"({decision.reason}, best={decision.best_eval_cp}cp, selected={decision.selected_eval_cp}cp)"
            )

    move = chess.Move.from_uci(selected.move)
    san = state.board.san(move)
    state.push_policy_move(selected.move)
    print(f"Bot plays {san} ({selected.move}): {format_policy_move(selected)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Play BenBot locally. The CLI defaults to deterministic argmax inspection; "
            "use --mode sample --temperature 0.8 to mirror live selection."
        )
    )
    parser.add_argument("--bot-color", choices=["white", "black"], default="black")
    parser.add_argument("--elo-self", type=int, default=1650, help="Bot/player Elo used by Maia2.")
    parser.add_argument("--elo-oppo", type=int, default=1650, help="Opponent Elo used by Maia2.")
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
        help="Optional reproducible sampling seed; omit for live human-like variation.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--blunder-veto-cp", type=int, default=400)
    parser.add_argument("--engine-time", type=float, default=0.05)
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
    state = PolicyGameState()
    rng = random.Random(args.seed) if args.seed is not None else None
    human_color = "black" if args.bot_color == "white" else "white"
    safety = None
    if args.stockfish_path:
        stockfish_path = Path(args.stockfish_path)
        if not stockfish_path.exists():
            raise FileNotFoundError(f"Stockfish executable not found: {stockfish_path}")
        safety = StockfishBlunderVeto(
            engine_path=str(stockfish_path),
            veto_cp=args.blunder_veto_cp,
            engine_time=args.engine_time,
        )

    try:
        print("Local personalized Maia2 game")
        print("Enter SAN or UCI moves. Commands: quit, resign, fen, moves, top.")
        print(f"Bot color: {args.bot_color}; human color: {human_color}")
        if safety is not None:
            print(
                f"Engine safety: enabled, veto threshold={args.blunder_veto_cp}cp, "
                f"time={args.engine_time}s"
            )

        maybe_policy_move(
            state,
            policy,
            args.bot_color,
            args.elo_self,
            args.elo_oppo,
            args.mode,
            args.top_k,
            args.temperature,
            safety,
            rng,
        )

        while True:
            print_position(state)
            result = state.result_or_none()
            if result is not None:
                print(f"Game over: {result}")
                break

            raw = input(f"{human_color}> ").strip()
            if raw.lower() in {"quit", "exit"}:
                break
            if raw.lower() == "resign":
                print("Human resigned.")
                break
            if raw.lower() == "fen":
                print(state.fen)
                continue
            if raw.lower() == "moves":
                print(state.prefix)
                continue
            if raw.lower() == "top":
                moves = policy.predict(
                    fen=state.fen,
                    elo_self=args.elo_self,
                    elo_oppo=args.elo_oppo,
                    you_color=args.bot_color,
                    uci_prefix_before=state.prefix,
                    top_n=args.top_k,
                    temperature=args.temperature,
                )
                for i, move in enumerate(moves, start=1):
                    print(f"{i}. {format_policy_move(move)}")
                continue

            try:
                move = parse_human_move(state, raw)
                push_move(state, move)
            except ValueError as exc:
                print(f"Invalid move: {exc}")
                continue

            maybe_policy_move(
                state,
                policy,
                args.bot_color,
                args.elo_self,
                args.elo_oppo,
                args.mode,
                args.top_k,
                args.temperature,
                safety,
                rng,
            )
    finally:
        if safety is not None:
            safety.close()


if __name__ == "__main__":
    main()
