#!/usr/bin/env python3
"""Download Chess.com games and build training-ready datasets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chess.pgn
import pandas as pd
import requests

BASE = "https://api.chess.com/pub/player"
USER_AGENT = "ben_chess_bot/0.1 (+https://github.com/benja/ben_chess_bot; contact: benja@example.com)"


@dataclass(frozen=True)
class Paths:
    root: Path
    username: str

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "chesscom" / self.username.lower()

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "data" / "manifests"


def ensure_dirs(paths: Paths) -> None:
    for d in [paths.raw_dir, paths.interim_dir, paths.processed_dir / "splits", paths.manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)


def month_range(start: str, end: str) -> Iterable[str]:
    cur = datetime.strptime(start, "%Y-%m")
    stop = datetime.strptime(end, "%Y-%m")
    while cur <= stop:
        yield cur.strftime("%Y/%m")
        year = cur.year + (cur.month // 12)
        month = (cur.month % 12) + 1
        cur = cur.replace(year=year, month=month)


def chesscom_get(url: str, retries: int = 4, backoff_sec: float = 1.5) -> dict:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    for attempt in range(retries + 1):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code < 400:
            return resp.json()

        retry_after = resp.headers.get("Retry-After")
        if resp.status_code in {403, 429, 500, 502, 503, 504} and attempt < retries:
            wait = float(retry_after) if retry_after else backoff_sec * (2 ** attempt)
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError(f"Exceeded retries for url: {url}")


def fetch_month(username: str, month: str) -> dict:
    url = f"{BASE}/{username}/games/{month}"
    return chesscom_get(url)


def save_json(path: Path, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    checksum = hashlib.sha256(blob).hexdigest()
    path.write_bytes(blob)
    return checksum


def fetch_archives(paths: Paths, start: str, end: str, pause_sec: float = 0.75) -> None:
    ensure_dirs(paths)
    manifest: list[dict] = []
    for month in month_range(start, end):
        out = paths.raw_dir / f"{month.replace('/', '-')}.json"
        payload = fetch_month(paths.username, month)
        checksum = save_json(out, payload)
        manifest.append(
            {
                "month": month,
                "file": str(out.relative_to(paths.root)),
                "fetched_at_utc": datetime.utcnow().isoformat(),
                "checksum_sha256": checksum,
                "game_count": len(payload.get("games", [])),
            }
        )
        print(f"Fetched {month}: {len(payload.get('games', []))} games")
        time.sleep(pause_sec)
    (paths.manifests_dir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2))


def game_id(game: dict) -> str:
    key = f"{game.get('url', '')}|{game.get('end_time', '')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def parse_result(game: dict, username: str) -> str:
    ul = username.lower()
    white = game.get("white", {}).get("username", "").lower() == ul
    your, opp = (game["white"], game["black"]) if white else (game["black"], game["white"])
    y, o = your.get("result", ""), opp.get("result", "")
    if y == "win":
        return "win"
    if o == "win":
        return "loss"
    return "draw"


def normalize_games(paths: Paths, rated_only: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for fp in sorted(paths.raw_dir.glob("*.json")):
        data = json.loads(fp.read_text())
        for g in data.get("games", []):
            if g.get("time_class") != "rapid":
                continue
            if rated_only and not bool(g.get("rated")):
                continue
            white = g.get("white", {})
            black = g.get("black", {})
            ul = paths.username.lower()
            if white.get("username", "").lower() == ul:
                you_color, your, opp = "white", white, black
            elif black.get("username", "").lower() == ul:
                you_color, your, opp = "black", black, white
            else:
                continue
            rows.append(
                {
                    "game_id": game_id(g),
                    "date_utc": pd.to_datetime(g.get("end_time", 0), unit="s", utc=True),
                    "you_color": you_color,
                    "your_username": paths.username,
                    "opponent_username": opp.get("username"),
                    "your_rating": your.get("rating"),
                    "opponent_rating": opp.get("rating"),
                    "result": parse_result(g, paths.username),
                    "time_control_raw": g.get("time_control"),
                    "time_class": g.get("time_class"),
                    "rated": bool(g.get("rated")),
                    "pgn": g.get("pgn"),
                    "source_url": g.get("url"),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"]).sort_values("date_utc")
    df.to_parquet(paths.interim_dir / "games.parquet", index=False)
    return df


def build_moves(paths: Paths, games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, g in games.iterrows():
        pgn = g.get("pgn")
        if not isinstance(pgn, str):
            continue
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            continue
        board = game.board()
        ply = 0
        for move in game.mainline_moves():
            ply += 1
            side = "white" if board.turn else "black"
            rows.append(
                {
                    "game_id": g["game_id"],
                    "date_utc": g["date_utc"],
                    "ply_index": ply,
                    "fullmove_number": board.fullmove_number,
                    "side_to_move": side,
                    "fen_before": board.fen(),
                    "uci_move": move.uci(),
                    "san_move": board.san(move),
                    "is_your_move": side == g["you_color"],
                }
            )
            board.push(move)
    df = pd.DataFrame(rows)
    df.to_parquet(paths.interim_dir / "moves.parquet", index=False)

    train = df[df["is_your_move"]].rename(columns={"fen_before": "fen", "uci_move": "target_move_uci"})
    train = train[["game_id", "date_utc", "ply_index", "fen", "target_move_uci"]].sort_values("date_utc")
    train.to_parquet(paths.processed_dir / "train_samples.parquet", index=False)
    return df


def write_splits(base_dir: Path, split_name: str, train_samples: pd.DataFrame, seed: int) -> dict:
    out_dir = base_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_name == "chron":
        ordered = train_samples.sort_values("date_utc").reset_index(drop=True)
    elif split_name == "random":
        ordered = train_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported split type: {split_name}")

    n = len(ordered)
    t_end = int(n * 0.7)
    v_end = int(n * 0.85)
    splits = {
        "train": ordered.iloc[:t_end],
        "val": ordered.iloc[t_end:v_end],
        "test": ordered.iloc[v_end:],
    }
    for name, frame in splits.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    return {k: int(len(v)) for k, v in splits.items()}


def create_splits(paths: Paths, train_samples: pd.DataFrame, split_mode: str, seed: int) -> dict:
    split_dir = paths.processed_dir / "splits"
    results: dict[str, dict[str, int]] = {}
    if split_mode in {"chron", "both"}:
        results["chron"] = write_splits(split_dir, "chron", train_samples, seed)
    if split_mode in {"random", "both"}:
        results["random"] = write_splits(split_dir, "random", train_samples, seed)
    return results


def build_stats(paths: Paths, games: pd.DataFrame, moves: pd.DataFrame, split_sizes: dict, split_mode: str, seed: int) -> None:
    stats = {
        "games": int(len(games)),
        "moves": int(len(moves)),
        "your_moves": int(moves["is_your_move"].sum()) if len(moves) else 0,
        "rated_only": True,
        "date_min": str(games["date_utc"].min()) if len(games) else None,
        "date_max": str(games["date_utc"].max()) if len(games) else None,
        "split_mode": split_mode,
        "split_seed": seed,
        "split_sizes": split_sizes,
    }
    (paths.manifests_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))


def report(paths: Paths) -> None:
    games_path = paths.interim_dir / "games.parquet"
    if not games_path.exists():
        raise FileNotFoundError("Run build first; games.parquet missing")
    games = pd.read_parquet(games_path)
    print("=== Dataset Report ===")
    print(f"Games: {len(games)}")
    print(f"Date range: {games['date_utc'].min()} -> {games['date_utc'].max()}")
    print("Results:")
    print(games["result"].value_counts(dropna=False).to_string())
    print("Top time controls:")
    print(games["time_control_raw"].value_counts().head(10).to_string())
    print("Rating summary:")
    print(games[["your_rating", "opponent_rating"]].describe().to_string())


def run_build(paths: Paths, split_mode: str, seed: int) -> None:
    ensure_dirs(paths)
    games = normalize_games(paths, rated_only=True)
    moves = build_moves(paths, games)
    train = pd.read_parquet(paths.processed_dir / "train_samples.parquet")
    split_sizes = create_splits(paths, train, split_mode=split_mode, seed=seed)
    build_stats(paths, games, moves, split_sizes, split_mode=split_mode, seed=seed)
    print(f"Built datasets: games={len(games)} moves={len(moves)} split_mode={split_mode} splits={split_sizes}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch")
    pf.add_argument("--username", required=True)
    pf.add_argument("--start", default="2025-11", help="YYYY-MM")
    pf.add_argument("--end", default="2026-04", help="YYYY-MM")

    pb = sub.add_parser("build")
    pb.add_argument("--username", required=True)
    pb.add_argument("--split-mode", choices=["chron", "random", "both"], default="both")
    pb.add_argument("--seed", type=int, default=42)

    pa = sub.add_parser("all")
    pa.add_argument("--username", required=True)
    pa.add_argument("--start", default="2025-11")
    pa.add_argument("--end", default="2026-04")
    pa.add_argument("--split-mode", choices=["chron", "random", "both"], default="both")
    pa.add_argument("--seed", type=int, default=42)

    pr = sub.add_parser("report")
    pr.add_argument("--username", required=True)

    args = p.parse_args()
    paths = Paths(Path.cwd(), args.username)

    if args.cmd == "fetch":
        fetch_archives(paths, args.start, args.end)
    elif args.cmd == "build":
        run_build(paths, split_mode=args.split_mode, seed=args.seed)
    elif args.cmd == "all":
        fetch_archives(paths, args.start, args.end)
        run_build(paths, split_mode=args.split_mode, seed=args.seed)
    elif args.cmd == "report":
        report(paths)


if __name__ == "__main__":
    main()
