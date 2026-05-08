from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict
from contextlib import suppress
from pathlib import Path
from typing import Literal

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine_safety import StockfishBlunderVeto
from game_state import PolicyGameState
from personalized_policy import PersonalizedMaia2Policy, PolicyMove


DEFAULT_ELO = 1650
DEFAULT_BOOKS_PATH = "artifacts/personal_books.json"
DEFAULT_STOCKFISH_PATH = "artifacts/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
DEFAULT_PLAY_MODE = "sample"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MOVE_DELAY_SECONDS = 0.5
DEFAULT_STOCKFISH_ENABLED = "1"


class PolicyConfig(BaseModel):
    model_type: Literal["rapid", "blitz"] = "rapid"
    device: Literal["cpu", "gpu"] = "cpu"
    strategy: Literal["fen", "prefix", "combined"] = "combined"
    alpha: float = 0.5
    min_count: int = 1
    books_path: str = DEFAULT_BOOKS_PATH


class EngineSafetyConfig(BaseModel):
    enabled: bool = False
    stockfish_path: str | None = None
    blunder_veto_cp: int = 400
    engine_time: float = 0.05


class NewGameRequest(BaseModel):
    human_color: Literal["white", "black"] = "white"
    elo_self: int = DEFAULT_ELO
    elo_oppo: int = DEFAULT_ELO
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class MoveRequest(BaseModel):
    move: str = Field(description="Human move in SAN or UCI notation.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class BotMoveRequest(BaseModel):
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class PredictRequest(BaseModel):
    fen: str
    you_color: Literal["white", "black"]
    uci_prefix_before: str = ""
    elo_self: int = DEFAULT_ELO
    elo_oppo: int = DEFAULT_ELO
    top_n: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=1.0, gt=0)


class GameSession:
    def __init__(self, human_color: str, elo_self: int, elo_oppo: int) -> None:
        self.state = PolicyGameState()
        self.human_color = human_color
        self.bot_color = "black" if human_color == "white" else "white"
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo
        self.last_move: dict[str, str] | None = None


app = FastAPI(title="Personalized Maia2 Chess Bot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CHESS_BOT_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, GameSession] = {}
policy: PersonalizedMaia2Policy | None = None
policy_lock = threading.Lock()
safety: StockfishBlunderVeto | None = None
safety_lock = threading.Lock()


def get_policy() -> PersonalizedMaia2Policy:
    global policy
    with policy_lock:
        if policy is None:
            cfg = PolicyConfig(
                model_type=os.getenv("CHESS_BOT_MODEL_TYPE", "rapid"),
                device=os.getenv("CHESS_BOT_DEVICE", "cpu"),
                strategy=os.getenv("CHESS_BOT_STRATEGY", "combined"),
                alpha=float(os.getenv("CHESS_BOT_ALPHA", "0.5")),
                min_count=int(os.getenv("CHESS_BOT_MIN_COUNT", "1")),
                books_path=os.getenv("CHESS_BOT_BOOKS_PATH", DEFAULT_BOOKS_PATH),
            )
            if not Path(cfg.books_path).exists():
                raise HTTPException(status_code=500, detail=f"Personal books not found: {cfg.books_path}")
            policy = PersonalizedMaia2Policy.load(
                books_path=cfg.books_path,
                model_type=cfg.model_type,
                device=cfg.device,
                strategy=cfg.strategy,
                alpha=cfg.alpha,
                min_count=cfg.min_count,
            )
        return policy


def get_safety() -> StockfishBlunderVeto | None:
    global safety
    enabled = os.getenv("CHESS_BOT_STOCKFISH_ENABLED", DEFAULT_STOCKFISH_ENABLED) == "1"
    if not enabled:
        return None
    with safety_lock:
        if safety is None:
            path = os.getenv("CHESS_BOT_STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)
            if not Path(path).exists():
                raise HTTPException(status_code=500, detail=f"Stockfish executable not found: {path}")
            safety = StockfishBlunderVeto(
                engine_path=path,
                veto_cp=int(os.getenv("CHESS_BOT_BLUNDER_VETO_CP", "400")),
                engine_time=float(os.getenv("CHESS_BOT_ENGINE_TIME", "0.05")),
            )
        return safety


def parse_move(board: chess.Board, raw: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(raw)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        return board.parse_san(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or illegal move: {raw}") from exc


def push_move(state: PolicyGameState, move: chess.Move) -> None:
    state.board.push(move)
    state.uci_history.append(move.uci())


def pop_move(state: PolicyGameState) -> chess.Move:
    move = state.board.pop()
    state.uci_history.pop()
    return move


def move_marker(move: chess.Move, side: str) -> dict[str, str]:
    return {
        "move": move.uci(),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "side": side,
    }


def serialize_move(move: PolicyMove, san: str | None = None) -> dict:
    payload = asdict(move)
    payload["san"] = san
    return payload


def serialize_candidates(board: chess.Board, moves: list[PolicyMove]) -> list[dict]:
    payloads = []
    for policy_move in moves:
        move = chess.Move.from_uci(policy_move.move)
        payloads.append(serialize_move(policy_move, san=board.san(move) if move in board.legal_moves else None))
    return payloads


def legal_moves(board: chess.Board) -> list[dict[str, str]]:
    moves = []
    for move in board.legal_moves:
        moves.append({"uci": move.uci(), "san": board.san(move)})
    return moves


def board_payload(session: GameSession, game_id: str, bot_move: dict | None = None) -> dict:
    state = session.state
    return {
        "game_id": game_id,
        "fen": state.fen,
        "prefix": state.prefix,
        "turn": "white" if state.board.turn == chess.WHITE else "black",
        "human_color": session.human_color,
        "bot_color": session.bot_color,
        "bot_move": bot_move,
        "last_move": session.last_move,
        "policy_config": {
            "mode": DEFAULT_PLAY_MODE,
            "top_k": DEFAULT_TOP_K,
            "temperature": DEFAULT_TEMPERATURE,
            "strategy": os.getenv("CHESS_BOT_STRATEGY", "combined"),
            "alpha": float(os.getenv("CHESS_BOT_ALPHA", "0.5")),
            "min_count": int(os.getenv("CHESS_BOT_MIN_COUNT", "1")),
            "stockfish_veto_cp": int(os.getenv("CHESS_BOT_BLUNDER_VETO_CP", "400")),
            "stockfish_enabled": os.getenv("CHESS_BOT_STOCKFISH_ENABLED", DEFAULT_STOCKFISH_ENABLED) == "1",
        },
        "game_over": state.board.is_game_over(claim_draw=True),
        "checkmate": state.board.is_checkmate(),
        "stalemate": state.board.is_stalemate(),
        "result": state.board.result(claim_draw=True) if state.board.is_game_over(claim_draw=True) else None,
        "legal_moves": legal_moves(state.board),
    }


def maybe_bot_move(session: GameSession, top_k: int, mode: str, temperature: float) -> dict | None:
    state = session.state
    bot_turn = (state.board.turn == chess.WHITE and session.bot_color == "white") or (
        state.board.turn == chess.BLACK and session.bot_color == "black"
    )
    if not bot_turn or state.board.is_game_over(claim_draw=True):
        return None

    delay = float(os.getenv("CHESS_BOT_MOVE_DELAY_SECONDS", str(DEFAULT_MOVE_DELAY_SECONDS)))
    if delay > 0:
        time.sleep(delay)

    active_policy = get_policy()
    candidates = active_policy.predict(
        fen=state.fen,
        elo_self=session.elo_self,
        elo_oppo=session.elo_oppo,
        you_color=session.bot_color,
        uci_prefix_before=state.prefix,
        top_n=top_k,
        temperature=temperature,
    )
    if not candidates:
        raise HTTPException(status_code=500, detail="Policy returned no legal moves")

    selected = candidates[0]
    if mode == "sample":
        selected = active_policy.choose_move(
            fen=state.fen,
            elo_self=session.elo_self,
            elo_oppo=session.elo_oppo,
            you_color=session.bot_color,
            uci_prefix_before=state.prefix,
            mode="sample",
            top_k=top_k,
            temperature=temperature,
        )

    safety_decision = None
    active_safety = get_safety()
    if active_safety is not None:
        safety_candidates = [selected] + [candidate for candidate in candidates if candidate.move != selected.move]
        try:
            with safety_lock:
                safety_decision = active_safety.choose_safe_move(state.board, safety_candidates)
            selected = safety_decision.selected
        except Exception:
            safety_decision = None

    move = chess.Move.from_uci(selected.move)
    if move not in state.board.legal_moves:
        selected = next((candidate for candidate in candidates if chess.Move.from_uci(candidate.move) in state.board.legal_moves), None)
        if selected is None:
            raise HTTPException(status_code=500, detail="Policy returned no legal moves")
        move = chess.Move.from_uci(selected.move)
    san = state.board.san(move)
    serialized_candidates = serialize_candidates(state.board, candidates)
    state.push_policy_move(selected.move)
    session.last_move = move_marker(move, session.bot_color)

    return {
        "selected": serialize_move(selected, san=san),
        "candidates": serialized_candidates,
        "safety": asdict(safety_decision) if safety_decision else None,
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "policy_loaded": policy is not None,
        "stockfish_enabled": os.getenv("CHESS_BOT_STOCKFISH_ENABLED", DEFAULT_STOCKFISH_ENABLED) == "1",
    }


@app.on_event("shutdown")
def shutdown() -> None:
    global safety
    if safety is not None:
        safety.close()
        safety = None


@app.post("/new-game")
def new_game(request: NewGameRequest) -> dict:
    game_id = str(uuid.uuid4())
    session = GameSession(
        human_color=request.human_color,
        elo_self=request.elo_self,
        elo_oppo=request.elo_oppo,
    )
    sessions[game_id] = session
    bot_move = maybe_bot_move(session, request.top_k, request.mode, request.temperature)
    return board_payload(session, game_id, bot_move=bot_move)


@app.get("/game/{game_id}")
def get_game(game_id: str) -> dict:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return board_payload(session, game_id)


@app.post("/game/{game_id}/bot-move")
def continue_bot_move(game_id: str, request: BotMoveRequest | None = None) -> dict:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    top_k = request.top_k if request else DEFAULT_TOP_K
    mode = request.mode if request else DEFAULT_PLAY_MODE
    temperature = request.temperature if request else DEFAULT_TEMPERATURE
    bot_move = maybe_bot_move(session, top_k, mode, temperature)
    return board_payload(session, game_id, bot_move=bot_move)


@app.post("/game/{game_id}/move")
def play_move(game_id: str, request: MoveRequest) -> dict:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    state = session.state

    human_turn = (state.board.turn == chess.WHITE and session.human_color == "white") or (
        state.board.turn == chess.BLACK and session.human_color == "black"
    )
    if not human_turn:
        raise HTTPException(status_code=400, detail="It is not the human player's turn")

    move = parse_move(state.board, request.move)
    previous_last_move = session.last_move
    push_move(state, move)
    session.last_move = move_marker(move, session.human_color)
    try:
        bot_move = maybe_bot_move(session, request.top_k, request.mode, request.temperature)
    except Exception:
        with suppress(Exception):
            pop_move(state)
        session.last_move = previous_last_move
        raise
    return board_payload(session, game_id, bot_move=bot_move)


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    active_policy = get_policy()
    moves = active_policy.predict(
        fen=request.fen,
        elo_self=request.elo_self,
        elo_oppo=request.elo_oppo,
        you_color=request.you_color,
        uci_prefix_before=request.uci_prefix_before,
        top_n=request.top_n,
        temperature=request.temperature,
    )
    return {"moves": [serialize_move(move) for move in moves]}


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
