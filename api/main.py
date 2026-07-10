from __future__ import annotations

import logging
import math
import os
import random
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Literal

import chess
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.runtime import (
    BoundedExecutor,
    SessionStore,
    SlidingWindowRateLimiter,
    WorkQueueFull,
    WorkTimedOut,
    configure_json_logger,
    log_event,
    process_memory_snapshot,
)
from engine_safety import StockfishBlunderVeto
from game_state import PolicyGameState
from personalized_policy import PersonalizedMaia2Policy, PolicyMove


DEFAULT_ELO = 1650
DEFAULT_BOOKS_PATH = "artifacts/personal_books.json"
DEFAULT_STOCKFISH_PATH = "artifacts/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
DEFAULT_PLAY_MODE = "sample"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_STRATEGY = "fen"
DEFAULT_ALPHA = 0.7
DEFAULT_MIN_COUNT = 1
DEFAULT_MOVE_DELAY_SECONDS = 0.5
DEFAULT_STOCKFISH_ENABLED = "1"
STARTED_AT = time.time()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


SESSION_TTL_SECONDS = _env_float("CHESS_BOT_SESSION_TTL_SECONDS", 1800.0, minimum=1.0)
SESSION_CAPACITY = _env_int("CHESS_BOT_SESSION_CAPACITY", 128, minimum=1)
RATE_LIMIT_REQUESTS = _env_int("CHESS_BOT_RATE_LIMIT_REQUESTS", 60, minimum=0)
RATE_LIMIT_WINDOW_SECONDS = _env_float("CHESS_BOT_RATE_LIMIT_WINDOW_SECONDS", 60.0, minimum=0.1)
RATE_LIMIT_MAX_CLIENTS = _env_int("CHESS_BOT_RATE_LIMIT_MAX_CLIENTS", 10_000, minimum=1)
INFERENCE_CONCURRENCY = _env_int("CHESS_BOT_INFERENCE_CONCURRENCY", 1, minimum=1)
INFERENCE_QUEUE_CAPACITY = _env_int("CHESS_BOT_INFERENCE_QUEUE_CAPACITY", 4, minimum=0)
INFERENCE_ADMISSION_TIMEOUT_SECONDS = _env_float(
    "CHESS_BOT_INFERENCE_ADMISSION_TIMEOUT_SECONDS", 0.25, minimum=0.0
)
INFERENCE_TIMEOUT_SECONDS = _env_float("CHESS_BOT_INFERENCE_TIMEOUT_SECONDS", 30.0, minimum=0.1)
MODEL_LOAD_TIMEOUT_SECONDS = _env_float("CHESS_BOT_MODEL_LOAD_TIMEOUT_SECONDS", 180.0, minimum=1.0)
SAFETY_TIMEOUT_SECONDS = _env_float("CHESS_BOT_SAFETY_TIMEOUT_SECONDS", 3.0, minimum=0.1)
SAFETY_LOAD_TIMEOUT_SECONDS = _env_float("CHESS_BOT_SAFETY_LOAD_TIMEOUT_SECONDS", 15.0, minimum=0.1)
DEPENDENCY_RETRY_SECONDS = _env_float("CHESS_BOT_DEPENDENCY_RETRY_SECONDS", 30.0, minimum=0.0)
STOCKFISH_ENABLED = _env_bool("CHESS_BOT_STOCKFISH_ENABLED", DEFAULT_STOCKFISH_ENABLED == "1")
REQUIRE_STOCKFISH = _env_bool("CHESS_BOT_REQUIRE_STOCKFISH", False)
TRUST_PROXY_HEADERS = _env_bool("CHESS_BOT_TRUST_PROXY_HEADERS", False)


logger = configure_json_logger()


class PolicyConfig(BaseModel):
    model_type: Literal["rapid", "blitz"] = "rapid"
    device: Literal["cpu", "gpu"] = "cpu"
    strategy: Literal["fen", "prefix", "combined"] = DEFAULT_STRATEGY
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    min_count: int = Field(default=DEFAULT_MIN_COUNT, ge=1)
    books_path: str = DEFAULT_BOOKS_PATH


class StockfishRuntimeConfig(BaseModel):
    veto_cp: int = Field(default=400, ge=0)
    engine_time: float = Field(default=0.05, gt=0)
    engine_timeout: float = Field(default=10.0, gt=0)


class NewGameRequest(BaseModel):
    human_color: Literal["white", "black"] = "white"
    elo_self: int = Field(default=DEFAULT_ELO, ge=100, le=4000)
    elo_oppo: int = Field(default=DEFAULT_ELO, ge=100, le=4000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class MoveRequest(BaseModel):
    move: str = Field(min_length=2, max_length=16, description="Human move in SAN or UCI notation.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class BotMoveRequest(BaseModel):
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    mode: Literal["argmax", "sample"] = DEFAULT_PLAY_MODE
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0)


class PredictRequest(BaseModel):
    fen: str = Field(min_length=15, max_length=128)
    you_color: Literal["white", "black"]
    uci_prefix_before: str = Field(default="", max_length=4096)
    elo_self: int = Field(default=DEFAULT_ELO, ge=100, le=4000)
    elo_oppo: int = Field(default=DEFAULT_ELO, ge=100, le=4000)
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
        self.lock = threading.Lock()


class DependencyUnavailable(RuntimeError):
    """A required runtime dependency could not be initialized."""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Function names are resolved when the server starts, after this module has
    # finished defining the lifecycle helpers below.
    startup()
    try:
        yield
    finally:
        shutdown()


app = FastAPI(title="Personalized Maia2 Chess Bot API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CHESS_BOT_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = SessionStore[GameSession](max_sessions=SESSION_CAPACITY, ttl_seconds=SESSION_TTL_SECONDS)
rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    max_keys=RATE_LIMIT_MAX_CLIENTS,
)
def _new_inference_executor() -> BoundedExecutor:
    return BoundedExecutor(
        max_workers=INFERENCE_CONCURRENCY,
        max_queue=INFERENCE_QUEUE_CAPACITY,
        name="benbot-inference",
    )


def _new_safety_executor() -> BoundedExecutor:
    return BoundedExecutor(max_workers=1, max_queue=1, name="benbot-stockfish")


inference_executor = _new_inference_executor()
safety_executor = _new_safety_executor()

policy: PersonalizedMaia2Policy | None = None
policy_lock = threading.Lock()
policy_retry_after = 0.0
policy_runtime_lock = threading.Lock()
policy_runtime_blocked = False
safety: StockfishBlunderVeto | None = None
safety_lock = threading.RLock()
safety_retry_after = 0.0
safety_needs_reload = False
safety_generation = 0
dependency_lock = threading.Lock()
readiness_lock = threading.Lock()
runtime_lifecycle_lock = threading.Lock()
runtime_shutting_down = False
dependency_init_lock = threading.Lock()
dependency_init_thread: threading.Thread | None = None
dependency_state: dict[str, dict[str, object]] = {
    "policy": {"status": "not_loaded", "error": None, "last_attempt": None, "last_ready": None},
    "stockfish": {
        "status": "not_loaded" if STOCKFISH_ENABLED else "disabled",
        "error": None,
        "last_attempt": None,
        "last_ready": None,
    },
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _public_error(component: str, exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"{component} executable or artifact was not found at the configured path"
    if isinstance(exc, WorkQueueFull):
        return f"{component} work queue is full"
    if isinstance(exc, WorkTimedOut):
        return f"{component} initialization or operation timed out"
    return f"{component} is unavailable"


def _set_dependency(name: str, status: str, *, error: str | None = None) -> None:
    now = _utc_now()
    with dependency_lock:
        record = dependency_state[name]
        record["status"] = status
        record["error"] = error
        if status == "loading" or status in {"ready", "degraded", "unavailable"}:
            record["last_attempt"] = now
        if status == "ready":
            record["last_ready"] = now


def _dependency_snapshot(name: str) -> dict[str, object]:
    with dependency_lock:
        return dict(dependency_state[name])


def _stockfish_snapshot() -> dict[str, object]:
    with safety_lock:
        snapshot = _dependency_snapshot("stockfish")
        snapshot["enabled"] = STOCKFISH_ENABLED
        snapshot["required"] = REQUIRE_STOCKFISH
        snapshot["available"] = (
            snapshot["status"] == "ready"
            and safety is not None
            and not safety_needs_reload
        )
        return snapshot


def _policy_config() -> PolicyConfig:
    return PolicyConfig(
        model_type=os.getenv("CHESS_BOT_MODEL_TYPE", "rapid"),
        device=os.getenv("CHESS_BOT_DEVICE", "cpu"),
        strategy=os.getenv("CHESS_BOT_STRATEGY", DEFAULT_STRATEGY),
        alpha=os.getenv("CHESS_BOT_ALPHA", str(DEFAULT_ALPHA)),
        min_count=os.getenv("CHESS_BOT_MIN_COUNT", str(DEFAULT_MIN_COUNT)),
        books_path=os.getenv("CHESS_BOT_BOOKS_PATH", DEFAULT_BOOKS_PATH),
    )


def _stockfish_runtime_config() -> StockfishRuntimeConfig:
    return StockfishRuntimeConfig(
        veto_cp=os.getenv("CHESS_BOT_BLUNDER_VETO_CP", "400"),
        engine_time=os.getenv("CHESS_BOT_ENGINE_TIME", "0.05"),
        engine_timeout=os.getenv("CHESS_BOT_ENGINE_TIMEOUT_SECONDS", "10"),
    )


def _mark_policy_runtime_unavailable(exc: BaseException) -> None:
    global policy_runtime_blocked
    public_error = _public_error("Maia2 inference", exc)
    with policy_runtime_lock:
        policy_runtime_blocked = True
    _set_dependency("policy", "unavailable", error=public_error)
    log_event(logger, logging.ERROR, "policy_runtime_unavailable", error=repr(exc))


def _policy_runtime_probe_allowed() -> bool:
    """Allow a real recovery inference only after abandoned work has drained."""
    with policy_runtime_lock:
        blocked = policy_runtime_blocked
    if not blocked:
        return True
    stats = inference_executor.stats()
    return stats["active"] == 0 and not inference_executor.closed


def _mark_policy_runtime_ready_after_probe() -> None:
    global policy_runtime_blocked
    with policy_runtime_lock:
        if not policy_runtime_blocked:
            return
        policy_runtime_blocked = False
    _set_dependency("policy", "ready")
    log_event(logger, logging.INFO, "policy_runtime_recovered", proof="successful_inference")


def get_policy() -> PersonalizedMaia2Policy:
    global policy, policy_retry_after
    with policy_lock:
        try:
            cfg = _policy_config()
        except Exception as exc:
            public_error = "Maia2 policy configuration is invalid"
            policy_retry_after = time.monotonic() + DEPENDENCY_RETRY_SECONDS
            _set_dependency("policy", "unavailable", error=public_error)
            logger.error(
                "dependency_configuration_invalid",
                extra={
                    "event": "dependency_configuration_invalid",
                    "fields": {"dependency": "policy", "error": repr(exc)},
                },
            )
            raise DependencyUnavailable(public_error) from exc
        if policy is not None:
            with policy_runtime_lock:
                blocked = policy_runtime_blocked
            if not blocked:
                _set_dependency("policy", "ready")
            return policy
        if time.monotonic() < policy_retry_after:
            error = str(_dependency_snapshot("policy").get("error") or "policy initialization failed")
            raise DependencyUnavailable(error)

        _set_dependency("policy", "loading")
        started = time.monotonic()
        try:
            if not Path(cfg.books_path).is_file():
                raise FileNotFoundError(cfg.books_path)
            loaded = PersonalizedMaia2Policy.load(
                books_path=cfg.books_path,
                model_type=cfg.model_type,
                device=cfg.device,
                strategy=cfg.strategy,
                alpha=cfg.alpha,
                min_count=cfg.min_count,
            )
        except Exception as exc:
            public_error = _public_error("Maia2 policy", exc)
            policy_retry_after = time.monotonic() + DEPENDENCY_RETRY_SECONDS
            _set_dependency("policy", "unavailable", error=public_error)
            logger.error(
                "dependency_failed",
                extra={
                    "event": "dependency_failed",
                    "fields": {"dependency": "policy", "error": repr(exc)},
                },
                exc_info=True,
            )
            raise DependencyUnavailable(public_error) from exc

        policy = loaded
        policy_retry_after = 0.0
        _set_dependency("policy", "ready")
        log_event(
            logger,
            logging.INFO,
            "dependency_ready",
            dependency="policy",
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return policy


def _resolve_executable(configured_path: str) -> str:
    candidate = Path(configured_path).expanduser()
    if candidate.is_file():
        return str(candidate)
    resolved = shutil.which(configured_path)
    if resolved:
        return resolved
    raise FileNotFoundError(configured_path)


def get_safety() -> StockfishBlunderVeto | None:
    global safety, safety_generation, safety_needs_reload, safety_retry_after
    if not STOCKFISH_ENABLED:
        _set_dependency("stockfish", "disabled")
        return None

    with safety_lock:
        if safety is not None:
            state = _dependency_snapshot("stockfish")
            if state["status"] == "ready" and not safety_needs_reload:
                return safety
            if time.monotonic() < safety_retry_after or safety_executor.stats()["active"] > 0:
                return None
            # A timeout abandons work that cannot be cancelled. Once that worker
            # drains, discard the possibly-corrupted UCI process and complete a
            # fresh handshake before reporting ready again.
            stale_safety = safety
            safety = None
            safety_needs_reload = False
            with suppress(Exception):
                stale_safety.close()
        if time.monotonic() < safety_retry_after:
            return None

        _set_dependency("stockfish", "loading")
        started = time.monotonic()
        try:
            configured_path = os.getenv("CHESS_BOT_STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)
            engine_path = _resolve_executable(configured_path)
            engine_config = _stockfish_runtime_config()
            safety = StockfishBlunderVeto(
                engine_path=engine_path,
                veto_cp=engine_config.veto_cp,
                engine_time=engine_config.engine_time,
                engine_timeout=engine_config.engine_timeout,
            )
        except Exception as exc:
            public_error = _public_error("Stockfish", exc)
            safety = None
            safety_needs_reload = False
            safety_retry_after = time.monotonic() + DEPENDENCY_RETRY_SECONDS
            _set_dependency("stockfish", "degraded", error=public_error)
            logger.error(
                "dependency_failed",
                extra={
                    "event": "dependency_failed",
                    "fields": {"dependency": "stockfish", "error": repr(exc)},
                },
                exc_info=True,
            )
            return None

        safety_retry_after = 0.0
        safety_needs_reload = False
        safety_generation += 1
        _set_dependency("stockfish", "ready")
        log_event(
            logger,
            logging.INFO,
            "dependency_ready",
            dependency="stockfish",
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return safety


def _mark_safety_degraded(exc: BaseException, *, discard: bool) -> None:
    global safety, safety_generation, safety_needs_reload, safety_retry_after
    public_error = _public_error("Stockfish", exc)
    with safety_lock:
        safety_retry_after = time.monotonic() + DEPENDENCY_RETRY_SECONDS
        safety_generation += 1
        failed = None
        if discard:
            failed = safety
            safety = None
            safety_needs_reload = False
        else:
            safety_needs_reload = True
        _set_dependency("stockfish", "degraded", error=public_error)
    log_event(logger, logging.ERROR, "stockfish_degraded", error=repr(exc), discard=discard)
    if failed is not None:
        with suppress(Exception):
            failed.close()


def _current_safety_handle() -> tuple[StockfishBlunderVeto, int] | None:
    active_safety = get_safety()
    if active_safety is None:
        return None
    with safety_lock:
        if active_safety is not safety or safety_needs_reload:
            return None
        return active_safety, safety_generation


def _mark_safety_ready_if_current(active_safety: StockfishBlunderVeto, generation: int) -> bool:
    """Publish ready only for a completion from the still-current engine."""
    with safety_lock:
        if (
            runtime_shutting_down
            or active_safety is not safety
            or safety_needs_reload
            or generation != safety_generation
        ):
            return False
        _set_dependency("stockfish", "ready")
        return True


def _run_model_work(fn, *, timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS):
    if runtime_shutting_down:
        raise HTTPException(status_code=503, detail="BenBot is shutting down", headers={"Retry-After": "1"})
    if not _policy_runtime_probe_allowed():
        raise HTTPException(
            status_code=503,
            detail="BenBot inference worker is recovering from a timed-out request",
            headers={"Retry-After": "1"},
        )
    try:
        result = inference_executor.run(
            fn,
            timeout_seconds=timeout_seconds,
            admission_timeout_seconds=INFERENCE_ADMISSION_TIMEOUT_SECONDS,
        )
        _mark_policy_runtime_ready_after_probe()
        return result
    except WorkQueueFull as exc:
        _mark_policy_runtime_unavailable(exc)
        log_event(logger, logging.WARNING, "inference_rejected", reason="queue_full")
        raise HTTPException(
            status_code=503,
            detail="BenBot is at inference capacity; retry shortly",
            headers={"Retry-After": "1"},
        ) from exc
    except WorkTimedOut as exc:
        _mark_policy_runtime_unavailable(exc)
        log_event(logger, logging.ERROR, "inference_timeout", timeout_seconds=timeout_seconds)
        raise HTTPException(status_code=504, detail="BenBot inference timed out") from exc
    except DependencyUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"}) from exc


def initialize_dependencies() -> tuple[dict[str, object], int]:
    """Actually load/check configured dependencies and build readiness state."""
    with readiness_lock:
        if runtime_shutting_down:
            _set_dependency("policy", "unavailable", error="Maia2 policy is unavailable")
            payload = readiness_payload()
            return payload, 503
        try:
            if policy is None:
                inference_executor.run(
                    get_policy,
                    timeout_seconds=MODEL_LOAD_TIMEOUT_SECONDS,
                    admission_timeout_seconds=INFERENCE_ADMISSION_TIMEOUT_SECONDS,
                )
            else:
                get_policy()
        except Exception as exc:
            if isinstance(exc, (WorkQueueFull, WorkTimedOut)):
                _mark_policy_runtime_unavailable(exc)

        if STOCKFISH_ENABLED and not runtime_shutting_down:
            try:
                if safety is None:
                    safety_executor.run(
                        get_safety,
                        timeout_seconds=SAFETY_LOAD_TIMEOUT_SECONDS,
                        admission_timeout_seconds=INFERENCE_ADMISSION_TIMEOUT_SECONDS,
                    )
                else:
                    get_safety()
            except (WorkQueueFull, WorkTimedOut) as exc:
                _mark_safety_degraded(exc, discard=False)
            except Exception as exc:
                _mark_safety_degraded(exc, discard=True)
        else:
            _set_dependency("stockfish", "disabled")

        payload = readiness_payload()
        return payload, 200 if payload["ready"] else 503


def _configuration_snapshot() -> tuple[dict[str, object], bool, bool]:
    try:
        policy_config = _policy_config()
        policy_values: dict[str, object] = {
            "model_type": policy_config.model_type,
            "device": policy_config.device,
            "strategy": policy_config.strategy,
            "alpha": policy_config.alpha,
            "min_count": policy_config.min_count,
        }
        policy_valid = True
    except Exception:
        policy_values = {"model_type": None, "device": None, "strategy": None, "alpha": None, "min_count": None}
        policy_valid = False

    try:
        engine_config = _stockfish_runtime_config()
        engine_values: dict[str, object] = {
            "stockfish_veto_cp": engine_config.veto_cp,
            "stockfish_engine_time_seconds": engine_config.engine_time,
            "stockfish_engine_timeout_seconds": engine_config.engine_timeout,
        }
        stockfish_config_valid = True
    except Exception:
        engine_values = {
            "stockfish_veto_cp": None,
            "stockfish_engine_time_seconds": None,
            "stockfish_engine_timeout_seconds": None,
        }
        stockfish_config_valid = False

    return (
        {
            **policy_values,
            **engine_values,
            "policy_config_valid": policy_valid,
            "stockfish_config_valid": stockfish_config_valid,
            "stockfish_enabled": STOCKFISH_ENABLED,
            "stockfish_required": REQUIRE_STOCKFISH,
        },
        policy_valid,
        stockfish_config_valid,
    )


def readiness_payload() -> dict[str, object]:
    configuration, policy_config_valid, stockfish_config_valid = _configuration_snapshot()
    policy_status = _dependency_snapshot("policy")
    stockfish_status = _stockfish_snapshot()
    policy_ready = policy_config_valid and policy_status["status"] == "ready"
    stockfish_ready = (not STOCKFISH_ENABLED) or (
        stockfish_config_valid and bool(stockfish_status["available"])
    )
    ready = policy_ready and (stockfish_ready or not REQUIRE_STOCKFISH)
    initializing = _dependency_initialization_running()
    if not policy_ready and (policy_status["status"] in {"not_loaded", "loading"} or initializing):
        status = "starting"
    elif not policy_ready or (REQUIRE_STOCKFISH and not stockfish_ready):
        status = "unavailable"
    elif not stockfish_ready:
        status = "degraded"
    else:
        status = "ready"
    return {
        "ready": ready,
        "status": status,
        "initializing": initializing,
        "checked_at": _utc_now(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "dependencies": {"policy": policy_status, "stockfish": stockfish_status},
        "sessions": sessions.stats(),
        "inference": inference_executor.stats(),
        "process": process_memory_snapshot(),
        "limits": {
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
            "safety_timeout_seconds": SAFETY_TIMEOUT_SECONDS,
        },
        "configuration": configuration,
    }


def cached_service_status() -> str:
    policy_status = _dependency_snapshot("policy")["status"]
    stockfish_status = _stockfish_snapshot()
    if policy_status in {"not_loaded", "loading"}:
        return "starting"
    if policy_status != "ready" or (REQUIRE_STOCKFISH and not stockfish_status["available"]):
        return "unavailable"
    if STOCKFISH_ENABLED and not stockfish_status["available"]:
        return "degraded"
    return "ready"


def _client_key(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()[:128]
    return request.client.host if request.client else "unknown"


def _rate_limited_path(path: str) -> bool:
    return path == "/new-game" or path == "/predict" or path.startswith("/game/")


@app.middleware("http")
async def observe_and_limit(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
    started = time.monotonic()
    if _rate_limited_path(request.url.path):
        decision = rate_limiter.check(_client_key(request))
        if not decision.allowed:
            retry_after = max(1, math.ceil(decision.retry_after))
            log_event(
                logger,
                logging.WARNING,
                "rate_limit_exceeded",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                retry_after=retry_after,
            )
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
            )

    try:
        response = await call_next(request)
    except Exception:
        logger.error(
            "request_failed",
            extra={
                "event": "request_failed",
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                },
            },
            exc_info=True,
        )
        raise
    response.headers["X-Request-ID"] = request_id
    log_event(
        logger,
        logging.INFO,
        "request_complete",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return response


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


def serialize_move(move: PolicyMove, san: str | None = None) -> dict[str, object]:
    payload = asdict(move)
    payload["san"] = san
    return payload


def serialize_candidates(board: chess.Board, moves: list[PolicyMove]) -> list[dict[str, object]]:
    payloads = []
    for policy_move in moves:
        try:
            move = chess.Move.from_uci(policy_move.move)
        except ValueError:
            payloads.append(serialize_move(policy_move))
            continue
        payloads.append(serialize_move(policy_move, san=board.san(move) if move in board.legal_moves else None))
    return payloads


def legal_moves(board: chess.Board) -> list[dict[str, str]]:
    return [{"uci": move.uci(), "san": board.san(move)} for move in board.legal_moves]


def board_payload(session: GameSession, game_id: str, bot_move: dict | None = None) -> dict[str, object]:
    state = session.state
    stockfish_status = _stockfish_snapshot()
    configuration, policy_config_valid, stockfish_config_valid = _configuration_snapshot()
    return {
        "game_id": game_id,
        "fen": state.fen,
        "prefix": state.prefix,
        "turn": "white" if state.board.turn == chess.WHITE else "black",
        "human_color": session.human_color,
        "bot_color": session.bot_color,
        "bot_move": bot_move,
        "last_move": session.last_move,
        "safety_status": stockfish_status,
        "policy_config": {
            "mode": DEFAULT_PLAY_MODE,
            "top_k": DEFAULT_TOP_K,
            "temperature": DEFAULT_TEMPERATURE,
            "strategy": configuration["strategy"],
            "alpha": configuration["alpha"],
            "min_count": configuration["min_count"],
            "config_valid": policy_config_valid and (stockfish_config_valid or not STOCKFISH_ENABLED),
            "policy_config_valid": policy_config_valid,
            "stockfish_config_valid": stockfish_config_valid,
            "stockfish_veto_cp": configuration["stockfish_veto_cp"],
            "stockfish_enabled": STOCKFISH_ENABLED,
            "stockfish_status": stockfish_status["status"],
            "stockfish_error": stockfish_status["error"],
        },
        "game_over": state.board.is_game_over(claim_draw=True),
        "checkmate": state.board.is_checkmate(),
        "stalemate": state.board.is_stalemate(),
        "result": state.board.result(claim_draw=True) if state.board.is_game_over(claim_draw=True) else None,
        "legal_moves": legal_moves(state.board),
    }


def _select_candidate(candidates: list[PolicyMove], mode: str) -> PolicyMove:
    if mode == "argmax":
        return candidates[0]
    weights = [max(0.0, candidate.probability) for candidate in candidates]
    if not any(weights):
        return candidates[0]
    return random.choices(candidates, weights=weights, k=1)[0]


def _safety_failure_payload(error: str) -> dict[str, object]:
    return {
        "status": "degraded",
        "available": False,
        "vetoed": False,
        "reason": "stockfish_unavailable",
        "error": error,
    }


def maybe_bot_move(session: GameSession, top_k: int, mode: str, temperature: float) -> dict | None:
    state = session.state
    bot_turn = (state.board.turn == chess.WHITE and session.bot_color == "white") or (
        state.board.turn == chess.BLACK and session.bot_color == "black"
    )
    if not bot_turn or state.board.is_game_over(claim_draw=True):
        return None

    delay = _env_float("CHESS_BOT_MOVE_DELAY_SECONDS", DEFAULT_MOVE_DELAY_SECONDS, minimum=0.0)
    if delay > 0:
        time.sleep(delay)

    fen = state.fen
    prefix = state.prefix

    def predict_candidates() -> list[PolicyMove]:
        active_policy = get_policy()
        return active_policy.predict(
            fen=fen,
            elo_self=session.elo_self,
            elo_oppo=session.elo_oppo,
            you_color=session.bot_color,
            uci_prefix_before=prefix,
            top_n=top_k,
            temperature=temperature,
        )

    candidates = _run_model_work(predict_candidates)
    if not candidates:
        raise HTTPException(status_code=500, detail="Policy returned no legal moves")

    selected = _select_candidate(candidates, mode)
    safety_payload: dict[str, object] | None = None
    safety_handle = _current_safety_handle()
    if safety_handle is not None:
        active_safety, active_safety_generation = safety_handle
        safety_candidates = [selected] + [candidate for candidate in candidates if candidate.move != selected.move]
        board_copy = state.board.copy(stack=True)
        try:
            decision = safety_executor.run(
                lambda: active_safety.choose_safe_move(board_copy, safety_candidates),
                timeout_seconds=SAFETY_TIMEOUT_SECONDS,
                admission_timeout_seconds=INFERENCE_ADMISSION_TIMEOUT_SECONDS,
            )
            if _mark_safety_ready_if_current(active_safety, active_safety_generation):
                selected = decision.selected
                safety_payload = {**asdict(decision), "status": "ready", "available": True, "error": None}
            else:
                error = str(_dependency_snapshot("stockfish").get("error") or "Stockfish is unavailable")
                safety_payload = _safety_failure_payload(error)
        except WorkTimedOut as exc:
            _mark_safety_degraded(exc, discard=False)
            safety_payload = _safety_failure_payload(_public_error("Stockfish", exc))
        except WorkQueueFull as exc:
            _mark_safety_degraded(exc, discard=False)
            safety_payload = _safety_failure_payload(_public_error("Stockfish", exc))
        except Exception as exc:
            _mark_safety_degraded(exc, discard=True)
            safety_payload = _safety_failure_payload(_public_error("Stockfish", exc))
    elif STOCKFISH_ENABLED:
        error = str(_dependency_snapshot("stockfish").get("error") or "Stockfish is unavailable")
        safety_payload = _safety_failure_payload(error)
    else:
        safety_payload = {"status": "disabled", "available": False, "vetoed": False, "error": None}

    try:
        move = chess.Move.from_uci(selected.move)
    except ValueError:
        move = chess.Move.null()
    if move not in state.board.legal_moves:
        selected = next(
            (
                candidate
                for candidate in candidates
                if _is_legal_uci(state.board, candidate.move)
            ),
            None,
        )
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
        "safety": safety_payload,
    }


def _is_legal_uci(board: chess.Board, move_uci: str) -> bool:
    try:
        return chess.Move.from_uci(move_uci) in board.legal_moves
    except ValueError:
        return False


@app.get("/health")
def health() -> dict[str, object]:
    """Fast liveness check; never loads the model or engine."""
    return {
        "ok": True,
        "status": "alive",
        "service_status": cached_service_status(),
        "policy_loaded": policy is not None,
        "stockfish_enabled": STOCKFISH_ENABLED,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.get("/ready")
def ready() -> JSONResponse:
    """Non-blocking readiness; starts one background dependency check at a time."""
    payload = readiness_payload()
    if not payload["ready"] or payload["status"] == "degraded":
        _start_dependency_initialization()
        payload = readiness_payload()
    status_code = 200 if payload["ready"] else 503
    return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store"})


@app.get("/status-badge")
def status_badge() -> JSONResponse:
    """Cached Shields endpoint data; does not trigger heavyweight initialization."""
    status = cached_service_status()
    colors = {"ready": "brightgreen", "degraded": "orange", "starting": "lightgrey", "unavailable": "red"}
    payload = {
        "schemaVersion": 1,
        "label": "BenBot",
        "message": status,
        "color": colors[status],
        "isError": status == "unavailable",
    }
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


@app.get("/status.svg")
def status_svg() -> Response:
    """Self-contained cached status badge for portfolios that cannot run JS."""
    status = cached_service_status()
    colors = {"ready": "#4c1", "degraded": "#e88b00", "starting": "#9f9f9f", "unavailable": "#e05d44"}
    message = escape(status)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="132" height="20" role="img" '
        f'aria-label="BenBot: {message}">'
        '<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" '
        'stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
        '<clipPath id="r"><rect width="132" height="20" rx="3"/></clipPath>'
        '<g clip-path="url(#r)"><rect width="62" height="20" fill="#555"/>'
        f'<rect x="62" width="70" height="20" fill="{colors[status]}"/>'
        '<rect width="132" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        'font-size="11"><text x="31" y="15" fill="#010101" fill-opacity=".3">BenBot</text>'
        '<text x="31" y="14">BenBot</text>'
        f'<text x="97" y="15" fill="#010101" fill-opacity=".3">{message}</text>'
        f'<text x="97" y="14">{message}</text></g></svg>'
    )
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


def _dependency_initialization_running() -> bool:
    with dependency_init_lock:
        return dependency_init_thread is not None and dependency_init_thread.is_alive()


def _start_dependency_initialization() -> bool:
    global dependency_init_thread
    if runtime_shutting_down:
        return False
    with dependency_init_lock:
        if dependency_init_thread is not None and dependency_init_thread.is_alive():
            return False
        now = time.monotonic()
        policy_status = _dependency_snapshot("policy")["status"]
        stockfish_status = _stockfish_snapshot()
        if policy_status != "ready":
            if policy is None and now < policy_retry_after:
                return False
            with policy_runtime_lock:
                blocked = policy_runtime_blocked
            if blocked:
                return False
        elif STOCKFISH_ENABLED and not stockfish_status["available"] and now < safety_retry_after:
            return False
        dependency_init_thread = threading.Thread(
            target=_background_initialize,
            name="benbot-dependency-init",
            daemon=True,
        )
        dependency_init_thread.start()
        return True


def _background_initialize() -> None:
    payload, status_code = initialize_dependencies()
    log_event(
        logger,
        logging.INFO if status_code == 200 else logging.ERROR,
        "startup_dependencies_checked",
        status=payload["status"],
        ready=payload["ready"],
    )


def startup() -> None:
    global inference_executor, runtime_shutting_down, safety_executor
    with runtime_lifecycle_lock:
        if inference_executor.closed:
            inference_executor = _new_inference_executor()
        if safety_executor.closed:
            safety_executor = _new_safety_executor()
        runtime_shutting_down = False
    log_event(
        logger,
        logging.INFO,
        "startup",
        session_capacity=SESSION_CAPACITY,
        session_ttl_seconds=SESSION_TTL_SECONDS,
        inference_concurrency=INFERENCE_CONCURRENCY,
        inference_queue_capacity=INFERENCE_QUEUE_CAPACITY,
        stockfish_enabled=STOCKFISH_ENABLED,
        stockfish_required=REQUIRE_STOCKFISH,
    )
    if _env_bool("CHESS_BOT_PRELOAD_ON_STARTUP", True):
        _start_dependency_initialization()


def shutdown() -> None:
    global runtime_shutting_down, safety, safety_generation, safety_needs_reload
    with runtime_lifecycle_lock:
        if runtime_shutting_down:
            return
        runtime_shutting_down = True

    # Reject/cancel queued model work first. A timed-out model call may still
    # finish in the background, but it does not share a closeable resource.
    inference_executor.shutdown(wait=False)
    # Stockfish is a shared UCI process. Drain its single worker before closing
    # the engine so shutdown cannot race an in-flight analyse/play call.
    safety_executor.shutdown(wait=True)
    with safety_lock:
        active_safety = safety
        safety = None
        safety_needs_reload = False
        safety_generation += 1
    if active_safety is not None:
        with suppress(Exception):
            active_safety.close()
    _set_dependency("stockfish", "not_loaded" if STOCKFISH_ENABLED else "disabled")
    log_event(logger, logging.INFO, "shutdown")


def _get_session(game_id: str) -> GameSession:
    session = sessions.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found or session expired")
    return session


@app.post("/new-game")
def new_game(request: NewGameRequest) -> dict[str, object]:
    game_id = str(uuid.uuid4())
    session = GameSession(
        human_color=request.human_color,
        elo_self=request.elo_self,
        elo_oppo=request.elo_oppo,
    )
    removed = sessions.put(game_id, session)
    if removed:
        log_event(logger, logging.INFO, "sessions_removed", count=len(removed), reason="expired_or_capacity")
    try:
        with session.lock:
            bot_move = maybe_bot_move(session, request.top_k, request.mode, request.temperature)
    except Exception:
        sessions.delete(game_id)
        raise
    log_event(logger, logging.INFO, "game_created", game_id=game_id, human_color=session.human_color)
    return board_payload(session, game_id, bot_move=bot_move)


@app.get("/game/{game_id}")
def get_game(game_id: str) -> dict[str, object]:
    session = _get_session(game_id)
    with session.lock:
        return board_payload(session, game_id)


@app.post("/game/{game_id}/bot-move")
def continue_bot_move(game_id: str, request: BotMoveRequest | None = None) -> dict[str, object]:
    session = _get_session(game_id)
    top_k = request.top_k if request else DEFAULT_TOP_K
    mode = request.mode if request else DEFAULT_PLAY_MODE
    temperature = request.temperature if request else DEFAULT_TEMPERATURE
    with session.lock:
        bot_move = maybe_bot_move(session, top_k, mode, temperature)
        return board_payload(session, game_id, bot_move=bot_move)


@app.post("/game/{game_id}/move")
def play_move(game_id: str, request: MoveRequest) -> dict[str, object]:
    session = _get_session(game_id)
    with session.lock:
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
            log_event(logger, logging.WARNING, "game_move_rolled_back", game_id=game_id, move=move.uci())
            raise
        log_event(logger, logging.INFO, "game_move", game_id=game_id, move=move.uci())
        return board_payload(session, game_id, bot_move=bot_move)


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid FEN") from exc

    def predict_moves() -> list[PolicyMove]:
        active_policy = get_policy()
        return active_policy.predict(
            fen=request.fen,
            elo_self=request.elo_self,
            elo_oppo=request.elo_oppo,
            you_color=request.you_color,
            uci_prefix_before=request.uci_prefix_before,
            top_n=request.top_n,
            temperature=request.temperature,
        )

    moves = _run_model_work(predict_moves)
    return {"moves": [serialize_move(move) for move in moves]}


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
