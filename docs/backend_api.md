# Backend API and readiness contract

FastAPI serves the personalized Maia2 policy, in-memory chess sessions, and the browser UI.

## Local run

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860/`. Startup can preload dependencies in the background; `GET /ready`
performs an active load/retry and is the authoritative dependency check.

## Service status

### `GET /health`

Fast liveness only. It never loads Maia2 or Stockfish, so a 200 response proves the web process is
alive but does not prove inference can succeed.

```json
{
  "ok": true,
  "status": "alive",
  "service_status": "starting",
  "policy_loaded": false,
  "stockfish_enabled": true,
  "uptime_seconds": 2.4
}
```

### `GET /ready`

Starts one background initialization/retry for Maia2 and enabled Stockfish, then immediately returns
cached dependency, session, queue, and timeout state. Callers should poll while the state is
`starting`; concurrent callers never queue behind the model-load timeout. The public states are:

- `ready`: Maia2 and every configured dependency required for readiness are available.
- `starting`: the single background initialization is still running (HTTP 503).
- `degraded`: Maia2 is ready, but optional Stockfish is unavailable. The policy can still play, but
  no safety veto is promised.
- `unavailable`: Maia2 failed, or Stockfish failed while `CHESS_BOT_REQUIRE_STOCKFISH=1`.

`ready` and optional-safety `degraded` return HTTP 200. `unavailable` returns HTTP 503. Example:

```json
{
  "ready": true,
  "status": "degraded",
  "checked_at": "2026-07-10T15:10:00Z",
  "dependencies": {
    "policy": {
      "status": "ready",
      "error": null,
      "last_attempt": "2026-07-10T15:09:55Z",
      "last_ready": "2026-07-10T15:09:59Z"
    },
    "stockfish": {
      "enabled": true,
      "required": false,
      "status": "degraded",
      "available": false,
      "error": "Stockfish unavailable",
      "last_attempt": "2026-07-10T15:09:59Z",
      "last_ready": null
    }
  },
  "sessions": {
    "active": 0,
    "capacity": 128,
    "ttl_seconds": 1800.0
  },
  "inference": {
    "active": 0,
    "capacity": 5,
    "workers": 1,
    "queue_capacity": 4
  },
  "limits": {
    "rate_limit_requests": 60,
    "rate_limit_window_seconds": 60.0,
    "inference_timeout_seconds": 30.0,
    "safety_timeout_seconds": 3.0
  }
}
```

Dependency errors are sanitized for public responses. Full exceptions remain in structured server
logs. Readiness responses use `Cache-Control: no-store`.

### `GET /status-badge` and `GET /status.svg`

These lightweight endpoints expose the **last-known** state without triggering model load.
`/status-badge` returns Shields endpoint JSON; `/status.svg` returns a self-contained 132×20 SVG.
Both report `starting`, `ready`, `degraded`, or `unavailable` and use `Cache-Control: no-store`.

```markdown
[![BenBot status](https://bwillcox-benbot.hf.space/status.svg)](https://bwillcox-benbot.hf.space/ready)
```

Use `/ready` for monitoring and for a live portfolio pill. Use the SVG when a no-JavaScript,
last-known badge is sufficient. Neither endpoint can execute if the hosting platform never schedules
the container, so an external URL probe is still required for alerts.

## Game endpoints

### `POST /new-game`

Starts an expiring, process-local game session.

```json
{
  "human_color": "white",
  "elo_self": 1650,
  "elo_oppo": 1650,
  "top_k": 5,
  "mode": "sample",
  "temperature": 0.8
}
```

If the bot is White, it replies immediately. The response includes FEN, legal moves, colors,
policy configuration, candidate trace, and both `safety_status` and the flattened
`policy_config.stockfish_status`/`stockfish_error` values used by the frontend degradation notice.
Elo inputs are bounded to 100–4000; out-of-range values return HTTP 422.

### `GET /game/{game_id}`

Returns current state. Missing, expired, or evicted sessions return HTTP 404.

### `POST /game/{game_id}/move`

Applies a human SAN or UCI move, then runs the bot if appropriate. If bot inference fails, the human
move and last-move marker are rolled back.
Move notation is limited to 16 characters; oversized input returns HTTP 422.

```json
{
  "move": "e4",
  "top_k": 5,
  "mode": "sample",
  "temperature": 0.8
}
```

### `POST /game/{game_id}/bot-move`

Continues a bot turn after a client reconnect or turn desynchronization.

### `POST /predict`

Stateless policy prediction for a FEN and optional move prefix.

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "you_color": "black",
  "uci_prefix_before": "e2e4",
  "elo_self": 1650,
  "elo_oppo": 1650,
  "top_n": 10,
  "temperature": 1.0
}
```

FEN is limited to 128 characters, the optional UCI prefix to 4096 characters, and Elo inputs to
100–4000; size/range violations return HTTP 422. A syntactically invalid in-range FEN returns HTTP
400. Policy queue saturation or dependency failure returns HTTP 503; inference timeout returns HTTP
504. Public dependency errors are sanitized; detailed exception text is emitted only to structured
server logs.

## Capacity and failure controls

- Sessions expire after a configurable TTL and the least-recently-used session is evicted at
  capacity.
- Game mutations are serialized per session.
- `/new-game`, `/predict`, and `/game/*` use a bounded per-client sliding-window rate limit. A rejected
  request returns HTTP 429 with `Retry-After`.
- Rate limiting uses the immediate peer by default. Forwarded client IP headers are honored only when
  `CHESS_BOT_TRUST_PROXY_HEADERS=1` is explicitly set for a deployment behind a trusted proxy.
- Maia2 and Stockfish run through separate bounded executors. Queue saturation fails fast; inference
  and safety calls have timeouts.
- Every response gets `X-Request-ID`; structured request, dependency, safety-degradation, and startup
  events are written to logs.

These are single-process controls. Multiple replicas do not share sessions, rate-limit counters, or
dependency state.

## Environment

```text
CHESS_BOT_CORS_ORIGINS=*                          # comma-separated origins
CHESS_BOT_MODEL_TYPE=rapid
CHESS_BOT_DEVICE=cpu
CHESS_BOT_STRATEGY=fen
CHESS_BOT_ALPHA=0.7
CHESS_BOT_MIN_COUNT=1
CHESS_BOT_BOOKS_PATH=artifacts/personal_books.json

CHESS_BOT_STOCKFISH_ENABLED=1
CHESS_BOT_REQUIRE_STOCKFISH=0
CHESS_BOT_STOCKFISH_PATH=artifacts/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe
CHESS_BOT_BLUNDER_VETO_CP=400
CHESS_BOT_ENGINE_TIME=0.05
CHESS_BOT_ENGINE_TIMEOUT_SECONDS=10

CHESS_BOT_SESSION_TTL_SECONDS=1800
CHESS_BOT_SESSION_CAPACITY=128
CHESS_BOT_RATE_LIMIT_REQUESTS=60                 # 0 disables request limiting
CHESS_BOT_RATE_LIMIT_WINDOW_SECONDS=60
CHESS_BOT_RATE_LIMIT_MAX_CLIENTS=10000
CHESS_BOT_TRUST_PROXY_HEADERS=0  # enable only behind a trusted proxy

CHESS_BOT_INFERENCE_CONCURRENCY=1
CHESS_BOT_INFERENCE_QUEUE_CAPACITY=4
CHESS_BOT_INFERENCE_ADMISSION_TIMEOUT_SECONDS=0.25
CHESS_BOT_INFERENCE_TIMEOUT_SECONDS=30
CHESS_BOT_MODEL_LOAD_TIMEOUT_SECONDS=180
CHESS_BOT_SAFETY_TIMEOUT_SECONDS=3
CHESS_BOT_SAFETY_LOAD_TIMEOUT_SECONDS=15
CHESS_BOT_DEPENDENCY_RETRY_SECONDS=30
CHESS_BOT_PRELOAD_ON_STARTUP=1
CHESS_BOT_MOVE_DELAY_SECONDS=0.5
CHESS_BOT_LOG_LEVEL=INFO
```

Set a Linux-compatible Stockfish path in the Space container; the shown default is the local Windows
artifact path. If the portfolio fetches `/ready` directly and CORS is restricted, include the
portfolio origin in `CHESS_BOT_CORS_ORIGINS`.

## Hosting shape

The portfolio can remain static, but Maia2/Torch and the native Stockfish process require a backend.
The deployed Space serves both UI and API; the portfolio embeds or links the Space and consumes the
readiness contract. See [portfolio integration](portfolio_integration.md) for exact wording, source
link markup, and dynamic badge code.
