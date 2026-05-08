# Backend API

FastAPI wrapper around the personalized Maia2 policy.

## Local Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8010
```

Open the playable board at:

```text
http://127.0.0.1:8010/
```

With Stockfish safety enabled:

```powershell
$env:CHESS_BOT_STOCKFISH_ENABLED="1"
$env:CHESS_BOT_STOCKFISH_PATH="artifacts\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
$env:CHESS_BOT_BLUNDER_VETO_CP="400"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8010
```

## Endpoints

### `GET /health`

Returns process health without loading Maia2.

### `POST /new-game`

Starts an in-memory game session.

```json
{
  "human_color": "white",
  "elo_self": 1650,
  "elo_oppo": 1650,
  "top_k": 5,
  "mode": "argmax",
  "temperature": 1.0
}
```

If the bot is White, it replies immediately with its first move.

### `GET /game/{game_id}`

Returns the current board state, legal moves, FEN, move prefix, result, and colors.

### `POST /game/{game_id}/move`

Applies a human SAN or UCI move, then returns the bot reply if it is the bot's turn.

```json
{
  "move": "e4",
  "top_k": 5,
  "mode": "argmax",
  "temperature": 1.0
}
```

### `POST /predict`

Stateless policy prediction for a FEN and move prefix.

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

## Environment

```text
CHESS_BOT_CORS_ORIGINS=*                       # comma-separated origins
CHESS_BOT_MODEL_TYPE=rapid
CHESS_BOT_DEVICE=cpu
CHESS_BOT_STRATEGY=combined
CHESS_BOT_ALPHA=0.5
CHESS_BOT_MIN_COUNT=1
CHESS_BOT_BOOKS_PATH=artifacts/personal_books.json
CHESS_BOT_STOCKFISH_ENABLED=0
CHESS_BOT_STOCKFISH_PATH=artifacts/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe
CHESS_BOT_BLUNDER_VETO_CP=400
CHESS_BOT_ENGINE_TIME=0.05
```

## Portfolio Hosting Shape

The portfolio can stay static. Host this backend separately because Maia2 loads a large Torch checkpoint and Stockfish is a native executable. A normal static iframe deployment is not a good fit for the model process.

Recommended shape:

```text
portfolio static site
  -> iframe/link to the chess backend root URL
      -> FastAPI serves both the board UI and model API
```

For a portfolio page, expose the bot metadata returned by the API:

- selected move
- SAN and UCI
- source: `prefix`, `fen`, or `maia2`
- Maia2 probability
- personal probability
- book count
- Stockfish safety decision, if enabled
