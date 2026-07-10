FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHESS_BOT_DEVICE=cpu \
    CHESS_BOT_STRATEGY=fen \
    CHESS_BOT_ALPHA=0.7 \
    CHESS_BOT_MIN_COUNT=1 \
    CHESS_BOT_STOCKFISH_ENABLED=1 \
    CHESS_BOT_STOCKFISH_PATH=/usr/games/stockfish \
    CHESS_BOT_PRELOAD_ON_STARTUP=1 \
    HOME=/home/user

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git stockfish \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group user

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0 \
    && python -m pip install -r requirements.txt

RUN mkdir -p /app/maia2_models /home/user/.cache \
    && chown -R user:user /app /home/user

COPY --chown=user:user . .

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=5).read()"]

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "10"]
