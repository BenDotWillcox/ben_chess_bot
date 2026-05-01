# ben_chess_bot data pipeline

Initial ingestion pipeline for Chess.com **rated rapid** games.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Default window is **2025-11 through 2026-04** (inclusive, so all of April 2026 is fetched).

```bash
python chesscom_pipeline.py all --username BigBeennn --split-mode both --seed 42
```

Or explicitly provide dates:

```bash
python chesscom_pipeline.py fetch --username BigBeennn --start 2025-11 --end 2026-04
python chesscom_pipeline.py build --username BigBeennn --split-mode both --seed 42
python chesscom_pipeline.py report --username BigBeennn
```

## Outputs

- `data/raw/chesscom/<username>/YYYY-MM.json`
- `data/interim/games.parquet` (rated rapid only)
- `data/interim/moves.parquet`
- `data/processed/train_samples.parquet`
- `data/processed/splits/chron/{train,val,test}.parquet`
- `data/processed/splits/random/{train,val,test}.parquet`
- `data/manifests/fetch_manifest.json`
- `data/manifests/dataset_stats.json`


## Chess.com API etiquette

- Script sends a descriptive `User-Agent` and `Accept: application/json` header.
- Fetching uses retries with exponential backoff for transient `403/429/5xx` responses.
- Calls are paced with a short delay between monthly requests.
- Update the `USER_AGENT` contact in `chesscom_pipeline.py` before heavy usage.
