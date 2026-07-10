# Reproducible evaluation

This project evaluates a count-based personal memory layered on pretrained Maia2. It is not a personally fine-tuned neural model.

## Data and chronological split

`chesscom_pipeline.py` downloads Chess.com monthly archives, keeps rated rapid games, parses every move, and emits one sample for each move made by the configured player. Splits are assigned at the **game** grain: games are sorted by `(date_utc, game_id)`, then the earliest 70% go to train, the next 15% to validation, and the final 15% to test. A game can never cross a boundary. The seeded random split is a diagnostic only and is not used for the published result.

Raw archives, Parquet files, model weights, and detailed predictions are gitignored because they contain personal game history or large binaries. The reviewed aggregate summaries and the minimum serving artifact, `artifacts/personal_books.json`, are tracked; its metadata records the training checksum, boundaries, and counts.

Build from already fetched archives and prove the data gates:

```bash
python chesscom_pipeline.py build --username BigBeennn --split-mode both --seed 42
python run_evaluation.py quality --output reports/heldout_evaluation/data_quality.json
```

The quality command fails on split overlap, reversed chronology, null required fields, duplicate `(game_id, ply_index)` keys, invalid FENs, illegal targets, player/side mismatches, invalid prefix moves, prefix-length mismatches, or a UCI prefix that does not reconstruct the recorded FEN. Prefix replay is required for trustworthy repetition/shuffle evaluation.

## Validation selection and held-out policy evaluation

The published setting is selected using validation only. The grid covers FEN, prefix, and combined memory strategies; alpha values `0.1, 0.2, 0.35, 0.5, 0.7`; and minimum counts `1, 2, 3`. Minimum validation log loss is primary; Brier score, top-1, top-3, and simpler settings break ties. The chosen setting is then evaluated once on the full test split.

```bash
python run_evaluation.py policy \
  --tune-on-validation \
  --seed 42 \
  --books-output artifacts/personal_books.json \
  --val-base-predictions-out reports/heldout_evaluation/val_base_predictions.jsonl \
  --base-predictions-out reports/heldout_evaluation/test_base_predictions.jsonl \
  --output-dir reports/heldout_evaluation
```

The detailed prediction caches are ignored, but can be supplied later with `--val-base-predictions-in` and `--base-predictions-in` to reproduce aggregation without another neural-model pass. Cache rows are checked against a SHA-256 sample key.

Metrics are defined as follows:

- **Top-1/top-3 accuracy:** whether Ben's actual next move is the highest-ranked or one of the three highest-ranked legal moves.
- **Log loss:** negative log probability of Ben's move, with probability clipped at `1e-15`.
- **Multiclass Brier score:** sum of squared error between the legal-move distribution and the one-hot actual move, averaged across positions.
- **Top-1 ECE:** expected calibration error over ten fixed equal-width bins of maximum move confidence. Every bin publishes count, accuracy, and mean confidence; this measures confidence calibration, not full multiclass calibration.
- **Personal-memory coverage:** positions where an exact prefix or FEN memory meets `min_count`, divided by every held-out sample. It is not semantic-position coverage.
- **Memory count:** total eligible personal observations at the matched key, bucketed as `0`, `1`, `2-4`, `5-9`, and `10+`.
- **Ben-likeness:** agreement and assigned probability on Ben's actual held-out move. This is an operational next-move measure, not a causal style or playing-strength claim.
- **Change rate:** fraction of held-out positions where personalization changes base Maia2's top move; improved and worsened matches are reported separately.

The aggregate, source, and count-bucket denominators reconcile in `summary.json`. Published results are in [the held-out report](../reports/heldout_evaluation/report.md) and its machine-readable `summary.json`.

## Stockfish safety evaluation

Safety uses the live sampling behavior (top five, temperature `0.8`) and seed `42` to make the 200-position subset and policy draws reproducible. It replays complete history, samples the personal policy, and applies the serving veto:

```bash
python run_evaluation.py safety \
  --stockfish-path /path/to/stockfish \
  --mode sample \
  --top-k 5 \
  --temperature 0.8 \
  --seed 42 \
  --limit 200 \
  --output-dir reports/safety_evaluation
```

The report includes veto/failure rates, finite-budget centipawn-loss distributions before and after safety, loss avoided, draw/shuffle risks and avoidances, added safety latency, Ben-move agreement before/after, and preservation of memory-covered choices. Python-chess maps mate to a 100,000-centipawn sentinel, so mate-scale and non-mate distributions are separated. The seed does **not** make the 50 ms Stockfish search deterministic: scores, vetoes, and latency remain timing-, hardware-, and engine-build-dependent. Results apply to the recorded run and do not establish win-rate effects.

See [the safety report](../reports/safety_evaluation/report.md) and its machine-readable `summary.json`.

## Evaluation seeds versus live play

Policy evaluation defaults to seed `42` and records it. In safety evaluation, that seed fixes subset selection and policy sampling only; time-limited engine output is not deterministic. `play_policy.py` and `predict_policy.py` accept optional `--seed` for debugging. Omitting it preserves variable human-like live sampling; the API also samples without a fixed evaluation seed.
