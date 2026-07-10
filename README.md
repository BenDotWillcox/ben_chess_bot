---
title: BenBot
emoji: ♟️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# BenBot

[![BenBot service status](https://bwillcox-benbot.hf.space/status.svg)](https://bwillcox-benbot.hf.space/ready)

BenBot is a playable chess policy that combines a **pretrained Maia2 rapid model** with a
**count-based prior from Ben's historical Chess.com games**. Maia2 supplies human-like legal-move
probabilities; the leakage-free validation run selected exact-position (FEN) memory to nudge those
probabilities toward moves Ben has played before. An optional Stockfish layer can veto a proposed
move when it crosses a configured blunder threshold.

BenBot is **not a personally fine-tuned neural network**. “Personalized policy layered on Maia2” is
the precise description of the deployed implementation.

- [Live Space](https://huggingface.co/spaces/Bwillcox/BenBot)
- [Public Space files / deployed source](https://huggingface.co/spaces/Bwillcox/BenBot/tree/main)
- [GitHub source](https://github.com/BenDotWillcox/ben_chess_bot)
- [Dependency readiness](https://bwillcox-benbot.hf.space/ready)

## What the app shows

The browser UI exposes the selected move, the top candidate probabilities, whether personal memory
was used, the active policy settings, and any Stockfish veto. It also reads `/ready` on startup and
periodically thereafter. Maia2 failure is shown as unavailable; optional Stockfish failure is shown
as a visible degraded-safety state while the core policy remains playable.

## System design

1. `chesscom_pipeline.py` downloads public Chess.com monthly archives, keeps rated rapid games, and
   converts each of Ben's turns into a next-move sample.
2. The samples are split chronologically by whole game. `build_personal_books.py` creates exact FEN
   and color-qualified UCI-prefix move-count books from the training split only.
3. At inference, pretrained Maia2 produces probabilities over legal moves for the current position
   and rating context.
4. With matching memory, BenBot blends the two distributions:

   `P(move) = (1 - alpha) * P_maia2(move) + alpha * P_personal(move)`

5. Temperature is applied and the policy either takes the argmax or samples from the top candidates.
6. When enabled and ready, Stockfish evaluates the proposed candidates and can replace a major
   blunder. If Stockfish is optional and unavailable, inference continues without that veto and the
   UI says so explicitly.

The runtime model is therefore the Maia2 checkpoint plus a small, inspectable serving artifact of
counts. Training the Maia2 network is not part of this project.

## Data card

### Collection and scope

- Source: public monthly Chess.com archive API for the account `BigBeennn`.
- Collection window: November 2025 through April 2026 inclusive.
- Filters: rated games whose Chess.com `time_class` is `rapid`.
- Filtered corpus: 510 games, 42,860 total plies, and 21,427 next-move samples for Ben.
- Stored sample context includes FEN before the move, UCI/SAN prefixes, ratings, color, result,
  time-control metadata, game ID, timestamp, and target move.
- The fetch manifest records source month, retrieval time, checksum, and raw game count. The dataset
  manifest records filters, counts, split metadata, and content fingerprints.

The source games were public, but public availability is not the same as consent for unrelated
reuse. This dataset and the derived book are intended only for this single-player demonstration.

### Split logic

The primary evaluation split is chronological and assigns **whole games**, not individual move rows,
to train/validation/test in a 70/15/15 ratio. Games are ordered by UTC completion time with game ID as
a deterministic tie-breaker. This prevents moves from one game crossing split boundaries. A seeded
random game-level split can also be generated for diagnostics, but it is not the headline result.

| Chronological partition | Games | Ben move samples | UTC range |
| --- | ---: | ---: | --- |
| Train | 357 | 14,710 | 2025-11-02 through 2026-03-03 |
| Validation | 76 | 3,079 | 2026-03-03 through 2026-04-05 |
| Test | 77 | 3,638 | 2026-04-06 through 2026-04-30 |

The training split alone builds both personal books. Validation selects personalization settings;
the test split is reserved for the final aggregate comparison. `--seed 42` controls random split
assignment, and the manifests include stable SHA-256 fingerprints for reproducibility.

### Privacy and retention

Raw archive responses and processed parquet files may contain usernames, ratings, URLs, timestamps,
and complete move histories. They are ignored by Git and should not be published as model assets.
The serving artifact removes usernames and URLs, but its FEN/prefix counts still encode recognizable
play patterns and partial game histories. Treat it as pseudonymous behavioral data, not anonymous
data. Delete both source data and derived books if the account owner withdraws permission.

## Evaluation

The evaluation compares base Maia2 with the personalized distribution on a later chronological test
period. Aggregate top-k accuracy, target log loss, memory coverage, and source/count slices are
computed from deterministic probability rankings. Live sampling remains stochastic by design.

The authoritative run uses all 3,638 test samples from 77 later games, seed 42, CPU inference, and no
sampling. Validation selected `strategy=fen`, `alpha=0.7`, and `min_count=1` from 45 candidates by
minimum log loss; the test partition was evaluated once after selection. Data-quality gates confirm
zero game overlap between every split pair.

| Metric | Base Maia2 rapid | Maia2 + personal prior | Delta |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 55.4975% | 59.7581% | +4.2606 pp |
| Top-3 accuracy | 82.5179% | 85.0192% | +2.5013 pp |
| Target log loss | 1.343631 | 1.227588 | -0.116043 |
| Multiclass Brier score | 0.578109 | 0.534999 | -0.043110 |
| Top-1 ECE (10 bins) | 0.019949 | 0.022074 | +0.002125 (worse) |
| Mean probability on Ben's move | 0.431567 | 0.467050 | +0.035483 |

Exact personal-memory coverage was 318/3,638 samples (8.7411%). Personalization changed the top
choice on 200 samples (5.4975%): 166 changes corrected a base top-1 miss and 11 replaced a correct
base choice with a miss. Source and memory-count slices, calibration bins, reconciliation checks,
and the run configuration are published in the [evaluation report](reports/heldout_evaluation/report.md)
and [machine-readable summary](reports/heldout_evaluation/summary.json).
Definitions, data gates, cache checks, and reproduction commands are in the
[evaluation methodology](docs/evaluation.md).

Interpretation is deliberately narrow: the metric asks whether the policy predicts this one player's
recorded next move. It does not measure playing strength, win rate, generalization to other players,
or whether the Stockfish veto improves outcomes. Exact-memory coverage is sparse and concentrated in
repeated positions. The modest ECE regression also means the accuracy/log-loss gain should not be
described as an across-the-board calibration improvement.

### Stockfish safety tradeoff

A separate seeded uniform sample of 200 held-out positions evaluates the live sampling configuration
(`top_k=5`, temperature `0.8`) with a 400-centipawn veto and 0.05-second engine budget. All 200
positions completed successfully. Seed 42 fixes the subset and policy draws, but the finite-time
Stockfish scores, vetoes, and latency remain timing- and hardware-dependent rather than deterministic.

- Stockfish changed 14/200 choices (7.00%), preserved 93.00% overall, and preserved all 19
  memory-backed choices in the sample.
- Same-pass candidate-set median/p95 centipawn loss fell from 1/249.6 before safety to 0/171.05
  after safety.
- Five mate-scale losses before safety fell to zero afterward. Because the 100,000-centipawn mate
  sentinel distorts the overall mean, the non-mate mean is the useful comparison: 38.65 to 31.87.
- Draw/shuffle risk was avoided on 8/12 flagged positions.
- Agreement with Ben's recorded move changed from 52.00% to 52.50% (+0.5 percentage points).
- Added safety latency was 310.53 ms median and 315.33 ms p95 on the evaluation machine.

See the [safety report](reports/safety_evaluation/report.md) and
[machine-readable safety summary](reports/safety_evaluation/summary.json). These estimates apply only
to this seeded subset, Stockfish build, hardware, and finite engine budget. Move agreement is an
observational style proxy, not a causal measure of style preservation or playing strength.

### Runtime reliability snapshot

A fresh local single-worker run reached Maia2 + Stockfish readiness 3.5 seconds after process start.
After excluding and recording one 921.62 ms first-inference warmup, five warm predictions averaged
29.25 ms (29.06 ms p50, 30.20 ms p95). A 20-request workload at concurrency 8 completed at 33.18
successful requests/second with 20 successes and no rejections, timeouts, or failures. API-process RSS
was 601.86 MiB after dependency load and 660.81 MiB after the workload; these values explicitly
exclude the Stockfish child process and are not total service memory.

This run used clean release revision `8e55c0c`, Windows AMD64 with 16 logical CPUs, and an elevated
200-request rate-limit ceiling so it measured inference rather than the default public limit. It is a
local release baseline, not a Hugging Face Space SLA. See the
[aggregate runtime report](reports/runtime_benchmark/report.md),
[machine-readable summary](reports/runtime_benchmark/summary.json), and
[reproduction methodology](docs/runtime_reliability.md).

## Hyperparameters

| Setting | Deployed default | Selection status |
| --- | ---: | --- |
| Maia2 model | `rapid` | Matches the filtered rapid-game domain |
| Personalization strategy | `fen` | Selected from FEN, prefix, and combined strategies by validation log loss |
| `alpha` | `0.7` | Selected in the same 45-candidate validation grid |
| `min_count` | `1` | Selected with the same grid; a single observation can influence the policy |
| Maximum stored prefix ply | `20` | Retained artifact/evaluated alternative; unused by the selected FEN strategy |
| Candidate count | `5` | UI/runtime choice; not selected by held-out performance |
| Sampling temperature | `0.8` | UI/runtime choice; not calibrated on held-out data |
| Stockfish veto | `400` centipawns | Evaluated, but not tuned, on the 200-position safety subset |
| Stockfish analysis time | `0.05` seconds | Engine budget; the measured end-to-end median was 310.53 ms |

Because `min_count=1` and `alpha=0.7`, one historical observation can have substantial influence.
The UI exposes these settings, and downstream claims should not imply more evidence than the grid
search provides.

## Intended use

BenBot is a portfolio demonstration of player-conditioned move prediction, inspectable retrieval,
and graceful degradation around an optional chess engine. It is suitable for casual play and for
studying how a personal prior changes a pretrained human-move model.

It is not an anti-cheating tool, a fair-play assistant for live rated games, a general model of all
chess players, or evidence of a particular playing rating. Do not use it during games where engine
assistance is prohibited.

## Deployment and operational constraints

- The Space uses Docker and serves FastAPI plus static browser assets on port `7860`.
- Maia2/Torch is substantially heavier than a static portfolio. The model process must run on a
  backend with enough memory; the first load can be slow and may require checkpoint download access.
- `artifacts/personal_books.json` must be present at runtime. Stockfish must be installed for the
  container architecture and `CHESS_BOT_STOCKFISH_PATH` must point to its executable.
- `GET /health` is shallow process liveness. `GET /ready` actually loads/checks Maia2 and enabled
  Stockfish; deploy routing and monitoring should use the endpoints for their distinct purposes.
- Stockfish may be configured as optional. Optional failure produces `degraded`, not a false “ready
  with safety” claim. Requiring Stockfish turns the same failure into unavailable readiness.
- Game sessions are process-local and are not durable across restarts or multiple replicas.
- A platform-level scheduler failure can prevent the container from running at all; no in-process
  endpoint can detect itself in that state, so an external monitor must probe the public URL.

See [the backend and readiness contract](docs/backend_api.md) for endpoints and environment variables,
the [runtime benchmark](reports/runtime_benchmark/report.md) and
[measurement method](docs/runtime_reliability.md), and
[portfolio integration](docs/portfolio_integration.md) for the public status/source presentation.

## Known limitations and failure modes

- Exact FEN/prefix lookup has low coverage outside repeated openings and does not generalize personal
  style to unseen positions.
- A count book can memorize rare behavior; `min_count=1` makes that risk explicit.
- The corpus covers one player, one time-control class, and six months, so results do not establish
  broader generalization.
- Live `sample` mode is intentionally stochastic unless a seed is supplied for reproducible tests.
- Maia2 checkpoint load, missing books, incompatible Torch wheels, insufficient memory, or network
  restrictions can make the core policy unavailable.
- Stockfish absence, process failure, or timeout removes the safety veto. The UI reports this state,
  but the bot can still choose severe blunders while degraded.
- The safety evaluation covers 200 seeded held-out positions, one Stockfish build, and one finite
  engine budget. It does not establish a win-rate effect or general latency on deployment hardware.
- The published runtime snapshot is a fresh-process local Windows measurement from clean release
  revision `8e55c0c`, not a Hugging Face Space SLA. Its RSS scope covers the API process only and
  excludes the Stockfish child process.
- Top-1 ECE was slightly worse after personalization (0.019949 to 0.022074), and per-bin estimates
  remain noisy at this dataset size; this is not evidence of universal calibration improvement.
- Platform scheduling is outside the application process. Readiness cannot repair or observe a Space
  that the hosting platform never starts.

## Personal contribution and third-party components

**Ben Willcox's project contribution** is the application and evaluation layer: selecting and
collecting the personal rapid-game corpus; implementing the reproducible ingestion/split pipeline;
building the position/prefix count artifacts; integrating those counts with Maia2 probabilities;
adding Stockfish veto logic; and building the FastAPI game service, inspectable browser UI,
evaluation scripts, and deployment configuration.

**Not Ben's original model/engine work:** Maia2, its pretrained weights, PyTorch, Stockfish, and
`python-chess` are third-party projects. BenBot does not claim authorship of those systems or that Ben
trained the Maia2 checkpoint. The novel project scope is the personal-data pipeline, count-based
personalization, evaluation, safety integration, and product presentation around them.

## Reproduce locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Fetch and build the dataset, then create the runtime book:

```bash
python chesscom_pipeline.py all --username BigBeennn --start 2025-11 --end 2026-04 --split-mode both --seed 42
python build_personal_books.py --train-split data/processed/splits/chron/train.parquet --output artifacts/personal_books.json --max-prefix-ply 20
```

Run the API and browser UI:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Then open `http://127.0.0.1:7860/`. The model may download/load on the first `/ready` request.

Raw data, processed datasets, checkpoints, detailed prediction caches, and unreviewed runtime outputs
are ignored by Git. Reviewed aggregate reports and the minimum serving artifact required for deployment
are tracked.
