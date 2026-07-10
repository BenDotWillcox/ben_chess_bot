# Held-out personalization evaluation

The evaluation uses a game-isolated chronological split. Hyperparameters must be selected on validation; this report evaluates the configured setting once on test.

| metric | base Maia2 | personalized | delta |
| --- | ---: | ---: | ---: |
| top-1 accuracy | 0.5550 | 0.5976 | +0.0426 |
| top-3 accuracy | 0.8252 | 0.8502 | +0.0250 |
| log loss | 1.3436 | 1.2276 | -0.1160 |
| multiclass Brier score | 0.5781 | 0.5350 | -0.0431 |
| top-1 ECE (10 bins) | 0.0199 | 0.0221 | +0.0021 |
| mean probability on Ben move | 0.4316 | 0.4671 | +0.0355 |

## Coverage and choice changes

- Test samples: **3638**.
- Exact personal-memory coverage: **318/3638 (8.74%)**.
- Base target-support coverage: **100.00%**.
- Personalization changed top-1 on **200 samples (5.50%)**.
- Changed decisions improved 166 and worsened 11 held-out top-1 matches.

## Data-quality gates

- Passed: **True**.
- Game overlap counts: `{"train_test": 0, "train_val": 0, "val_test": 0}`.
- Data as of: **2026-04-30 17:55:56+00:00**.

## Caveats

- Coverage is exact prefix/FEN count-book coverage, not semantic position similarity; most middlegame/endgame positions remain base Maia2.
- Log loss clips target probabilities at the epsilon recorded in `summary.json`.
- Top-1 ECE is confidence calibration only; with a few thousand personal games, per-bin estimates can be noisy and it is not full multiclass calibration.
- This observational next-move evaluation does not establish game win-rate improvement.
- GPU kernels can retain platform-level nondeterminism; the published run records its seed and device.

## Validation-only hyperparameter selection

- Selection rule: minimum validation log loss; Brier, top-1, top-3, and simpler blend settings break ties.
- Selected `fen` / alpha `0.7` / min-count `1` from 45 candidates.
- Validation log loss/top-1: `1.3093` / `0.5713`.
- The held-out test split was not used for this selection.

## Chronological split evidence

| split | games | Ben-move samples | start UTC | end UTC | prefix replay |
| --- | ---: | ---: | --- | --- | ---: |
| train | 357 | 14710 | 2025-11-02 01:12:07+00:00 | 2026-03-03 15:47:53+00:00 | 100.00% |
| val | 76 | 3079 | 2026-03-03 18:19:01+00:00 | 2026-04-05 16:56:03+00:00 | 100.00% |
| test | 77 | 3638 | 2026-04-06 00:27:24+00:00 | 2026-04-30 17:55:56+00:00 | 100.00% |

## Performance by memory source

| slice | n | memory coverage | change rate | base top-1 | personal top-1 | base top-3 | personal top-3 | base log loss | personal log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fen | 318 | 100.00% | 62.89% | 0.3396 | 0.8270 | 0.6478 | 0.9340 | 2.0744 | 0.7468 |
| maia2 | 3320 | 0.00% | 0.00% | 0.5756 | 0.5756 | 0.8422 | 0.8422 | 1.2736 | 1.2736 |

## Performance by memory observation count

| slice | n | memory coverage | change rate | base top-1 | personal top-1 | base top-3 | personal top-3 | base log loss | personal log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3320 | 0.00% | 0.00% | 0.5756 | 0.5756 | 0.8422 | 0.8422 | 1.2736 | 1.2736 |
| 1 | 45 | 100.00% | 48.89% | 0.4000 | 0.6000 | 0.6667 | 0.7556 | 1.7844 | 1.5919 |
| 2-4 | 53 | 100.00% | 49.06% | 0.3962 | 0.6604 | 0.7358 | 0.8868 | 2.0135 | 1.1478 |
| 5-9 | 62 | 100.00% | 51.61% | 0.4839 | 0.7903 | 0.7742 | 0.9516 | 1.5032 | 0.7232 |
| 10+ | 158 | 100.00% | 75.95% | 0.2468 | 0.9620 | 0.5633 | 0.9937 | 2.4015 | 0.3809 |
