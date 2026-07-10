"""Offline-testable policy, data-quality, and safety evaluation primitives.

The neural model and Stockfish are intentionally kept outside the metric
functions.  Tests can pass cached probability dictionaries and safety records,
while the CLI in ``run_evaluation.py`` supplies the real Maia2/Stockfish calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import chess
import pandas as pd

from personalized_policy import apply_temperature, normalize, position_key, prefix_key


DEFAULT_SEED = 42
LOG_LOSS_EPSILON = 1e-15
REQUIRED_SAMPLE_COLUMNS = (
    "fen",
    "move",
    "elo_self",
    "elo_oppo",
    "game_id",
    "date_utc",
    "ply_index",
    "ply_prefix_before",
    "uci_prefix_before",
    "you_color",
)
AUDIT_REQUIRED_SAMPLE_COLUMNS = REQUIRED_SAMPLE_COLUMNS + (
    "target_move_uci",
    "side_to_move",
    "time_class",
    "rated",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (when present) for evaluation only."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def sample_key(row: Any) -> str:
    raw = "|".join(
        str(getattr(row, field)) for field in ("game_id", "ply_index", "fen", "move")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sorted_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Validate, normalize, and deterministically order a legal-move distribution."""
    cleaned: dict[str, float] = {}
    for move, probability in probabilities.items():
        value = float(probability)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid probability for {move}: {probability}")
        cleaned[str(move)] = value
    if not cleaned or sum(cleaned.values()) <= 0:
        raise ValueError("Move probabilities must contain positive mass")
    normalized = normalize(cleaned)
    return dict(sorted(normalized.items(), key=lambda item: (-item[1], item[0])))


def _legal_counts(
    counts: Mapping[str, int] | None,
    legal_moves: set[str],
) -> dict[str, int]:
    if not counts:
        return {}
    return {
        move: int(count)
        for move, count in counts.items()
        if move in legal_moves and int(count) > 0
    }


def select_personal_memory(
    row: Any,
    books: Mapping[str, Any],
    legal_moves: Iterable[str],
    strategy: str,
    min_count: int,
) -> tuple[dict[str, int], str]:
    """Match serving-time prefix/FEN precedence using only legal observations."""
    if strategy not in {"fen", "prefix", "combined"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    legal = set(legal_moves)
    fen_counts = _legal_counts(books.get("fen_book", {}).get(position_key(row.fen)), legal)
    prefix_counts = _legal_counts(
        books.get("prefix_book", {}).get(prefix_key(row.you_color, row.uci_prefix_before)),
        legal,
    )

    candidates: list[tuple[str, dict[str, int]]]
    if strategy == "fen":
        candidates = [("fen", fen_counts)]
    elif strategy == "prefix":
        candidates = [("prefix", prefix_counts)]
    else:
        candidates = [("prefix", prefix_counts), ("fen", fen_counts)]
    for source, counts in candidates:
        if sum(counts.values()) >= min_count:
            return counts, source
    return {}, "maia2"


def blend_probabilities(
    base_probabilities: Mapping[str, float],
    personal_counts: Mapping[str, int],
    alpha: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    base = sorted_probabilities(base_probabilities)
    if not personal_counts:
        return base, {}
    total = sum(personal_counts.values())
    if total <= 0:
        return base, {}
    personal = {move: count / total for move, count in personal_counts.items()}
    blended = {
        move: (1.0 - alpha) * probability + alpha * personal.get(move, 0.0)
        for move, probability in base.items()
    }
    return sorted_probabilities(blended), personal


def count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    return "10+"


def _log_loss(probability: float, epsilon: float) -> float:
    return -math.log(max(float(probability), epsilon))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean": round(_mean(values), 6) if values else None,
        "p50": round(percentile(values, 0.50), 6) if values else None,
        "p90": round(percentile(values, 0.90), 6) if values else None,
        "p95": round(percentile(values, 0.95), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def top1_calibration(
    records: Sequence[Mapping[str, Any]], prefix: str, *, bins: int = 10
) -> dict[str, Any]:
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets: list[list[Mapping[str, Any]]] = [[] for _ in range(bins)]
    for row in records:
        confidence = float(row[f"{prefix}_top1_confidence"])
        bucket_index = min(int(confidence * bins), bins - 1)
        buckets[bucket_index].append(row)
    details = []
    weighted_error = 0.0
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        accuracy = _mean([float(row[f"{prefix}_top1_hit"]) for row in bucket]) if bucket else None
        confidence = _mean([float(row[f"{prefix}_top1_confidence"]) for row in bucket]) if bucket else None
        if bucket:
            weighted_error += count / len(records) * abs(float(accuracy) - float(confidence))
        details.append(
            {
                "lower_inclusive": round(index / bins, 6),
                "upper_inclusive" if index == bins - 1 else "upper_exclusive": round(
                    (index + 1) / bins, 6
                ),
                "count": count,
                "accuracy": round(float(accuracy), 6) if accuracy is not None else None,
                "mean_confidence": round(float(confidence), 6) if confidence is not None else None,
            }
        )
    if sum(bucket["count"] for bucket in details) != len(records):
        raise AssertionError("Calibration-bin denominator does not reconcile")
    return {
        "definition": f"top-1 expected calibration error across {bins} equal-width confidence bins",
        "ece": round(weighted_error, 6),
        "bins": details,
        "denominator": len(records),
    }


def _policy_metrics(records: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "top1_accuracy": round(_mean([float(row[f"{prefix}_top1_hit"]) for row in records]), 6),
        "top3_accuracy": round(_mean([float(row[f"{prefix}_top3_hit"]) for row in records]), 6),
        "log_loss": round(_mean([float(row[f"{prefix}_log_loss"]) for row in records]), 6),
        "mean_target_probability": round(
            _mean([float(row[f"{prefix}_target_probability"]) for row in records]), 6
        ),
        "target_support_coverage": round(
            _mean([float(row[f"{prefix}_target_probability"] > 0.0) for row in records]), 6
        ),
        "multiclass_brier_score": round(
            _mean([float(row[f"{prefix}_brier_score"]) for row in records]), 6
        ),
        "top1_calibration": top1_calibration(records, prefix),
    }


def _policy_group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = len(records)
    changed = [row for row in records if row["top1_changed"]]
    return {
        "samples": samples,
        "denominator": samples,
        "memory_covered_samples": sum(row["memory_source"] != "maia2" for row in records),
        "base_maia2": _policy_metrics(records, "base"),
        "personalized": _policy_metrics(records, "personalized"),
        "top1_change_rate": round(_mean([float(row["top1_changed"]) for row in records]), 6),
        "changed_samples": len(changed),
        "improved_samples": sum(row["personalized_improved_top1"] for row in records),
        "worsened_samples": sum(row["personalized_worsened_top1"] for row in records),
    }


def summarize_policy_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No policy evaluation records")
    total = len(records)
    overall = _policy_group_summary(records)
    memory_covered = sum(row["memory_source"] != "maia2" for row in records)
    target_in_memory = sum(row["memory_target_count"] > 0 for row in records)

    by_source: dict[str, Any] = {}
    for source in ("prefix", "fen", "maia2"):
        group = [row for row in records if row["memory_source"] == source]
        if group:
            by_source[source] = _policy_group_summary(group)
    by_count: dict[str, Any] = {}
    for bucket in ("0", "1", "2-4", "5-9", "10+"):
        group = [row for row in records if row["memory_count_bucket"] == bucket]
        if group:
            by_count[bucket] = _policy_group_summary(group)

    source_denominator = sum(group["samples"] for group in by_source.values())
    count_denominator = sum(group["samples"] for group in by_count.values())
    if source_denominator != total or count_denominator != total:
        raise AssertionError("Memory group denominators do not reconcile to total samples")

    base = overall["base_maia2"]
    personalized = overall["personalized"]
    reconciliation_metrics: dict[str, Any] = {}
    for policy_name in ("base_maia2", "personalized"):
        for metric in (
            "top1_accuracy",
            "top3_accuracy",
            "log_loss",
            "mean_target_probability",
            "multiclass_brier_score",
        ):
            weighted = sum(
                group["samples"] * group[policy_name][metric] for group in by_source.values()
            ) / total
            overall_value = overall[policy_name][metric]
            reconciliation_metrics[f"{policy_name}.{metric}"] = {
                "overall": overall_value,
                "source_weighted": round(weighted, 6),
                "absolute_delta": round(abs(weighted - overall_value), 9),
                "within_rounding_tolerance": abs(weighted - overall_value) <= 2e-6,
            }
    return {
        "samples": total,
        "coverage": {
            "denominator": total,
            "personal_memory_samples": memory_covered,
            "personal_memory_rate": round(memory_covered / total, 6),
            "target_seen_in_memory_samples": target_in_memory,
            "target_seen_in_memory_rate": round(target_in_memory / total, 6),
            "base_target_support_rate": base["target_support_coverage"],
        },
        "base_maia2": base,
        "personalized": personalized,
        "personalization": {
            "top1_changed_samples": overall["changed_samples"],
            "top1_change_rate": overall["top1_change_rate"],
            "top1_improved_samples": overall["improved_samples"],
            "top1_worsened_samples": overall["worsened_samples"],
            "ben_likeness": {
                "definition": "agreement/probability on Ben's actual held-out next move",
                "base_top1_match_rate": base["top1_accuracy"],
                "personalized_top1_match_rate": personalized["top1_accuracy"],
                "top1_lift": round(personalized["top1_accuracy"] - base["top1_accuracy"], 6),
                "target_probability_lift": round(
                    personalized["mean_target_probability"] - base["mean_target_probability"], 6
                ),
                "log_loss_improvement": round(base["log_loss"] - personalized["log_loss"], 6),
            },
        },
        "by_memory_source": by_source,
        "by_memory_count": by_count,
        "reconciliation": {
            "source_samples": source_denominator,
            "count_bucket_samples": count_denominator,
            "matches_total": source_denominator == count_denominator == total,
            "source_weighted_metrics": reconciliation_metrics,
            "weighted_metrics_passed": all(
                item["within_rounding_tolerance"] for item in reconciliation_metrics.values()
            ),
        },
    }


def evaluate_policy_probabilities(
    source: pd.DataFrame,
    base_probabilities: Sequence[Mapping[str, float]],
    books: Mapping[str, Any],
    *,
    strategy: str = "combined",
    alpha: float = 0.5,
    min_count: int = 1,
    epsilon: float = LOG_LOSS_EPSILON,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(source) != len(base_probabilities):
        raise ValueError("Source rows and prediction rows have different lengths")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between 0 and 1")
    missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"Evaluation split is missing columns: {missing}")

    records: list[dict[str, Any]] = []
    for row, raw_base in zip(source.reset_index(drop=True).itertuples(index=False), base_probabilities):
        base = sorted_probabilities(raw_base)
        board = chess.Board(row.fen)
        legal_moves = {move.uci() for move in board.legal_moves}
        if set(base) - legal_moves:
            raise ValueError(f"Prediction contains illegal moves for sample {sample_key(row)}")
        if row.move not in legal_moves:
            raise ValueError(f"Held-out target is illegal for sample {sample_key(row)}")

        counts, source_name = select_personal_memory(
            row, books, legal_moves, strategy=strategy, min_count=min_count
        )
        personalized, personal_probs = blend_probabilities(base, counts, alpha)
        base_moves = list(base)
        personalized_moves = list(personalized)
        base_target = float(base.get(row.move, 0.0))
        personalized_target = float(personalized.get(row.move, 0.0))
        base_hit = base_moves[0] == row.move
        personalized_hit = personalized_moves[0] == row.move
        memory_total = sum(counts.values())
        record = {
            "sample_key": sample_key(row),
            "game_id": str(row.game_id),
            "date_utc": str(row.date_utc),
            "ply_index": int(row.ply_index),
            "fen": row.fen,
            "actual_move": row.move,
            "you_color": row.you_color,
            "uci_prefix_before": row.uci_prefix_before,
            "base_probabilities": base,
            "personalized_probabilities": personalized,
            "personal_probabilities": personal_probs,
            "memory_counts": counts,
            "memory_source": source_name,
            "memory_count": memory_total,
            "memory_target_count": int(counts.get(row.move, 0)),
            "memory_count_bucket": count_bucket(memory_total),
            "base_top1_move": base_moves[0],
            "personalized_top1_move": personalized_moves[0],
            "base_top1_hit": base_hit,
            "personalized_top1_hit": personalized_hit,
            "base_top1_confidence": float(base[base_moves[0]]),
            "personalized_top1_confidence": float(personalized[personalized_moves[0]]),
            "base_top3_hit": row.move in base_moves[:3],
            "personalized_top3_hit": row.move in personalized_moves[:3],
            "base_target_probability": base_target,
            "personalized_target_probability": personalized_target,
            "base_log_loss": _log_loss(base_target, epsilon),
            "personalized_log_loss": _log_loss(personalized_target, epsilon),
            "base_brier_score": sum(probability * probability for probability in base.values())
            - (2.0 * base_target)
            + 1.0,
            "personalized_brier_score": sum(
                probability * probability for probability in personalized.values()
            )
            - (2.0 * personalized_target)
            + 1.0,
            "top1_changed": base_moves[0] != personalized_moves[0],
            "personalized_improved_top1": personalized_hit and not base_hit,
            "personalized_worsened_top1": base_hit and not personalized_hit,
        }
        records.append(record)

    summary = summarize_policy_records(records)
    summary["configuration"] = {
        "strategy": strategy,
        "alpha": alpha,
        "min_count": min_count,
        "log_loss_epsilon": epsilon,
        "count_bucket_definition": {"0": "0", "1": "1", "2-4": "2 to 4", "5-9": "5 to 9", "10+": "10 or more"},
    }
    return summary, records


def audit_splits(splits: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Prove grain, legality, chronology, timezone, and game isolation."""
    expected = ("train", "val", "test")
    missing_splits = [name for name in expected if name not in splits]
    if missing_splits:
        raise ValueError(f"Missing splits for audit: {missing_splits}")

    details: dict[str, Any] = {}
    game_sets: dict[str, set[str]] = {}
    all_dates: list[pd.Timestamp] = []
    all_valid = True
    for name in expected:
        frame = splits[name].reset_index(drop=True)
        missing_columns = [column for column in AUDIT_REQUIRED_SAMPLE_COLUMNS if column not in frame.columns]
        null_counts = {
            column: int(frame[column].isna().sum())
            for column in AUDIT_REQUIRED_SAMPLE_COLUMNS
            if column in frame.columns
        }
        duplicate_keys = (
            int(frame.duplicated(subset=["game_id", "ply_index"]).sum())
            if {"game_id", "ply_index"}.issubset(frame.columns)
            else None
        )
        invalid_fens = 0
        illegal_targets = 0
        side_mismatches = 0
        prefix_reconstruction_mismatches = 0
        invalid_prefix_moves = 0
        ply_prefix_mismatches = 0
        prefix_reconstruction_successes = 0
        target_alias_mismatches = 0
        non_rapid_samples = 0
        unrated_samples = 0
        if not missing_columns:
            for row in frame.itertuples(index=False):
                try:
                    board = chess.Board(row.fen)
                except ValueError:
                    invalid_fens += 1
                    continue
                try:
                    move = chess.Move.from_uci(row.move)
                except ValueError:
                    illegal_targets += 1
                    continue
                if move not in board.legal_moves:
                    illegal_targets += 1
                expected_color = "white" if board.turn == chess.WHITE else "black"
                if (
                    row.you_color != expected_color
                    or row.side_to_move != expected_color
                    or row.side_to_move != row.you_color
                ):
                    side_mismatches += 1
                if row.target_move_uci != row.move:
                    target_alias_mismatches += 1
                if row.time_class != "rapid":
                    non_rapid_samples += 1
                if not bool(row.rated):
                    unrated_samples += 1
                prefix_moves = str(row.uci_prefix_before).split()
                if len(prefix_moves) != int(row.ply_prefix_before):
                    ply_prefix_mismatches += 1
                replay = chess.Board()
                replay_valid = True
                for move_uci in prefix_moves:
                    try:
                        prefix_move = chess.Move.from_uci(move_uci)
                    except ValueError:
                        invalid_prefix_moves += 1
                        replay_valid = False
                        break
                    if prefix_move not in replay.legal_moves:
                        invalid_prefix_moves += 1
                        replay_valid = False
                        break
                    replay.push(prefix_move)
                if replay_valid and position_key(replay.fen()) == position_key(row.fen):
                    prefix_reconstruction_successes += 1
                else:
                    prefix_reconstruction_mismatches += 1

        dates = pd.to_datetime(frame["date_utc"], utc=True) if "date_utc" in frame else pd.Series(dtype="datetime64[ns, UTC]")
        all_dates.extend(dates.tolist())
        date_ordered = bool(dates.is_monotonic_increasing)
        timezone_utc = all(timestamp.tz is not None and str(timestamp.tz) == "UTC" for timestamp in dates)
        game_sets[name] = set(frame["game_id"].astype(str)) if "game_id" in frame else set()
        passed = (
            not missing_columns
            and not any(null_counts.values())
            and duplicate_keys == 0
            and invalid_fens == 0
            and illegal_targets == 0
            and side_mismatches == 0
            and prefix_reconstruction_mismatches == 0
            and invalid_prefix_moves == 0
            and ply_prefix_mismatches == 0
            and target_alias_mismatches == 0
            and non_rapid_samples == 0
            and unrated_samples == 0
            and date_ordered
            and timezone_utc
        )
        all_valid = all_valid and passed
        details[name] = {
            "samples": int(len(frame)),
            "games": int(frame["game_id"].nunique()) if "game_id" in frame else 0,
            "grain": "one row per Ben move",
            "missing_columns": missing_columns,
            "null_counts": null_counts,
            "duplicate_game_ply_keys": duplicate_keys,
            "invalid_fens": invalid_fens,
            "illegal_target_moves": illegal_targets,
            "side_to_move_mismatches": side_mismatches,
            "prefix_reconstruction_successes": prefix_reconstruction_successes,
            "prefix_reconstruction_mismatches": prefix_reconstruction_mismatches,
            "prefix_reconstruction_coverage": round(
                prefix_reconstruction_successes / len(frame), 6
            )
            if len(frame)
            else None,
            "invalid_prefix_moves": invalid_prefix_moves,
            "ply_prefix_mismatches": ply_prefix_mismatches,
            "target_alias_mismatches": target_alias_mismatches,
            "non_rapid_samples": non_rapid_samples,
            "unrated_samples": unrated_samples,
            "date_min": str(dates.min()) if len(dates) else None,
            "date_max": str(dates.max()) if len(dates) else None,
            "date_ordered": date_ordered,
            "timezone_utc": timezone_utc,
            "passed": passed,
        }

    overlaps: dict[str, int] = {}
    for index, left in enumerate(expected):
        for right in expected[index + 1 :]:
            overlaps[f"{left}_{right}"] = len(game_sets[left] & game_sets[right])
    boundary_checks = {
        f"{left}_before_{right}": pd.to_datetime(splits[left]["date_utc"], utc=True).max()
        <= pd.to_datetime(splits[right]["date_utc"], utc=True).min()
        for left, right in zip(expected, expected[1:])
    }
    passed = all_valid and not any(overlaps.values()) and all(boundary_checks.values())
    return {
        "passed": passed,
        "expected_grain": "one unique (game_id, ply_index) row for each rated-rapid move made by Ben",
        "as_of_utc": str(max(all_dates)) if all_dates else None,
        "splits": details,
        "game_id_overlap_counts": overlaps,
        "chronological_boundaries": boundary_checks,
        "stable_boundary_tie_breaker": "game_id ascending when date_utc ties",
    }


def select_safety_subset(
    source: pd.DataFrame,
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> tuple[pd.DataFrame, list[Mapping[str, Any]]]:
    if len(source) != len(prediction_records):
        raise ValueError("Source and prediction records differ in length")
    if limit is None or limit >= len(source):
        indices = list(range(len(source)))
    else:
        if limit < 1:
            raise ValueError("limit must be positive")
        indices = sorted(random.Random(seed).sample(range(len(source)), limit))
    return source.iloc[indices].reset_index(drop=True), [prediction_records[index] for index in indices]


def select_policy_move(
    probabilities: Mapping[str, float],
    *,
    mode: str,
    top_k: int,
    temperature: float,
    rng: random.Random,
) -> str:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    adjusted = sorted_probabilities(apply_temperature(dict(probabilities), temperature))
    moves = list(adjusted)[:top_k]
    if mode == "argmax":
        return moves[0]
    if mode == "sample":
        return rng.choices(moves, weights=[adjusted[move] for move in moves], k=1)[0]
    raise ValueError("mode must be 'argmax' or 'sample'")


def summarize_safety_records(
    records: Sequence[Mapping[str, Any]],
    *,
    attempted_samples: int | None = None,
) -> dict[str, Any]:
    attempted = attempted_samples if attempted_samples is not None else len(records)
    successful = len(records)
    if attempted < successful:
        raise ValueError("attempted_samples cannot be smaller than successful records")
    if not records:
        raise ValueError("No successful safety evaluation records")

    vetoed = [row for row in records if row["vetoed"]]
    original_losses = [float(row["original_centipawn_loss"]) for row in records]
    selected_losses = [float(row["selected_centipawn_loss"]) for row in records]
    avoided_losses = [float(row["centipawn_loss_avoided"]) for row in records]
    mate_scale_threshold = 90_000.0
    non_mate_rows = [
        row
        for row in records
        if float(row["original_centipawn_loss"]) < mate_scale_threshold
        and float(row["selected_centipawn_loss"]) < mate_scale_threshold
    ]
    latencies = [float(row["safety_latency_ms"]) for row in records]
    at_risk = [row for row in records if row.get("original_draw_or_shuffle_reason")]
    avoided_risk = [
        row for row in at_risk if not row.get("selected_draw_or_shuffle_reason")
    ]
    memory_records = [row for row in records if row.get("memory_source") != "maia2"]
    reasons = Counter(str(row["reason"]) for row in vetoed)
    positive_avoidance = sum(value > 0 for value in avoided_losses)
    for row in records:
        if not row["vetoed"] and (
            row["original_move"] != row["selected_move"]
            or float(row["original_centipawn_loss"]) != float(row["selected_centipawn_loss"])
            or float(row["centipawn_loss_avoided"]) != 0.0
        ):
            raise AssertionError("Non-veto safety record changed the move or centipawn loss")
    if positive_avoidance > len(vetoed):
        raise AssertionError("Positive centipawn-loss avoidance cannot exceed veto count")

    before_matches = sum(row["original_move"] == row["actual_move"] for row in records)
    after_matches = sum(row["selected_move"] == row["actual_move"] for row in records)
    memory_preserved = sum(
        row["selected_move"] == row["original_move"] for row in memory_records
    )
    return {
        "attempted_samples": attempted,
        "successful_samples": successful,
        "failed_samples": attempted - successful,
        "failure_rate": round((attempted - successful) / attempted, 6) if attempted else 0.0,
        "veto": {
            "vetoed_samples": len(vetoed),
            "veto_rate": round(len(vetoed) / successful, 6),
            "reasons": dict(sorted(reasons.items())),
        },
        "centipawn_loss": {
            "definition": "loss versus the best policy candidate in the same Stockfish evaluation pass",
            "before_safety": distribution(original_losses),
            "after_safety": distribution(selected_losses),
            "avoided": distribution(avoided_losses),
            "avoided_on_vetoed_samples": distribution(
                [float(row["centipawn_loss_avoided"]) for row in vetoed]
            ),
            "avoided_on_positive_samples": distribution(
                [value for value in avoided_losses if value > 0]
            ),
            "positive_avoidance_samples": positive_avoidance,
            "mate_scale_threshold": mate_scale_threshold,
            "mate_scale_before_samples": sum(value >= mate_scale_threshold for value in original_losses),
            "mate_scale_after_samples": sum(value >= mate_scale_threshold for value in selected_losses),
            "non_mate_samples": len(non_mate_rows),
            "non_mate_before_safety": distribution(
                [float(row["original_centipawn_loss"]) for row in non_mate_rows]
            ),
            "non_mate_after_safety": distribution(
                [float(row["selected_centipawn_loss"]) for row in non_mate_rows]
            ),
            "non_mate_avoided": distribution(
                [float(row["centipawn_loss_avoided"]) for row in non_mate_rows]
            ),
            "non_mate_positive_avoided": distribution(
                [
                    float(row["centipawn_loss_avoided"])
                    for row in non_mate_rows
                    if float(row["centipawn_loss_avoided"]) > 0
                ]
            ),
        },
        "draw_shuffle_avoidance": {
            "at_risk_samples": len(at_risk),
            "avoided_samples": len(avoided_risk),
            "avoidance_rate": round(len(avoided_risk) / len(at_risk), 6) if at_risk else None,
        },
        "added_latency_ms": distribution(latencies),
        "personalization_impact": {
            "definition": "agreement with Ben's actual held-out move before versus after safety",
            "ben_match_before_samples": before_matches,
            "ben_match_after_samples": after_matches,
            "ben_match_before_rate": round(before_matches / successful, 6),
            "ben_match_after_rate": round(after_matches / successful, 6),
            "ben_match_rate_change": round((after_matches - before_matches) / successful, 6),
            "original_choice_preserved_rate": round(
                sum(row["selected_move"] == row["original_move"] for row in records) / successful,
                6,
            ),
            "memory_covered_samples": len(memory_records),
            "memory_choice_preserved_rate": round(memory_preserved / len(memory_records), 6)
            if memory_records
            else None,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
