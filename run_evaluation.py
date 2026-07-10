#!/usr/bin/env python3
"""Reproducible held-out Maia2 personalization and Stockfish safety evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import chess
import pandas as pd

from build_personal_books import build_books, training_fingerprint
from engine_safety import StockfishBlunderVeto
from evaluation_pipeline import (
    DEFAULT_SEED,
    audit_splits,
    evaluate_policy_probabilities,
    read_jsonl,
    sample_key,
    seed_everything,
    select_policy_move,
    select_safety_subset,
    sorted_probabilities,
    summarize_safety_records,
    write_json,
    write_jsonl,
)
from personalized_policy import PolicyMove, apply_temperature, position_key


DEFAULT_TRAIN = Path("data/processed/splits/chron/train.parquet")
DEFAULT_VAL = Path("data/processed/splits/chron/val.parquet")
DEFAULT_TEST = Path("data/processed/splits/chron/test.parquet")
VALIDATED_STRATEGY = "fen"
VALIDATED_ALPHA = 0.7
VALIDATED_MIN_COUNT = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "maia2": package_version("maia2"),
            "torch": package_version("torch"),
            "python-chess": package_version("chess"),
            "pandas": package_version("pandas"),
        },
    }


def safety_reproducibility_metadata(seed: int) -> dict[str, Any]:
    """Scope reproducibility honestly for a finite-time Stockfish study."""
    return {
        "seed": seed,
        "deterministic_evaluation": False,
        "seeded_subset_selection_reproducible": True,
        "seeded_policy_sampling_reproducible": True,
        "engine_output_deterministic": False,
        "determinism_scope": (
            "The seed fixes held-out subset selection and policy sampling only; "
            "finite-time Stockfish scores, vetoes, and latency remain timing- and hardware-dependent."
        ),
    }


def load_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(args.train_split).reset_index(drop=True),
        "val": pd.read_parquet(args.val_split).reset_index(drop=True),
        "test": pd.read_parquet(args.test_split).reset_index(drop=True),
    }


def run_maia_predictions(
    source: pd.DataFrame,
    *,
    model_type: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> list[dict[str, float]]:
    seed_everything(seed)
    from maia2 import inference, model

    required = ["fen", "move", "elo_self", "elo_oppo"]
    maia_model = model.from_pretrained(model_type, device)
    predictions, _ = inference.inference_batch(
        source[required].copy(),
        maia_model,
        verbose=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return [sorted_probabilities(probabilities) for probabilities in predictions["move_probs"]]


def load_base_prediction_cache(path: Path, source: pd.DataFrame) -> list[dict[str, float]]:
    cached = read_jsonl(path)
    if len(cached) != len(source):
        raise ValueError(f"Prediction cache has {len(cached)} rows; split has {len(source)}")
    probabilities: list[dict[str, float]] = []
    for row, record in zip(source.itertuples(index=False), cached):
        expected = sample_key(row)
        if record.get("sample_key") != expected:
            raise ValueError(f"Prediction cache key mismatch: expected {expected}")
        probabilities.append(sorted_probabilities(record["base_probabilities"]))
    return probabilities


def write_base_prediction_cache(
    path: Path,
    source: pd.DataFrame,
    probabilities: Sequence[Mapping[str, float]],
) -> None:
    records = [
        {"sample_key": sample_key(row), "base_probabilities": dict(probs)}
        for row, probs in zip(source.itertuples(index=False), probabilities)
    ]
    write_jsonl(path, records)


def policy_report(summary: Mapping[str, Any]) -> str:
    base = summary["base_maia2"]
    personal = summary["personalized"]
    coverage = summary["coverage"]
    ben = summary["personalization"]["ben_likeness"]
    lines = [
        "# Held-out personalization evaluation",
        "",
        "The evaluation uses a game-isolated chronological split. Hyperparameters must be selected on validation; this report evaluates the configured setting once on test.",
        "",
        "| metric | base Maia2 | personalized | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| top-1 accuracy | {base['top1_accuracy']:.4f} | {personal['top1_accuracy']:.4f} | {personal['top1_accuracy'] - base['top1_accuracy']:+.4f} |",
        f"| top-3 accuracy | {base['top3_accuracy']:.4f} | {personal['top3_accuracy']:.4f} | {personal['top3_accuracy'] - base['top3_accuracy']:+.4f} |",
        f"| log loss | {base['log_loss']:.4f} | {personal['log_loss']:.4f} | {personal['log_loss'] - base['log_loss']:+.4f} |",
        f"| multiclass Brier score | {base['multiclass_brier_score']:.4f} | {personal['multiclass_brier_score']:.4f} | {personal['multiclass_brier_score'] - base['multiclass_brier_score']:+.4f} |",
        f"| top-1 ECE (10 bins) | {base['top1_calibration']['ece']:.4f} | {personal['top1_calibration']['ece']:.4f} | {personal['top1_calibration']['ece'] - base['top1_calibration']['ece']:+.4f} |",
        f"| mean probability on Ben move | {base['mean_target_probability']:.4f} | {personal['mean_target_probability']:.4f} | {ben['target_probability_lift']:+.4f} |",
        "",
        "## Coverage and choice changes",
        "",
        f"- Test samples: **{summary['samples']}**.",
        f"- Exact personal-memory coverage: **{coverage['personal_memory_samples']}/{coverage['denominator']} ({coverage['personal_memory_rate']:.2%})**.",
        f"- Base target-support coverage: **{coverage['base_target_support_rate']:.2%}**.",
        f"- Personalization changed top-1 on **{summary['personalization']['top1_changed_samples']} samples ({summary['personalization']['top1_change_rate']:.2%})**.",
        f"- Changed decisions improved {summary['personalization']['top1_improved_samples']} and worsened {summary['personalization']['top1_worsened_samples']} held-out top-1 matches.",
        "",
        "## Data-quality gates",
        "",
        f"- Passed: **{summary['data_quality']['passed']}**.",
        f"- Game overlap counts: `{json.dumps(summary['data_quality']['game_id_overlap_counts'], sort_keys=True)}`.",
        f"- Data as of: **{summary['data_quality']['as_of_utc']}**.",
        "",
        "## Caveats",
        "",
        "- Coverage is exact prefix/FEN count-book coverage, not semantic position similarity; most middlegame/endgame positions remain base Maia2.",
        "- Log loss clips target probabilities at the epsilon recorded in `summary.json`.",
        "- Top-1 ECE is confidence calibration only; with a few thousand personal games, per-bin estimates can be noisy and it is not full multiclass calibration.",
        "- This observational next-move evaluation does not establish game win-rate improvement.",
        "- GPU kernels can retain platform-level nondeterminism; the published run records its seed and device.",
        "",
    ]
    if summary.get("validation_tuning"):
        tuning = summary["validation_tuning"]
        selected = tuning["selected"]
        lines.extend(
            [
                "## Validation-only hyperparameter selection",
                "",
                f"- Selection rule: {tuning['selection_rule']}.",
                f"- Selected `{selected['strategy']}` / alpha `{selected['alpha']}` / min-count `{selected['min_count']}` from {len(tuning['candidates'])} candidates.",
                f"- Validation log loss/top-1: `{selected['log_loss']:.4f}` / `{selected['top1_accuracy']:.4f}`.",
                "- The held-out test split was not used for this selection.",
                "",
            ]
        )
    lines.extend(
        [
            "## Chronological split evidence",
            "",
            "| split | games | Ben-move samples | start UTC | end UTC | prefix replay |",
            "| --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for split_name in ("train", "val", "test"):
        split = summary["data_quality"]["splits"][split_name]
        lines.append(
            f"| {split_name} | {split['games']} | {split['samples']} | {split['date_min']} | {split['date_max']} | {split['prefix_reconstruction_coverage']:.2%} |"
        )

    def add_slice_table(title: str, groups: Mapping[str, Any]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| slice | n | memory coverage | change rate | base top-1 | personal top-1 | base top-3 | personal top-3 | base log loss | personal log loss |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, group in groups.items():
            group_coverage = group["memory_covered_samples"] / group["samples"]
            lines.append(
                f"| {name} | {group['samples']} | {group_coverage:.2%} | {group['top1_change_rate']:.2%} | "
                f"{group['base_maia2']['top1_accuracy']:.4f} | {group['personalized']['top1_accuracy']:.4f} | "
                f"{group['base_maia2']['top3_accuracy']:.4f} | {group['personalized']['top3_accuracy']:.4f} | "
                f"{group['base_maia2']['log_loss']:.4f} | {group['personalized']['log_loss']:.4f} |"
            )

    add_slice_table("Performance by memory source", summary["by_memory_source"])
    add_slice_table(
        "Performance by memory observation count",
        {
            key: summary["by_memory_count"][key]
            for key in ("0", "1", "2-4", "5-9", "10+")
            if key in summary["by_memory_count"]
        },
    )
    lines.append("")
    return "\n".join(lines)


def run_policy(args: argparse.Namespace) -> None:
    splits = load_splits(args)
    quality = audit_splits(splits)
    if not quality["passed"]:
        raise ValueError("Dataset quality audit failed; refusing to publish evaluation")

    train = splits["train"]
    source = splits[args.eval_split_name]
    full_eval_samples = len(source)
    if args.limit is not None:
        source = source.head(args.limit).reset_index(drop=True)

    if args.books:
        books = json.loads(Path(args.books).read_text(encoding="utf-8"))
        expected_hash = training_fingerprint(train)
        actual_hash = books.get("metadata", {}).get("training_sha256")
        if actual_hash != expected_hash:
            raise ValueError(
                "Personal books do not match the chronological training split; rebuild them first"
            )
    else:
        books = build_books(train, max_prefix_ply=args.max_prefix_ply)
    if args.books_output:
        write_json(args.books_output, books)

    def predictions_for(
        frame: pd.DataFrame,
        cache_in: Path | None,
        cache_out: Path | None,
    ) -> list[dict[str, float]]:
        if cache_in:
            probabilities = load_base_prediction_cache(cache_in, frame)
        else:
            probabilities = run_maia_predictions(
                frame,
                model_type=args.model_type,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
        if cache_out:
            write_base_prediction_cache(cache_out, frame, probabilities)
        return probabilities

    tuning: dict[str, Any] | None = None
    selected_strategy = args.strategy
    selected_alpha = args.alpha
    selected_min_count = args.min_count
    if args.tune_on_validation:
        if args.eval_split_name != "test":
            raise ValueError("Validation tuning is only valid before a final test evaluation")
        validation = splits["val"]
        if args.limit is not None:
            validation = validation.head(args.limit).reset_index(drop=True)
        validation_probabilities = predictions_for(
            validation, args.val_base_predictions_in, args.val_base_predictions_out
        )
        candidates: list[dict[str, Any]] = []
        for strategy in args.strategies:
            for alpha in args.alphas:
                for min_count in args.min_counts:
                    candidate_summary, _ = evaluate_policy_probabilities(
                        validation,
                        validation_probabilities,
                        books,
                        strategy=strategy,
                        alpha=alpha,
                        min_count=min_count,
                    )
                    metrics = candidate_summary["personalized"]
                    candidates.append(
                        {
                            "strategy": strategy,
                            "alpha": alpha,
                            "min_count": min_count,
                            "samples": candidate_summary["samples"],
                            "coverage": candidate_summary["coverage"]["personal_memory_rate"],
                            "top1_accuracy": metrics["top1_accuracy"],
                            "top3_accuracy": metrics["top3_accuracy"],
                            "log_loss": metrics["log_loss"],
                            "multiclass_brier_score": metrics["multiclass_brier_score"],
                            "top1_ece": metrics["top1_calibration"]["ece"],
                            "top1_change_rate": candidate_summary["personalization"]["top1_change_rate"],
                        }
                    )
        selected = min(
            candidates,
            key=lambda item: (
                item["log_loss"],
                item["multiclass_brier_score"],
                -item["top1_accuracy"],
                -item["top3_accuracy"],
                item["alpha"],
                -item["min_count"],
                item["strategy"],
            ),
        )
        selected_strategy = selected["strategy"]
        selected_alpha = selected["alpha"]
        selected_min_count = selected["min_count"]
        tuning = {
            "split": "val",
            "full_split_samples": len(splits["val"]),
            "evaluated_samples": len(validation),
            "is_full_split": len(validation) == len(splits["val"]),
            "selection_rule": "minimum validation log loss; Brier, top-1, top-3, and simpler blend settings break ties",
            "selected": selected,
            "candidates": candidates,
        }

    base_probabilities = predictions_for(
        source, args.base_predictions_in, args.base_predictions_out
    )

    summary, records = evaluate_policy_probabilities(
        source,
        base_probabilities,
        books,
        strategy=selected_strategy,
        alpha=selected_alpha,
        min_count=selected_min_count,
    )
    summary.update(
        {
            "schema_version": 1,
            "evaluation": {
                "split": args.eval_split_name,
                "evaluated_samples": len(source),
                "full_split_samples": full_eval_samples,
                "is_full_split": len(source) == full_eval_samples,
                "model_type": args.model_type,
                "device": args.device,
                "seed": args.seed,
                "deterministic_evaluation": True,
                "sampling_used": False,
                "train_split": str(args.train_split),
                "val_split": str(args.val_split),
                "test_split": str(args.test_split),
            },
            "book_metadata": books.get("metadata", {}),
            "data_quality": quality,
            "validation_tuning": tuning,
            "runtime_provenance": runtime_provenance(),
        }
    )
    model_artifact = Path("maia2_models") / f"{args.model_type}_model.pt"
    if model_artifact.exists():
        summary["runtime_provenance"]["model_artifact"] = {
            "path": str(model_artifact),
            "bytes": model_artifact.stat().st_size,
            "sha256": file_sha256(model_artifact),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    (args.output_dir / "report.md").write_text(policy_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def board_with_history(row: Any) -> chess.Board:
    """Rebuild the move stack; safety metrics are invalid without this history."""
    board = chess.Board()
    prefix_moves = str(row.uci_prefix_before).split()
    if len(prefix_moves) != int(row.ply_prefix_before):
        raise ValueError("Prefix length does not match ply_prefix_before")
    for move_uci in prefix_moves:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal prefix move: {move_uci}")
        board.push(move)
    if position_key(board.fen()) != position_key(row.fen):
        raise ValueError("Prefix does not reconstruct FEN")
    return board


def policy_candidates(
    record: Mapping[str, Any],
    *,
    selected_move: str,
    top_k: int,
    temperature: float,
) -> list[PolicyMove]:
    probabilities = sorted_probabilities(
        apply_temperature(dict(record["personalized_probabilities"]), temperature)
    )
    ordered_moves = list(probabilities)[:top_k]
    if selected_move not in ordered_moves:
        raise AssertionError("Selected move is outside safety candidate set")
    ordered_moves = [selected_move] + [move for move in ordered_moves if move != selected_move]
    return [
        PolicyMove(
            move=move,
            probability=float(probabilities[move]),
            maia_probability=float(record["base_probabilities"].get(move, 0.0)),
            personal_probability=float(record["personal_probabilities"].get(move, 0.0)),
            source=str(record["memory_source"]),
            count=int(record["memory_counts"].get(move, 0)),
        )
        for move in ordered_moves
    ]


def run_stockfish_safety(
    source: pd.DataFrame,
    prediction_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    source, prediction_records = select_safety_subset(
        source, prediction_records, limit=args.limit, seed=args.seed
    )
    rng = random.Random(args.seed)
    safety = StockfishBlunderVeto(
        engine_path=str(args.stockfish_path),
        veto_cp=args.veto_cp,
        engine_time=args.engine_time,
        avoid_draw_when_winning_cp=args.avoid_draw_when_winning_cp,
        avoid_shuffle_cp=args.avoid_shuffle_cp,
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    engine_id = dict(safety.engine.id)
    try:
        for row, prediction in zip(source.itertuples(index=False), prediction_records):
            try:
                if prediction["sample_key"] != sample_key(row):
                    raise ValueError("Prediction/source sample key mismatch")
                selected_move = select_policy_move(
                    prediction["personalized_probabilities"],
                    mode=args.mode,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    rng=rng,
                )
                candidates = policy_candidates(
                    prediction,
                    selected_move=selected_move,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
                board = board_with_history(row)
                original = candidates[0]
                original_risk = safety.drawish_reason(safety.board_after_move(board, original.move))
                if original_risk is None:
                    original_risk = safety.shuffle_reason(board, original.move)

                started = time.perf_counter()
                decision = safety.choose_safe_move(board, candidates)
                safety_latency_ms = (time.perf_counter() - started) * 1000.0
                original_eval = int(decision.original_eval_cp)
                selected_eval = int(decision.selected_eval_cp)
                reference_eval = max(int(decision.best_eval_cp), original_eval, selected_eval)

                selected_risk = safety.drawish_reason(
                    safety.board_after_move(board, decision.selected.move)
                )
                if selected_risk is None:
                    selected_risk = safety.shuffle_reason(board, decision.selected.move)
                original_loss = max(0, reference_eval - original_eval)
                selected_loss = max(0, reference_eval - selected_eval)
                avoided_loss = original_loss - selected_loss
                if not decision.vetoed:
                    if decision.selected.move != original.move or original_eval != selected_eval:
                        raise AssertionError("Non-veto decision changed move or same-pass evaluation")
                    if original_loss != selected_loss or avoided_loss != 0:
                        raise AssertionError("Non-veto decision cannot avoid centipawn loss")
                records.append(
                    {
                        "sample_key": prediction["sample_key"],
                        "game_id": str(row.game_id),
                        "ply_index": int(row.ply_index),
                        "actual_move": row.move,
                        "memory_source": prediction["memory_source"],
                        "memory_count": prediction["memory_count"],
                        "original_move": original.move,
                        "selected_move": decision.selected.move,
                        "vetoed": bool(decision.vetoed),
                        "reason": decision.reason,
                        "original_eval_cp": original_eval,
                        "selected_eval_cp": selected_eval,
                        "reference_eval_cp": reference_eval,
                        "original_centipawn_loss": original_loss,
                        "selected_centipawn_loss": selected_loss,
                        "centipawn_loss_avoided": avoided_loss,
                        "original_draw_or_shuffle_reason": original_risk,
                        "selected_draw_or_shuffle_reason": selected_risk,
                        "safety_latency_ms": safety_latency_ms,
                    }
                )
            except Exception as exc:
                failures.append(
                    {"sample_key": str(prediction.get("sample_key", "unknown")), "error": repr(exc)}
                )
                if args.fail_fast:
                    raise
    finally:
        safety.close()

    summary = summarize_safety_records(records, attempted_samples=len(source))
    summary.update(
        {
            "schema_version": 1,
            "configuration": {
                **safety_reproducibility_metadata(args.seed),
                "subset_selection": "seeded uniform sample without replacement",
                "requested_limit": args.limit,
                "mode": args.mode,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "veto_cp": args.veto_cp,
                "engine_time_seconds": args.engine_time,
                "avoid_draw_when_winning_cp": args.avoid_draw_when_winning_cp,
                "avoid_shuffle_cp": args.avoid_shuffle_cp,
                "stockfish_executable": args.stockfish_path.name,
                "stockfish_executable_bytes": args.stockfish_path.stat().st_size,
                "stockfish_executable_sha256": file_sha256(args.stockfish_path),
                "stockfish_engine_id": engine_id,
                "runtime_provenance": runtime_provenance(),
            },
            "caveats": [
                "Safety results apply only to the recorded seeded held-out subset and engine time budget.",
                "Seed 42 fixes subset selection and policy sampling, not finite-time Stockfish output.",
                "Centipawn values, veto decisions, and latency are timing- and hardware-dependent and can vary across reruns.",
                "Ben-likeness is held-out move agreement, not a causal style or win-rate estimate.",
                "Draw/shuffle rates can be sparse; the report retains the raw numerator and denominator.",
                "Mate scores use python-chess's 100,000-centipawn sentinel; mate-scale and non-mate distributions are reported separately.",
            ],
        }
    )
    return summary, records, failures


def safety_report(summary: Mapping[str, Any]) -> str:
    veto = summary["veto"]
    cp = summary["centipawn_loss"]
    latency = summary["added_latency_ms"]
    draw = summary["draw_shuffle_avoidance"]
    impact = summary["personalization_impact"]
    return "\n".join(
        [
            "# Stockfish safety-layer evaluation",
            "",
            f"Seeded held-out sample: **{summary['successful_samples']}/{summary['attempted_samples']} successful**.",
            "Seed 42 fixes the sampled positions and policy draws; the 50 ms Stockfish searches and latency are not deterministic.",
            "",
            f"- Veto rate: **{veto['vetoed_samples']}/{summary['successful_samples']} ({veto['veto_rate']:.2%})**.",
            f"- Median/p95 centipawn loss before: **{cp['before_safety']['p50']:.2f} / {cp['before_safety']['p95']:.2f}**; after: **{cp['after_safety']['p50']:.2f} / {cp['after_safety']['p95']:.2f}**.",
            f"- Centipawn loss avoided, median/p95: **{cp['avoided']['p50']:.2f} / {cp['avoided']['p95']:.2f}** ({cp['positive_avoidance_samples']} positive samples).",
            f"- Conditional loss avoided on positive samples, median/p95: **{cp['avoided_on_positive_samples']['p50']:.2f} / {cp['avoided_on_positive_samples']['p95']:.2f}**.",
            f"- Non-mate conditional loss avoided, median/p95: **{cp['non_mate_positive_avoided']['p50']:.2f} / {cp['non_mate_positive_avoided']['p95']:.2f}**.",
            f"- Non-mate mean loss before/after: **{cp['non_mate_before_safety']['mean']:.2f} / {cp['non_mate_after_safety']['mean']:.2f}**; mate-scale losses: **{cp['mate_scale_before_samples']} / {cp['mate_scale_after_samples']}**.",
            f"- Added safety latency, median/p95: **{latency['p50']:.2f} / {latency['p95']:.2f} ms**.",
            f"- Draw/shuffle risks avoided: **{draw['avoided_samples']}/{draw['at_risk_samples']}**.",
            f"- Held-out Ben move match before/after: **{impact['ben_match_before_rate']:.2%} / {impact['ben_match_after_rate']:.2%}**.",
            f"- Original personalized choice preserved: **{impact['original_choice_preserved_rate']:.2%}**.",
            f"- Full held-out history replay gate: **{summary['data_quality_evidence']['full_test_samples']}/{summary['data_quality_evidence']['full_test_samples']} positions reconstructed ({summary['data_quality_evidence']['prefix_reconstruction_mismatches']} mismatches)**.",
            "",
            "## Caveats",
            "",
            *[f"- {caveat}" for caveat in summary["caveats"]],
            "",
        ]
    )


def run_safety(args: argparse.Namespace) -> None:
    source = pd.read_parquet(args.eval_split).reset_index(drop=True)
    predictions = read_jsonl(args.policy_records)
    if len(source) != len(predictions):
        raise ValueError("Policy prediction file must cover the complete evaluation split")
    summary, records, failures = run_stockfish_safety(source, predictions, args)
    if args.data_quality:
        quality = json.loads(args.data_quality.read_text(encoding="utf-8"))
        test_quality = quality.get("splits", {}).get("test", {})
        if (
            not quality.get("passed")
            or test_quality.get("samples") != len(source)
            or test_quality.get("prefix_reconstruction_coverage") != 1.0
        ):
            raise ValueError("Safety evaluation requires passing full-test prefix reconstruction evidence")
        summary["data_quality_evidence"] = {
            "path": str(args.data_quality),
            "full_test_samples": len(source),
            "prefix_reconstruction_coverage": 1.0,
            "prefix_reconstruction_mismatches": test_quality["prefix_reconstruction_mismatches"],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    (args.output_dir / "report.md").write_text(safety_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_quality(args: argparse.Namespace) -> None:
    quality = audit_splits(load_splits(args))
    print(json.dumps(quality, indent=2, sort_keys=True))
    if args.output:
        write_json(args.output, quality)
    if not quality["passed"]:
        raise SystemExit(1)


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-split", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-split", type=Path, default=DEFAULT_TEST)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    quality = commands.add_parser("quality", help="Audit chronological split integrity and legal-move grain.")
    add_split_arguments(quality)
    quality.add_argument("--output", type=Path, default=None)
    quality.set_defaults(func=run_quality)

    policy = commands.add_parser("policy", help="Compare personalized policy with base Maia2.")
    add_split_arguments(policy)
    policy.add_argument("--eval-split-name", choices=["val", "test"], default="test")
    policy.add_argument("--model-type", choices=["rapid", "blitz"], default="rapid")
    policy.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    policy.add_argument("--batch-size", type=int, default=128)
    policy.add_argument("--num-workers", type=int, default=0)
    policy.add_argument("--seed", type=int, default=DEFAULT_SEED)
    policy.add_argument("--limit", type=int, default=None, help="Smoke-only row limit; omit for publication.")
    policy.add_argument(
        "--strategy",
        choices=["fen", "prefix", "combined"],
        default=VALIDATED_STRATEGY,
        help=f"Personal-memory strategy (validated default: {VALIDATED_STRATEGY}).",
    )
    policy.add_argument(
        "--alpha",
        type=float,
        default=VALIDATED_ALPHA,
        help=f"Personal blend weight (validated default: {VALIDATED_ALPHA}).",
    )
    policy.add_argument(
        "--min-count",
        type=int,
        default=VALIDATED_MIN_COUNT,
        help=f"Minimum memory count (validated default: {VALIDATED_MIN_COUNT}).",
    )
    policy.add_argument(
        "--tune-on-validation",
        action="store_true",
        help="Select strategy/alpha/min-count on validation before evaluating test.",
    )
    policy.add_argument(
        "--strategies", nargs="+", choices=["fen", "prefix", "combined"], default=["fen", "prefix", "combined"]
    )
    policy.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5, 0.7])
    policy.add_argument("--min-counts", type=int, nargs="+", default=[1, 2, 3])
    policy.add_argument("--max-prefix-ply", type=int, default=20)
    policy.add_argument("--books", type=Path, default=None)
    policy.add_argument("--books-output", type=Path, default=None)
    policy.add_argument("--base-predictions-in", type=Path, default=None)
    policy.add_argument("--base-predictions-out", type=Path, default=None)
    policy.add_argument("--val-base-predictions-in", type=Path, default=None)
    policy.add_argument("--val-base-predictions-out", type=Path, default=None)
    policy.add_argument("--output-dir", type=Path, default=Path("reports/heldout_evaluation"))
    policy.set_defaults(func=run_policy)

    safety = commands.add_parser("safety", help="Measure Stockfish veto tradeoffs on held-out positions.")
    safety.add_argument("--eval-split", type=Path, default=DEFAULT_TEST)
    safety.add_argument(
        "--policy-records",
        type=Path,
        default=Path("reports/heldout_evaluation/predictions.jsonl"),
    )
    safety.add_argument(
        "--data-quality",
        type=Path,
        default=Path("reports/heldout_evaluation/data_quality.json"),
        help="Passing full-split audit required for repetition/shuffle claims.",
    )
    safety.add_argument("--stockfish-path", type=Path, required=True)
    safety.add_argument("--seed", type=int, default=DEFAULT_SEED)
    safety.add_argument("--limit", type=int, default=200)
    safety.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    safety.add_argument("--top-k", type=int, default=5)
    safety.add_argument("--temperature", type=float, default=1.0)
    safety.add_argument("--veto-cp", type=int, default=400)
    safety.add_argument("--engine-time", type=float, default=0.05)
    safety.add_argument("--avoid-draw-when-winning-cp", type=int, default=400)
    safety.add_argument("--avoid-shuffle-cp", type=int, default=150)
    safety.add_argument("--fail-fast", action="store_true")
    safety.add_argument("--output-dir", type=Path, default=Path("reports/safety_evaluation"))
    safety.set_defaults(func=run_safety)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
