from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

import chess
import pandas as pd

from chesscom_pipeline import write_splits
from engine_safety import StockfishBlunderVeto
from evaluation_pipeline import (
    LOG_LOSS_EPSILON,
    audit_splits,
    evaluate_policy_probabilities,
    select_policy_move,
    summarize_safety_records,
)
from personalized_policy import PersonalizedMaia2Policy, PolicyMove
from run_evaluation import (
    VALIDATED_ALPHA,
    VALIDATED_MIN_COUNT,
    VALIDATED_STRATEGY,
    build_parser,
    safety_reproducibility_metadata,
)


def sample_row(
    game_id: str,
    date: str,
    fen: str,
    move: str,
    color: str,
    prefix: str,
    ply: int,
) -> dict:
    return {
        "fen": fen,
        "move": move,
        "elo_self": 1600,
        "elo_oppo": 1600,
        "game_id": game_id,
        "date_utc": pd.Timestamp(date, tz="UTC"),
        "ply_index": ply,
        "ply_prefix_before": ply - 1,
        "fullmove_number": (ply + 1) // 2,
        "uci_prefix_before": prefix,
        "you_color": color,
        "target_move_uci": move,
        "side_to_move": color,
        "time_class": "rapid",
        "rated": True,
    }


class ChronologicalSplitTests(unittest.TestCase):
    def test_split_assigns_complete_games_and_is_deterministic(self) -> None:
        rows = []
        for index in range(10):
            for ply in (1, 3):
                rows.append(
                    sample_row(
                        f"g{index:02}",
                        f"2026-01-{index + 1:02} 12:00:00",
                        chess.STARTING_FEN,
                        "e2e4",
                        "white",
                        "",
                        ply,
                    )
                )
        source = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first = write_splits(base, "chron", source, seed=42)
            frames = {
                name: pd.read_parquet(base / "chron" / f"{name}.parquet")
                for name in ("train", "val", "test")
            }
            second = write_splits(base, "chron", source, seed=999)

        self.assertEqual(first, second)  # seed does not affect chronology
        self.assertEqual({name: stats["games"] for name, stats in first.items()}, {"train": 7, "val": 1, "test": 2})
        game_sets = {name: set(frame.game_id) for name, frame in frames.items()}
        self.assertFalse(game_sets["train"] & game_sets["val"])
        self.assertFalse(game_sets["train"] & game_sets["test"])
        self.assertFalse(game_sets["val"] & game_sets["test"])
        self.assertLessEqual(frames["train"].date_utc.max(), frames["val"].date_utc.min())
        self.assertLessEqual(frames["val"].date_utc.max(), frames["test"].date_utc.min())
        self.assertTrue(all(frame.groupby("game_id").size().eq(2).all() for frame in frames.values()))


class PolicyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        board_after_e4 = chess.Board()
        board_after_e4.push_uci("e2e4")
        board_after_d4 = chess.Board()
        board_after_d4.push_uci("d2d4")
        self.source = pd.DataFrame(
            [
                sample_row("g1", "2026-01-01", chess.STARTING_FEN, "e2e4", "white", "", 1),
                sample_row("g2", "2026-01-02", board_after_e4.fen(), "c7c5", "black", "e2e4", 2),
                sample_row("g3", "2026-01-03", board_after_d4.fen(), "d7d5", "black", "d2d4", 2),
            ]
        )
        self.base = [
            {"d2d4": 0.6, "e2e4": 0.4},
            {"e7e5": 0.7, "c7c5": 0.3},
            {"g8f6": 0.5, "d7d5": 0.5},
        ]
        self.books = {
            "metadata": {},
            "fen_book": {},
            "prefix_book": {
                "white|": {"e2e4": 3, "d2d4": 1},
                "black|e2e4": {"c7c5": 1},
            },
        }

    def test_policy_metrics_cover_accuracy_loss_calibration_and_groups(self) -> None:
        summary, records = evaluate_policy_probabilities(
            self.source, self.base, self.books, strategy="combined", alpha=0.7, min_count=1
        )
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["coverage"]["personal_memory_samples"], 2)
        self.assertAlmostEqual(summary["coverage"]["personal_memory_rate"], 2 / 3, places=6)
        self.assertEqual(summary["personalization"]["top1_changed_samples"], 2)
        self.assertEqual(summary["personalization"]["top1_improved_samples"], 2)
        self.assertEqual(sum(group["samples"] for group in summary["by_memory_source"].values()), 3)
        self.assertEqual(sum(group["samples"] for group in summary["by_memory_count"].values()), 3)
        self.assertTrue(summary["reconciliation"]["weighted_metrics_passed"])
        for policy in ("base_maia2", "personalized"):
            calibration = summary[policy]["top1_calibration"]
            self.assertEqual(calibration["denominator"], 3)
            self.assertEqual(sum(bucket["count"] for bucket in calibration["bins"]), 3)
            self.assertGreaterEqual(summary[policy]["multiclass_brier_score"], 0.0)
        self.assertEqual({row["memory_count_bucket"] for row in records}, {"0", "1", "2-4"})

    def test_log_loss_clips_zero_target_probability(self) -> None:
        source = self.source.head(1)
        summary, records = evaluate_policy_probabilities(
            source,
            [{"d2d4": 1.0}],
            {"fen_book": {}, "prefix_book": {}},
        )
        self.assertAlmostEqual(records[0]["base_log_loss"], -math.log(LOG_LOSS_EPSILON))
        self.assertEqual(records[0]["base_brier_score"], 2.0)
        self.assertEqual(summary["base_maia2"]["target_support_coverage"], 0.0)

    def test_seeded_sampling_repeats_but_argmax_needs_no_sampling(self) -> None:
        probabilities = {"e2e4": 0.4, "d2d4": 0.35, "g1f3": 0.25}
        first_rng = random.Random(17)
        second_rng = random.Random(17)
        first = [
            select_policy_move(
                probabilities, mode="sample", top_k=3, temperature=0.8, rng=first_rng
            )
            for _ in range(20)
        ]
        second = [
            select_policy_move(
                probabilities, mode="sample", top_k=3, temperature=0.8, rng=second_rng
            )
            for _ in range(20)
        ]
        self.assertEqual(first, second)
        self.assertEqual(
            select_policy_move(
                probabilities, mode="argmax", top_k=3, temperature=1.0, rng=random.Random()
            ),
            "e2e4",
        )

        policy = PersonalizedMaia2Policy(
            maia_model=object(),
            prepared=[],
            fen_book={},
            prefix_book={},
            inference_each=lambda *_: (probabilities, None),
        )
        live_first = random.Random(23)
        live_second = random.Random(23)
        sequence_one = [
            policy.choose_move(
                chess.STARTING_FEN,
                1600,
                1600,
                "white",
                mode="sample",
                top_k=3,
                rng=live_first,
            ).move
            for _ in range(20)
        ]
        sequence_two = [
            policy.choose_move(
                chess.STARTING_FEN,
                1600,
                1600,
                "white",
                mode="sample",
                top_k=3,
                rng=live_second,
            ).move
            for _ in range(20)
        ]
        self.assertEqual(sequence_one, sequence_two)


class QualityAndSafetyTests(unittest.TestCase):
    def test_cli_defaults_and_safety_determinism_scope_are_explicit(self) -> None:
        policy_args = build_parser().parse_args(["policy"])
        self.assertEqual(policy_args.strategy, VALIDATED_STRATEGY)
        self.assertEqual(policy_args.alpha, VALIDATED_ALPHA)
        self.assertEqual(policy_args.min_count, VALIDATED_MIN_COUNT)
        self.assertEqual(
            (policy_args.strategy, policy_args.alpha, policy_args.min_count),
            ("fen", 0.7, 1),
        )

        metadata = safety_reproducibility_metadata(42)
        self.assertEqual(metadata["seed"], 42)
        self.assertFalse(metadata["deterministic_evaluation"])
        self.assertTrue(metadata["seeded_subset_selection_reproducible"])
        self.assertTrue(metadata["seeded_policy_sampling_reproducible"])
        self.assertFalse(metadata["engine_output_deterministic"])
        self.assertIn("Stockfish", metadata["determinism_scope"])

    def test_quality_gate_reconciles_grain_legality_timezone_and_overlap(self) -> None:
        frames = {}
        for index, name in enumerate(("train", "val", "test")):
            frames[name] = pd.DataFrame(
                [
                    sample_row(
                        f"{name}-game",
                        f"2026-02-0{index + 1}",
                        chess.STARTING_FEN,
                        "e2e4",
                        "white",
                        "",
                        1,
                    )
                ]
            )
        audit = audit_splits(frames)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["game_id_overlap_counts"], {"train_val": 0, "train_test": 0, "val_test": 0})
        self.assertTrue(all(split["timezone_utc"] for split in audit["splits"].values()))

        frames["test"].loc[0, "game_id"] = "train-game"
        failed = audit_splits(frames)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["game_id_overlap_counts"]["train_test"], 1)

    def test_safety_summary_reports_all_requested_denominators(self) -> None:
        records = [
            {
                "vetoed": True,
                "reason": "vetoed_a_delta_500_cp",
                "original_centipawn_loss": 500,
                "selected_centipawn_loss": 20,
                "centipawn_loss_avoided": 480,
                "safety_latency_ms": 12,
                "original_draw_or_shuffle_reason": "repeated_position",
                "selected_draw_or_shuffle_reason": None,
                "memory_source": "prefix",
                "original_move": "a2a3",
                "selected_move": "e2e4",
                "actual_move": "e2e4",
            },
            {
                "vetoed": False,
                "reason": "within_threshold",
                "original_centipawn_loss": 10,
                "selected_centipawn_loss": 10,
                "centipawn_loss_avoided": 0,
                "safety_latency_ms": 8,
                "original_draw_or_shuffle_reason": None,
                "selected_draw_or_shuffle_reason": None,
                "memory_source": "maia2",
                "original_move": "d2d4",
                "selected_move": "d2d4",
                "actual_move": "d2d4",
            },
        ]
        summary = summarize_safety_records(records, attempted_samples=3)
        self.assertEqual(summary["successful_samples"], 2)
        self.assertEqual(summary["failed_samples"], 1)
        self.assertEqual(summary["veto"]["veto_rate"], 0.5)
        self.assertEqual(summary["draw_shuffle_avoidance"]["avoidance_rate"], 1.0)
        self.assertEqual(summary["centipawn_loss"]["avoided"]["mean"], 240.0)
        self.assertEqual(summary["added_latency_ms"]["p50"], 10.0)
        self.assertEqual(summary["personalization_impact"]["ben_match_before_samples"], 1)
        self.assertEqual(summary["personalization_impact"]["ben_match_after_samples"], 2)

    def test_same_pass_original_eval_prevents_non_veto_avoidance_contamination(self) -> None:
        board = chess.Board()
        original = PolicyMove("e2e4", 0.6, 0.6, 0.0, "maia2", 0)
        alternative = PolicyMove("d2d4", 0.4, 0.4, 0.0, "maia2", 0)
        veto = object.__new__(StockfishBlunderVeto)
        veto.veto_cp = 100
        veto.avoid_draw_when_winning_cp = 100
        veto.avoid_shuffle_cp = 150
        veto.evaluate_board = lambda _board, _turn: 0
        veto.evaluate_after_move = lambda _board, move, _turn: {"e2e4": -250, "d2d4": 20}[move]
        veto.drawish_reason = lambda _board: None
        veto.shuffle_reason = lambda _board, _move: None

        decision = veto.choose_safe_move(board, [original, alternative])
        self.assertTrue(decision.vetoed)
        self.assertEqual(decision.original_eval_cp, -250)
        self.assertEqual(decision.selected_eval_cp, 20)

        contaminated = [
            {
                "vetoed": False,
                "reason": "within_threshold",
                "original_centipawn_loss": 9,
                "selected_centipawn_loss": 0,
                "centipawn_loss_avoided": 9,
                "safety_latency_ms": 1,
                "original_draw_or_shuffle_reason": None,
                "selected_draw_or_shuffle_reason": None,
                "memory_source": "maia2",
                "original_move": "e2e4",
                "selected_move": "e2e4",
                "actual_move": "e2e4",
            }
        ]
        with self.assertRaisesRegex(AssertionError, "Non-veto"):
            summarize_safety_records(contaminated)


if __name__ == "__main__":
    unittest.main()
