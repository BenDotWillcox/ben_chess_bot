"""Optional Stockfish blunder-veto guardrail for policy moves."""
from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine

from personalized_policy import PolicyMove


@dataclass(frozen=True)
class SafetyDecision:
    selected: PolicyMove
    original: PolicyMove
    vetoed: bool
    reason: str
    best_eval_cp: int | None
    original_eval_cp: int | None
    selected_eval_cp: int | None


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    pov = score.pov(turn)
    mate_score = pov.score(mate_score=100000)
    if mate_score is None:
        raise ValueError("Engine score did not include cp or mate value")
    return int(mate_score)


class StockfishBlunderVeto:
    def __init__(
        self,
        engine_path: str,
        veto_cp: int = 400,
        engine_time: float = 0.05,
        engine_timeout: float = 10.0,
        avoid_draw_when_winning_cp: int | None = None,
        avoid_shuffle_cp: int = 150,
    ) -> None:
        self.engine_path = engine_path
        self.veto_cp = veto_cp
        self.avoid_draw_when_winning_cp = avoid_draw_when_winning_cp or veto_cp
        self.avoid_shuffle_cp = avoid_shuffle_cp
        self.engine_time = engine_time
        self.engine_timeout = engine_timeout
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=engine_timeout)

    def close(self) -> None:
        self.engine.quit()

    def evaluate_after_move(self, board: chess.Board, move_uci: str, root_turn: chess.Color) -> int:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal candidate move: {move_uci}")
        next_board = board.copy(stack=True)
        next_board.push(move)
        terminal_eval = self.terminal_eval(next_board, root_turn)
        if terminal_eval is not None:
            return terminal_eval
        info = self.engine.analyse(next_board, chess.engine.Limit(time=self.engine_time))
        return score_to_cp(info["score"], root_turn)

    def evaluate_board(self, board: chess.Board, root_turn: chess.Color) -> int:
        terminal_eval = self.terminal_eval(board, root_turn)
        if terminal_eval is not None:
            return terminal_eval
        info = self.engine.analyse(board, chess.engine.Limit(time=self.engine_time))
        return score_to_cp(info["score"], root_turn)

    def terminal_eval(self, board: chess.Board, root_turn: chess.Color) -> int | None:
        if board.is_checkmate():
            return -100000 if board.turn == root_turn else 100000
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fivefold_repetition()
            or board.is_seventyfive_moves()
        ):
            return 0
        return None

    def board_after_move(self, board: chess.Board, move_uci: str) -> chess.Board:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal candidate move: {move_uci}")
        next_board = board.copy(stack=True)
        next_board.push(move)
        return next_board

    def drawish_reason(self, board: chess.Board) -> str | None:
        if board.is_stalemate():
            return "stalemate"
        if board.is_insufficient_material():
            return "insufficient_material"
        if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            return "threefold_repetition"
        if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
            return "fifty_move_rule"
        if board.is_repetition(2):
            return "repeated_position"
        return None

    def shuffle_reason(self, board: chess.Board, move_uci: str) -> str | None:
        move = chess.Move.from_uci(move_uci)
        if len(board.move_stack) >= 2:
            previous_own_move = board.move_stack[-2]
            if move.from_square == previous_own_move.to_square and move.to_square == previous_own_move.from_square:
                return "undo_last_own_move"

        next_board = self.board_after_move(board, move_uci)
        if next_board.is_repetition(2):
            return "repeated_position"
        return None

    def engine_policy_move(self, board: chess.Board, root_turn: chess.Color) -> tuple[PolicyMove, int, str | None] | None:
        result = self.engine.play(board, chess.engine.Limit(time=self.engine_time))
        if result.move is None:
            return None
        move_uci = result.move.uci()
        next_board = self.board_after_move(board, move_uci)
        return (
            PolicyMove(
                move=move_uci,
                probability=0.0,
                maia_probability=0.0,
                personal_probability=0.0,
                source="stockfish_veto",
                count=0,
            ),
            self.evaluate_after_move(board, move_uci, root_turn),
            self.drawish_reason(next_board),
        )

    def choose_safe_move(self, board: chess.Board, candidates: list[PolicyMove]) -> SafetyDecision:
        if not candidates:
            raise ValueError("No candidate moves to safety-check")

        root_turn = board.turn
        current_eval = self.evaluate_board(board, root_turn)
        evals = {
            candidate.move: self.evaluate_after_move(board, candidate.move, root_turn)
            for candidate in candidates
        }
        drawish_reasons = {
            candidate.move: self.drawish_reason(self.board_after_move(board, candidate.move))
            for candidate in candidates
        }
        shuffle_reasons = {
            candidate.move: self.shuffle_reason(board, candidate.move)
            for candidate in candidates
        }

        best_move, best_eval = max(evals.items(), key=lambda item: item[1])
        original = candidates[0]
        original_eval = evals[original.move]
        original_drawish = drawish_reasons[original.move]
        original_shuffle = shuffle_reasons[original.move]
        clearly_winning = max(current_eval, best_eval) >= self.avoid_draw_when_winning_cp

        if original_shuffle is not None:
            for candidate in candidates:
                candidate_eval = evals[candidate.move]
                if shuffle_reasons[candidate.move] is None and best_eval - candidate_eval <= self.avoid_shuffle_cp:
                    return SafetyDecision(
                        selected=candidate,
                        original=original,
                        vetoed=candidate.move != original.move,
                        reason=f"avoided_{original_shuffle}_{original.move}",
                        best_eval_cp=best_eval,
                        original_eval_cp=original_eval,
                        selected_eval_cp=candidate_eval,
                    )

        if clearly_winning and original_drawish is not None:
            for candidate in candidates:
                candidate_eval = evals[candidate.move]
                if drawish_reasons[candidate.move] is None and best_eval - candidate_eval < self.veto_cp:
                    return SafetyDecision(
                        selected=candidate,
                        original=original,
                        vetoed=candidate.move != original.move,
                        reason=f"avoided_{original_drawish}_{original.move}",
                        best_eval_cp=best_eval,
                        original_eval_cp=original_eval,
                        selected_eval_cp=candidate_eval,
                    )

            engine_fallback = self.engine_policy_move(board, root_turn)
            if engine_fallback is not None:
                engine_move, engine_eval, engine_drawish = engine_fallback
                if engine_drawish is None and engine_eval > 0:
                    return SafetyDecision(
                        selected=engine_move,
                        original=original,
                        vetoed=True,
                        reason=f"stockfish_fallback_avoided_{original_drawish}_{original.move}",
                        best_eval_cp=max(best_eval, engine_eval),
                        original_eval_cp=original_eval,
                        selected_eval_cp=engine_eval,
                    )

        if best_eval - original_eval < self.veto_cp:
            return SafetyDecision(
                selected=original,
                original=original,
                vetoed=False,
                reason="within_threshold",
                best_eval_cp=best_eval,
                original_eval_cp=original_eval,
                selected_eval_cp=original_eval,
            )

        for candidate in candidates:
            candidate_eval = evals[candidate.move]
            if best_eval - candidate_eval < self.veto_cp:
                return SafetyDecision(
                    selected=candidate,
                    original=original,
                    vetoed=candidate.move != original.move,
                    reason=f"vetoed_{original.move}_delta_{best_eval - original_eval}_cp",
                    best_eval_cp=best_eval,
                    original_eval_cp=original_eval,
                    selected_eval_cp=candidate_eval,
                )

        best_candidate = next(candidate for candidate in candidates if candidate.move == best_move)
        return SafetyDecision(
            selected=best_candidate,
            original=original,
            vetoed=best_candidate.move != original.move,
            reason=f"fallback_best_candidate_delta_{best_eval - original_eval}_cp",
            best_eval_cp=best_eval,
            original_eval_cp=original_eval,
            selected_eval_cp=best_eval,
        )
