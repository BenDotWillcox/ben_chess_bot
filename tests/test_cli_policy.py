from __future__ import annotations

import json
import random
import sys
from types import SimpleNamespace

import chess

import play_policy
import predict_policy
from game_state import PolicyGameState
from personalized_policy import PolicyMove


def _candidates() -> list[PolicyMove]:
    return [
        PolicyMove("e2e4", 0.2, 0.2, 0.0, "maia2", 0),
        PolicyMove("d2d4", 0.8, 0.8, 0.0, "maia2", 0),
    ]


class PredictOnlyPolicy:
    def __init__(self, candidates: list[PolicyMove]) -> None:
        self.candidates = candidates
        self.predict_calls = 0

    def predict(self, **_kwargs) -> list[PolicyMove]:
        self.predict_calls += 1
        return list(self.candidates)


def test_cli_parsers_use_validated_personalization_defaults() -> None:
    play_args = play_policy.build_parser().parse_args([])
    predict_args = predict_policy.build_parser().parse_args(
        ["--fen", chess.STARTING_FEN, "--you-color", "white"]
    )

    assert (play_args.strategy, play_args.alpha, play_args.min_count) == ("fen", 0.7, 1)
    assert (predict_args.strategy, predict_args.alpha, predict_args.min_count) == ("fen", 0.7, 1)
    assert play_args.mode == predict_args.mode == "argmax"


def test_terminal_sampling_reuses_the_single_prediction_distribution(capsys) -> None:
    candidates = _candidates()
    policy = PredictOnlyPolicy(candidates)
    state = PolicyGameState()
    seed = 17
    expected = random.Random(seed).choices(
        candidates,
        weights=[candidate.probability for candidate in candidates],
        k=1,
    )[0]

    play_policy.maybe_policy_move(
        state=state,
        policy=policy,  # type: ignore[arg-type]
        bot_color="white",
        elo_self=1650,
        elo_oppo=1650,
        mode="sample",
        top_k=2,
        temperature=0.8,
        safety=None,
        rng=random.Random(seed),
    )

    assert policy.predict_calls == 1
    assert state.uci_history == [expected.move]
    assert expected.move in capsys.readouterr().out


def test_prediction_cli_samples_from_displayed_moves_without_second_inference(
    monkeypatch, capsys
) -> None:
    candidates = _candidates()
    policy = PredictOnlyPolicy(candidates)
    load_config: dict[str, object] = {}

    def fake_load(**kwargs):
        load_config.update(kwargs)
        return policy

    seed = 23
    expected = random.Random(seed).choices(
        candidates,
        weights=[candidate.probability for candidate in candidates],
        k=1,
    )[0]
    monkeypatch.setattr(
        predict_policy,
        "PersonalizedMaia2Policy",
        SimpleNamespace(load=fake_load),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_policy.py",
            "--fen",
            chess.STARTING_FEN,
            "--you-color",
            "white",
            "--mode",
            "sample",
            "--seed",
            str(seed),
            "--top-n",
            "2",
        ],
    )

    predict_policy.main()
    payload = json.loads(capsys.readouterr().out)

    assert policy.predict_calls == 1
    assert payload["selected"]["move"] == expected.move
    assert [move["move"] for move in payload["moves"]] == [move.move for move in candidates]
    assert payload["config"] == {
        "model_type": "rapid",
        "strategy": "fen",
        "alpha": 0.7,
        "min_count": 1,
        "mode": "sample",
        "temperature": 1.0,
        "seed": seed,
    }
    assert load_config["strategy"] == "fen"
    assert load_config["alpha"] == 0.7
    assert load_config["min_count"] == 1
