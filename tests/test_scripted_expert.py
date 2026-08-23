from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from env.pick_place import PickPlace
from expert.evaluate import create_parser as create_evaluation_parser
from expert.evaluate import main as evaluate_main
from expert.scripted import DLSController, ExpertConfig, run_episode


@pytest.fixture
def env() -> PickPlace:
    instance = PickPlace()
    yield instance
    instance.close()


def test_dls_controller_returns_a_valid_environment_action(env: PickPlace) -> None:
    reset_info = env.reset(np.random.default_rng(0))
    controller = DLSController(env, ExpertConfig())
    controller.set_grasp_yaw(reset_info["cube_yaw"])

    action = controller.action(env.tcp + np.array([0.01, -0.01, 0.02]), 1.0)

    assert action.shape == (8,)
    assert np.all(np.isfinite(action))
    np.testing.assert_array_less(
        action[:7], env.model.actuator_ctrlrange[:7, 1] + 1e-12
    )
    np.testing.assert_array_less(
        env.model.actuator_ctrlrange[:7, 0] - 1e-12, action[:7]
    )
    assert action[7] == 1.0


def test_scripted_expert_succeeds_and_records_auditable_steps(
    env: PickPlace,
) -> None:
    result = run_episode(env, seed=0, record_steps=True)

    assert result.success
    assert result.failure_stage is None
    assert result.attempts >= 1
    assert result.control_steps == len(result.steps)
    assert result.sim_time_s == pytest.approx(result.control_steps / env.cfg.control_hz)
    assert {step.stage for step in result.steps} >= {
        "pregrasp",
        "descend",
        "close",
        "lift",
        "transport",
        "lower",
        "open",
        "retreat",
        "settle",
    }
    assert all(step.action.shape == (8,) for step in result.steps)
    assert all(np.all(np.isfinite(step.action)) for step in result.steps)
    assert json.loads(json.dumps(result.to_dict()))["seed"] == 0


def test_scripted_expert_reports_the_stage_when_motion_budget_is_too_small(
    env: PickPlace,
) -> None:
    config = ExpertConfig(max_move_steps=1, max_attempts=1)

    result = run_episode(env, seed=0, config=config)

    assert not result.success
    assert result.failure_stage == "pregrasp"


def test_scripted_expert_meets_the_fixed_100_seed_gate(env: PickPlace) -> None:
    results = [run_episode(env, seed=seed) for seed in range(100)]
    successes = sum(result.success for result in results)

    assert successes >= 90
    assert [result.seed for result in results] == list(range(100))


def test_evaluation_cli_writes_jsonl_and_gate_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "m1-eval"

    exit_code = evaluate_main(
        [
            "--num-seeds",
            "2",
            "--required-successes",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    episode_lines = (output_dir / "episodes.jsonl").read_text().splitlines()
    episodes = [json.loads(line) for line in episode_lines]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert [episode["seed"] for episode in episodes] == [0, 1]
    assert summary["num_episodes"] == 2
    assert summary["gate"]["required_successes"] == 1
    assert summary["gate"]["passed"]


def test_evaluation_cli_allows_a_zero_gate_for_known_failure_replays() -> None:
    args = create_evaluation_parser().parse_args(["--required-successes", "0"])

    assert args.required_successes == 0
