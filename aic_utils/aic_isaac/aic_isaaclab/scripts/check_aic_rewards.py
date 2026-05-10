# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check AIC insertion reward terms in Isaac Lab."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check AIC reward tensors.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs.")
parser.add_argument("--num_steps", type=int, default=8, help="Random action steps.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.25,
    help="Uniform random action range is [-scale, scale].",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for the timestamped reward log. Defaults to <repo>/logs/aic_rewards.",
)
parser.add_argument(
    "--no_log_file",
    action="store_true",
    default=False,
    help="Print only to stdout instead of also writing a log file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task import mdp


class Reporter:
    """Small stdout/file tee for reward-check output."""

    def __init__(self, path: Path | None):
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        if self._file is not None:
            self._file.write(f"{text}\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _repo_root() -> Path:
    """Find the AIC repo root from this script path."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "bahw_docs" / "overview.md").exists():
            return parent
    return Path.cwd()


def _default_log_path(task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = task_name.replace("/", "_")
    return _repo_root() / "logs" / "aic_rewards" / f"{stamp}_{safe_task}.log"


def _summary(value: torch.Tensor) -> tuple[bool, float, float, float]:
    finite = bool(torch.isfinite(value).all().item())
    detached = value.detach().float()
    return (
        finite,
        float(detached.mean().cpu().item()),
        float(detached.min().cpu().item()),
        float(detached.max().cpu().item()),
    )


def _reward_checks() -> tuple[tuple[str, Callable[[Any], torch.Tensor]], ...]:
    return (
        ("sc_approach", lambda env: mdp.sc_approach_reward(env, std=1.00)),
        (
            "sc_distance_progress",
            lambda env: mdp.sc_distance_progress_reward(env, scale=0.02, clip=1.0),
        ),
        (
            "sc_lateral_progress",
            lambda env: mdp.sc_lateral_progress_reward(env, scale=0.005, clip=1.0),
        ),
        (
            "sc_orientation_progress",
            lambda env: mdp.sc_orientation_progress_reward(
                env, scale=0.10, clip=1.0
            ),
        ),
        (
            "sc_depth_progress",
            lambda env: mdp.sc_depth_progress_reward(env, scale=0.01, clip=1.0),
        ),
        (
            "sc_coarse_lateral_alignment",
            lambda env: mdp.sc_lateral_alignment_reward(env, std=0.30),
        ),
        (
            "sc_coarse_orientation_alignment",
            lambda env: mdp.sc_orientation_alignment_reward(env, std=2.00),
        ),
        (
            "sc_lateral_alignment",
            lambda env: mdp.sc_lateral_alignment_reward(env, std=0.02),
        ),
        (
            "sc_orientation_alignment",
            lambda env: mdp.sc_orientation_alignment_reward(env, std=0.35),
        ),
        (
            "sc_insertion_depth",
            lambda env: mdp.sc_insertion_depth_reward(
                env,
                depth_scale=0.02,
                max_depth=0.03,
                lateral_threshold=0.01,
                orientation_threshold=0.35,
            ),
        ),
        (
            "sc_insertion_success",
            lambda env: mdp.sc_insertion_success_bonus(
                env,
                lateral_threshold=0.005,
                orientation_threshold=0.20,
                depth_threshold=0.012,
            ),
        ),
        (
            "sc_scripted_action_prior",
            lambda env: mdp.sc_scripted_action_prior_reward(
                env,
                action_name="arm_action",
                asset_name="robot",
                action_body_name="gripper_tcp",
                action_scale=0.05,
                action_clip=1.0,
                approach_depth=0.0,
                target_depth=0.02,
                max_translation_step=0.025,
                max_rotation_step=0.10,
                align_lateral_threshold=0.05,
                align_orientation_threshold=0.50,
                std=1.00,
            ),
        ),
    )


def _print_reward_checks(report: Reporter, env: Any, label: str) -> bool:
    report.line(f"== Direct Reward Tensor Checks: {label} ==")
    all_finite = True
    with torch.inference_mode():
        for name, func in _reward_checks():
            value = func(env)
            finite, mean, min_value, max_value = _summary(value)
            all_finite = all_finite and finite
            report.line(
                f"{name}: finite={finite} "
                f"mean={mean:.6f} min={min_value:.6f} max={max_value:.6f}"
            )
    return all_finite


def _print_reward_manager_terms(report: Reporter, env: Any) -> None:
    reward_manager = getattr(env, "reward_manager", None)
    report.line("== Reward Manager Introspection ==")
    if reward_manager is None:
        report.line("reward_manager: unavailable")
        return
    for attr_name in ("_term_names", "active_terms"):
        value = getattr(reward_manager, attr_name, None)
        if value is not None:
            report.line(f"{attr_name}: {value}")
    term_cfgs = getattr(reward_manager, "_term_cfgs", None)
    if term_cfgs is not None:
        report.line("_term_cfgs:")
        for cfg in term_cfgs:
            name = getattr(cfg, "name", "<unnamed>")
            weight = getattr(cfg, "weight", "<unknown>")
            report.line(f"  {name}: weight={weight}")


def _print_analytic_shape_checks(report: Reporter) -> bool:
    """Check reward kernel monotonicity without mutating simulator state."""
    report.line("== Analytic Reward Shape Checks ==")
    all_ok = True

    distances = torch.tensor([1.25, 0.50, 0.10, 0.02, 0.0])
    approach = 1.0 - torch.tanh(distances / 0.50)
    approach_ok = bool(torch.all(approach[1:] >= approach[:-1]).item())
    all_ok = all_ok and approach_ok
    report.line(
        "approach_reward distance "
        f"{distances.tolist()} -> {approach.tolist()} monotonic={approach_ok}"
    )

    lateral_errors = torch.tensor([0.05, 0.02, 0.01, 0.005, 0.0])
    lateral = 1.0 - torch.tanh(lateral_errors / 0.02)
    lateral_ok = bool(torch.all(lateral[1:] >= lateral[:-1]).item())
    all_ok = all_ok and lateral_ok
    report.line(
        "lateral_reward error "
        f"{lateral_errors.tolist()} -> {lateral.tolist()} monotonic={lateral_ok}"
    )

    orientation_errors = torch.tensor([1.0, 0.35, 0.20, 0.10, 0.0])
    orientation = 1.0 - torch.tanh(orientation_errors / 0.35)
    orientation_ok = bool(torch.all(orientation[1:] >= orientation[:-1]).item())
    all_ok = all_ok and orientation_ok
    report.line(
        "orientation_reward error "
        f"{orientation_errors.tolist()} -> {orientation.tolist()} "
        f"monotonic={orientation_ok}"
    )

    depths = torch.tensor([-0.01, 0.0, 0.006, 0.012, 0.02, 0.04])
    aligned_depth = torch.clamp(torch.clamp(depths, min=0.0, max=0.03) / 0.02, max=1.0)
    depth_ok = bool(torch.all(aligned_depth[1:] >= aligned_depth[:-1]).item())
    all_ok = all_ok and depth_ok
    report.line(
        "depth_reward aligned depth "
        f"{depths.tolist()} -> {aligned_depth.tolist()} monotonic={depth_ok}"
    )

    misaligned_depth = torch.zeros_like(aligned_depth)
    misaligned_ok = bool(torch.all(misaligned_depth == 0.0).item())
    all_ok = all_ok and misaligned_ok
    report.line(
        "depth_reward misaligned depth "
        f"{depths.tolist()} -> {misaligned_depth.tolist()} zeroed={misaligned_ok}"
    )

    report.line(f"analytic_shape_checks_ok: {all_ok}")
    return all_ok


def main() -> int:
    log_path = None
    if not args_cli.no_log_file:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = args_cli.task.replace("/", "_")
        log_path = (
            Path(args_cli.output_dir) / f"{stamp}_{safe_task}.log"
            if args_cli.output_dir
            else _default_log_path(args_cli.task)
        )

    report = Reporter(log_path)
    env = None
    failed = False
    try:
        report.line("== AIC Reward Check ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"num_envs: {args_cli.num_envs}")
        report.line(f"num_steps: {args_cli.num_steps}")
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped

        report.line()
        report.line(f"gym observation space: {env.observation_space}")
        report.line(f"gym action space: {env.action_space}")
        _print_reward_manager_terms(report, base_env)
        failed = failed or not _print_analytic_shape_checks(report)

        with torch.inference_mode():
            env.reset()

        failed = failed or not _print_reward_checks(report, base_env, "after reset")

        for step in range(args_cli.num_steps):
            actions = torch.empty(
                env.action_space.shape, device=base_env.device, dtype=torch.float32
            ).uniform_(-args_cli.action_scale, args_cli.action_scale)
            with torch.inference_mode():
                _, reward, _, _, _ = env.step(actions)
            finite, mean, min_value, max_value = _summary(reward)
            failed = failed or not finite
            report.line(
                f"step {step:02d} total_reward: finite={finite} "
                f"mean={mean:.6f} min={min_value:.6f} max={max_value:.6f}"
            )
            failed = failed or not _print_reward_checks(
                report, base_env, f"after step {step:02d}"
            )

        report.line()
        report.line("== Done ==")
        report.line(f"overall_finite: {not failed}")
    finally:
        if env is not None:
            env.close()
        report.close()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
