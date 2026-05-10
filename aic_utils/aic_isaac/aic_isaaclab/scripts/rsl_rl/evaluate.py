# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint on the AIC task and report insertion metrics."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint.")
parser.add_argument(
    "--num_envs", type=int, default=16, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the env.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_eval_episodes",
    type=int,
    default=256,
    help="Number of evaluation episodes to complete.",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=None,
    help="Manual evaluation timeout. Defaults to env.max_episode_length.",
)
parser.add_argument("--lateral_threshold", type=float, default=0.005)
parser.add_argument("--orientation_threshold", type=float, default=0.20)
parser.add_argument("--depth_threshold", type=float, default=0.012)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for the timestamped evaluation log. Defaults to <repo>/logs/aic_eval.",
)
parser.add_argument(
    "--no_log_file",
    action="store_true",
    default=False,
    help="Print only to stdout instead of also writing a log file.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task import mdp


class Reporter:
    """Small stdout/file tee for evaluation output."""

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
    return _repo_root() / "logs" / "aic_eval" / f"{stamp}_{safe_task}.log"


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _target_name(target_id: int) -> str:
    if 0 <= target_id < len(mdp.SC_TARGET_NAMES):
        return mdp.SC_TARGET_NAMES[target_id]
    return f"unknown_{target_id}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> int:
    """Evaluate a trained RSL-RL policy."""
    if args_cli.task is None:
        raise ValueError("--task is required.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "sc_insertion_success"):
            env_cfg.terminations.sc_insertion_success = None

    resume_path = _resolve_checkpoint(agent_cfg)
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
    wrapped_env = None
    try:
        report.line("== AIC RSL-RL Evaluation ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"agent: {args_cli.agent}")
        report.line(f"checkpoint: {resume_path}")
        report.line(f"num_envs: {env_cfg.scene.num_envs}")
        report.line(f"num_eval_episodes: {args_cli.num_eval_episodes}")
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped
        max_episode_steps = args_cli.max_episode_steps
        if max_episode_steps is None:
            max_episode_steps = int(getattr(base_env, "max_episode_length"))
        report.line(f"max_episode_steps: {max_episode_steps}")

        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        agent_cfg_dict = cli_args.runner_cfg_to_dict(agent_cfg)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(
                wrapped_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
            )
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(
                wrapped_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
            )
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

        completed = 0
        success_count = 0
        timeout_count = 0
        lateral_miss_count = 0
        orientation_miss_count = 0
        depth_shortfall_count = 0
        episode_lengths: list[int] = []
        success_lengths: list[int] = []
        terminal_lateral: list[float] = []
        terminal_orientation: list[float] = []
        terminal_depth: list[float] = []
        success_lateral: list[float] = []
        success_depth: list[float] = []
        per_target = {
            target_name: {"episodes": 0, "successes": 0}
            for target_name in mdp.SC_TARGET_NAMES
        }
        last_progress_report = 0

        while completed < args_cli.num_eval_episodes:
            with torch.inference_mode():
                base_env.reset()
                obs = wrapped_env.get_observations()

            active = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
            episode_steps = torch.zeros(
                base_env.num_envs, device=base_env.device, dtype=torch.long
            )
            target_ids = mdp.active_sc_target_ids(base_env).detach().clone()

            while bool(active.any().item()) and completed < args_cli.num_eval_episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                    actions = actions.clone()
                    actions[~active] = 0.0
                    obs, _, _, _ = wrapped_env.step(actions)
                    if policy_nn is not None and hasattr(policy_nn, "reset"):
                        policy_nn.reset(~active)

                    episode_steps[active] += 1
                    lateral = mdp.sc_lateral_error(base_env)
                    orientation = mdp.sc_orientation_error(base_env)
                    depth = mdp.sc_insertion_depth(base_env)
                    success = mdp.sc_insertion_success(
                        base_env,
                        lateral_threshold=args_cli.lateral_threshold,
                        orientation_threshold=args_cli.orientation_threshold,
                        depth_threshold=args_cli.depth_threshold,
                    )
                    timeout = episode_steps >= max_episode_steps
                    done = active & (success | timeout)

                for env_id in done.nonzero(as_tuple=False).flatten().tolist():
                    if completed >= args_cli.num_eval_episodes:
                        break
                    completed += 1
                    active[env_id] = False

                    target_name = _target_name(int(target_ids[env_id].detach().cpu().item()))
                    per_target.setdefault(target_name, {"episodes": 0, "successes": 0})
                    per_target[target_name]["episodes"] += 1

                    step_count = int(episode_steps[env_id].detach().cpu().item())
                    lat = float(lateral[env_id].detach().cpu().item())
                    ori = float(orientation[env_id].detach().cpu().item())
                    dep = float(depth[env_id].detach().cpu().item())
                    succeeded = bool(success[env_id].detach().cpu().item())

                    episode_lengths.append(step_count)
                    terminal_lateral.append(lat)
                    terminal_orientation.append(ori)
                    terminal_depth.append(dep)

                    if succeeded:
                        success_count += 1
                        per_target[target_name]["successes"] += 1
                        success_lengths.append(step_count)
                        success_lateral.append(lat)
                        success_depth.append(dep)
                    else:
                        if bool(timeout[env_id].detach().cpu().item()):
                            timeout_count += 1
                        if lat >= args_cli.lateral_threshold:
                            lateral_miss_count += 1
                        if ori >= args_cli.orientation_threshold:
                            orientation_miss_count += 1
                        if dep <= args_cli.depth_threshold:
                            depth_shortfall_count += 1

                if (
                    completed > last_progress_report
                    and completed % max(base_env.num_envs, 1) == 0
                ):
                    report.line(
                        f"progress: {completed}/{args_cli.num_eval_episodes} "
                        f"successes={success_count}"
                    )
                    last_progress_report = completed

        success_rate = success_count / max(completed, 1)
        report.line()
        report.line("== Summary ==")
        report.line(f"episodes: {completed}")
        report.line(f"successes: {success_count}")
        report.line(f"success_rate: {success_rate:.6f}")
        report.line(f"mean_episode_length: {_mean_or_nan(episode_lengths):.3f}")
        report.line(f"mean_episode_length_on_success: {_mean_or_nan(success_lengths):.3f}")
        report.line(f"mean_lateral_error_at_termination: {_mean_or_nan(terminal_lateral):.6f}")
        report.line(
            "mean_orientation_error_at_termination: "
            f"{_mean_or_nan(terminal_orientation):.6f}"
        )
        report.line(f"mean_insertion_depth_at_termination: {_mean_or_nan(terminal_depth):.6f}")
        report.line(f"mean_success_lateral_error: {_mean_or_nan(success_lateral):.6f}")
        report.line(f"mean_success_insertion_depth: {_mean_or_nan(success_depth):.6f}")
        report.line("failure_breakdown:")
        report.line(f"  timeout: {timeout_count}")
        report.line(f"  lateral_miss: {lateral_miss_count}")
        report.line(f"  orientation_miss: {orientation_miss_count}")
        report.line(f"  depth_shortfall: {depth_shortfall_count}")
        report.line("per_target:")
        for target_name, stats in per_target.items():
            episodes = stats["episodes"]
            successes = stats["successes"]
            rate = successes / episodes if episodes else float("nan")
            report.line(
                f"  {target_name}: episodes={episodes} "
                f"successes={successes} success_rate={rate:.6f}"
            )
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        report.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
