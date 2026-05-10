# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether privileged SC geometry can drive the existing IK action to insertion."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a scripted SC insertion controller.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of envs.")
parser.add_argument("--max_steps", type=int, default=1500, help="Maximum sim steps.")
parser.add_argument("--report_every", type=int, default=50, help="Report interval.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.05,
    help="Raw action divisor matching DifferentialInverseKinematicsActionCfg.scale.",
)
parser.add_argument(
    "--action_clip",
    type=float,
    default=1.0,
    help="Absolute raw action clip. Set <=0 to disable.",
)
parser.add_argument(
    "--approach_depth",
    type=float,
    default=-0.04,
    help="Scripted outside-port depth while aligning, in meters.",
)
parser.add_argument(
    "--target_depth",
    type=float,
    default=0.016,
    help="Scripted final insertion depth, in meters.",
)
parser.add_argument(
    "--max_translation_step",
    type=float,
    default=0.025,
    help="Maximum processed translation command per env step, in meters.",
)
parser.add_argument(
    "--max_rotation_step",
    type=float,
    default=0.10,
    help="Maximum processed angle-axis command per env step, in radians.",
)
parser.add_argument("--align_lateral_threshold", type=float, default=0.02)
parser.add_argument("--align_orientation_threshold", type=float, default=0.50)
parser.add_argument("--success_lateral_threshold", type=float, default=0.005)
parser.add_argument("--success_orientation_threshold", type=float, default=0.20)
parser.add_argument("--success_depth_threshold", type=float, default=0.012)
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
    help="Directory for the timestamped log. Defaults to <repo>/logs/aic_scripted_insert.",
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
    """Small stdout/file tee for scripted-check output."""

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
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "bahw_docs" / "overview.md").exists():
            return parent
    return Path.cwd()


def _default_log_path(task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = task_name.replace("/", "_")
    return _repo_root() / "logs" / "aic_scripted_insert" / f"{stamp}_{safe_task}.log"


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., 0:1], -quat[..., 1:]), dim=-1)


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1.0e-9)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = _quat_normalize(quat)
    q_vec = quat[..., 1:]
    q_w = quat[..., 0:1]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _clip_by_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vec * scale


def _summary(value: torch.Tensor) -> str:
    detached = value.detach().float()
    return (
        f"mean={detached.mean().item():.6f} "
        f"min={detached.min().item():.6f} max={detached.max().item():.6f}"
    )


def _scripted_actions(env: Any, inactive: torch.Tensor) -> torch.Tensor:
    """Compute raw relative-IK actions from privileged SC insertion geometry."""
    robot = env.scene["robot"]

    plug_pos_w, _ = mdp.sc_plug_tip_pose(env)
    plug_axis_w = mdp.sc_plug_axis(env)
    entry_pos_w, _ = mdp.sc_port_entry_pose(env)
    port_axis_w = mdp.sc_port_insertion_axis(env)

    lateral = mdp.sc_lateral_error(env)
    orientation = mdp.sc_orientation_error(env)
    aligned = (
        (lateral < args_cli.align_lateral_threshold)
        & (orientation < args_cli.align_orientation_threshold)
        & (~inactive)
    )
    target_depth = torch.full(
        (env.num_envs,), args_cli.approach_depth, device=env.device
    )
    target_depth[aligned] = args_cli.target_depth
    target_pos_w = entry_pos_w + target_depth.unsqueeze(-1) * port_axis_w

    delta_pos_w = _clip_by_norm(
        target_pos_w - plug_pos_w,
        max_norm=args_cli.max_translation_step,
    )

    cross = torch.cross(plug_axis_w, port_axis_w, dim=-1)
    sin_angle = torch.norm(cross, dim=-1)
    cos_angle = torch.sum(plug_axis_w * port_axis_w, dim=-1)
    angle = torch.atan2(sin_angle, cos_angle)
    axis_w = cross / torch.clamp(sin_angle.unsqueeze(-1), min=1.0e-9)
    rot_step = torch.clamp(angle, max=args_cli.max_rotation_step)
    rot_vec_w = axis_w * rot_step.unsqueeze(-1)
    rot_vec_w = torch.where(sin_angle.unsqueeze(-1) > 1.0e-6, rot_vec_w, 0.0)

    root_inv = _quat_conjugate(robot.data.root_quat_w)
    delta_pos_root = _quat_apply(root_inv, delta_pos_w)
    rot_vec_root = _quat_apply(root_inv, rot_vec_w)

    processed = torch.cat((delta_pos_root, rot_vec_root), dim=-1)
    processed[inactive] = 0.0

    action_shape = env.action_space.shape
    if len(action_shape) == 1:
        actions = torch.zeros((env.num_envs, action_shape[0]), device=env.device)
    else:
        actions = torch.zeros(action_shape, device=env.device)
    actions[:, : processed.shape[1]] = processed / args_cli.action_scale
    if args_cli.action_clip > 0:
        actions = torch.clamp(actions, -args_cli.action_clip, args_cli.action_clip)
    return actions


def _success_mask(env: Any) -> torch.Tensor:
    return mdp.sc_insertion_success(
        env,
        lateral_threshold=args_cli.success_lateral_threshold,
        orientation_threshold=args_cli.success_orientation_threshold,
        depth_threshold=args_cli.success_depth_threshold,
    )


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "sc_insertion_success"):
            env_cfg.terminations.sc_insertion_success = None

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
    try:
        report.line("== AIC Scripted SC Insertion Check ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"num_envs: {args_cli.num_envs}")
        report.line(f"max_steps: {args_cli.max_steps}")
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        report.line(f"action_space: {env.action_space}")
        env.reset()

        target_ids = mdp.active_sc_target_ids(env).detach().clone()
        first_success_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )

        success = _success_mask(env)
        first_success_step[success] = 0

        for step in range(args_cli.max_steps + 1):
            lateral = mdp.sc_lateral_error(env)
            orientation = mdp.sc_orientation_error(env)
            depth = mdp.sc_insertion_depth(env)
            success = _success_mask(env)
            newly_successful = success & (first_success_step < 0)
            first_success_step[newly_successful] = step

            if step == 0 or step % args_cli.report_every == 0 or bool(success.all()):
                report.line(
                    f"step={step} successes={int(success.sum().item())}/{env.num_envs} "
                    f"lateral({_summary(lateral)}) "
                    f"orientation({_summary(orientation)}) "
                    f"depth({_summary(depth)})"
                )
            if bool(success.all()) or step >= args_cli.max_steps:
                break

            with torch.inference_mode():
                env.step(_scripted_actions(env, inactive=success))

        report.line()
        report.line("== Summary ==")
        report.line(f"successes: {int((first_success_step >= 0).sum().item())}/{env.num_envs}")
        report.line(f"first_success_steps: {first_success_step.detach().cpu().tolist()}")
        report.line("per_target:")
        for target_id, target_name in enumerate(mdp.SC_TARGET_NAMES):
            mask = target_ids == target_id
            episodes = int(mask.sum().item())
            successes = int(((first_success_step >= 0) & mask).sum().item())
            rate = successes / episodes if episodes else float("nan")
            report.line(
                f"  {target_name}: episodes={episodes} "
                f"successes={successes} success_rate={rate:.6f}"
            )

        final_lateral = mdp.sc_lateral_error(env)
        final_orientation = mdp.sc_orientation_error(env)
        final_depth = mdp.sc_insertion_depth(env)
        report.line(f"final_lateral: {_summary(final_lateral)}")
        report.line(f"final_orientation: {_summary(final_orientation)}")
        report.line(f"final_depth: {_summary(final_depth)}")
        return 0 if bool((first_success_step >= 0).all().item()) else 1
    finally:
        if env is not None:
            env.close()
        report.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
