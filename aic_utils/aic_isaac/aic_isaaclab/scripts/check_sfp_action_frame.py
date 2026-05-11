# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe how raw relative-IK actions move the SFP tip in the port frame."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check SFP action-frame effects.")
parser.add_argument("--task", type=str, default="AIC-SFP-Task-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs.")
parser.add_argument(
    "--raw_action",
    type=float,
    default=1.0,
    help="Raw action magnitude applied to each probed action axis.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1,
    help="Number of environment steps to apply each action.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for the timestamped log. Defaults to <repo>/logs/aic_action_frame.",
)
parser.add_argument(
    "--no_log_file",
    action="store_true",
    default=False,
    help="Print only to stdout instead of also writing a log file.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
    """Small stdout/file tee for diagnostic output."""

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
    return _repo_root() / "logs" / "aic_action_frame" / f"{stamp}_{safe_task}.log"


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1.0e-9)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in Isaac's wxyz convention."""
    quat = _quat_normalize(quat)
    q_vec = quat[..., 1:]
    q_w = quat[..., 0:1]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _axis_from_quat(
    quat_w: torch.Tensor, local_axis: tuple[float, float, float]
) -> torch.Tensor:
    axis = torch.tensor(local_axis, device=quat_w.device, dtype=quat_w.dtype)
    axis = axis.unsqueeze(0).expand(quat_w.shape[0], -1)
    return _quat_apply(quat_w, axis)


def _sfp_metrics(env: Any) -> dict[str, torch.Tensor]:
    plug_pos_w, _ = mdp.sfp_plug_tip_pose(env)
    entry_pos_w, entry_quat_w = mdp.sfp_port_entry_pose(env)
    insertion_axis_w = mdp.sfp_port_insertion_axis(env)

    delta = plug_pos_w - entry_pos_w
    depth = torch.sum(delta * insertion_axis_w, dim=-1, keepdim=True)
    lateral = delta - depth * insertion_axis_w

    port_x_w = _axis_from_quat(entry_quat_w, (1.0, 0.0, 0.0))
    port_y_w = _axis_from_quat(entry_quat_w, (0.0, 1.0, 0.0))
    return {
        "lateral": torch.norm(lateral, dim=-1),
        "lateral_x": torch.sum(lateral * port_x_w, dim=-1),
        "lateral_y": torch.sum(lateral * port_y_w, dim=-1),
        "depth": depth.squeeze(-1),
        "orientation": mdp.sfp_orientation_error(env),
    }


def _mean(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu().item())


def _disable_sfp_terminations(env_cfg: Any) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    for term_name in (
        "time_out",
        "sfp_insertion_success",
        "sfp_corridor_violation",
        "sfp_corridor_lateral_violation",
        "sfp_corridor_orientation_violation",
        "sfp_corridor_min_depth_violation",
        "sfp_corridor_max_depth_violation",
    ):
        if hasattr(terminations, term_name):
            setattr(terminations, term_name, None)


def _print_summary(
    report: Reporter,
    label: str,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    parts = [f"action={label}"]
    for key in ("lateral_x", "lateral_y", "depth", "lateral", "orientation"):
        delta = after[key] - before[key]
        parts.append(f"d_{key}_mean={_mean(delta):+.6f}")
    parts.append(f"after_lateral_mean={_mean(after['lateral']):.6f}")
    parts.append(f"after_depth_mean={_mean(after['depth']):.6f}")
    parts.append(f"after_orientation_mean={_mean(after['orientation']):.6f}")
    report.line("  " + " ".join(parts))


def main() -> int:
    log_path = None
    if not args_cli.no_log_file:
        log_path = (
            Path(args_cli.output_dir)
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args_cli.task}.log"
            if args_cli.output_dir
            else _default_log_path(args_cli.task)
        )

    report = Reporter(log_path)
    env = None
    try:
        report.line("== AIC SFP Action Frame Check ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"num_envs: {args_cli.num_envs}")
        report.line(f"raw_action: {args_cli.raw_action}")
        report.line(f"num_steps: {args_cli.num_steps}")
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        _disable_sfp_terminations(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped

        action_dim = env.action_space.shape[-1]
        report.line(f"action_dim: {action_dim}")
        report.line("translation probes:")

        probes = (
            ("tx+", 0, args_cli.raw_action),
            ("tx-", 0, -args_cli.raw_action),
            ("ty+", 1, args_cli.raw_action),
            ("ty-", 1, -args_cli.raw_action),
            ("tz+", 2, args_cli.raw_action),
            ("tz-", 2, -args_cli.raw_action),
        )
        if action_dim >= 6:
            probes = probes + (
                ("rx+", 3, args_cli.raw_action),
                ("rx-", 3, -args_cli.raw_action),
                ("ry+", 4, args_cli.raw_action),
                ("ry-", 4, -args_cli.raw_action),
                ("rz+", 5, args_cli.raw_action),
                ("rz-", 5, -args_cli.raw_action),
            )

        with torch.inference_mode():
            for label, index, value in probes:
                env.reset()
                before = _sfp_metrics(base_env)
                target_ids = mdp.active_sfp_target_ids(base_env).detach().cpu()
                actions = torch.zeros(
                    env.action_space.shape,
                    device=base_env.device,
                    dtype=torch.float32,
                )
                actions[:, index] = value
                for _ in range(args_cli.num_steps):
                    env.step(actions)
                after = _sfp_metrics(base_env)
                _print_summary(report, label, before, after)
                for target_id, target_name in enumerate(mdp.SFP_TARGET_NAMES):
                    mask = target_ids == target_id
                    if not bool(mask.any()):
                        continue
                    report.line(
                        "    "
                        f"{target_name}: n={int(mask.sum().item())} "
                        f"d_lateral_x_mean={_mean(after['lateral_x'][mask] - before['lateral_x'][mask]):+.6f} "
                        f"d_lateral_y_mean={_mean(after['lateral_y'][mask] - before['lateral_y'][mask]):+.6f} "
                        f"d_depth_mean={_mean(after['depth'][mask] - before['depth'][mask]):+.6f}"
                    )

        report.line("== Done ==")
        return 0
    finally:
        if env is not None:
            env.close()
        report.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
