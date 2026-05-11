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
    "--custom_action",
    action="append",
    default=[],
    help=(
        "Comma-separated raw action vector to probe. Provide either 3 "
        "translation values, or the full action dimension. May be repeated."
    ),
)
parser.add_argument(
    "--custom_sequence",
    action="append",
    default=[],
    help=(
        "Semicolon-separated action phases to probe, for example "
        "'1,1,0@15;0,0,-1@135'. Each action may provide 3 translation values "
        "or the full action dimension. Phase steps default to --num_steps."
    ),
)
parser.add_argument(
    "--custom_only",
    action="store_true",
    default=False,
    help="Run only custom probes and skip the standard axis probes.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1,
    help="Number of environment steps to apply each action.",
)
parser.add_argument(
    "--lateral_threshold",
    type=float,
    default=0.020,
    help="Lateral success threshold for after-action summary.",
)
parser.add_argument(
    "--orientation_threshold",
    type=float,
    default=0.50,
    help="Orientation success threshold for after-action summary.",
)
parser.add_argument(
    "--depth_threshold",
    type=float,
    default=0.005,
    help="Insertion-depth success threshold for after-action summary.",
)
parser.add_argument(
    "--print_joint_positions",
    action="store_true",
    default=False,
    help="Print mean arm joint positions after each probe, grouped by SFP target.",
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


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


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


def _joint_indices(asset: Any, joint_names: tuple[str, ...]) -> list[int]:
    available = getattr(asset, "joint_names", None)
    if available is None:
        available = getattr(getattr(asset, "data", None), "joint_names", None)
    if available is None:
        raise RuntimeError(f"Asset {asset!r} does not expose joint_names.")
    return [list(available).index(name) for name in joint_names]


def _format_floats(values: torch.Tensor) -> str:
    values_list = values.detach().cpu().tolist()
    return "(" + ", ".join(f"{float(value):.10f}" for value in values_list) + ")"


def _parse_custom_action(raw: str, action_dim: int) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) == 3 and action_dim >= 3:
        values = values + [0.0] * (action_dim - 3)
    if len(values) != action_dim:
        raise ValueError(
            f"Custom action '{raw}' has {len(values)} values after padding. "
            f"Expected 3 or {action_dim} values."
        )
    return values


def _parse_action_phase(raw: str, action_dim: int) -> tuple[list[float], int]:
    action_text, _, steps_text = raw.partition("@")
    steps = args_cli.num_steps
    if steps_text.strip():
        steps = int(steps_text.strip())
    if steps < 1:
        raise ValueError(f"Custom action phase '{raw}' must have at least 1 step.")
    return _parse_custom_action(action_text, action_dim), steps


def _parse_custom_sequence(
    raw: str,
    action_dim: int,
) -> list[tuple[list[float], int]]:
    phases = [
        _parse_action_phase(phase.strip(), action_dim)
        for phase in raw.split(";")
        if phase.strip()
    ]
    if not phases:
        raise ValueError(f"Custom sequence '{raw}' did not contain any phases.")
    return phases


def _success_mask(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        (metrics["lateral"] < args_cli.lateral_threshold)
        & (metrics["orientation"] < args_cli.orientation_threshold)
        & (metrics["depth"] > args_cli.depth_threshold)
    )


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
    trajectory: dict[str, torch.Tensor] | None = None,
) -> None:
    success = _success_mask(after)
    parts = [f"action={label}"]
    for key in ("lateral_x", "lateral_y", "depth", "lateral", "orientation"):
        delta = after[key] - before[key]
        parts.append(f"d_{key}_mean={_mean(delta):+.6f}")
    parts.append(f"after_lateral_mean={_mean(after['lateral']):.6f}")
    parts.append(f"after_depth_mean={_mean(after['depth']):.6f}")
    parts.append(f"after_orientation_mean={_mean(after['orientation']):.6f}")
    parts.append(f"successes={int(success.sum().item())}/{success.numel()}")
    if trajectory is not None:
        ever_success = trajectory["ever_success"]
        first_success_step = trajectory["first_success_step"]
        successful_steps = first_success_step[ever_success]
        if bool(ever_success.any()):
            mean_first_success = _mean(successful_steps.float())
        else:
            mean_first_success = float("nan")
        parts.append(
            f"ever_successes={int(ever_success.sum().item())}/{ever_success.numel()}"
        )
        parts.append(f"mean_first_success_step={mean_first_success:.2f}")
        parts.append(
            f"best_aligned_depth_mean={_mean(trajectory['best_aligned_depth']):.6f}"
        )
    report.line("  " + " ".join(parts))


def _print_joint_summary(
    report: Reporter,
    env: Any,
    target_ids: torch.Tensor,
    success: torch.Tensor,
) -> None:
    robot = env.scene["robot"]
    joint_ids = _joint_indices(robot, ARM_JOINT_NAMES)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    report.line(f"    joint_names={ARM_JOINT_NAMES}")
    report.line(f"    mean_joint_pos_all={_format_floats(joint_pos.mean(dim=0))}")
    if bool(success.any()):
        report.line(
            "    "
            f"mean_joint_pos_success={_format_floats(joint_pos[success].mean(dim=0))}"
        )
    for target_id, target_name in enumerate(mdp.SFP_TARGET_NAMES):
        mask = target_ids.to(joint_pos.device) == target_id
        if not bool(mask.any()):
            continue
        report.line(
            "    "
            f"{target_name}_mean_joint_pos={_format_floats(joint_pos[mask].mean(dim=0))}"
        )


def _empty_trajectory_metrics(
    base_env: Any,
    initial_metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "ever_success": torch.zeros(
            base_env.num_envs, device=base_env.device, dtype=torch.bool
        ),
        "first_success_step": torch.full(
            (base_env.num_envs,), -1, device=base_env.device, dtype=torch.long
        ),
        "best_aligned_depth": initial_metrics["depth"].clone(),
    }


def _update_trajectory_metrics(
    trajectory: dict[str, torch.Tensor],
    metrics: dict[str, torch.Tensor],
    step_index: int,
) -> None:
    success = _success_mask(metrics)
    first_success = success & ~trajectory["ever_success"]
    trajectory["first_success_step"][first_success] = step_index
    trajectory["ever_success"] |= success

    aligned = (
        (metrics["lateral"] < args_cli.lateral_threshold)
        & (metrics["orientation"] < args_cli.orientation_threshold)
    )
    best_candidate = torch.where(
        aligned,
        metrics["depth"],
        trajectory["best_aligned_depth"],
    )
    trajectory["best_aligned_depth"] = torch.maximum(
        trajectory["best_aligned_depth"],
        best_candidate,
    )


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
        report.line(f"custom_action: {args_cli.custom_action}")
        report.line(f"custom_sequence: {args_cli.custom_sequence}")
        report.line(f"custom_only: {args_cli.custom_only}")
        report.line(f"print_joint_positions: {args_cli.print_joint_positions}")
        report.line(f"num_steps: {args_cli.num_steps}")
        report.line(
            "success_thresholds: "
            f"lateral<{args_cli.lateral_threshold} "
            f"orientation<{args_cli.orientation_threshold} "
            f"depth>{args_cli.depth_threshold}"
        )
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

        probes: list[tuple[str, list[tuple[list[float], int]]]] = []
        if not args_cli.custom_only:
            for label, index, value in (
                ("tx+", 0, args_cli.raw_action),
                ("tx-", 0, -args_cli.raw_action),
                ("ty+", 1, args_cli.raw_action),
                ("ty-", 1, -args_cli.raw_action),
                ("tz+", 2, args_cli.raw_action),
                ("tz-", 2, -args_cli.raw_action),
            ):
                action = [0.0] * action_dim
                action[index] = value
                probes.append((label, [(action, args_cli.num_steps)]))
            if action_dim >= 6:
                for label, index, value in (
                    ("rx+", 3, args_cli.raw_action),
                    ("rx-", 3, -args_cli.raw_action),
                    ("ry+", 4, args_cli.raw_action),
                    ("ry-", 4, -args_cli.raw_action),
                    ("rz+", 5, args_cli.raw_action),
                    ("rz-", 5, -args_cli.raw_action),
                ):
                    action = [0.0] * action_dim
                    action[index] = value
                    probes.append((label, [(action, args_cli.num_steps)]))

        for custom_index, raw_custom_action in enumerate(args_cli.custom_action):
            custom_action = _parse_custom_action(raw_custom_action, action_dim)
            custom_label = (
                f"custom{custom_index}[" + ",".join(f"{v:g}" for v in custom_action) + "]"
            )
            probes.append((custom_label, [(custom_action, args_cli.num_steps)]))

        for sequence_index, raw_sequence in enumerate(args_cli.custom_sequence):
            phases = _parse_custom_sequence(raw_sequence, action_dim)
            phase_labels = []
            for action_values, steps in phases:
                action_label = ",".join(f"{value:g}" for value in action_values)
                phase_labels.append(f"{action_label}@{steps}")
            custom_label = f"sequence{sequence_index}[" + ";".join(phase_labels) + "]"
            probes.append((custom_label, phases))

        if not probes:
            raise ValueError(
                "No probes requested. Provide --custom_action, "
                "--custom_sequence, or omit --custom_only."
            )

        report.line("probes:")
        for label, phases in probes:
            report.line(f"  {label}: {phases}")

        with torch.inference_mode():
            for label, phases in probes:
                env.reset()
                before = _sfp_metrics(base_env)
                target_ids = mdp.active_sfp_target_ids(base_env).detach().cpu()
                trajectory = _empty_trajectory_metrics(base_env, before)
                step_index = 0
                for action_values, phase_steps in phases:
                    actions = torch.zeros(
                        env.action_space.shape,
                        device=base_env.device,
                        dtype=torch.float32,
                    )
                    actions[:] = torch.tensor(
                        action_values,
                        device=base_env.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    for _ in range(phase_steps):
                        step_index += 1
                        env.step(actions)
                        _update_trajectory_metrics(
                            trajectory,
                            _sfp_metrics(base_env),
                            step_index,
                        )
                after = _sfp_metrics(base_env)
                _print_summary(report, label, before, after, trajectory)
                success = _success_mask(after)
                if args_cli.print_joint_positions:
                    _print_joint_summary(report, base_env, target_ids, success)
                for target_id, target_name in enumerate(mdp.SFP_TARGET_NAMES):
                    mask = target_ids == target_id
                    if not bool(mask.any()):
                        continue
                    target_success = success[mask]
                    success_text = (
                        f"successes={int(target_success.sum().item())}/"
                        f"{target_success.numel()}"
                    )
                    if trajectory is not None:
                        target_ever_success = trajectory["ever_success"][
                            mask.to(trajectory["ever_success"].device)
                        ]
                        success_text += (
                            f" ever_successes={int(target_ever_success.sum().item())}/"
                            f"{target_ever_success.numel()}"
                        )
                    report.line(
                        "    "
                        f"{target_name}: n={int(mask.sum().item())} "
                        f"{success_text} "
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
