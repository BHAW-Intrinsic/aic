# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether privileged connector geometry can drive the existing IK action."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a scripted connector insertion controller.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument(
    "--connector",
    choices=("auto", "sc", "sfp"),
    default="auto",
    help="Connector geometry to script. 'auto' selects SFP for SFP tasks, else SC.",
)
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
    "--control_frame",
    choices=("tip", "wrist_legacy"),
    default="tip",
    help=(
        "Use 'tip' to solve the wrist target from the desired SC tip pose. "
        "Use 'wrist_legacy' to send tip deltas directly as action-body deltas."
    ),
)
parser.add_argument(
    "--action_body_name",
    type=str,
    default="gripper_tcp",
    help="Articulation body controlled by the differential IK action.",
)
parser.add_argument(
    "--diagnostic_body_names",
    type=str,
    default=(
        "wrist_3_link,gripper_tcp,ati_tool_link,tool0,sc_plug_link,"
        "sfp_module_link,sfp_tip_link"
    ),
    help=(
        "Comma-separated body names whose relative transforms to the active "
        "connector tip are logged at reset and summary time."
    ),
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
parser.add_argument("--success_lateral_threshold", type=float, default=None)
parser.add_argument("--success_orientation_threshold", type=float, default=None)
parser.add_argument("--success_depth_threshold", type=float, default=None)
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


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

SC_DEFAULT_SUCCESS_LATERAL = 0.005
SC_DEFAULT_SUCCESS_ORIENTATION = 0.20
SC_DEFAULT_SUCCESS_DEPTH = 0.012
SFP_DEFAULT_SUCCESS_LATERAL = 0.004
SFP_DEFAULT_SUCCESS_ORIENTATION = 0.20
SFP_DEFAULT_SUCCESS_DEPTH = 0.015


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


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in Isaac's wxyz convention."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


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


def _quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle, dim=-1)
    axis = axis_angle / torch.clamp(angle.unsqueeze(-1), min=1.0e-9)
    half_angle = 0.5 * angle
    quat = torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            axis * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )
    identity = torch.zeros_like(quat)
    identity[:, 0] = 1.0
    quat = torch.where(angle.unsqueeze(-1) > 1.0e-9, quat, identity)
    return _quat_normalize(quat)


def _axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _quat_normalize(quat)
    quat = torch.where(quat[:, 0:1] < 0.0, -quat, quat)
    vec = quat[:, 1:]
    sin_half = torch.norm(vec, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, torch.clamp(quat[:, 0], min=1.0e-9))
    axis = vec / torch.clamp(sin_half.unsqueeze(-1), min=1.0e-9)
    axis_angle = axis * angle.unsqueeze(-1)
    return torch.where(sin_half.unsqueeze(-1) > 1.0e-9, axis_angle, 0.0)


def _summary(value: torch.Tensor) -> str:
    detached = value.detach().float()
    return (
        f"mean={detached.mean().item():.6f} "
        f"min={detached.min().item():.6f} max={detached.max().item():.6f}"
    )


def _vector_component_summary(value: torch.Tensor, labels: tuple[str, ...]) -> str:
    parts = []
    for index, label in enumerate(labels):
        parts.append(f"{label}({_summary(value[:, index])})")
    return " ".join(parts)


def _body_index(asset: Any, body_name: str) -> int:
    body_names = getattr(asset, "body_names", None)
    if body_names is None:
        body_names = getattr(getattr(asset, "data", None), "body_names", None)
    if body_names is None:
        raise RuntimeError(f"Asset {asset!r} does not expose body_names.")
    try:
        return list(body_names).index(body_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Body '{body_name}' not found. Available bodies: {list(body_names)}"
        ) from exc


def _joint_indices(asset: Any, joint_names: tuple[str, ...]) -> list[int]:
    available = getattr(asset, "joint_names", None)
    if available is None:
        available = getattr(getattr(asset, "data", None), "joint_names", None)
    if available is None:
        raise RuntimeError(f"Asset {asset!r} does not expose joint_names.")

    available_list = list(available)
    indices = []
    for joint_name in joint_names:
        try:
            indices.append(available_list.index(joint_name))
        except ValueError as exc:
            raise RuntimeError(
                f"Joint '{joint_name}' not found. Available joints: {available_list}"
            ) from exc
    return indices


def _body_pose(env: Any, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    body_id = _body_index(robot, body_name)
    return robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]


def _connector_name() -> str:
    if args_cli.connector != "auto":
        return args_cli.connector
    return "sfp" if "sfp" in args_cli.task.lower() else "sc"


def _connector_label() -> str:
    return _connector_name().upper()


def _target_names() -> tuple[str, ...]:
    return mdp.SFP_TARGET_NAMES if _connector_name() == "sfp" else mdp.SC_TARGET_NAMES


def _active_target_ids(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.active_sfp_target_ids(env)
    return mdp.active_sc_target_ids(env)


def _plug_tip_pose(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if _connector_name() == "sfp":
        return mdp.sfp_plug_tip_pose(env)
    return mdp.sc_plug_tip_pose(env)


def _plug_axis(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.sfp_plug_axis(env)
    return mdp.sc_plug_axis(env)


def _port_entry_pose(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if _connector_name() == "sfp":
        return mdp.sfp_port_entry_pose(env)
    return mdp.sc_port_entry_pose(env)


def _port_insertion_axis(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.sfp_port_insertion_axis(env)
    return mdp.sc_port_insertion_axis(env)


def _lateral_error(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.sfp_lateral_error(env)
    return mdp.sc_lateral_error(env)


def _orientation_error(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.sfp_orientation_error(env)
    return mdp.sc_orientation_error(env)


def _insertion_depth(env: Any) -> torch.Tensor:
    if _connector_name() == "sfp":
        return mdp.sfp_insertion_depth(env)
    return mdp.sc_insertion_depth(env)


def _port_frame_delta(env: Any) -> torch.Tensor:
    plug_pos_w, _ = _plug_tip_pose(env)
    entry_pos_w, entry_quat_w = _port_entry_pose(env)
    return _quat_apply(_quat_conjugate(entry_quat_w), plug_pos_w - entry_pos_w)


def _success_thresholds() -> tuple[float, float, float]:
    if _connector_name() == "sfp":
        lateral_default = SFP_DEFAULT_SUCCESS_LATERAL
        orientation_default = SFP_DEFAULT_SUCCESS_ORIENTATION
        depth_default = SFP_DEFAULT_SUCCESS_DEPTH
    else:
        lateral_default = SC_DEFAULT_SUCCESS_LATERAL
        orientation_default = SC_DEFAULT_SUCCESS_ORIENTATION
        depth_default = SC_DEFAULT_SUCCESS_DEPTH
    lateral = (
        lateral_default
        if args_cli.success_lateral_threshold is None
        else args_cli.success_lateral_threshold
    )
    orientation = (
        orientation_default
        if args_cli.success_orientation_threshold is None
        else args_cli.success_orientation_threshold
    )
    depth = (
        depth_default
        if args_cli.success_depth_threshold is None
        else args_cli.success_depth_threshold
    )
    return lateral, orientation, depth


def _body_to_tip_pose(env: Any, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    body_pos_w, body_quat_w = _body_pose(env, body_name)
    tip_pos_w, tip_quat_w = _plug_tip_pose(env)
    body_inv = _quat_conjugate(body_quat_w)
    rel_pos = _quat_apply(body_inv, tip_pos_w - body_pos_w)
    rel_quat = _quat_mul(body_inv, tip_quat_w)
    return rel_pos, _quat_normalize(rel_quat)


def _parse_body_names(names: str) -> list[str]:
    parsed = []
    for name in names.split(","):
        name = name.strip()
        if name and name not in parsed:
            parsed.append(name)
    if args_cli.action_body_name not in parsed:
        parsed.insert(0, args_cli.action_body_name)
    return parsed


def _available_body_names(env: Any) -> list[str]:
    robot = env.scene["robot"]
    body_names = getattr(robot, "body_names", None)
    if body_names is None:
        body_names = getattr(getattr(robot, "data", None), "body_names", None)
    return [str(name) for name in body_names] if body_names is not None else []


def _diagnostic_offsets(
    env: Any,
    body_names: list[str],
    report: Reporter,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    available = set(_available_body_names(env))
    offsets = {}
    for body_name in body_names:
        if body_name not in available:
            report.line(f"diagnostic_body_unavailable: {body_name}")
            continue
        offsets[body_name] = _body_to_tip_pose(env, body_name)
    return offsets


def _report_diagnostic_offsets(
    report: Reporter,
    prefix: str,
    offsets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for body_name, (rel_pos, rel_quat) in offsets.items():
        report.line(
            f"{prefix}_{body_name}_to_{_connector_name()}_tip_pos env0: "
            f"{rel_pos[0].detach().cpu().tolist()}"
        )
        report.line(
            f"{prefix}_{body_name}_to_{_connector_name()}_tip_quat env0: "
            f"{rel_quat[0].detach().cpu().tolist()}"
        )


def _root_frame_pose(
    env: Any,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_inv = _quat_conjugate(robot.data.root_quat_w)
    pos_root = _quat_apply(root_inv, pos_w - robot.data.root_pos_w)
    quat_root = _quat_mul(root_inv, quat_w)
    return pos_root, _quat_normalize(quat_root)


def _scripted_actions(env: Any, inactive: torch.Tensor) -> torch.Tensor:
    """Compute raw relative-IK actions from privileged insertion geometry."""
    robot = env.scene["robot"]

    plug_pos_w, plug_quat_w = _plug_tip_pose(env)
    plug_axis_w = _plug_axis(env)
    entry_pos_w, _ = _port_entry_pose(env)
    port_axis_w = _port_insertion_axis(env)

    lateral = _lateral_error(env)
    orientation = _orientation_error(env)
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

    if args_cli.control_frame == "tip":
        desired_tip_pos_w = plug_pos_w + delta_pos_w
        desired_tip_quat_w = _quat_mul(_quat_from_axis_angle(rot_vec_w), plug_quat_w)

        body_pos_w, body_quat_w = _body_pose(env, args_cli.action_body_name)
        tip_pos_in_body, tip_quat_in_body = _body_to_tip_pose(
            env, args_cli.action_body_name
        )
        desired_body_quat_w = _quat_mul(
            desired_tip_quat_w, _quat_conjugate(tip_quat_in_body)
        )
        desired_body_pos_w = desired_tip_pos_w - _quat_apply(
            desired_body_quat_w, tip_pos_in_body
        )

        body_pos_root, body_quat_root = _root_frame_pose(env, body_pos_w, body_quat_w)
        desired_body_pos_root, desired_body_quat_root = _root_frame_pose(
            env, desired_body_pos_w, desired_body_quat_w
        )
        delta_pos_root = desired_body_pos_root - body_pos_root
        delta_quat_root = _quat_mul(
            desired_body_quat_root, _quat_conjugate(body_quat_root)
        )
        rot_vec_root = _axis_angle_from_quat(delta_quat_root)
        rot_vec_root = _clip_by_norm(rot_vec_root, args_cli.max_rotation_step)
    else:
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
    lateral_threshold, orientation_threshold, depth_threshold = _success_thresholds()
    if _connector_name() == "sfp":
        return mdp.sfp_insertion_success_mask(
            env,
            lateral_threshold=lateral_threshold,
            orientation_threshold=orientation_threshold,
            depth_threshold=depth_threshold,
        )
    return mdp.sc_insertion_success_mask(
        env,
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
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
        if hasattr(env_cfg.terminations, "sfp_insertion_success"):
            env_cfg.terminations.sfp_insertion_success = None

    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action.body_name = args_cli.action_body_name
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.body_name = args_cli.action_body_name

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
        report.line(f"== AIC Scripted {_connector_label()} Insertion Check ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"connector: {_connector_name()}")
        report.line(f"num_envs: {args_cli.num_envs}")
        report.line(f"max_steps: {args_cli.max_steps}")
        report.line(f"control_frame: {args_cli.control_frame}")
        report.line(f"action_body_name: {args_cli.action_body_name}")
        lateral_threshold, orientation_threshold, depth_threshold = _success_thresholds()
        report.line(
            "success_thresholds: "
            f"lateral={lateral_threshold} "
            f"orientation={orientation_threshold} depth={depth_threshold}"
        )
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        report.line(f"action_space: {env.action_space}")
        env.reset()

        robot = env.scene["robot"]
        arm_joint_indices = _joint_indices(robot, ARM_JOINT_NAMES)
        target_ids = _active_target_ids(env).detach().clone()
        diagnostic_body_names = _parse_body_names(args_cli.diagnostic_body_names)
        initial_offsets = _diagnostic_offsets(env, diagnostic_body_names, report)
        _report_diagnostic_offsets(report, "initial", initial_offsets)
        first_success_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        first_success_joint_pos = torch.full(
            (env.num_envs, len(arm_joint_indices)),
            float("nan"),
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )

        success = _success_mask(env)
        first_success_step[success] = 0
        if bool(success.any()):
            first_success_joint_pos[success] = robot.data.joint_pos[success][
                :, arm_joint_indices
            ]

        for step in range(args_cli.max_steps + 1):
            lateral = _lateral_error(env)
            orientation = _orientation_error(env)
            depth = _insertion_depth(env)
            port_delta = _port_frame_delta(env)
            success = _success_mask(env)
            newly_successful = success & (first_success_step < 0)
            first_success_step[newly_successful] = step
            if bool(newly_successful.any()):
                first_success_joint_pos[newly_successful] = robot.data.joint_pos[
                    newly_successful
                ][:, arm_joint_indices]

            if step == 0 or step % args_cli.report_every == 0 or bool(success.all()):
                report.line(
                    f"step={step} successes={int(success.sum().item())}/{env.num_envs} "
                    f"lateral({_summary(lateral)}) "
                    f"orientation({_summary(orientation)}) "
                    f"depth({_summary(depth)}) "
                    f"port_delta({_vector_component_summary(port_delta, ('x', 'y', 'z'))})"
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
        for target_id, target_name in enumerate(_target_names()):
            mask = target_ids == target_id
            episodes = int(mask.sum().item())
            successes = int(((first_success_step >= 0) & mask).sum().item())
            rate = successes / episodes if episodes else float("nan")
            report.line(
                f"  {target_name}: episodes={episodes} "
                f"successes={successes} success_rate={rate:.6f}"
            )
        report.line(f"arm_joint_names: {list(ARM_JOINT_NAMES)}")
        for env_id in range(env.num_envs):
            step = int(first_success_step[env_id].detach().cpu().item())
            if step < 0:
                continue
            target_name = _target_names()[int(target_ids[env_id].cpu().item())]
            values = first_success_joint_pos[env_id].detach().cpu().tolist()
            report.line(
                f"first_success_joint_pos env={env_id} target={target_name} "
                f"step={step}: {values}"
            )
        report.line("first_success_joint_pos_mean_per_target:")
        for target_id, target_name in enumerate(_target_names()):
            mask = (target_ids == target_id) & (first_success_step >= 0)
            if not bool(mask.any()):
                report.line(f"  {target_name}: unavailable")
                continue
            values = first_success_joint_pos[mask].mean(dim=0).detach().cpu().tolist()
            report.line(f"  {target_name}: {values}")

        final_lateral = _lateral_error(env)
        final_orientation = _orientation_error(env)
        final_depth = _insertion_depth(env)
        final_port_delta = _port_frame_delta(env)
        final_offsets = _diagnostic_offsets(env, diagnostic_body_names, report)
        report.line(f"final_lateral: {_summary(final_lateral)}")
        report.line(f"final_orientation: {_summary(final_orientation)}")
        report.line(f"final_depth: {_summary(final_depth)}")
        report.line(
            "final_port_frame_delta: "
            f"{_vector_component_summary(final_port_delta, ('x', 'y', 'z'))}"
        )
        for body_name, (final_pos, _) in final_offsets.items():
            if body_name not in initial_offsets:
                continue
            initial_pos, _ = initial_offsets[body_name]
            offset_drift = torch.norm(final_pos - initial_pos, dim=-1)
            report.line(
                f"{body_name}_to_{_connector_name()}_tip_pos_drift: "
                f"{_summary(offset_drift)}"
            )
        _report_diagnostic_offsets(report, "final", final_offsets)
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
