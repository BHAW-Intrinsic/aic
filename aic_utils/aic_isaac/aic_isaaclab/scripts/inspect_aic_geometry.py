# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect AIC task geometry exposed by the Isaac Lab scene.

The first insertion-correctness milestone is to verify whether the imported USD
assets expose semantic plug-tip and port-entrance frames that match the Gazebo
SDF names. This script creates the task, prints the runtime scene/body names,
searches the USD stage for the known SC and SFP semantic names, and writes the
same report to ``logs/aic_geometry/``.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


SEMANTIC_NAMES = (
    "sc_tip_link",
    "sc_port_base_link_entrance",
    "sfp_tip_link",
    "sfp_port_0_link_entrance",
    "sfp_port_1_link_entrance",
)

PRIMARY_ASSET_NAMES = ("robot", "sc_port", "sc_port_2", "nic_card")
PLUG_BODY_TERMS = ("plug", "tip", "sc_tip", "sfp_tip", "connector", "cable")


parser = argparse.ArgumentParser(description="Inspect AIC Isaac Lab geometry.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create.")
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
    help=(
        "Directory for the timestamped inspection log. Defaults to "
        "<repo>/logs/aic_geometry."
    ),
)
parser.add_argument(
    "--no_log_file",
    action="store_true",
    default=False,
    help="Print only to stdout instead of also writing a log file.",
)
parser.add_argument(
    "--max_usd_matches",
    type=int,
    default=200,
    help="Maximum USD prim matches to print for broad plug/tip searches.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import omni.usd
import torch
from pxr import Usd, UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task.mdp import geometry as aic_geometry


class Reporter:
    """Small stdout/file tee for inspection output."""

    def __init__(self, path: Path | None):
        self.path = path
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
    return _repo_root() / "logs" / "aic_geometry" / f"{stamp}_{safe_task}.log"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _tensor_row(value: Any, env_index: int = 0) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    if value.numel() == 0:
        return []
    row = value[env_index] if value.ndim > 1 else value
    return [float(v) for v in row.detach().cpu().flatten().tolist()]


def _format_vector(value: list[float] | None, precision: int = 6) -> str:
    if value is None:
        return "unavailable"
    return "[" + ", ".join(f"{v:.{precision}f}" for v in value) + "]"


def _format_scalar(value: torch.Tensor | None, precision: int = 6) -> str:
    row = _tensor_row(value)
    if not row:
        return "unavailable"
    return f"{row[0]:.{precision}f}"


def _format_optional_scalar(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{precision}f}"


def _tensor_shape(value: torch.Tensor | None) -> str:
    if not isinstance(value, torch.Tensor):
        return "unavailable"
    return str(tuple(value.shape))


def _scene_collection_names(scene: Any, attr_name: str) -> list[str]:
    collection = getattr(scene, attr_name, None)
    if isinstance(collection, dict):
        return sorted(collection.keys())
    return []


def _scene_asset_names(scene: Any) -> list[str]:
    names: set[str] = set()
    for attr_name in (
        "articulations",
        "rigid_objects",
        "deformable_objects",
        "sensors",
        "extras",
    ):
        names.update(_scene_collection_names(scene, attr_name))
    try:
        names.update(scene.keys())
    except Exception:
        pass
    return sorted(names)


def _asset_body_names(asset: Any) -> list[str]:
    body_names = getattr(asset, "body_names", None)
    if body_names is None:
        body_names = getattr(getattr(asset, "data", None), "body_names", None)
    return [str(name) for name in _as_list(body_names)]


def _asset_joint_names(asset: Any) -> list[str]:
    return [str(name) for name in _as_list(getattr(asset, "joint_names", None))]


def _asset_root_pose(asset: Any) -> tuple[list[float] | None, list[float] | None]:
    data = getattr(asset, "data", None)
    if data is None:
        return None, None
    pos = _tensor_row(getattr(data, "root_pos_w", None))
    quat = _tensor_row(getattr(data, "root_quat_w", None))
    if pos is None or quat is None:
        root_state = getattr(data, "root_state_w", None)
        if isinstance(root_state, torch.Tensor) and root_state.shape[-1] >= 7:
            pos = _tensor_row(root_state[:, 0:3])
            quat = _tensor_row(root_state[:, 3:7])
    return pos, quat


def _asset_body_pose(
    asset: Any, body_id: int
) -> tuple[list[float] | None, list[float] | None]:
    data = getattr(asset, "data", None)
    if data is None:
        return None, None
    body_pos = getattr(data, "body_pos_w", None)
    body_quat = getattr(data, "body_quat_w", None)
    if not isinstance(body_pos, torch.Tensor) or not isinstance(body_quat, torch.Tensor):
        return None, None
    if body_pos.ndim < 3 or body_id >= body_pos.shape[1]:
        return None, None
    pos = [float(v) for v in body_pos[0, body_id].detach().cpu().tolist()]
    quat = [float(v) for v in body_quat[0, body_id].detach().cpu().tolist()]
    return pos, quat


def _matches_any(text: str, needles: tuple[str, ...] | list[str]) -> bool:
    lower_text = text.lower()
    return any(needle.lower() in lower_text for needle in needles)


def _print_scene_summary(report: Reporter, scene: Any) -> list[str]:
    report.line("== Scene Collections ==")
    for attr_name in (
        "articulations",
        "rigid_objects",
        "deformable_objects",
        "sensors",
        "extras",
    ):
        names = _scene_collection_names(scene, attr_name)
        report.line(f"{attr_name}: {names if names else '[]'}")

    asset_names = _scene_asset_names(scene)
    report.line()
    report.line("== All Scene Asset Names ==")
    for name in asset_names:
        report.line(f"- {name}")
    return asset_names


def _print_asset_details(report: Reporter, scene: Any, asset_names: list[str]) -> None:
    report.line()
    report.line("== Primary Asset Body Names And Root Poses ==")
    for name in PRIMARY_ASSET_NAMES:
        if name not in asset_names:
            report.line(f"{name}: MISSING from scene")
            continue
        asset = scene[name]
        report.line(f"{name}: {type(asset).__name__}")
        body_names = _asset_body_names(asset)
        joint_names = _asset_joint_names(asset)
        report.line(f"  body_names ({len(body_names)}): {body_names}")
        if joint_names:
            report.line(f"  joint_names ({len(joint_names)}): {joint_names}")
        pos, quat = _asset_root_pose(asset)
        report.line(f"  root_pos_w env0:  {_format_vector(pos)}")
        report.line(f"  root_quat_w env0: {_format_vector(quat)}")


def _print_body_pose_matches(
    report: Reporter, scene: Any, asset_names: list[str]
) -> None:
    report.line()
    report.line("== Available Plug/Tip-Like Body Poses ==")
    found_any = False
    for asset_name in asset_names:
        asset = scene[asset_name]
        if _matches_any(asset_name, PLUG_BODY_TERMS):
            found_any = True
            pos, quat = _asset_root_pose(asset)
            report.line(f"{asset_name} root")
            report.line(f"  root_pos_w env0:  {_format_vector(pos)}")
            report.line(f"  root_quat_w env0: {_format_vector(quat)}")
        body_names = _asset_body_names(asset)
        for body_id, body_name in enumerate(body_names):
            if not _matches_any(body_name, PLUG_BODY_TERMS):
                continue
            found_any = True
            pos, quat = _asset_body_pose(asset, body_id)
            report.line(f"{asset_name}.{body_name} body_id={body_id}")
            report.line(f"  body_pos_w env0:  {_format_vector(pos)}")
            report.line(f"  body_quat_w env0: {_format_vector(quat)}")
    if not found_any:
        report.line("No plug/tip-like runtime body names found.")


def _runtime_name_matches(scene: Any, asset_names: list[str], target: str) -> list[str]:
    matches: list[str] = []
    for asset_name in asset_names:
        if target.lower() in asset_name.lower():
            matches.append(f"asset:{asset_name}")
        asset = scene[asset_name]
        for body_name in _asset_body_names(asset):
            if target.lower() in body_name.lower():
                matches.append(f"body:{asset_name}.{body_name}")
        for joint_name in _asset_joint_names(asset):
            if target.lower() in joint_name.lower():
                matches.append(f"joint:{asset_name}.{joint_name}")
    return matches


def _usd_prim_matches(stage: Any, target: str) -> list[str]:
    matches: list[str] = []
    target_lower = target.lower()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if target_lower in path.lower():
            matches.append(path)
    return matches


def _first_env0_usd_match(stage: Any, target: str) -> str | None:
    """Return the first USD prim path for ``target`` in env_0."""
    matches = _usd_prim_matches(stage, target)
    for path in matches:
        if "/env_0/" in path:
            return path
    return matches[0] if matches else None


def _usd_world_pose(
    stage: Any, prim_path: str
) -> tuple[list[float] | None, list[float] | None]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    try:
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
    except Exception:
        return None, None
    imaginary = rotation.GetImaginary()
    pos = [float(translation[i]) for i in range(3)]
    quat = [float(rotation.GetReal())] + [float(imaginary[i]) for i in range(3)]
    return pos, quat


def _quat_abs_dot(
    quat_a: list[float] | None, quat_b: list[float] | None
) -> float | None:
    if quat_a is None or quat_b is None:
        return None
    norm_a = math.sqrt(sum(v * v for v in quat_a))
    norm_b = math.sqrt(sum(v * v for v in quat_b))
    if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
        return None
    return abs(sum(a * b for a, b in zip(quat_a, quat_b)) / (norm_a * norm_b))


def _quat_angle_error(
    quat_a: list[float] | None, quat_b: list[float] | None
) -> float | None:
    dot = _quat_abs_dot(quat_a, quat_b)
    if dot is None:
        return None
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _position_delta(
    helper_pos: list[float] | None, usd_pos: list[float] | None
) -> tuple[list[float] | None, float | None]:
    if helper_pos is None or usd_pos is None:
        return None, None
    delta = [helper_pos[i] - usd_pos[i] for i in range(3)]
    norm = math.sqrt(sum(value * value for value in delta))
    return delta, norm


def _print_semantic_search(
    report: Reporter, scene: Any, asset_names: list[str], stage: Any
) -> None:
    report.line()
    report.line("== Gazebo Semantic Name Search ==")
    for target in SEMANTIC_NAMES:
        runtime_matches = _runtime_name_matches(scene, asset_names, target)
        usd_matches = _usd_prim_matches(stage, target)
        report.line(f"{target}:")
        report.line(f"  runtime matches: {runtime_matches if runtime_matches else 'none'}")
        report.line(f"  USD prim matches: {usd_matches if usd_matches else 'none'}")


def _print_usd_semantic_world_poses(report: Reporter, stage: Any) -> None:
    report.line()
    report.line("== USD Semantic Frame World Poses Env0 ==")
    for target in SEMANTIC_NAMES:
        prim_path = _first_env0_usd_match(stage, target)
        if prim_path is None:
            report.line(f"{target}: no USD prim match")
            continue
        pos, quat = _usd_world_pose(stage, prim_path)
        report.line(f"{target}: {prim_path}")
        report.line(f"  usd_pos_w:  {_format_vector(pos)}")
        report.line(f"  usd_quat_w: {_format_vector(quat)}")


def _print_broad_usd_search(report: Reporter, stage: Any, max_matches: int) -> None:
    report.line()
    report.line("== Broad USD Plug/Tip/Entrance Search ==")
    terms = ("sc_", "sfp_", "plug", "tip", "entrance", "port")
    matches: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if _matches_any(path, terms):
            matches.append(path)
        if len(matches) >= max_matches:
            break
    for path in matches:
        report.line(f"- {path}")
    if not matches:
        report.line("No broad USD matches found.")
    if len(matches) >= max_matches:
        report.line(f"... stopped after --max_usd_matches={max_matches}")


def _print_geometry_helper_values(report: Reporter, env: Any) -> None:
    report.line()
    report.line("== AIC Geometry Helper Values ==")
    try:
        active_ids = aic_geometry.active_sc_target_ids(env)
        active_names = aic_geometry.active_sc_target_names(env)
        plug_pos, plug_quat = aic_geometry.sc_plug_tip_pose(env)
        port_pos, port_quat = aic_geometry.sc_port_entry_pose(env)
        plug_axis = aic_geometry.sc_plug_axis(env)
        port_axis = aic_geometry.sc_port_insertion_axis(env)
        plug_to_port = aic_geometry.sc_plug_to_port_vector(env)
        lateral_error = aic_geometry.sc_lateral_error(env)
        insertion_depth = aic_geometry.sc_insertion_depth(env)
        orientation_error = aic_geometry.sc_orientation_error(env)
    except Exception as exc:
        report.line(f"Geometry helper inspection failed: {type(exc).__name__}: {exc}")
        return

    report.line(f"active_sc_target_ids all envs: {active_ids.detach().cpu().tolist()}")
    report.line(f"active_sc_target_names: {active_names}")
    report.line(f"sc_port_entry_pos_local: {aic_geometry.SC_PORT_ENTRY_POS_LOCAL}")
    report.line(
        "sc_port_insertion_axis_local: "
        f"{aic_geometry.SC_PORT_INSERTION_AXIS_LOCAL}"
    )
    report.line(f"sc_plug_axis_local: {aic_geometry.SC_PLUG_AXIS_LOCAL}")
    report.line("helper tensor shapes:")
    report.line(f"  plug_tip_pos_w:        {_tensor_shape(plug_pos)}")
    report.line(f"  plug_tip_quat_w:       {_tensor_shape(plug_quat)}")
    report.line(f"  port_entry_pos_w:      {_tensor_shape(port_pos)}")
    report.line(f"  port_entry_quat_w:     {_tensor_shape(port_quat)}")
    report.line(f"  plug_axis_w:           {_tensor_shape(plug_axis)}")
    report.line(f"  port_insertion_axis_w: {_tensor_shape(port_axis)}")
    report.line(f"  plug_to_port_vec_w:    {_tensor_shape(plug_to_port)}")
    report.line(f"  lateral_error:         {_tensor_shape(lateral_error)}")
    report.line(f"  insertion_depth:       {_tensor_shape(insertion_depth)}")
    report.line(f"  orientation_error:     {_tensor_shape(orientation_error)}")
    report.line(f"plug_tip_pos_w env0:       {_format_vector(_tensor_row(plug_pos))}")
    report.line(f"plug_tip_quat_w env0:      {_format_vector(_tensor_row(plug_quat))}")
    report.line(f"port_entry_pos_w env0:     {_format_vector(_tensor_row(port_pos))}")
    report.line(f"port_entry_quat_w env0:    {_format_vector(_tensor_row(port_quat))}")
    report.line(f"plug_axis_w env0:          {_format_vector(_tensor_row(plug_axis))}")
    report.line(f"port_insertion_axis_w env0:{_format_vector(_tensor_row(port_axis))}")
    report.line(f"plug_to_port_vec_w env0:   {_format_vector(_tensor_row(plug_to_port))}")
    report.line(f"lateral_error env0:        {_format_scalar(lateral_error)}")
    report.line(f"insertion_depth env0:      {_format_scalar(insertion_depth)}")
    report.line(f"orientation_error env0:    {_format_scalar(orientation_error)}")

    report.line("per-target SC helper poses env0:")
    for target_name in aic_geometry.SC_TARGET_NAMES:
        target_pos, target_quat = aic_geometry.sc_port_entry_pose_for_target(
            env, target_name
        )
        target_axis = aic_geometry.sc_port_insertion_axis_for_target(env, target_name)
        report.line(f"  {target_name}:")
        report.line(
            f"    port_entry_pos_w:      {_format_vector(_tensor_row(target_pos))}"
        )
        report.line(
            f"    port_entry_quat_w:     {_format_vector(_tensor_row(target_quat))}"
        )
        report.line(
            f"    port_insertion_axis_w: {_format_vector(_tensor_row(target_axis))}"
        )

    report.line()
    report.line("== AIC SFP Geometry Helper Values ==")
    try:
        active_ids = aic_geometry.active_sfp_target_ids(env)
        active_names = aic_geometry.active_sfp_target_names(env)
        plug_pos, plug_quat = aic_geometry.sfp_plug_tip_pose(env)
        port_pos, port_quat = aic_geometry.sfp_port_entry_pose(env)
        plug_axis = aic_geometry.sfp_plug_axis(env)
        port_axis = aic_geometry.sfp_port_insertion_axis(env)
        plug_to_port = aic_geometry.sfp_plug_to_port_vector(env)
        lateral_error = aic_geometry.sfp_lateral_error(env)
        insertion_depth = aic_geometry.sfp_insertion_depth(env)
        orientation_error = aic_geometry.sfp_orientation_error(env)
    except Exception as exc:
        report.line(
            f"SFP geometry helper inspection failed: {type(exc).__name__}: {exc}"
        )
        return

    report.line(f"active_sfp_target_ids all envs: {active_ids.detach().cpu().tolist()}")
    report.line(f"active_sfp_target_names: {active_names}")
    report.line(f"sfp_port_link_pos_local: {aic_geometry.SFP_PORT_LINK_POS_LOCAL}")
    report.line(f"sfp_port_entry_pos_local: {aic_geometry.SFP_PORT_ENTRY_POS_LOCAL}")
    report.line(
        "sfp_port_insertion_axis_local: "
        f"{aic_geometry.SFP_PORT_INSERTION_AXIS_LOCAL}"
    )
    report.line(f"sfp_plug_axis_local: {aic_geometry.SFP_PLUG_AXIS_LOCAL}")
    report.line("SFP helper tensor shapes:")
    report.line(f"  plug_tip_pos_w:        {_tensor_shape(plug_pos)}")
    report.line(f"  plug_tip_quat_w:       {_tensor_shape(plug_quat)}")
    report.line(f"  port_entry_pos_w:      {_tensor_shape(port_pos)}")
    report.line(f"  port_entry_quat_w:     {_tensor_shape(port_quat)}")
    report.line(f"  plug_axis_w:           {_tensor_shape(plug_axis)}")
    report.line(f"  port_insertion_axis_w: {_tensor_shape(port_axis)}")
    report.line(f"  plug_to_port_vec_w:    {_tensor_shape(plug_to_port)}")
    report.line(f"  lateral_error:         {_tensor_shape(lateral_error)}")
    report.line(f"  insertion_depth:       {_tensor_shape(insertion_depth)}")
    report.line(f"  orientation_error:     {_tensor_shape(orientation_error)}")
    report.line(f"sfp_plug_tip_pos_w env0:       {_format_vector(_tensor_row(plug_pos))}")
    report.line(f"sfp_plug_tip_quat_w env0:      {_format_vector(_tensor_row(plug_quat))}")
    report.line(f"sfp_port_entry_pos_w env0:     {_format_vector(_tensor_row(port_pos))}")
    report.line(f"sfp_port_entry_quat_w env0:    {_format_vector(_tensor_row(port_quat))}")
    report.line(f"sfp_plug_axis_w env0:          {_format_vector(_tensor_row(plug_axis))}")
    report.line(f"sfp_port_insertion_axis_w env0:{_format_vector(_tensor_row(port_axis))}")
    report.line(f"sfp_plug_to_port_vec_w env0:   {_format_vector(_tensor_row(plug_to_port))}")
    report.line(f"sfp_lateral_error env0:        {_format_scalar(lateral_error)}")
    report.line(f"sfp_insertion_depth env0:      {_format_scalar(insertion_depth)}")
    report.line(f"sfp_orientation_error env0:    {_format_scalar(orientation_error)}")

    report.line("per-target SFP helper poses env0:")
    for target_name in aic_geometry.SFP_TARGET_NAMES:
        target_pos, target_quat = aic_geometry.sfp_port_entry_pose_for_target(
            env, target_name
        )
        target_axis = aic_geometry.sfp_port_insertion_axis_for_target(env, target_name)
        report.line(f"  {target_name}:")
        report.line(
            f"    port_entry_pos_w:      {_format_vector(_tensor_row(target_pos))}"
        )
        report.line(
            f"    port_entry_quat_w:     {_format_vector(_tensor_row(target_quat))}"
        )
        report.line(
            f"    port_insertion_axis_w: {_format_vector(_tensor_row(target_axis))}"
        )


def _print_helper_usd_deltas(report: Reporter, env: Any, stage: Any) -> None:
    report.line()
    report.line("== SFP Helper Vs USD Semantic Frames Env0 ==")
    report.line(
        "SC plug training currently uses the gripped virtual helper, so this "
        "section compares only the SFP helper frames that should match USD."
    )
    helper_specs = (
        ("sfp_tip_link", "sfp plug tip", aic_geometry.sfp_plug_tip_pose),
        (
            "sfp_port_0_link_entrance",
            "sfp_port_0 entry",
            lambda env: aic_geometry.sfp_port_entry_pose_for_target(env, "sfp_port_0"),
        ),
        (
            "sfp_port_1_link_entrance",
            "sfp_port_1 entry",
            lambda env: aic_geometry.sfp_port_entry_pose_for_target(env, "sfp_port_1"),
        ),
    )
    for target, label, helper_fn in helper_specs:
        prim_path = _first_env0_usd_match(stage, target)
        if prim_path is None:
            report.line(f"{label}: no USD prim match for {target}")
            continue
        try:
            helper_pos_tensor, helper_quat_tensor = helper_fn(env)
        except Exception as exc:
            report.line(f"{label}: helper failed: {type(exc).__name__}: {exc}")
            continue
        helper_pos = _tensor_row(helper_pos_tensor)
        helper_quat = _tensor_row(helper_quat_tensor)
        usd_pos, usd_quat = _usd_world_pose(stage, prim_path)
        pos_delta, pos_delta_norm = _position_delta(helper_pos, usd_pos)
        quat_angle_error = _quat_angle_error(helper_quat, usd_quat)
        report.line(f"{label}: {prim_path}")
        report.line(f"  helper_pos_w:       {_format_vector(helper_pos)}")
        report.line(f"  usd_pos_w:          {_format_vector(usd_pos)}")
        report.line(f"  helper_minus_usd_m: {_format_vector(pos_delta)}")
        report.line(
            f"  pos_delta_norm_m:   {_format_optional_scalar(pos_delta_norm)}"
        )
        report.line(f"  helper_quat_w:      {_format_vector(helper_quat)}")
        report.line(f"  usd_quat_w:         {_format_vector(usd_quat)}")
        report.line(
            "  quat_angle_error_rad: "
            f"{_format_optional_scalar(quat_angle_error)}"
        )


def main() -> None:
    """Create the task and print geometry names/poses needed by stage 0."""
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
        report.line("== AIC Geometry Inspection ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"num_envs: {args_cli.num_envs}")
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

        with torch.inference_mode():
            env.reset()
            zero_actions = torch.zeros(
                env.action_space.shape, device=base_env.device, dtype=torch.float32
            )
            env.step(zero_actions)

        scene = base_env.scene
        stage = omni.usd.get_context().get_stage()
        asset_names = _print_scene_summary(report, scene)
        _print_asset_details(report, scene, asset_names)
        _print_body_pose_matches(report, scene, asset_names)
        _print_semantic_search(report, scene, asset_names, stage)
        _print_usd_semantic_world_poses(report, stage)
        _print_geometry_helper_values(report, base_env)
        _print_helper_usd_deltas(report, base_env, stage)
        _print_broad_usd_search(report, stage, args_cli.max_usd_matches)

        report.line()
        report.line("== Done ==")
    finally:
        if env is not None:
            env.close()
        report.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
