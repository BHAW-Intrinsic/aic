# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared plug-to-port geometry helpers for the AIC task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


SC_TARGET_NAMES = ("sc_port", "sc_port_2")
SC_ACTIVE_TARGET_ATTR = "aic_active_sc_target_ids"
SC_PLUG_TIP_BODY = "sc_tip_link"
SC_PLUG_AXIS_LOCAL = (0.0, 0.0, 1.0)
SC_PREV_DISTANCE_ATTR = "aic_prev_sc_distance"
SC_PREV_LATERAL_ATTR = "aic_prev_sc_lateral_error"
SC_PREV_ORIENTATION_ATTR = "aic_prev_sc_orientation_error"
SC_PREV_DEPTH_ATTR = "aic_prev_sc_insertion_depth"

# Derived from aic_assets/models/SC Port/model.sdf:
# - sc_port_base_link pose relative to sc_port_link:
#   xyz=(0, -0.002, 0), rpy=(pi/2, pi, 0)
# - sc_port_base_link_entrance pose relative to sc_port_base_link:
#   xyz=(0, 0, -0.01564), rpy=(0, 0, 0)
# With the SDF Rz*Ry*Rx convention, the entrance lies at approximately
# (0, 0.01364, 0) in the sc_port_link frame.
SC_PORT_ENTRY_POS_LOCAL = (0.0, 0.01364, 0.0)
SC_PORT_ENTRY_QUAT_LOCAL = (0.0, 0.0, 0.7071067811865476, -0.7071067811865476)

# The entrance is on the +Y face of the SC port model, so the insertion direction
# from the entrance into the port is -Y in the sc_port_link frame.
SC_PORT_INSERTION_AXIS_LOCAL = (0.0, -1.0, 0.0)


def _device(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.device:
    return torch.device(getattr(env, "device", "cpu"))


def _constant_tensor(
    values: tuple[float, ...], env: ManagerBasedEnv | ManagerBasedRLEnv
) -> torch.Tensor:
    return torch.tensor(values, device=_device(env), dtype=torch.float32)


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1.0e-9)


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


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in Isaac's wxyz convention."""
    quat = _quat_normalize(quat)
    q_vec = quat[..., 1:]
    q_w = quat[..., 0:1]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _expand_vec(
    values: tuple[float, float, float],
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    num_envs: int | None = None,
) -> torch.Tensor:
    if num_envs is None:
        num_envs = env.num_envs
    return _constant_tensor(values, env).unsqueeze(0).expand(num_envs, -1)


def _expand_quat(
    values: tuple[float, float, float, float],
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    num_envs: int | None = None,
) -> torch.Tensor:
    if num_envs is None:
        num_envs = env.num_envs
    quat = _constant_tensor(values, env)
    quat = _quat_normalize(quat.unsqueeze(0)).squeeze(0)
    return quat.unsqueeze(0).expand(num_envs, -1)


def _compose_pose(
    parent_pos_w: torch.Tensor,
    parent_quat_w: torch.Tensor,
    child_pos_parent: torch.Tensor,
    child_quat_parent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    child_pos_w = parent_pos_w + _quat_apply(parent_quat_w, child_pos_parent)
    child_quat_w = _quat_mul(parent_quat_w, child_quat_parent)
    return child_pos_w, _quat_normalize(child_quat_w)


def _body_index(asset, body_name: str) -> int:
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


def _gather_active(values: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    gather_ids = target_ids.view(-1, 1, 1).expand(-1, 1, values.shape[-1])
    return values.gather(dim=1, index=gather_ids).squeeze(1)


def active_sc_target_ids(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env active SC target ids, defaulting to ``sc_port``."""
    target_ids = getattr(env, SC_ACTIVE_TARGET_ATTR, None)
    if not isinstance(target_ids, torch.Tensor) or target_ids.shape[0] != env.num_envs:
        target_ids = torch.zeros(env.num_envs, device=_device(env), dtype=torch.long)
        setattr(env, SC_ACTIVE_TARGET_ATTR, target_ids)
    return target_ids


def active_sc_target_names(env: ManagerBasedEnv | ManagerBasedRLEnv) -> list[str]:
    """Return active SC target names for debugging."""
    target_ids = active_sc_target_ids(env).detach().cpu().tolist()
    return [SC_TARGET_NAMES[int(target_id)] for target_id in target_ids]


def sample_active_sc_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Sample the active SC target for each resetting environment."""
    active_ids = active_sc_target_ids(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=active_ids.device)
    active_ids[env_ids] = torch.randint(
        low=0,
        high=len(SC_TARGET_NAMES),
        size=(len(env_ids),),
        device=active_ids.device,
        dtype=active_ids.dtype,
    )


def _reset_metric_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    attr_name: str,
    values: torch.Tensor,
) -> None:
    buffer = getattr(env, attr_name, None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.shape != (env.num_envs,)
        or buffer.device != values.device
    ):
        buffer = torch.zeros(env.num_envs, device=values.device, dtype=values.dtype)
        setattr(env, attr_name, buffer)
    buffer[env_ids] = values[env_ids].detach()


def reset_sc_progress_buffers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Initialize stateful SC progress reward buffers for resetting envs."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=_device(env))
    _reset_metric_buffer(
        env,
        env_ids,
        SC_PREV_DISTANCE_ATTR,
        torch.norm(sc_plug_to_port_vector(env), dim=-1),
    )
    _reset_metric_buffer(env, env_ids, SC_PREV_LATERAL_ATTR, sc_lateral_error(env))
    _reset_metric_buffer(
        env, env_ids, SC_PREV_ORIENTATION_ATTR, sc_orientation_error(env)
    )
    _reset_metric_buffer(env, env_ids, SC_PREV_DEPTH_ATTR, sc_insertion_depth(env))


def sc_plug_tip_pose(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    asset_name: str = "robot",
    body_name: str = SC_PLUG_TIP_BODY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the SC plug tip pose in world frame."""
    asset = env.scene[asset_name]
    body_id = _body_index(asset, body_name)
    return asset.data.body_pos_w[:, body_id], asset.data.body_quat_w[:, body_id]


def sc_plug_axis(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    asset_name: str = "robot",
    body_name: str = SC_PLUG_TIP_BODY,
) -> torch.Tensor:
    """Return the SC plug insertion axis in world frame."""
    _, plug_quat_w = sc_plug_tip_pose(env, asset_name=asset_name, body_name=body_name)
    local_axis = _expand_vec(SC_PLUG_AXIS_LOCAL, env)
    return _quat_apply(plug_quat_w, local_axis)


def active_sc_port_root_pose(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_names: tuple[str, ...] = SC_TARGET_NAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active SC port root pose in world frame."""
    target_ids = active_sc_target_ids(env)
    root_pos = torch.stack(
        [env.scene[name].data.root_pos_w for name in target_names], dim=1
    )
    root_quat = torch.stack(
        [env.scene[name].data.root_quat_w for name in target_names], dim=1
    )
    return _gather_active(root_pos, target_ids), _gather_active(root_quat, target_ids)


def sc_port_root_pose_for_target(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a named SC port root pose in world frame."""
    asset = env.scene[target_name]
    return asset.data.root_pos_w, asset.data.root_quat_w


def _sc_port_entry_pose_from_root(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    port_pos_w: torch.Tensor,
    port_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    entry_pos_local = _expand_vec(SC_PORT_ENTRY_POS_LOCAL, env)
    entry_quat_local = _expand_quat(SC_PORT_ENTRY_QUAT_LOCAL, env)
    return _compose_pose(port_pos_w, port_quat_w, entry_pos_local, entry_quat_local)


def sc_port_entry_pose(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_names: tuple[str, ...] = SC_TARGET_NAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active SC port entrance helper pose in world frame."""
    port_pos_w, port_quat_w = active_sc_port_root_pose(env, target_names=target_names)
    return _sc_port_entry_pose_from_root(env, port_pos_w, port_quat_w)


def sc_port_entry_pose_for_target(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a named SC port entrance helper pose in world frame."""
    port_pos_w, port_quat_w = sc_port_root_pose_for_target(env, target_name)
    return _sc_port_entry_pose_from_root(env, port_pos_w, port_quat_w)


def sc_port_insertion_axis(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_names: tuple[str, ...] = SC_TARGET_NAMES,
) -> torch.Tensor:
    """Return the active SC port insertion axis in world frame."""
    port_pos_w, port_quat_w = active_sc_port_root_pose(env, target_names=target_names)
    del port_pos_w
    local_axis = _expand_vec(SC_PORT_INSERTION_AXIS_LOCAL, env)
    return _quat_apply(port_quat_w, local_axis)


def sc_port_insertion_axis_for_target(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    target_name: str,
) -> torch.Tensor:
    """Return a named SC port insertion axis in world frame."""
    port_pos_w, port_quat_w = sc_port_root_pose_for_target(env, target_name)
    del port_pos_w
    local_axis = _expand_vec(SC_PORT_INSERTION_AXIS_LOCAL, env)
    return _quat_apply(port_quat_w, local_axis)


def sc_plug_to_port_vector(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.Tensor:
    """Return vector from SC plug tip to active SC port entrance in world frame."""
    plug_pos_w, _ = sc_plug_tip_pose(env)
    entry_pos_w, _ = sc_port_entry_pose(env)
    return entry_pos_w - plug_pos_w


def sc_insertion_depth(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.Tensor:
    """Return signed SC insertion depth along the active port axis.

    Depth is zero at the helper entrance frame and increases as the plug tip
    moves from the entrance into the port.
    """
    plug_pos_w, _ = sc_plug_tip_pose(env)
    entry_pos_w, _ = sc_port_entry_pose(env)
    axis_w = sc_port_insertion_axis(env)
    return torch.sum((plug_pos_w - entry_pos_w) * axis_w, dim=-1)


def sc_lateral_error(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.Tensor:
    """Return plug-tip distance from the active port insertion axis."""
    plug_pos_w, _ = sc_plug_tip_pose(env)
    entry_pos_w, _ = sc_port_entry_pose(env)
    axis_w = sc_port_insertion_axis(env)
    delta = plug_pos_w - entry_pos_w
    depth = torch.sum(delta * axis_w, dim=-1, keepdim=True)
    lateral = delta - depth * axis_w
    return torch.norm(lateral, dim=-1)


def sc_orientation_error(env: ManagerBasedEnv | ManagerBasedRLEnv) -> torch.Tensor:
    """Return angular error between SC plug axis and active port insertion axis."""
    plug_axis_w = sc_plug_axis(env)
    port_axis_w = sc_port_insertion_axis(env)
    dot = torch.sum(plug_axis_w * port_axis_w, dim=-1)
    return torch.acos(torch.clamp(dot, min=-1.0, max=1.0))


def sc_insertion_success_from_errors(
    lateral_error: torch.Tensor,
    orientation_error: torch.Tensor,
    insertion_depth: torch.Tensor,
    lateral_threshold: float = 0.005,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.012,
) -> torch.Tensor:
    """Return the SC insertion success mask from precomputed geometry errors."""
    return (
        (lateral_error < lateral_threshold)
        & (orientation_error < orientation_threshold)
        & (insertion_depth > depth_threshold)
    )


def sc_insertion_success_mask(
    env: ManagerBasedEnv | ManagerBasedRLEnv,
    lateral_threshold: float = 0.005,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.012,
) -> torch.Tensor:
    """Return the SC insertion success mask for the active target."""
    return sc_insertion_success_from_errors(
        sc_lateral_error(env),
        sc_orientation_error(env),
        sc_insertion_depth(env),
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
    )
