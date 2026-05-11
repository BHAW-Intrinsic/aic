# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the aic task (UR5e assembly with task board).

Includes:
- Command-tracking rewards with exponential / tanh kernels (inspired by the
  gear-assembly deploy environment).
- SC insertion rewards using shared plug-to-port geometry helpers.
- A sparse reaching bonus.
- Smoothness and safety penalties (torques, joint acceleration, action rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from . import geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


SC_SCRIPTED_ACTION_PRIOR_ATTR = "aic_sc_scripted_action_prior"


# ---------------------------------------------------------------------------
# Command-pose tracking (position)
# ---------------------------------------------------------------------------


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward position tracking using a Gaussian (exponential) kernel.

    Unlike tanh, this kernel drops off very steeply beyond *sigma*, providing
    almost no gradient far from the target while giving a strong signal
    close-in — ideal for fine insertion tasks.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    dist_sq = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


# ---------------------------------------------------------------------------
# Command-pose tracking (orientation)
# ---------------------------------------------------------------------------


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize orientation error (shortest-path angular distance in rad)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward orientation tracking using the tanh kernel.

    Maps the angular error through ``1 - tanh(error / std)`` so that perfectly
    aligned orientations yield 1.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ang_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


# ---------------------------------------------------------------------------
# Sparse reaching bonus
# ---------------------------------------------------------------------------


def ee_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse +1 bonus when the EE is within *threshold* (m) of the command position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


# ---------------------------------------------------------------------------
# SC insertion rewards
# ---------------------------------------------------------------------------


def sc_lateral_alignment_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
) -> torch.Tensor:
    """Reward centering the SC plug tip on the active port insertion axis."""
    lateral_error = geometry.sc_lateral_error(env)
    return 1.0 - torch.tanh(lateral_error / std)


def sc_orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
) -> torch.Tensor:
    """Reward aligning the SC plug axis with the active port insertion axis."""
    orientation_error = geometry.sc_orientation_error(env)
    return 1.0 - torch.tanh(orientation_error / std)


def sc_approach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.50,
) -> torch.Tensor:
    """Reward moving the SC plug tip toward the active port entrance."""
    distance = torch.norm(geometry.sc_plug_to_port_vector(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def _metric_progress_reward(
    env: ManagerBasedRLEnv,
    attr_name: str,
    current: torch.Tensor,
    improvement: str,
    scale: float,
    clip: float,
) -> torch.Tensor:
    """Reward one-step progress in a scalar geometry metric."""
    previous = getattr(env, attr_name, None)
    if (
        not isinstance(previous, torch.Tensor)
        or previous.shape != current.shape
        or previous.device != current.device
    ):
        previous = current.detach().clone()
        setattr(env, attr_name, previous)

    if improvement == "decrease":
        progress = previous - current
    elif improvement == "increase":
        progress = current - previous
    else:
        raise ValueError(f"Unsupported progress direction: {improvement}")

    with torch.no_grad():
        previous.copy_(current.detach())
    return torch.clamp(progress / scale, min=-clip, max=clip)


def sc_distance_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.02,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward reducing plug-tip distance to the active SC port entrance."""
    distance = torch.norm(geometry.sc_plug_to_port_vector(env), dim=-1)
    return _metric_progress_reward(
        env,
        geometry.SC_PREV_DISTANCE_ATTR,
        distance,
        improvement="decrease",
        scale=scale,
        clip=clip,
    )


def sc_lateral_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.005,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward reducing lateral error to the active SC port axis."""
    return _metric_progress_reward(
        env,
        geometry.SC_PREV_LATERAL_ATTR,
        geometry.sc_lateral_error(env),
        improvement="decrease",
        scale=scale,
        clip=clip,
    )


def sc_orientation_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.10,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward reducing plug-to-port angular error."""
    return _metric_progress_reward(
        env,
        geometry.SC_PREV_ORIENTATION_ATTR,
        geometry.sc_orientation_error(env),
        improvement="decrease",
        scale=scale,
        clip=clip,
    )


def sc_depth_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward increasing signed insertion depth, including pre-insertion approach."""
    return _metric_progress_reward(
        env,
        geometry.SC_PREV_DEPTH_ATTR,
        geometry.sc_insertion_depth(env),
        improvement="increase",
        scale=scale,
        clip=clip,
    )


def sc_insertion_depth_reward(
    env: ManagerBasedRLEnv,
    depth_scale: float = 0.02,
    max_depth: float = 0.03,
    lateral_threshold: float = 0.01,
    orientation_threshold: float = 0.35,
) -> torch.Tensor:
    """Reward insertion depth only when lateral and angular alignment are acceptable."""
    lateral_error = geometry.sc_lateral_error(env)
    orientation_error = geometry.sc_orientation_error(env)
    depth = torch.clamp(geometry.sc_insertion_depth(env), min=0.0, max=max_depth)
    aligned = (lateral_error < lateral_threshold) & (
        orientation_error < orientation_threshold
    )
    return aligned.float() * torch.clamp(depth / depth_scale, max=1.0)


def sc_insertion_success_bonus(
    env: ManagerBasedRLEnv,
    lateral_threshold: float = 0.005,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.012,
) -> torch.Tensor:
    """Sparse bonus for a plausible SC insertion state."""
    return geometry.sc_insertion_success_mask(
        env,
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
    ).float()


# ---------------------------------------------------------------------------
# SFP insertion rewards
# ---------------------------------------------------------------------------


def sfp_lateral_alignment_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
) -> torch.Tensor:
    """Reward centering the SFP plug tip on the active port insertion axis."""
    lateral_error = geometry.sfp_lateral_error(env)
    return 1.0 - torch.tanh(lateral_error / std)


def sfp_lateral_error_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 0.06,
    clip: float = 1.0,
) -> torch.Tensor:
    """Penalize SFP lateral error inside the near-port curriculum corridor."""
    lateral_error = geometry.sfp_lateral_error(env)
    return torch.clamp(lateral_error / max(scale, 1.0e-6), min=0.0, max=clip)


def sfp_lateral_corridor_penalty(
    env: ManagerBasedRLEnv,
    soft_limit: float = 0.020,
    hard_limit: float = 0.060,
    clip: float = 1.0,
    violation_cost: float = 1.0,
) -> torch.Tensor:
    """Penalize approaching or crossing the SFP lateral curriculum boundary."""
    lateral_error = geometry.sfp_lateral_error(env)
    scale = max(hard_limit - soft_limit, 1.0e-6)
    margin_cost = torch.clamp((lateral_error - soft_limit) / scale, min=0.0, max=clip)
    terminal_cost = (lateral_error > hard_limit).float() * violation_cost
    return margin_cost + terminal_cost


def sfp_depth_backout_penalty(
    env: ManagerBasedRLEnv,
    soft_min_depth: float = -0.010,
    hard_min_depth: float = -0.080,
    clip: float = 1.0,
    violation_cost: float = 1.0,
) -> torch.Tensor:
    """Penalize backing the SFP module out of the near-port curriculum band."""
    depth = geometry.sfp_insertion_depth(env)
    scale = max(soft_min_depth - hard_min_depth, 1.0e-6)
    margin_cost = torch.clamp(
        (soft_min_depth - depth) / scale,
        min=0.0,
        max=clip,
    )
    terminal_cost = (depth < hard_min_depth).float() * violation_cost
    return margin_cost + terminal_cost


def sfp_orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
) -> torch.Tensor:
    """Reward aligning the SFP plug axis with the active port insertion axis."""
    orientation_error = geometry.sfp_orientation_error(env)
    return 1.0 - torch.tanh(orientation_error / std)


def sfp_approach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.50,
    active_until_depth: float | None = None,
) -> torch.Tensor:
    """Reward moving the SFP plug tip toward the active port entrance."""
    distance = torch.norm(geometry.sfp_plug_to_port_vector(env), dim=-1)
    reward = 1.0 - torch.tanh(distance / std)
    if active_until_depth is not None:
        reward = reward * (geometry.sfp_insertion_depth(env) < active_until_depth).float()
    return reward


def sfp_distance_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.02,
    clip: float = 1.0,
    active_until_depth: float | None = None,
) -> torch.Tensor:
    """Reward reducing plug-tip distance to the active SFP port entrance."""
    distance = torch.norm(geometry.sfp_plug_to_port_vector(env), dim=-1)
    reward = _metric_progress_reward(
        env,
        geometry.SFP_PREV_DISTANCE_ATTR,
        distance,
        improvement="decrease",
        scale=scale,
        clip=clip,
    )
    if active_until_depth is not None:
        reward = reward * (geometry.sfp_insertion_depth(env) < active_until_depth).float()
    return reward


def sfp_lateral_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.005,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward reducing lateral error to the active SFP port axis."""
    return _metric_progress_reward(
        env,
        geometry.SFP_PREV_LATERAL_ATTR,
        geometry.sfp_lateral_error(env),
        improvement="decrease",
        scale=scale,
        clip=clip,
    )


def sfp_orientation_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.10,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward reducing SFP plug-to-port angular error."""
    return _metric_progress_reward(
        env,
        geometry.SFP_PREV_ORIENTATION_ATTR,
        geometry.sfp_orientation_error(env),
        improvement="decrease",
        scale=scale,
        clip=clip,
    )


def sfp_depth_progress_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward increasing signed SFP insertion depth."""
    return _metric_progress_reward(
        env,
        geometry.SFP_PREV_DEPTH_ATTR,
        geometry.sfp_insertion_depth(env),
        improvement="increase",
        scale=scale,
        clip=clip,
    )


def sfp_insertion_depth_reward(
    env: ManagerBasedRLEnv,
    depth_scale: float = 0.025,
    max_depth: float = 0.045,
    min_depth: float = 0.0,
    lateral_threshold: float = 0.008,
    orientation_threshold: float = 0.35,
) -> torch.Tensor:
    """Reward SFP insertion depth once lateral/angular alignment is acceptable.

    ``min_depth`` can be slightly negative for early curricula so a plug just
    outside the entrance still receives a shaped ramp toward positive depth.
    """
    lateral_error = geometry.sfp_lateral_error(env)
    orientation_error = geometry.sfp_orientation_error(env)
    depth = torch.clamp(geometry.sfp_insertion_depth(env), min=min_depth, max=max_depth)
    aligned = (lateral_error < lateral_threshold) & (
        orientation_error < orientation_threshold
    )
    depth_span = max(depth_scale - min_depth, 1.0e-6)
    return aligned.float() * torch.clamp((depth - min_depth) / depth_span, max=1.0)


def sfp_insertion_success_bonus(
    env: ManagerBasedRLEnv,
    lateral_threshold: float = 0.004,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.015,
) -> torch.Tensor:
    """Sparse bonus for a plausible SFP insertion state."""
    return geometry.sfp_insertion_success_mask(
        env,
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
    ).float()


def _relative_ik_delta_pos_w(
    env: ManagerBasedRLEnv,
    action_name: str,
    asset_name: str,
    action_scale: float,
) -> torch.Tensor:
    if action_name:
        actual_action = env.action_manager.get_term(action_name).raw_actions
    else:
        actual_action = env.action_manager.action
    delta_pos_root = actual_action[:, :3] * action_scale

    robot = env.scene[asset_name]
    return geometry._quat_apply(robot.data.root_quat_w, delta_pos_root)


def _raw_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    if action_name:
        return env.action_manager.get_term(action_name).raw_actions
    return env.action_manager.action


def _sfp_signed_lateral_components(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    plug_pos_w, _ = geometry.sfp_plug_tip_pose(env)
    entry_pos_w, entry_quat_w = geometry.sfp_port_entry_pose(env)
    insertion_axis_w = geometry.sfp_port_insertion_axis(env)

    plug_delta = plug_pos_w - entry_pos_w
    depth = torch.sum(plug_delta * insertion_axis_w, dim=-1)
    lateral_vec = plug_delta - depth.unsqueeze(-1) * insertion_axis_w
    lateral_error = torch.norm(lateral_vec, dim=-1)

    port_x = geometry._quat_apply(
        entry_quat_w,
        geometry._expand_vec((1.0, 0.0, 0.0), env),
    )
    port_y = geometry._quat_apply(
        entry_quat_w,
        geometry._expand_vec((0.0, 1.0, 0.0), env),
    )
    signed_x = torch.sum(lateral_vec * port_x, dim=-1)
    signed_y = torch.sum(lateral_vec * port_y, dim=-1)
    return signed_x, signed_y, lateral_error, depth


def sfp_port_frame_lateral_action_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    command_scale: float = 0.02,
    min_lateral_error: float = 0.002,
    lateral_scale: float = 0.006,
    lateral_threshold: float = 0.060,
    orientation_threshold: float = 0.80,
    min_depth: float = -0.080,
    max_depth: float = 0.060,
) -> torch.Tensor:
    """Reward raw IK commands that empirically reduce SFP port-frame lateral error.

    The SFP action-frame diagnostic shows raw ``x+`` increases signed port-frame
    x, while raw ``y+`` decreases signed port-frame y. This term uses that
    measured mapping directly instead of assuming the raw translation command is
    already expressed in the active SFP port frame.
    """
    raw = _raw_action(env, action_name)
    signed_x, signed_y, lateral_error, depth = _sfp_signed_lateral_components(env)
    lateral_denom = torch.clamp(lateral_error, min=1.0e-6)

    desired_raw_x = -signed_x / lateral_denom
    desired_raw_y = signed_y / lateral_denom
    correction_command = raw[:, 0] * desired_raw_x + raw[:, 1] * desired_raw_y
    scaled_command = torch.clamp(
        correction_command / max(command_scale, 1.0e-6), min=0.0, max=1.0
    )
    lateral_gain = torch.clamp(
        (lateral_error - min_lateral_error)
        / max(lateral_scale - min_lateral_error, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    orientation_error = geometry.sfp_orientation_error(env)
    active = (
        (lateral_error > min_lateral_error)
        & (lateral_error < lateral_threshold)
        & (orientation_error < orientation_threshold)
        & (depth > min_depth)
        & (depth < max_depth)
    )
    return active.float() * lateral_gain * scaled_command


def sfp_port_frame_depth_action_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    command_scale: float = 0.02,
    realized_depth_scale: float = 2.0e-5,
    min_depth: float = -0.080,
    target_depth: float = 0.005,
    lateral_threshold: float = 0.120,
    orientation_threshold: float = 1.50,
) -> torch.Tensor:
    """Reward raw ``z-`` commands and penalize raw ``z+`` commands.

    The SFP action-frame diagnostic shows raw ``z-`` is the clearest positive
    depth command. This signed coarse term stays active before fine lateral
    alignment, but it only pays when the last action produced measured positive
    signed-depth progress. That prevents PPO from maximizing raw inward intent
    while contact or IK leaves the plug tip outside the port.
    """
    raw = _raw_action(env, action_name)
    inward_command = -raw[:, 2]
    scaled_command = torch.clamp(
        inward_command / max(command_scale, 1.0e-6), min=-1.0, max=1.0
    )

    _, _, lateral_error, depth = _sfp_signed_lateral_components(env)
    orientation_error = geometry.sfp_orientation_error(env)
    realized_progress = _metric_progress_reward(
        env,
        geometry.SFP_PREV_DEPTH_ACTION_ATTR,
        depth,
        improvement="increase",
        scale=realized_depth_scale,
        clip=1.0,
    )
    realized_gain = torch.clamp(realized_progress, min=0.0, max=1.0)
    depth_gain = torch.clamp(
        (target_depth - depth) / max(target_depth - min_depth, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    active = (
        (depth < target_depth)
        & (lateral_error < lateral_threshold)
        & (orientation_error < orientation_threshold)
    )
    return active.float() * depth_gain * realized_gain * scaled_command


def sfp_lateral_correction_action_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    asset_name: str = "robot",
    action_scale: float = 0.01,
    command_scale: float = 0.004,
    min_lateral_error: float = 0.002,
    lateral_scale: float = 0.012,
    lateral_threshold: float = 0.060,
    orientation_threshold: float = 0.80,
    min_depth: float = -0.080,
    max_depth: float = 0.060,
) -> torch.Tensor:
    """Reward relative-IK commands that reduce SFP lateral error."""
    delta_pos_w = _relative_ik_delta_pos_w(
        env,
        action_name=action_name,
        asset_name=asset_name,
        action_scale=action_scale,
    )

    plug_pos_w, _ = geometry.sfp_plug_tip_pose(env)
    entry_pos_w, _ = geometry.sfp_port_entry_pose(env)
    insertion_axis_w = geometry.sfp_port_insertion_axis(env)
    plug_delta = plug_pos_w - entry_pos_w
    depth = torch.sum(plug_delta * insertion_axis_w, dim=-1)
    lateral_vec = plug_delta - depth.unsqueeze(-1) * insertion_axis_w
    lateral_error = torch.norm(lateral_vec, dim=-1)
    lateral_dir = lateral_vec / torch.clamp(lateral_error, min=1.0e-6).unsqueeze(-1)

    correction_command = torch.sum(delta_pos_w * -lateral_dir, dim=-1)
    scaled_command = torch.clamp(
        correction_command / max(command_scale, 1.0e-6), min=0.0, max=1.0
    )
    lateral_gain = torch.clamp(
        (lateral_error - min_lateral_error)
        / max(lateral_scale - min_lateral_error, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    orientation_error = geometry.sfp_orientation_error(env)
    active = (
        (lateral_error > min_lateral_error)
        & (lateral_error < lateral_threshold)
        & (orientation_error < orientation_threshold)
        & (depth > min_depth)
        & (depth < max_depth)
    )
    return active.float() * lateral_gain * scaled_command


def sfp_port_approach_action_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    asset_name: str = "robot",
    action_scale: float = 0.01,
    command_scale: float = 0.004,
    min_distance: float = 0.001,
    max_distance: float = 0.120,
    min_depth: float = -0.080,
    depth_threshold: float = 0.005,
    orientation_threshold: float = 1.20,
) -> torch.Tensor:
    """Reward relative-IK commands that move the SFP tip toward the port entry."""
    delta_pos_w = _relative_ik_delta_pos_w(
        env,
        action_name=action_name,
        asset_name=asset_name,
        action_scale=action_scale,
    )

    to_entry = geometry.sfp_plug_to_port_vector(env)
    distance = torch.norm(to_entry, dim=-1)
    direction = to_entry / torch.clamp(distance, min=1.0e-6).unsqueeze(-1)
    approach_command = torch.sum(delta_pos_w * direction, dim=-1)
    scaled_command = torch.clamp(
        approach_command / max(command_scale, 1.0e-6), min=0.0, max=1.0
    )

    depth = geometry.sfp_insertion_depth(env)
    orientation_error = geometry.sfp_orientation_error(env)
    active = (
        (distance > min_distance)
        & (distance < max_distance)
        & (depth > min_depth)
        & (depth < depth_threshold)
        & (orientation_error < orientation_threshold)
    )
    return active.float() * scaled_command


def sfp_insertion_action_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    asset_name: str = "robot",
    action_scale: float = 0.05,
    command_scale: float = 0.025,
    realized_depth_scale: float = 2.0e-5,
    lateral_threshold: float = 0.010,
    orientation_threshold: float = 0.35,
    lateral_std: float = 0.0,
) -> torch.Tensor:
    """Reward relative-IK translation commands that move the SFP tip inward.

    This is a privileged PPO shaping term: it uses the simulator's port axis in
    the reward only, while the actor still receives eval-compatible observations.
    """
    delta_pos_w = _relative_ik_delta_pos_w(
        env,
        action_name=action_name,
        asset_name=asset_name,
        action_scale=action_scale,
    )
    insertion_axis_w = geometry.sfp_port_insertion_axis(env)
    inward_command = torch.sum(delta_pos_w * insertion_axis_w, dim=-1)
    depth = geometry.sfp_insertion_depth(env)
    realized_progress = _metric_progress_reward(
        env,
        geometry.SFP_PREV_INSERTION_ACTION_ATTR,
        depth,
        improvement="increase",
        scale=realized_depth_scale,
        clip=1.0,
    )
    realized_gain = torch.clamp(realized_progress, min=0.0, max=1.0)

    lateral_error = geometry.sfp_lateral_error(env)
    orientation_error = geometry.sfp_orientation_error(env)
    aligned = (lateral_error < lateral_threshold) & (
        orientation_error < orientation_threshold
    )
    scaled_command = torch.clamp(
        inward_command / max(command_scale, 1.0e-6), min=0.0, max=1.0
    )
    if lateral_std > 0.0:
        lateral_gain = 1.0 - torch.tanh(lateral_error / lateral_std)
        scaled_command = scaled_command * lateral_gain
    return aligned.float() * realized_gain * scaled_command


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., 0:1], -quat[..., 1:]), dim=-1)


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
    return geometry._quat_normalize(quat)


def _axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = geometry._quat_normalize(quat)
    quat = torch.where(quat[:, 0:1] < 0.0, -quat, quat)
    vec = quat[:, 1:]
    sin_half = torch.norm(vec, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, torch.clamp(quat[:, 0], min=1.0e-9))
    axis = vec / torch.clamp(sin_half.unsqueeze(-1), min=1.0e-9)
    axis_angle = axis * angle.unsqueeze(-1)
    return torch.where(sin_half.unsqueeze(-1) > 1.0e-9, axis_angle, 0.0)


def _body_pose(
    env: ManagerBasedRLEnv,
    asset_name: str,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_name]
    body_id = geometry._body_index(asset, body_name)
    return asset.data.body_pos_w[:, body_id], asset.data.body_quat_w[:, body_id]


def _body_to_tip_pose(
    env: ManagerBasedRLEnv,
    asset_name: str,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_pos_w, body_quat_w = _body_pose(env, asset_name, body_name)
    tip_pos_w, tip_quat_w = geometry.sc_plug_tip_pose(env, asset_name=asset_name)
    body_inv = _quat_conjugate(body_quat_w)
    rel_pos = geometry._quat_apply(body_inv, tip_pos_w - body_pos_w)
    rel_quat = geometry._quat_mul(body_inv, tip_quat_w)
    return rel_pos, geometry._quat_normalize(rel_quat)


def _root_frame_pose(
    env: ManagerBasedRLEnv,
    asset_name: str,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_name]
    root_inv = _quat_conjugate(asset.data.root_quat_w)
    pos_root = geometry._quat_apply(root_inv, pos_w - asset.data.root_pos_w)
    quat_root = geometry._quat_mul(root_inv, quat_w)
    return pos_root, geometry._quat_normalize(quat_root)


def _scripted_sc_raw_action(
    env: ManagerBasedRLEnv,
    asset_name: str,
    action_body_name: str,
    action_scale: float,
    action_clip: float,
    approach_depth: float,
    target_depth: float,
    max_translation_step: float,
    max_rotation_step: float,
    align_lateral_threshold: float,
    align_orientation_threshold: float,
) -> torch.Tensor:
    """Compute the same privileged relative-IK action used by the scripted check."""
    plug_pos_w, plug_quat_w = geometry.sc_plug_tip_pose(env, asset_name=asset_name)
    plug_axis_w = geometry.sc_plug_axis(env, asset_name=asset_name)
    entry_pos_w, _ = geometry.sc_port_entry_pose(env)
    port_axis_w = geometry.sc_port_insertion_axis(env)

    lateral = geometry.sc_lateral_error(env)
    orientation = geometry.sc_orientation_error(env)
    aligned = (lateral < align_lateral_threshold) & (
        orientation < align_orientation_threshold
    )
    desired_depth = torch.full(
        (env.num_envs,),
        approach_depth,
        device=plug_pos_w.device,
        dtype=plug_pos_w.dtype,
    )
    desired_depth[aligned] = target_depth
    target_tip_pos_w = entry_pos_w + desired_depth.unsqueeze(-1) * port_axis_w

    delta_pos_w = _clip_by_norm(
        target_tip_pos_w - plug_pos_w,
        max_norm=max_translation_step,
    )
    desired_tip_pos_w = plug_pos_w + delta_pos_w

    cross = torch.cross(plug_axis_w, port_axis_w, dim=-1)
    sin_angle = torch.norm(cross, dim=-1)
    cos_angle = torch.sum(plug_axis_w * port_axis_w, dim=-1)
    angle = torch.atan2(sin_angle, cos_angle)
    axis_w = cross / torch.clamp(sin_angle.unsqueeze(-1), min=1.0e-9)
    rot_step = torch.clamp(angle, max=max_rotation_step)
    rot_vec_w = axis_w * rot_step.unsqueeze(-1)
    rot_vec_w = torch.where(sin_angle.unsqueeze(-1) > 1.0e-6, rot_vec_w, 0.0)

    desired_tip_quat_w = geometry._quat_mul(
        _quat_from_axis_angle(rot_vec_w), plug_quat_w
    )
    body_pos_w, body_quat_w = _body_pose(env, asset_name, action_body_name)
    tip_pos_in_body, tip_quat_in_body = _body_to_tip_pose(
        env, asset_name, action_body_name
    )
    desired_body_quat_w = geometry._quat_mul(
        desired_tip_quat_w, _quat_conjugate(tip_quat_in_body)
    )
    desired_body_pos_w = desired_tip_pos_w - geometry._quat_apply(
        desired_body_quat_w, tip_pos_in_body
    )

    body_pos_root, body_quat_root = _root_frame_pose(
        env, asset_name, body_pos_w, body_quat_w
    )
    desired_body_pos_root, desired_body_quat_root = _root_frame_pose(
        env, asset_name, desired_body_pos_w, desired_body_quat_w
    )
    delta_pos_root = desired_body_pos_root - body_pos_root
    delta_quat_root = geometry._quat_mul(
        desired_body_quat_root, _quat_conjugate(body_quat_root)
    )
    rot_vec_root = _clip_by_norm(
        _axis_angle_from_quat(delta_quat_root), max_rotation_step
    )

    raw_action = torch.cat((delta_pos_root, rot_vec_root), dim=-1) / action_scale
    if action_clip > 0.0:
        raw_action = torch.clamp(raw_action, -action_clip, action_clip)
    return raw_action


def sc_scripted_raw_action(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    action_body_name: str = "gripper_tcp",
    action_scale: float = 0.05,
    action_clip: float = 1.0,
    approach_depth: float = 0.0,
    target_depth: float = 0.02,
    max_translation_step: float = 0.025,
    max_rotation_step: float = 0.10,
    align_lateral_threshold: float = 0.05,
    align_orientation_threshold: float = 0.50,
) -> torch.Tensor:
    """Return the privileged scripted raw action for SC insertion."""
    return _scripted_sc_raw_action(
        env,
        asset_name=asset_name,
        action_body_name=action_body_name,
        action_scale=action_scale,
        action_clip=action_clip,
        approach_depth=approach_depth,
        target_depth=target_depth,
        max_translation_step=max_translation_step,
        max_rotation_step=max_rotation_step,
        align_lateral_threshold=align_lateral_threshold,
        align_orientation_threshold=align_orientation_threshold,
    )


def reset_sc_scripted_action_prior_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name: str = "robot",
    action_body_name: str = "gripper_tcp",
    action_scale: float = 0.05,
    action_clip: float = 1.0,
    approach_depth: float = 0.0,
    target_depth: float = 0.02,
    max_translation_step: float = 0.025,
    max_rotation_step: float = 0.10,
    align_lateral_threshold: float = 0.05,
    align_orientation_threshold: float = 0.50,
) -> None:
    """Cache the scripted action for reset states before the first policy step."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    desired_action = _scripted_sc_raw_action(
        env,
        asset_name=asset_name,
        action_body_name=action_body_name,
        action_scale=action_scale,
        action_clip=action_clip,
        approach_depth=approach_depth,
        target_depth=target_depth,
        max_translation_step=max_translation_step,
        max_rotation_step=max_rotation_step,
        align_lateral_threshold=align_lateral_threshold,
        align_orientation_threshold=align_orientation_threshold,
    )
    buffer = getattr(env, SC_SCRIPTED_ACTION_PRIOR_ATTR, None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.shape != desired_action.shape
        or buffer.device != desired_action.device
    ):
        buffer = torch.zeros_like(desired_action)
        setattr(env, SC_SCRIPTED_ACTION_PRIOR_ATTR, buffer)
    buffer[env_ids] = desired_action[env_ids].detach()


def sc_scripted_action_prior_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
    asset_name: str = "robot",
    action_body_name: str = "gripper_tcp",
    action_scale: float = 0.05,
    action_clip: float = 1.0,
    approach_depth: float = 0.0,
    target_depth: float = 0.02,
    max_translation_step: float = 0.025,
    max_rotation_step: float = 0.10,
    align_lateral_threshold: float = 0.05,
    align_orientation_threshold: float = 0.50,
    std: float = 1.00,
) -> torch.Tensor:
    """Reward matching the privileged scripted relative-IK SC insertion action.

    This is a Step 6 teacher/curriculum term. It uses privileged geometry in the
    reward only; it does not add hidden geometry to actor observations.
    """
    desired_action = _scripted_sc_raw_action(
        env,
        asset_name=asset_name,
        action_body_name=action_body_name,
        action_scale=action_scale,
        action_clip=action_clip,
        approach_depth=approach_depth,
        target_depth=target_depth,
        max_translation_step=max_translation_step,
        max_rotation_step=max_rotation_step,
        align_lateral_threshold=align_lateral_threshold,
        align_orientation_threshold=align_orientation_threshold,
    )
    cached_action = getattr(env, SC_SCRIPTED_ACTION_PRIOR_ATTR, None)
    if (
        not isinstance(cached_action, torch.Tensor)
        or cached_action.shape != desired_action.shape
        or cached_action.device != desired_action.device
    ):
        cached_action = desired_action.detach().clone()
        setattr(env, SC_SCRIPTED_ACTION_PRIOR_ATTR, cached_action)

    if action_name:
        actual_action = env.action_manager.get_term(action_name).raw_actions
    else:
        actual_action = env.action_manager.action
    actual_action = actual_action[:, : cached_action.shape[1]]
    action_error = torch.norm(actual_action - cached_action, dim=-1)
    reward = 1.0 - torch.tanh(action_error / std)
    with torch.no_grad():
        cached_action.copy_(desired_action.detach())
    return reward


# ---------------------------------------------------------------------------
# Smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques (L2 squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear acceleration of selected bodies (encourages gentle motion)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1
    )
