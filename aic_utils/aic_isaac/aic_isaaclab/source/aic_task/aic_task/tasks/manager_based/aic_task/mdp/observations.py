# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the AIC task (e.g. contact sensing)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from . import geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_column(value: torch.Tensor) -> torch.Tensor:
    """Return scalar per-env values as a concatenation-friendly column."""
    if value.ndim == 1:
        return value.unsqueeze(-1)
    return value


def active_sc_target_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Eval-compatible SC target metadata as one-hot ``[sc_port, sc_port_2]``."""
    target_ids = geometry.active_sc_target_ids(env)
    return torch.nn.functional.one_hot(
        target_ids, num_classes=len(geometry.SC_TARGET_NAMES)
    ).to(dtype=torch.float32)


def sc_plug_to_port_vec(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged vector from SC plug tip to active SC port entrance."""
    return geometry.sc_plug_to_port_vector(env)


def sc_lateral_error_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged SC plug-tip lateral error as ``(num_envs, 1)``."""
    return _as_column(geometry.sc_lateral_error(env))


def sc_insertion_depth_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged SC insertion depth as ``(num_envs, 1)``."""
    return _as_column(geometry.sc_insertion_depth(env))


def sc_orientation_error_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged SC plug-to-port orientation error as ``(num_envs, 1)``."""
    return _as_column(geometry.sc_orientation_error(env))


def sc_active_port_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged active SC port entrance pose, xyz + quat wxyz."""
    pos_w, quat_w = geometry.sc_port_entry_pose(env)
    return torch.cat((pos_w, quat_w), dim=-1)


def sc_plug_tip_pose_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged SC plug tip pose, xyz + quat wxyz."""
    pos_w, quat_w = geometry.sc_plug_tip_pose(env)
    return torch.cat((pos_w, quat_w), dim=-1)


def contact_net_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Net contact forces (world frame) from the contact sensor, flattened for policy obs.

    Uses the current timestep net forces (no history). Body selection is via sensor_cfg.body_ids
    if set by the manager, or sensor_cfg.body_names matched against the sensor's body_names.

    Returns:
        Tensor of shape (num_envs, num_bodies * 3) in world frame (x,y,z per body).
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = contact_sensor.data.net_forces_w  # (N, B, 3)
    body_ids = sensor_cfg.body_ids
    if body_ids is None or body_ids == slice(None):
        if getattr(sensor_cfg, "body_names", None) is not None:
            names = (
                [sensor_cfg.body_names]
                if isinstance(sensor_cfg.body_names, str)
                else sensor_cfg.body_names
            )
            pattern = re.compile(names[0] if len(names) == 1 else "|".join(names))
            body_ids = [
                i for i, b in enumerate(contact_sensor.body_names) if pattern.search(b)
            ]
            if body_ids:
                net = net[:, body_ids, :]
    else:
        net = net[:, body_ids, :]
    return net.reshape(env.num_envs, -1)
