# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the AIC task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sc_insertion_success(
    env: ManagerBasedRLEnv,
    lateral_threshold: float = 0.005,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.012,
) -> torch.Tensor:
    """Terminate when the SC plug is plausibly inserted in the active port."""
    return geometry.sc_insertion_success_mask(
        env,
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
    )


def sfp_insertion_success(
    env: ManagerBasedRLEnv,
    lateral_threshold: float = 0.004,
    orientation_threshold: float = 0.20,
    depth_threshold: float = 0.015,
) -> torch.Tensor:
    """Terminate when the SFP module is plausibly inserted in the active port."""
    return geometry.sfp_insertion_success_mask(
        env,
        lateral_threshold=lateral_threshold,
        orientation_threshold=orientation_threshold,
        depth_threshold=depth_threshold,
    )


def sfp_insertion_corridor_violation(
    env: ManagerBasedRLEnv,
    lateral_limit: float = 0.06,
    orientation_limit: float = 0.80,
    min_depth: float = -0.08,
    max_depth: float = 0.06,
) -> torch.Tensor:
    """Terminate early when SFP attempts leave the near-port curriculum corridor."""
    lateral_error = geometry.sfp_lateral_error(env)
    orientation_error = geometry.sfp_orientation_error(env)
    depth = geometry.sfp_insertion_depth(env)
    return (
        (lateral_error > lateral_limit)
        | (orientation_error > orientation_limit)
        | (depth < min_depth)
        | (depth > max_depth)
    )


def sfp_corridor_lateral_violation(
    env: ManagerBasedRLEnv,
    lateral_limit: float = 0.06,
) -> torch.Tensor:
    """Terminate early when SFP lateral error leaves the curriculum corridor."""
    return geometry.sfp_lateral_error(env) > lateral_limit


def sfp_corridor_orientation_violation(
    env: ManagerBasedRLEnv,
    orientation_limit: float = 0.80,
) -> torch.Tensor:
    """Terminate early when SFP orientation error leaves the curriculum corridor."""
    return geometry.sfp_orientation_error(env) > orientation_limit


def sfp_corridor_min_depth_violation(
    env: ManagerBasedRLEnv,
    min_depth: float = -0.08,
) -> torch.Tensor:
    """Terminate early when the SFP module backs too far out of the port."""
    return geometry.sfp_insertion_depth(env) < min_depth


def sfp_corridor_max_depth_violation(
    env: ManagerBasedRLEnv,
    max_depth: float = 0.06,
) -> torch.Tensor:
    """Terminate early when the SFP module overshoots the curriculum depth."""
    return geometry.sfp_insertion_depth(env) > max_depth
