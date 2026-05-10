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
