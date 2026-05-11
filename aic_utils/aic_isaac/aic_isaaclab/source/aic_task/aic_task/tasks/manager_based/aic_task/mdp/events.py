from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING

import omni.usd
import torch
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, Sdf, UsdGeom, UsdLux

from . import geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Matches the regex form Isaac Lab uses to instantiate per-env prim paths.
_ENV_REGEX_RE = re.compile(r"env_(?:\.\*|\[\^/\]\*)")

# Orientations captured from PhysX on the first reset and reused on every
# subsequent reset. Holding the quaternion fixed keeps the composed child
# transforms from referenced USDs (e.g. port frames) correctly aligned.
_cached_orientations: dict[str, torch.Tensor] = {}

SC_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Mean first-success joint positions from scripted virtual-tip insertion checks.
# These are curriculum seeds, not final deployment assumptions.
SC_NEAR_PORT_JOINT_PRESETS = {
    "sc_port": (
        0.8141875863075256,
        -1.8485052585601807,
        -1.8315728902816772,
        -1.0275382995605469,
        1.5704457759857178,
        2.171452760696411,
    ),
    "sc_port_2": (
        0.7603225708007812,
        -1.8013938665390015,
        -1.8958141803741455,
        -1.0111992359161377,
        1.570515513420105,
        2.1116960048675537,
    ),
}

# Mean joint positions after the Step 8 SFP lateral pre-correction diagnostic:
# starting from the earlier near-port seeds, apply raw action (0.5, 0.5, 0.0)
# for 30 steps. From these seeds, pure raw z-negative insertion produced the
# first high-success deterministic SFP final-insertion diagnostic.
SFP_NEAR_PORT_JOINT_PRESETS = {
    "sfp_port_0": (
        0.8343623281,
        -1.5769010782,
        -1.8567240238,
        -1.0969889164,
        1.8369734287,
        2.1079621315,
    ),
    "sfp_port_1": (
        0.8025181293,
        -1.6159480810,
        -1.8159053326,
        -1.1020359993,
        1.8379788399,
        2.1117913723,
    ),
}


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
) -> None:
    """Randomize the dome light's intensity and color on reset.

    The light is a single shared prim, so the randomization is global across
    all environments regardless of ``env_ids``.
    """
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return
    light = UsdLux.DomeLight(light_prim)

    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    color_min, color_max = color_range
    r = torch.empty(1).uniform_(color_min[0], color_max[0]).item()
    g = torch.empty(1).uniform_(color_min[1], color_max[1]).item()
    b = torch.empty(1).uniform_(color_min[2], color_max[2]).item()
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))


def _sample_axis(pose_range: dict, snap_step: dict, axis: str) -> float:
    """Sample an axis offset, snapping to a grid step when configured."""
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    step = snap_step.get(axis, 0.0)
    if step > 0 and (hi - lo) > 0:
        n_lo = math.ceil(lo / step)
        n_hi = math.floor(hi / step)
        return random.randint(n_lo, n_hi) * step
    return torch.empty(1).uniform_(lo, hi).item()


def _write_usd_xform_pose(
    stage,
    prim_path_template: str,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
) -> None:
    """Mirror a per-env rigid body pose onto its USD Xform.

    The prim translate is authored relative to its env root, so the world
    position is converted to env-local coordinates before writing.
    """
    ids = env_ids.tolist()
    local_pos = (world_pos - env_origins).tolist()
    rot = world_rot.tolist()

    for i, env_id in enumerate(ids):
        prim_path = _ENV_REGEX_RE.sub(f"env_{env_id}", prim_path_template)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xf = UsdGeom.Xformable(prim)
        tx, ty, tz = local_pos[i]
        qw, qx, qy, qz = rot[i]

        for op in xf.GetOrderedXformOps():
            name = op.GetOpName()
            if "translate" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Float3:
                    op.Set(Gf.Vec3f(tx, ty, tz))
                else:
                    op.Set(Gf.Vec3d(tx, ty, tz))
            elif "orient" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Quatf:
                    op.Set(Gf.Quatf(qw, qx, qy, qz))
                else:
                    op.Set(Gf.Quatd(qw, qx, qy, qz))


def randomize_board_and_parts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    board_scene_name: str = "task_board",
    board_default_pos: tuple = (0.0, 0.0, 0.0),
    board_range: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0)},
    parts: list[dict] = (),
    sync_usd_xforms: bool = True,
) -> None:
    """Randomize the task board and its attached parts on reset.

    The board position is drawn from ``board_range`` around ``board_default_pos``.
    Each part is offset from the board by a fixed ``offset`` plus a random
    delta from ``pose_range`` (optionally snapped to ``snap_step``).

    When ``sync_usd_xforms`` is True (default) the pose is mirrored onto the
    USD Xform so the viewport tracks physics state. Training workloads should
    set this False to skip the per-env USD writes.
    """
    device = env.device
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    stage = omni.usd.get_context().get_stage() if sync_usd_xforms else None

    all_names = [board_scene_name] + [p["scene_name"] for p in parts]
    if not _cached_orientations:
        for name in all_names:
            _cached_orientations[name] = (
                env.scene[name].data.root_state_w[:, 3:7].clone()
            )

    # Board pose.
    board_asset = env.scene[board_scene_name]
    board_rot = _cached_orientations[board_scene_name][env_ids]
    board_pos = torch.tensor([board_default_pos], device=device).expand(n, -1).clone()
    board_pos[:, 0] += torch.empty(n, device=device).uniform_(
        *board_range.get("x", (0.0, 0.0))
    )
    board_pos[:, 1] += torch.empty(n, device=device).uniform_(
        *board_range.get("y", (0.0, 0.0))
    )
    board_world_pos = board_pos + env_origins

    board_asset.write_root_pose_to_sim(
        torch.cat([board_world_pos, board_rot], dim=-1), env_ids=env_ids
    )
    board_asset.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids
    )
    if sync_usd_xforms:
        _write_usd_xform_pose(
            stage,
            board_asset.cfg.prim_path,
            env_ids,
            env_origins,
            board_world_pos,
            board_rot,
        )

    # Part poses, anchored to the board.
    for part_cfg in parts:
        pname = part_cfg["scene_name"]
        part_asset = env.scene[pname]
        part_rot = _cached_orientations[pname][env_ids]

        ox, oy, oz = part_cfg["offset"]
        pr = part_cfg.get("pose_range", {})
        snap = part_cfg.get("snap_step", {})

        part_pos = board_world_pos.clone()
        for idx in range(n):
            part_pos[idx, 0] += ox + _sample_axis(pr, snap, "x")
            part_pos[idx, 1] += oy + _sample_axis(pr, snap, "y")
            part_pos[idx, 2] = board_world_pos[idx, 2] + oz

        part_asset.write_root_pose_to_sim(
            torch.cat([part_pos, part_rot], dim=-1), env_ids=env_ids
        )
        part_asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=device), env_ids=env_ids
        )
        if sync_usd_xforms:
            _write_usd_xform_pose(
                stage,
                part_asset.cfg.prim_path,
                env_ids,
                env_origins,
                part_pos,
                part_rot,
            )


def _joint_indices(asset, joint_names: tuple[str, ...]) -> list[int]:
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


def reset_robot_near_sc_port(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    probability: float = 1.0,
    blend: float = 0.85,
    position_noise: float = 0.015,
    velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset the arm near the active SC port as a Step 6 insertion curriculum.

    The presets come from successful scripted insertions with the virtual
    gripped SC tip helper. ``blend`` keeps the reset outside the exact success
    pose so PPO still has to learn the final alignment and insertion.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    joint_ids = _joint_indices(asset, SC_ARM_JOINT_NAMES)
    iter_env_ids = env_ids[:, None]
    device = env.device

    default_joint_pos = asset.data.default_joint_pos[iter_env_ids, joint_ids].clone()
    default_joint_vel = asset.data.default_joint_vel[iter_env_ids, joint_ids].clone()

    presets = torch.tensor(
        [SC_NEAR_PORT_JOINT_PRESETS[name] for name in geometry.SC_TARGET_NAMES],
        device=device,
        dtype=default_joint_pos.dtype,
    )
    target_ids = geometry.active_sc_target_ids(env)[env_ids]
    preset_joint_pos = presets[target_ids]

    joint_pos = default_joint_pos + blend * (preset_joint_pos - default_joint_pos)
    if position_noise > 0.0:
        joint_pos += torch.empty_like(joint_pos).uniform_(
            -position_noise, position_noise
        )

    if probability < 1.0:
        use_curriculum = torch.rand(len(env_ids), device=device) < probability
        joint_pos = torch.where(use_curriculum[:, None], joint_pos, default_joint_pos)

    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel
    if velocity_range != (0.0, 0.0):
        joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(*velocity_range)
        joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, joint_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )


def reset_robot_near_sfp_port(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    probability: float = 1.0,
    blend: float = 1.0,
    position_noise: float = 0.01,
    velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset the arm near the active SFP port for a PPO insertion curriculum."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    joint_ids = _joint_indices(asset, SC_ARM_JOINT_NAMES)
    iter_env_ids = env_ids[:, None]
    device = env.device

    default_joint_pos = asset.data.default_joint_pos[iter_env_ids, joint_ids].clone()
    default_joint_vel = asset.data.default_joint_vel[iter_env_ids, joint_ids].clone()

    presets = torch.tensor(
        [SFP_NEAR_PORT_JOINT_PRESETS[name] for name in geometry.SFP_TARGET_NAMES],
        device=device,
        dtype=default_joint_pos.dtype,
    )
    target_ids = geometry.active_sfp_target_ids(env)[env_ids]
    preset_joint_pos = presets[target_ids]

    joint_pos = default_joint_pos + blend * (preset_joint_pos - default_joint_pos)
    if position_noise > 0.0:
        joint_pos += torch.empty_like(joint_pos).uniform_(
            -position_noise, position_noise
        )

    if probability < 1.0:
        use_curriculum = torch.rand(len(env_ids), device=device) < probability
        joint_pos = torch.where(use_curriculum[:, None], joint_pos, default_joint_pos)

    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel
    if velocity_range != (0.0, 0.0):
        joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(*velocity_range)
        joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, joint_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )
