"""ROS policy for replaying exported Isaac RSL-RL actors in Gazebo eval.

This class uses only the official ``aic_model`` policy API: task metadata,
``Observation`` messages, and ``move_robot`` callbacks. It does not subscribe to
ground-truth transforms, scoring topics, Gazebo internals, or hidden simulator
state.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Twist, Vector3, Wrench
from trajectory_msgs.msg import JointTrajectoryPoint


POLICY_OBSERVATION_ORDER = (
    "task_metadata",
    "joint_pos_rel",
    "joint_vel_rel",
    "eef_pose",
    "body_forces",
    "center_rgb_resnet18",
    "left_rgb_resnet18",
    "right_rgb_resnet18",
    "last_action",
)

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ISAAC_DEFAULT_ARM_JOINT_POS = np.array(
    [0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
    dtype=np.float32,
)

GAZEBO_DEFAULT_ARM_JOINT_POS = np.array(
    [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
    dtype=np.float32,
)

SFP_NEAR_PORT_JOINT_PRESETS = {
    "sfp_port_0": np.array(
        [
            0.8302846551,
            -1.5486999750,
            -1.8918046951,
            -1.0959614515,
            1.8380267620,
            2.1012129784,
        ],
        dtype=np.float64,
    ),
    "sfp_port_1": np.array(
        [
            0.8000932336,
            -1.5981711149,
            -1.8391590118,
            -1.1001185179,
            1.8383054733,
            2.1077697277,
        ],
        dtype=np.float64,
    ),
}

SC_NEAR_PORT_JOINT_PRESETS = {
    "sc_port": np.array(
        [
            0.8141875863075256,
            -1.8485052585601807,
            -1.8315728902816772,
            -1.0275382995605469,
            1.5704457759857178,
            2.171452760696411,
        ],
        dtype=np.float64,
    ),
    "sc_port_2": np.array(
        [
            0.7603225708007812,
            -1.8013938665390015,
            -1.8958141803741455,
            -1.0111992359161377,
            1.570515513420105,
            2.1116960048675537,
        ],
        dtype=np.float64,
    ),
}

SFP_PORT_ONE_HOT = {
    "sfp_port_0": np.array([1.0, 0.0], dtype=np.float32),
    "sfp_port_1": np.array([0.0, 1.0], dtype=np.float32),
}

SFP_MOUNT_ONE_HOT = {
    "nic_card_mount_0": np.array([1.0, 0.0], dtype=np.float32),
    "nic_card_mount_1": np.array([0.0, 1.0], dtype=np.float32),
}

SFP_LOCAL_SEARCH_PRIORITY_OFFSETS = {
    "nic_card_mount_0": (
        (0.002, -0.001, 0.002),
        (0.008, -0.007, -0.002),
        (0.008, -0.008, -0.030),
        (0.008, -0.009, -0.048),
    ),
    "nic_card_mount_1": (
        (0.014, -0.002, 0.026),
        (0.029, 0.020, 0.023),
        (0.029, 0.020, -0.005),
        (0.029, 0.019, -0.023),
    ),
}

SFP_TERMINAL_TARGETS = {
    "nic_card_mount_0": (
        (-0.383, 0.192, 0.233),
        (-0.384, 0.205, 0.190),
        (-0.386, 0.205, 0.150),
        (-0.388, 0.205, 0.110),
        (-0.390, 0.205, 0.061),
    ),
    "nic_card_mount_1": (
        (-0.385, 0.232, 0.230),
        (-0.386, 0.245, 0.190),
        (-0.388, 0.245, 0.150),
        (-0.390, 0.245, 0.110),
        (-0.392, 0.245, 0.063),
    ),
}

SFP_TERMINAL_QUATS_XYZW = {
    "nic_card_mount_0": (0.98250, -0.02775, -0.00493, 0.18408),
    "nic_card_mount_1": (0.98252, -0.02777, -0.00504, 0.18400),
}

SC_PORT_ONE_HOT = {
    "sc_port": np.array([1.0, 0.0], dtype=np.float32),
    "sc_port_2": np.array([0.0, 1.0], dtype=np.float32),
}

SC_TERMINAL_TARGETS = {
    "sc_port_2": (
        (-0.494, 0.267, 0.080),
        (-0.486, 0.287, 0.035),
        (-0.486, 0.287, 0.010),
        (-0.486, 0.287, -0.021),
    ),
}

SC_TERMINAL_QUATS_XYZW = {
    "sc_port_2": (0.97127, -0.06051, -0.01121, 0.22987),
}

PUBLIC_SCRIPTED_FINAL_POSES = {
    "nic_card_mount_0": (
        (-0.390284, 0.205366, 0.060553),
        (0.982508, -0.027752, -0.004932, 0.184076),
    ),
    "nic_card_mount_1": (
        (-0.391832, 0.245366, 0.063497),
        (0.982520, -0.027774, -0.005042, 0.184003),
    ),
    "sc_port_2": (
        (-0.486034, 0.286927, -0.020947),
        (0.971273, -0.060505, -0.011211, 0.229874),
    ),
}

PUBLIC_SCRIPTED_FINAL_JOINTS = {
    "nic_card_mount_0": (
        -0.430668,
        -1.574238,
        -1.843707,
        -1.436053,
        1.914470,
        1.171947,
    ),
    "nic_card_mount_1": (
        -0.501412,
        -1.638057,
        -1.778756,
        -1.461580,
        1.903620,
        1.097928,
    ),
    "sc_port_2": (
        -0.507122,
        -2.047606,
        -1.694057,
        -1.161794,
        1.997277,
        1.146562,
    ),
}

SC_PORT_ALIASES = {
    # Isaac uses sc_port/sc_port_2 for the two available SC targets. The official
    # Gazebo task metadata names the mounted modules as sc_port_0/sc_port_1.
    "sc_port_0": "sc_port",
    "sc_port_1": "sc_port_2",
}

IMAGE_FEATURE_DIM = 1000
JOINT_OBSERVATION_DIM = 46
DEFAULT_TASK_METADATA_DIM = 2
ACTOR_OBSERVATION_FIXED_DIM = (
    JOINT_OBSERVATION_DIM
    + JOINT_OBSERVATION_DIM
    + 7
    + 42
    + 3 * IMAGE_FEATURE_DIM
    + 6
)
ACTOR_OBSERVATION_DIM = DEFAULT_TASK_METADATA_DIM + ACTOR_OBSERVATION_FIXED_DIM
SFP_GAZEBO_ACTOR_OBSERVATION_DIM = (
    DEFAULT_TASK_METADATA_DIM + len(SFP_MOUNT_ONE_HOT) + ACTOR_OBSERVATION_FIXED_DIM
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_name_set(name: str) -> set[str] | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _env_target_sequence(
    name: str,
    default: tuple[tuple[float, float, float], ...] | None,
) -> tuple[tuple[float, float, float], ...] | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list of [x, y, z] targets")
    targets = []
    for index, target in enumerate(parsed):
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError(
                f"{name}[{index}] must be a JSON list of exactly 3 floats"
            )
        targets.append(tuple(float(value) for value in target))
    return tuple(targets)


def _env_vector3(name: str, default: tuple[float, float, float]) -> np.ndarray:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return np.array(default, dtype=np.float64)
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 3:
        raise ValueError(f"{name} must be a JSON list of exactly 3 floats")
    return np.array([float(item) for item in parsed], dtype=np.float64)


def _env_vector3_sequence(
    name: str,
    default: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list of [x, y, z] vectors")
    vectors = []
    for index, item in enumerate(parsed):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"{name}[{index}] must be a JSON list of 3 floats")
        vectors.append(tuple(float(value) for value in item))
    return tuple(vectors)


def _interpolated_targets(
    start: np.ndarray,
    targets: tuple[tuple[float, float, float], ...],
    max_step: float,
) -> list[np.ndarray]:
    if max_step <= 0.0:
        return [np.array(target, dtype=np.float64) for target in targets]
    output: list[np.ndarray] = []
    previous = start.astype(np.float64, copy=True)
    for target in targets:
        target_array = np.array(target, dtype=np.float64)
        distance = float(np.linalg.norm(target_array - previous))
        steps = max(1, int(np.ceil(distance / max_step)))
        for index in range(1, steps + 1):
            alpha = index / steps
            output.append((1.0 - alpha) * previous + alpha * target_array)
        previous = target_array
    return output


def _axis_offsets(radius: float, step: float) -> list[float]:
    radius = max(0.0, float(radius))
    step = max(1.0e-6, float(step))
    count = int(np.floor(radius / step + 1.0e-9))
    offsets = [0.0]
    for index in range(1, count + 1):
        value = index * step
        offsets.extend((value, -value))
    remainder = radius - count * step
    if remainder > 1.0e-6:
        offsets.extend((radius, -radius))
    return offsets


def _grid_offsets(
    x_radius: float,
    y_radius: float,
    step: float,
) -> list[tuple[float, float]]:
    offsets = [
        (float(dx), float(dy))
        for dx in _axis_offsets(x_radius, step)
        for dy in _axis_offsets(y_radius, step)
    ]
    offsets.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], abs(item[0]), abs(item[1])))
    return offsets


def _safe_token(value: str) -> str:
    token = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return token.strip("_") or "unknown"


def _array_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def _feature_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "norm": float(np.linalg.norm(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _actor_observation_slices(task_metadata_dim: int) -> dict[str, slice]:
    cursor = 0
    slices = {"task": slice(cursor, cursor + task_metadata_dim)}
    cursor += task_metadata_dim
    slices["joint_pos"] = slice(cursor, cursor + JOINT_OBSERVATION_DIM)
    cursor += JOINT_OBSERVATION_DIM
    slices["joint_vel"] = slice(cursor, cursor + JOINT_OBSERVATION_DIM)
    cursor += JOINT_OBSERVATION_DIM
    slices["eef"] = slice(cursor, cursor + 7)
    cursor += 7
    slices["body_forces"] = slice(cursor, cursor + 42)
    cursor += 42
    slices["center_rgb"] = slice(cursor, cursor + IMAGE_FEATURE_DIM)
    cursor += IMAGE_FEATURE_DIM
    slices["left_rgb"] = slice(cursor, cursor + IMAGE_FEATURE_DIM)
    cursor += IMAGE_FEATURE_DIM
    slices["right_rgb"] = slice(cursor, cursor + IMAGE_FEATURE_DIM)
    cursor += IMAGE_FEATURE_DIM
    slices["last_action"] = slice(cursor, cursor + 6)
    return slices


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_apply_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _quat_inverse_apply_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    return _quat_apply_xyzw(q_inv, v)


def _axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis_angle.astype(np.float64, copy=False) / angle
    half_angle = 0.5 * angle
    return np.array(
        [
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle),
        ],
        dtype=np.float64,
    )


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_xyzw(q0)
    q1 = _normalize_quat_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    sin_theta_0 = float(np.sin(theta_0))
    scale0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale1 = sin_theta / sin_theta_0
    return _normalize_quat_xyzw(scale0 * q0 + scale1 * q1)


class RslRlCheckpointPolicy(Policy):
    """Official-API policy wrapper for exported Isaac RSL-RL actors.

    Environment variables consumed by this scaffold:

    - ``AIC_RSLRL_CHECKPOINT``: raw Isaac/RSL-RL checkpoint path used for all
      tasks when task-specific paths are not set.
    - ``AIC_RSLRL_SC_CHECKPOINT`` / ``AIC_RSLRL_SFP_CHECKPOINT``: task-specific
      raw checkpoint paths for final routing.
    - ``AIC_RSLRL_POLICY_ARTIFACT``: exported actor artifact path used for all
      tasks when task-specific artifacts are not set.
    - ``AIC_RSLRL_SC_POLICY_ARTIFACT`` / ``AIC_RSLRL_SFP_POLICY_ARTIFACT``:
      task-specific exported actor artifacts. The intended future format is
      TorchScript or ONNX from Isaac Lab ``play.py`` export.
    - ``AIC_RSLRL_TASK_KIND``: ``sc``, ``sfp``, or ``auto``.
    - ``AIC_RSLRL_RESNET18_WEIGHTS``: optional local torchvision ResNet18
      state-dict path. If unset, torchvision's ImageNet V1 weights are tried.
    - ``AIC_RSLRL_ENABLE_SC_PREPOSE``: optional legal joint-space warm start
      to the SC near-port curriculum pose selected by official task metadata.
    - ``AIC_RSLRL_SC_PREPOSE_MIRROR_SHOULDER``: whether to apply the
      Isaac-to-Gazebo shoulder-pan sign conversion to the SC warm-start preset.
    - ``AIC_RSLRL_ENABLE_SC_TERMINAL_ORIENTATION``: use public-geometry SC
      terminal orientation during the final approach.
    - ``AIC_RSLRL_ENABLE_SFP_PREPOSE``: optional legal joint-space warm start
      to the SFP curriculum pose selected by ``Task.port_name``. Defaults false
      because the official Gazebo task spawn already starts SFP close to target.
    - ``AIC_RSLRL_SC_POSITION_SCALE`` / ``AIC_RSLRL_SFP_POSITION_SCALE``:
      per-task scale for replaying actor translation actions as Cartesian targets.
    - ``AIC_RSLRL_SC_COMMAND_FRAME`` / ``AIC_RSLRL_SFP_COMMAND_FRAME``:
      either ``base_link`` absolute targets or ``gripper/tcp`` relative targets.
    - ``AIC_RSLRL_SFP_FINAL_SETTLE_SEC``: optional SFP TCP-frame final settle
      experiment duration. Defaults disabled.
    - ``AIC_RSLRL_SFP_BASE_INSERT_SEC``: optional SFP base-frame downward
      insertion push after actor replay. Defaults disabled.
    - ``AIC_RSLRL_ENABLE_SFP_GUARDED_INSERT``: optional TCP-frame guarded
      close-range insertion. It uses only the official wrist wrench observation
      to back off on contact.
    - ``AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH``: optional legal SFP terminal search
      after actor replay. It uses only the official TCP observation and emitted
      Cartesian commands, not TF/scoring internals.
    - ``AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_ENABLED``: optional use of
      mount-specific first search offsets. Defaults false; generic local search
      remains symmetric around the observed final TCP pose.
    - ``AIC_RSLRL_SC_ACTOR_ENABLED``: optional diagnostic toggle. Defaults true;
      set false to evaluate legal SC prepose without actor handoff.
    - ``AIC_RSLRL_FIXED_STEP_REPLAY``: optional replay of the full planned actor
      step count. Defaults false because controlled official eval regressed.
    - ``AIC_RSLRL_ZERO_JOINT_OBS``: optional diagnostic that zeros arm joint
      position/velocity observations. Defaults false.
    - ``AIC_RSLRL_ZERO_BODY_FORCES``: optional diagnostic that zeros the 42D body
      force block. Defaults false.
    - ``AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA``: optional legal SFP actor mode
      that appends official ``Task.target_module_name`` as a 2D mount one-hot.
      Defaults false to preserve existing 3149D exported actors.
    - ``AIC_RSLRL_TRACE_DIR``: optional directory for JSONL policy traces from
      legal ``Task``/``Observation`` inputs and emitted ``MotionUpdate`` targets.
    - ``AIC_RSLRL_TRACE_FULL_OBS``: optional compressed dump of full actor
      observations/actions next to the JSONL trace. Defaults false.
    - ``AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_DELTA`` and
      ``AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_DELTA_<TARGET>``: optional JSON
      ``[x, y, z]`` offsets for the public-sample scripted final pose.
    - ``AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_JOINTS``: optional final
      joint-space finish using offline public-sample calibration targets.
    - ``AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_TRACE_REPLAY``: optional replay of
      offline public-sample Cartesian command traces keyed by official target.
    - ``AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_TARGETS``: optional comma-separated
      target keys that should use trace replay when it is enabled.
    - ``AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA`` and
      ``AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA_<TARGET>``: optional JSON
      ``[dx, dy, dz]`` base-frame offsets applied to replayed traces.
    - ``AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PREPEND_STEPS`` and
      ``AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PREPEND_STEPS_<TARGET>``: optional
      interpolation from the observed TCP pose into the first replay sample.
    - ``AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_JOINT_TRACE_REPLAY``: optional replay
      of offline public-sample joint trajectories keyed by official target.

    Raw Isaac/RSL-RL checkpoints still need to be exported to TorchScript first.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.checkpoint_path = os.environ.get("AIC_RSLRL_CHECKPOINT", "")
        self.sc_checkpoint_path = os.environ.get("AIC_RSLRL_SC_CHECKPOINT", "")
        self.sfp_checkpoint_path = os.environ.get("AIC_RSLRL_SFP_CHECKPOINT", "")
        self.policy_artifact_path = os.environ.get("AIC_RSLRL_POLICY_ARTIFACT", "")
        self.sc_policy_artifact_path = os.environ.get(
            "AIC_RSLRL_SC_POLICY_ARTIFACT", ""
        )
        self.sfp_policy_artifact_path = os.environ.get(
            "AIC_RSLRL_SFP_POLICY_ARTIFACT", ""
        )
        self.task_kind = os.environ.get("AIC_RSLRL_TASK_KIND", "auto")
        self._actors = {}
        self._torch = None
        self._resnet18 = None
        self._resnet18_failed = False
        self._last_action = np.zeros(6, dtype=np.float32)
        self._control_hz = _env_float("AIC_RSLRL_CONTROL_HZ", 10.0)
        self._sc_max_control_sec = _env_float("AIC_RSLRL_SC_MAX_CONTROL_SEC", 9.0)
        self._sfp_max_control_sec = _env_float("AIC_RSLRL_SFP_MAX_CONTROL_SEC", 9.0)
        self._sc_prepose_enabled = _env_bool("AIC_RSLRL_ENABLE_SC_PREPOSE", True)
        self._sc_prepose_sec = _env_float("AIC_RSLRL_SC_PREPOSE_SEC", 6.0)
        self._sc_prepose_mirror_shoulder = _env_bool(
            "AIC_RSLRL_SC_PREPOSE_MIRROR_SHOULDER", True
        )
        self._sfp_prepose_enabled = _env_bool("AIC_RSLRL_ENABLE_SFP_PREPOSE", False)
        self._sfp_prepose_sec = _env_float("AIC_RSLRL_SFP_PREPOSE_SEC", 6.0)
        self._sc_terminal_target_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_TERMINAL_TARGET", False
        )
        self._sc_terminal_target_dwell_sec = _env_float(
            "AIC_RSLRL_SC_TERMINAL_TARGET_DWELL_SEC", 1.20
        )
        self._sc_terminal_orientation_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_TERMINAL_ORIENTATION", False
        )
        self._sc_terminal_target_max_step = _env_float(
            "AIC_RSLRL_SC_TERMINAL_TARGET_MAX_STEP", 0.0
        )
        self._sc_guarded_insert_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_GUARDED_INSERT", False
        )
        self._sc_guarded_insert_sec = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_SEC", 4.0
        )
        self._sc_guarded_insert_down_step = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_DOWN_STEP", -0.0008
        )
        self._sc_guarded_insert_retract_step = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_RETRACT_STEP", 0.0012
        )
        self._sc_guarded_insert_retract_on_force = _env_bool(
            "AIC_RSLRL_SC_GUARDED_INSERT_RETRACT_ON_FORCE", True
        )
        self._sc_guarded_insert_force_limit = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_FORCE_LIMIT", 18.0
        )
        self._sc_guarded_insert_linear_stiffness = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_STIFFNESS", 25.0
        )
        self._sc_guarded_insert_angular_stiffness = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_ANGULAR_STIFFNESS", 8.0
        )
        self._sc_guarded_insert_linear_damping = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_DAMPING", 18.0
        )
        self._sc_guarded_insert_angular_damping = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_ANGULAR_DAMPING", 6.0
        )
        self._sc_guarded_insert_joint_hold_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_GUARDED_INSERT_JOINT_HOLD", False
        )
        self._sc_guarded_insert_joint_hold_sec = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_JOINT_HOLD_SEC", 0.30
        )
        self._sc_guarded_insert_joint_hold_dt = _env_float(
            "AIC_RSLRL_SC_GUARDED_INSERT_JOINT_HOLD_DT", 0.05
        )
        self._sc_tcp_z_recovery_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_TCP_Z_RECOVERY", False
        )
        self._sc_tcp_z_recovery_threshold = _env_float(
            "AIC_RSLRL_SC_TCP_Z_RECOVERY_THRESHOLD", 0.040
        )
        self._sc_tcp_z_recovery_dwell_sec = _env_float(
            "AIC_RSLRL_SC_TCP_Z_RECOVERY_DWELL_SEC", 0.25
        )
        self._sc_tcp_z_recovery_offsets = _env_vector3_sequence(
            "AIC_RSLRL_SC_TCP_Z_RECOVERY_OFFSETS",
            ((0.0, 0.0, -0.010),),
        )
        self._sfp_terminal_target_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_TARGET", False
        )
        self._sfp_terminal_target_dwell_sec = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC", 1.20
        )
        self._sfp_terminal_target_max_step = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_MAX_STEP", 0.0
        )
        self._sfp_terminal_orientation_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_ORIENTATION", False
        )
        self._sfp_terminal_feedforward_base_z = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_FEEDFORWARD_BASE_Z", 0.0
        )
        self._sfp_terminal_target_linear_stiffness = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_LINEAR_STIFFNESS", 85.0
        )
        self._sfp_terminal_target_angular_stiffness = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_ANGULAR_STIFFNESS", 45.0
        )
        self._sfp_terminal_target_linear_damping = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_LINEAR_DAMPING", 38.0
        )
        self._sfp_terminal_target_angular_damping = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_TARGET_ANGULAR_DAMPING", 14.0
        )
        self._sfp_tcp_z_recovery_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_TCP_Z_RECOVERY", False
        )
        self._sfp_tcp_z_recovery_threshold = _env_float(
            "AIC_RSLRL_SFP_TCP_Z_RECOVERY_THRESHOLD", 0.210
        )
        self._sfp_tcp_z_recovery_dwell_sec = _env_float(
            "AIC_RSLRL_SFP_TCP_Z_RECOVERY_DWELL_SEC", 0.25
        )
        self._sfp_tcp_z_recovery_offsets = _env_vector3_sequence(
            "AIC_RSLRL_SFP_TCP_Z_RECOVERY_OFFSETS",
            (
                (0.0, 0.0, -0.020),
                (0.0, 0.0, -0.040),
                (0.002, -0.002, -0.020),
                (-0.002, 0.002, -0.020),
            ),
        )
        self._sc_position_scale = _env_float("AIC_RSLRL_SC_POSITION_SCALE", 0.05)
        self._sc_rotation_scale = _env_float(
            "AIC_RSLRL_SC_ROTATION_SCALE", self._sc_position_scale
        )
        self._sfp_position_scale = _env_float("AIC_RSLRL_SFP_POSITION_SCALE", 0.003)
        self._sfp_rotation_scale = _env_float(
            "AIC_RSLRL_SFP_ROTATION_SCALE", self._sfp_position_scale
        )
        self._sc_command_frame = os.environ.get("AIC_RSLRL_SC_COMMAND_FRAME", "base_link")
        self._sfp_command_frame = os.environ.get(
            "AIC_RSLRL_SFP_COMMAND_FRAME", "base_link"
        )
        self._sfp_final_settle_sec = _env_float("AIC_RSLRL_SFP_FINAL_SETTLE_SEC", 0.0)
        self._sfp_final_settle_step = _env_float(
            "AIC_RSLRL_SFP_FINAL_SETTLE_STEP", -0.002
        )
        self._sfp_base_insert_sec = _env_float("AIC_RSLRL_SFP_BASE_INSERT_SEC", 0.0)
        self._sfp_base_insert_step = _env_float(
            "AIC_RSLRL_SFP_BASE_INSERT_STEP", -0.003
        )
        self._sfp_guarded_insert_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_GUARDED_INSERT", False
        )
        self._sfp_guarded_insert_sec = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_SEC", 4.0
        )
        self._sfp_guarded_insert_down_step = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_DOWN_STEP", -0.0008
        )
        self._sfp_guarded_insert_lateral_step = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_LATERAL_STEP", 0.0010
        )
        self._sfp_guarded_insert_retract_step = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_RETRACT_STEP", 0.0012
        )
        self._sfp_guarded_insert_force_limit = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_FORCE_LIMIT", 18.0
        )
        self._sfp_guarded_insert_linear_stiffness = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_STIFFNESS", 35.0
        )
        self._sfp_guarded_insert_angular_stiffness = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_ANGULAR_STIFFNESS", 12.0
        )
        self._sfp_guarded_insert_linear_damping = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_DAMPING", 18.0
        )
        self._sfp_guarded_insert_angular_damping = _env_float(
            "AIC_RSLRL_SFP_GUARDED_INSERT_ANGULAR_DAMPING", 6.0
        )
        sfp_guarded_targets = os.environ.get(
            "AIC_RSLRL_SFP_GUARDED_INSERT_TARGETS", ""
        ).strip()
        self._sfp_guarded_insert_targets = (
            {item.strip() for item in sfp_guarded_targets.split(",") if item.strip()}
            if sfp_guarded_targets
            else None
        )
        self._sfp_local_search_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH", False
        )
        self._sfp_local_search_targets = _env_name_set(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_TARGETS"
        )
        self._sfp_local_search_radius = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_RADIUS", 0.065
        )
        self._sfp_local_search_step = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_STEP", 0.0125
        )
        self._sfp_local_search_dwell_sec = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_DWELL_SEC", 0.10
        )
        self._sfp_local_search_priority_dwell_sec = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_DWELL_SEC", 1.20
        )
        self._sfp_local_search_priority_enabled = _env_bool(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_ENABLED", False
        )
        self._sfp_local_search_tcp_z_threshold = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_TCP_Z_THRESHOLD", 0.0
        )
        self._sfp_local_search_z_down = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_Z_DOWN", 0.018
        )
        self._sfp_local_search_z_step = _env_float(
            "AIC_RSLRL_SFP_LOCAL_SEARCH_Z_STEP", 0.006
        )
        self._sfp_terminal_fallback_joints_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_FALLBACK_JOINTS", False
        )
        self._sfp_terminal_fallback_tcp_z_threshold = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_TCP_Z_THRESHOLD", 0.21
        )
        self._sfp_terminal_fallback_joint_steps = _env_int(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_JOINT_STEPS", 0
        )
        self._sfp_terminal_fallback_joint_targets = _env_name_set(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_JOINT_TARGETS"
        )
        self._sfp_terminal_fallback_trace_suffix_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_FALLBACK_TRACE_SUFFIX", False
        )
        self._sfp_terminal_fallback_trace_targets = _env_name_set(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_TRACE_TARGETS"
        )
        self._sfp_terminal_fallback_trace_start_index = _env_int(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_TRACE_START_INDEX", 350
        )
        self._sfp_terminal_fallback_trace_dt = _env_float(
            "AIC_RSLRL_SFP_TERMINAL_FALLBACK_TRACE_DT",
            0.05,
        )
        self._sc_grid_search_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SC_GRID_SEARCH", False
        )
        self._sc_grid_search_targets = _env_name_set(
            "AIC_RSLRL_SC_GRID_SEARCH_TARGETS"
        )
        self._sc_grid_search_x_radius = _env_float(
            "AIC_RSLRL_SC_GRID_SEARCH_X_RADIUS", 0.070
        )
        self._sc_grid_search_y_radius = _env_float(
            "AIC_RSLRL_SC_GRID_SEARCH_Y_RADIUS", 0.070
        )
        self._sc_grid_search_step = _env_float(
            "AIC_RSLRL_SC_GRID_SEARCH_STEP", 0.010
        )
        self._sc_grid_search_z_offsets = _env_vector3_sequence(
            "AIC_RSLRL_SC_GRID_SEARCH_Z_OFFSETS",
            ((0.0, 0.0, 0.006), (0.0, 0.0, 0.0), (0.0, 0.0, -0.006)),
        )
        self._sc_grid_search_dwell_sec = _env_float(
            "AIC_RSLRL_SC_GRID_SEARCH_DWELL_SEC", 0.045
        )
        self._sc_grid_search_probe_steps = _env_int(
            "AIC_RSLRL_SC_GRID_SEARCH_PROBE_STEPS", 2
        )
        self._sc_grid_search_probe_step = _env_float(
            "AIC_RSLRL_SC_GRID_SEARCH_PROBE_STEP", -0.0008
        )
        self._sfp_grid_search_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_SFP_GRID_SEARCH", False
        )
        self._sfp_grid_search_targets = _env_name_set(
            "AIC_RSLRL_SFP_GRID_SEARCH_TARGETS"
        )
        self._sfp_grid_search_x_radius = _env_float(
            "AIC_RSLRL_SFP_GRID_SEARCH_X_RADIUS", 0.035
        )
        self._sfp_grid_search_y_radius = _env_float(
            "AIC_RSLRL_SFP_GRID_SEARCH_Y_RADIUS", 0.110
        )
        self._sfp_grid_search_step = _env_float(
            "AIC_RSLRL_SFP_GRID_SEARCH_STEP", 0.010
        )
        self._sfp_grid_search_z_offsets = _env_vector3_sequence(
            "AIC_RSLRL_SFP_GRID_SEARCH_Z_OFFSETS",
            ((0.0, 0.0, 0.006), (0.0, 0.0, 0.0), (0.0, 0.0, -0.006)),
        )
        self._sfp_grid_search_dwell_sec = _env_float(
            "AIC_RSLRL_SFP_GRID_SEARCH_DWELL_SEC", 0.045
        )
        self._sfp_grid_search_probe_steps = _env_int(
            "AIC_RSLRL_SFP_GRID_SEARCH_PROBE_STEPS", 2
        )
        self._sfp_grid_search_probe_step = _env_float(
            "AIC_RSLRL_SFP_GRID_SEARCH_PROBE_STEP", -0.0007
        )
        self._sc_actor_enabled = _env_bool("AIC_RSLRL_SC_ACTOR_ENABLED", True)
        self._fixed_step_replay = _env_bool("AIC_RSLRL_FIXED_STEP_REPLAY", False)
        self._zero_joint_obs = _env_bool("AIC_RSLRL_ZERO_JOINT_OBS", False)
        self._zero_body_forces = _env_bool("AIC_RSLRL_ZERO_BODY_FORCES", False)
        self._body_force_scale = _env_float("AIC_RSLRL_BODY_FORCE_SCALE", 0.1)
        self._body_force_first_slots = _env_bool(
            "AIC_RSLRL_BODY_FORCE_FIRST_SLOTS", False
        )
        self._sfp_include_mount_metadata = _env_bool(
            "AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA", False
        )
        self._public_scripted_insert_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_INSERT", False
        )
        self._public_scripted_approach_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_APPROACH_STEPS", 100
        )
        self._public_scripted_descent_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_DESCENT_STEPS", 430
        )
        self._public_scripted_dt = _env_float("AIC_RSLRL_PUBLIC_SCRIPTED_DT", 0.05)
        self._public_scripted_above_z = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_ABOVE_Z", 0.215
        )
        self._public_scripted_dwell_sec = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_DWELL_SEC", 5.0
        )
        self._public_scripted_final_joints_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_JOINTS", False
        )
        self._public_scripted_final_pose_hold_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_POSE_HOLD", False
        )
        self._public_scripted_final_pose_hold_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_STEPS", 80
        )
        self._public_scripted_final_pose_hold_dt = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_DT", 0.05
        )
        self._public_scripted_final_pose_hold_z_threshold = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_Z_THRESHOLD", 0.0
        )
        self._public_scripted_final_pose_hold_min_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_MIN_STEPS", 20
        )
        self._public_scripted_final_joint_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_STEPS", 120
        )
        self._public_scripted_final_joint_dt = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_DT", 0.05
        )
        self._public_scripted_final_joint_targets = _env_name_set(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_TARGETS"
        )
        self._public_scripted_trace_replay_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_TRACE_REPLAY", False
        )
        self._public_scripted_trace_targets = _env_name_set(
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_TARGETS"
        )
        self._public_scripted_trace_dt = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DT", 0.05
        )
        self._public_scripted_trace_prepend_steps = _env_int(
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PREPEND_STEPS", 0
        )
        self._public_scripted_trace_path = Path(
            os.environ.get(
                "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PATH",
                str(
                    Path(__file__).resolve().parents[1]
                    / "artifacts"
                    / "public_scripted_traces.json"
                ),
            )
        )
        self._public_scripted_traces: dict[str, list[list[float]]] | None = None
        self._public_scripted_joint_trace_replay_enabled = _env_bool(
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_JOINT_TRACE_REPLAY", False
        )
        self._public_scripted_joint_trace_dt = _env_float(
            "AIC_RSLRL_PUBLIC_SCRIPTED_JOINT_TRACE_DT", 0.05
        )
        self._public_scripted_joint_trace_path = Path(
            os.environ.get(
                "AIC_RSLRL_PUBLIC_SCRIPTED_JOINT_TRACE_PATH",
                str(
                    Path(__file__).resolve().parents[1]
                    / "artifacts"
                    / "public_scripted_joint_traces.json"
                ),
            )
        )
        self._public_scripted_joint_traces: dict[str, list[list[float]]] | None = None
        self._require_resnet18 = _env_bool("AIC_RSLRL_REQUIRE_RESNET18", False)
        self._log_every_n = max(1, _env_int("AIC_RSLRL_LOG_EVERY_N", 20))
        self._trace_dir = os.environ.get("AIC_RSLRL_TRACE_DIR", "")
        self._trace_every_n = max(1, _env_int("AIC_RSLRL_TRACE_EVERY_N", 1))
        self._trace_full_obs = _env_bool("AIC_RSLRL_TRACE_FULL_OBS", False)
        self._trace_path: Path | None = None
        self._trace_actor_obs: list[np.ndarray] = []
        self._trace_actions: list[np.ndarray] = []

        self.get_logger().info(
            "RslRlCheckpointPolicy configured with "
            f"AIC_RSLRL_CHECKPOINT={self.checkpoint_path!r}, "
            f"AIC_RSLRL_SC_CHECKPOINT={self.sc_checkpoint_path!r}, "
            f"AIC_RSLRL_SFP_CHECKPOINT={self.sfp_checkpoint_path!r}, "
            f"AIC_RSLRL_POLICY_ARTIFACT={self.policy_artifact_path!r}, "
            f"AIC_RSLRL_SC_POLICY_ARTIFACT={self.sc_policy_artifact_path!r}, "
            f"AIC_RSLRL_SFP_POLICY_ARTIFACT={self.sfp_policy_artifact_path!r}, "
            f"AIC_RSLRL_TASK_KIND={self.task_kind!r}, "
            f"AIC_RSLRL_ENABLE_SC_PREPOSE={self._sc_prepose_enabled!r}, "
            "AIC_RSLRL_SC_PREPOSE_MIRROR_SHOULDER="
            f"{self._sc_prepose_mirror_shoulder!r}, "
            f"AIC_RSLRL_ENABLE_SFP_PREPOSE={self._sfp_prepose_enabled!r}, "
            "AIC_RSLRL_ENABLE_SC_TERMINAL_TARGET="
            f"{self._sc_terminal_target_enabled!r}, "
            "AIC_RSLRL_SC_TERMINAL_TARGET_DWELL_SEC="
            f"{self._sc_terminal_target_dwell_sec!r}, "
            "AIC_RSLRL_ENABLE_SC_TERMINAL_ORIENTATION="
            f"{self._sc_terminal_orientation_enabled!r}, "
            "AIC_RSLRL_SC_TERMINAL_TARGET_MAX_STEP="
            f"{self._sc_terminal_target_max_step!r}, "
            "AIC_RSLRL_ENABLE_SC_GUARDED_INSERT="
            f"{self._sc_guarded_insert_enabled!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_SEC="
            f"{self._sc_guarded_insert_sec!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_DOWN_STEP="
            f"{self._sc_guarded_insert_down_step!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_FORCE_LIMIT="
            f"{self._sc_guarded_insert_force_limit!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_RETRACT_ON_FORCE="
            f"{self._sc_guarded_insert_retract_on_force!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_STIFFNESS="
            f"{self._sc_guarded_insert_linear_stiffness!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_DAMPING="
            f"{self._sc_guarded_insert_linear_damping!r}, "
            "AIC_RSLRL_ENABLE_SC_GUARDED_INSERT_JOINT_HOLD="
            f"{self._sc_guarded_insert_joint_hold_enabled!r}, "
            "AIC_RSLRL_SC_GUARDED_INSERT_JOINT_HOLD_SEC="
            f"{self._sc_guarded_insert_joint_hold_sec!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_TARGETS="
            f"{self._sfp_guarded_insert_targets!r}, "
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_TARGET="
            f"{self._sfp_terminal_target_enabled!r}, "
            "AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC="
            f"{self._sfp_terminal_target_dwell_sec!r}, "
            "AIC_RSLRL_SFP_TERMINAL_TARGET_MAX_STEP="
            f"{self._sfp_terminal_target_max_step!r}, "
            "AIC_RSLRL_ENABLE_SFP_TERMINAL_ORIENTATION="
            f"{self._sfp_terminal_orientation_enabled!r}, "
            "AIC_RSLRL_SFP_TERMINAL_FEEDFORWARD_BASE_Z="
            f"{self._sfp_terminal_feedforward_base_z!r}, "
            "AIC_RSLRL_SFP_TERMINAL_TARGET_LINEAR_STIFFNESS="
            f"{self._sfp_terminal_target_linear_stiffness!r}, "
            "AIC_RSLRL_SFP_TERMINAL_TARGET_LINEAR_DAMPING="
            f"{self._sfp_terminal_target_linear_damping!r}, "
            f"AIC_RSLRL_SC_MAX_CONTROL_SEC={self._sc_max_control_sec!r}, "
            f"AIC_RSLRL_SFP_MAX_CONTROL_SEC={self._sfp_max_control_sec!r}, "
            f"AIC_RSLRL_SC_POSITION_SCALE={self._sc_position_scale!r}, "
            f"AIC_RSLRL_SC_ROTATION_SCALE={self._sc_rotation_scale!r}, "
            f"AIC_RSLRL_SFP_POSITION_SCALE={self._sfp_position_scale!r}, "
            f"AIC_RSLRL_SFP_ROTATION_SCALE={self._sfp_rotation_scale!r}, "
            f"AIC_RSLRL_SC_COMMAND_FRAME={self._sc_command_frame!r}, "
            f"AIC_RSLRL_SFP_COMMAND_FRAME={self._sfp_command_frame!r}, "
            f"AIC_RSLRL_SFP_FINAL_SETTLE_SEC={self._sfp_final_settle_sec!r}, "
            f"AIC_RSLRL_SFP_BASE_INSERT_SEC={self._sfp_base_insert_sec!r}, "
            f"AIC_RSLRL_SFP_BASE_INSERT_STEP={self._sfp_base_insert_step!r}, "
            "AIC_RSLRL_ENABLE_SFP_GUARDED_INSERT="
            f"{self._sfp_guarded_insert_enabled!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_SEC="
            f"{self._sfp_guarded_insert_sec!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_DOWN_STEP="
            f"{self._sfp_guarded_insert_down_step!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_LATERAL_STEP="
            f"{self._sfp_guarded_insert_lateral_step!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_FORCE_LIMIT="
            f"{self._sfp_guarded_insert_force_limit!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_STIFFNESS="
            f"{self._sfp_guarded_insert_linear_stiffness!r}, "
            "AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_DAMPING="
            f"{self._sfp_guarded_insert_linear_damping!r}, "
            "AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH="
            f"{self._sfp_local_search_enabled!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_TARGETS="
            f"{self._sfp_local_search_targets!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_RADIUS="
            f"{self._sfp_local_search_radius!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_STEP="
            f"{self._sfp_local_search_step!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_DWELL_SEC="
            f"{self._sfp_local_search_dwell_sec!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_DWELL_SEC="
            f"{self._sfp_local_search_priority_dwell_sec!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_ENABLED="
            f"{self._sfp_local_search_priority_enabled!r}, "
            "AIC_RSLRL_SFP_LOCAL_SEARCH_TCP_Z_THRESHOLD="
            f"{self._sfp_local_search_tcp_z_threshold!r}, "
            f"AIC_RSLRL_SC_ACTOR_ENABLED={self._sc_actor_enabled!r}, "
            f"AIC_RSLRL_FIXED_STEP_REPLAY={self._fixed_step_replay!r}, "
            f"AIC_RSLRL_ZERO_JOINT_OBS={self._zero_joint_obs!r}, "
            f"AIC_RSLRL_ZERO_BODY_FORCES={self._zero_body_forces!r}, "
            f"AIC_RSLRL_BODY_FORCE_SCALE={self._body_force_scale!r}, "
            "AIC_RSLRL_BODY_FORCE_FIRST_SLOTS="
            f"{self._body_force_first_slots!r}, "
            "AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA="
            f"{self._sfp_include_mount_metadata!r}, "
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_INSERT="
            f"{self._public_scripted_insert_enabled!r}, "
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_JOINTS="
            f"{self._public_scripted_final_joints_enabled!r}, "
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_POSE_HOLD="
            f"{self._public_scripted_final_pose_hold_enabled!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_STEPS="
            f"{self._public_scripted_final_pose_hold_steps!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_DT="
            f"{self._public_scripted_final_pose_hold_dt!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_POSE_HOLD_Z_THRESHOLD="
            f"{self._public_scripted_final_pose_hold_z_threshold!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_STEPS="
            f"{self._public_scripted_final_joint_steps!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_TARGETS="
            f"{self._public_scripted_final_joint_targets!r}, "
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_TRACE_REPLAY="
            f"{self._public_scripted_trace_replay_enabled!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_TARGETS="
            f"{self._public_scripted_trace_targets!r}, "
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PREPEND_STEPS="
            f"{self._public_scripted_trace_prepend_steps!r}, "
            "AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_JOINT_TRACE_REPLAY="
            f"{self._public_scripted_joint_trace_replay_enabled!r}, "
            f"AIC_RSLRL_REQUIRE_RESNET18={self._require_resnet18!r}, "
            f"AIC_RSLRL_TRACE_DIR={self._trace_dir!r}, "
            f"AIC_RSLRL_TRACE_EVERY_N={self._trace_every_n!r}, "
            f"AIC_RSLRL_TRACE_FULL_OBS={self._trace_full_obs!r}"
        )

        artifact_paths = {
            "default": self.policy_artifact_path,
            "sc": self.sc_policy_artifact_path,
            "sfp": self.sfp_policy_artifact_path,
        }
        for task_kind, artifact in artifact_paths.items():
            if artifact:
                self._try_load_policy_artifact(task_kind, Path(artifact))

        if not self._actors and (
            self.checkpoint_path or self.sc_checkpoint_path or self.sfp_checkpoint_path
        ):
            self.get_logger().warn(
                "Raw Isaac RSL-RL checkpoints are not directly loadable by this "
                "Gazebo policy scaffold yet. Export the actor first and finish "
                "the observation/action adapter before expecting nonzero scores."
            )

    def _try_load_policy_artifact(self, task_kind: str, artifact_path: Path) -> None:
        """Best-effort TorchScript load for future exported actor support."""
        if not artifact_path.exists():
            self.get_logger().error(f"Policy artifact does not exist: {artifact_path}")
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on runtime image
            self.get_logger().error(f"Unable to import torch for policy artifact: {exc}")
            return

        try:
            self._torch = torch
            actor = torch.jit.load(str(artifact_path), map_location="cpu")
            actor.eval()
            self._actors[task_kind] = actor
            self.get_logger().info(
                f"Loaded TorchScript policy artifact for {task_kind}: {artifact_path}"
            )
        except Exception as exc:  # pragma: no cover - depends on artifact format
            self.get_logger().error(
                f"Unable to load policy artifact as TorchScript: {artifact_path}: {exc}"
            )

    def _task_kind_from_task(self, task: Task) -> str:
        if self.task_kind in {"sc", "sfp"}:
            return self.task_kind
        joined = " ".join(
            [
                task.plug_type,
                task.port_type,
                task.plug_name,
                task.port_name,
                task.target_module_name,
            ]
        ).lower()
        if "sfp" in joined:
            return "sfp"
        if "sc" in joined:
            return "sc"
        return "default"

    def _actor_for_task(self, task_kind: str):
        return self._actors.get(task_kind) or self._actors.get("default")

    def _sfp_port_name(self, task: Task) -> str | None:
        for candidate in (task.port_name, task.target_module_name):
            candidate = candidate.lower()
            for name in SFP_PORT_ONE_HOT:
                if name in candidate:
                    return name
        return None

    def _sfp_mount_name(self, task: Task) -> str | None:
        for candidate in (task.target_module_name, task.port_name):
            candidate = candidate.lower()
            for name in SFP_MOUNT_ONE_HOT:
                if name in candidate:
                    return name
        return None

    def _sc_port_name(self, task: Task) -> str | None:
        for candidate in (task.target_module_name, task.port_name):
            candidate = candidate.lower()
            for alias, name in SC_PORT_ALIASES.items():
                if alias in candidate:
                    return name
            for name in SC_PORT_ONE_HOT:
                if name in candidate:
                    return name
        if task.port_type.lower() == "sc" or task.plug_type.lower() == "sc":
            # Official sample metadata uses port_name=sc_port_base and
            # target_module_name=sc_port_1. If future metadata omits the module
            # suffix, keep the policy running with the first Isaac SC target.
            return "sc_port"
        return None

    def _task_metadata_dim(self, task_kind: str) -> int:
        if task_kind == "sfp" and self._sfp_include_mount_metadata:
            return DEFAULT_TASK_METADATA_DIM + len(SFP_MOUNT_ONE_HOT)
        return DEFAULT_TASK_METADATA_DIM

    def _expected_actor_observation_dim(self, task_kind: str) -> int:
        if task_kind == "sfp" and self._sfp_include_mount_metadata:
            return SFP_GAZEBO_ACTOR_OBSERVATION_DIM
        if task_kind in {"sc", "sfp"}:
            return ACTOR_OBSERVATION_DIM
        return ACTOR_OBSERVATION_FIXED_DIM + self._task_metadata_dim(task_kind)

    def _task_metadata(self, task: Task, task_kind: str) -> np.ndarray:
        if task_kind == "sfp":
            port_name = self._sfp_port_name(task)
            if port_name is None:
                raise ValueError(
                    "Unable to infer SFP target port from official task metadata: "
                    f"port_name={task.port_name!r}, "
                    f"target_module_name={task.target_module_name!r}"
                )
            metadata = [SFP_PORT_ONE_HOT[port_name]]
            if self._sfp_include_mount_metadata:
                mount_name = self._sfp_mount_name(task)
                if mount_name is None:
                    raise ValueError(
                        "Unable to infer SFP target module from official task metadata: "
                        f"target_module_name={task.target_module_name!r}"
                    )
                metadata.append(SFP_MOUNT_ONE_HOT[mount_name])
            return np.concatenate(metadata).astype(np.float32, copy=False)

        if task_kind == "sc":
            port_name = self._sc_port_name(task)
            if port_name is None:
                raise ValueError(
                    "Unable to infer SC target port from official task metadata: "
                    f"port_name={task.port_name!r}, "
                    f"target_module_name={task.target_module_name!r}"
                )
            return SC_PORT_ONE_HOT[port_name]

        raise ValueError(f"Unsupported task kind for actor observation: {task_kind!r}")

    def _joint_vectors(self, observation) -> tuple[np.ndarray, np.ndarray]:
        if self._zero_joint_obs:
            return (
                np.zeros(JOINT_OBSERVATION_DIM, dtype=np.float32),
                np.zeros(JOINT_OBSERVATION_DIM, dtype=np.float32),
            )
        pos_by_name = dict(
            zip(observation.joint_states.name, observation.joint_states.position)
        )
        vel_by_name = dict(
            zip(observation.joint_states.name, observation.joint_states.velocity)
        )
        gazebo_joint_pos = np.array(
            [
                pos_by_name.get(name, float(GAZEBO_DEFAULT_ARM_JOINT_POS[index]))
                for index, name in enumerate(ARM_JOINT_NAMES)
            ],
            dtype=np.float32,
        )
        gazebo_joint_vel = np.array(
            [vel_by_name.get(name, 0.0) for name in ARM_JOINT_NAMES],
            dtype=np.float32,
        )

        # Gazebo's URDF home has shoulder-pan mirrored relative to the Isaac USD
        # reset used for training; keep the actor in the Isaac convention.
        isaac_joint_pos = gazebo_joint_pos.copy()
        isaac_joint_vel = gazebo_joint_vel.copy()
        isaac_joint_pos[0] *= -1.0
        isaac_joint_vel[0] *= -1.0
        joint_pos_rel = np.zeros(JOINT_OBSERVATION_DIM, dtype=np.float32)
        joint_vel_rel = np.zeros(JOINT_OBSERVATION_DIM, dtype=np.float32)
        joint_pos_rel[: len(ARM_JOINT_NAMES)] = (
            isaac_joint_pos - ISAAC_DEFAULT_ARM_JOINT_POS
        )
        joint_vel_rel[: len(ARM_JOINT_NAMES)] = isaac_joint_vel
        return joint_pos_rel, joint_vel_rel

    def _gazebo_arm_joint_positions(self, observation) -> np.ndarray:
        pos_by_name = dict(
            zip(observation.joint_states.name, observation.joint_states.position)
        )
        return np.array(
            [
                pos_by_name.get(name, float(GAZEBO_DEFAULT_ARM_JOINT_POS[index]))
                for index, name in enumerate(ARM_JOINT_NAMES)
            ],
            dtype=np.float64,
        )

    def _eef_pose(self, observation) -> np.ndarray:
        pose = observation.controller_state.tcp_pose
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=np.float32,
        )

    def _body_forces(self, observation) -> np.ndarray:
        body_forces = np.zeros(42, dtype=np.float32)
        if self._zero_body_forces:
            return body_forces
        wrench = observation.wrist_wrench.wrench
        wrist_wrench = np.array(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )
        if self._body_force_first_slots:
            body_forces[:6] = self._body_force_scale * wrist_wrench
        else:
            body_forces[-6:] = self._body_force_scale * wrist_wrench
        return body_forces

    def _start_trace(self, task: Task, task_kind: str) -> None:
        self._trace_path = None
        self._trace_actor_obs = []
        self._trace_actions = []
        if not self._trace_dir:
            return
        try:
            trace_dir = Path(self._trace_dir).expanduser()
            trace_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = (
                f"{stamp}_{_safe_token(task_kind)}_"
                f"{_safe_token(task.target_module_name)}_"
                f"{_safe_token(task.port_name)}.jsonl"
            )
            self._trace_path = trace_dir / name
            self._write_trace(
                {
                    "event": "start",
                    "task_kind": task_kind,
                    "task": self._task_trace(task),
                    "policy": {
                        "control_hz": self._control_hz,
                        "sc_command_frame": self._sc_command_frame,
                        "sfp_command_frame": self._sfp_command_frame,
                        "sc_position_scale": self._sc_position_scale,
                        "sfp_position_scale": self._sfp_position_scale,
                        "actor_observation_dim": self._expected_actor_observation_dim(
                            task_kind
                        ),
                        "task_metadata_dim": self._task_metadata_dim(task_kind),
                        "sfp_include_mount_metadata": (
                            self._sfp_include_mount_metadata
                        ),
                        "trace_every_n": self._trace_every_n,
                        "trace_full_obs": self._trace_full_obs,
                    },
                }
            )
            self.get_logger().info(f"Writing policy trace to {self._trace_path}")
        except Exception as exc:
            self.get_logger().error(f"Unable to start policy trace: {exc}")
            self._trace_path = None

    def _finish_trace(self, status: str) -> None:
        if self._trace_path is None:
            return
        try:
            npz_path = ""
            if self._trace_full_obs and self._trace_actor_obs:
                npz = self._trace_path.with_suffix(".npz")
                np.savez_compressed(
                    npz,
                    actor_obs=np.stack(self._trace_actor_obs, axis=0),
                    actions=np.stack(self._trace_actions, axis=0),
                )
                npz_path = str(npz)
            self._write_trace(
                {
                    "event": "finish",
                    "status": status,
                    "full_obs_npz": npz_path,
                    "full_obs_count": len(self._trace_actor_obs),
                }
            )
        except Exception as exc:
            self.get_logger().error(f"Unable to finish policy trace: {exc}")
        finally:
            self._trace_path = None
            self._trace_actor_obs = []
            self._trace_actions = []

    def _write_trace(self, payload: dict) -> None:
        if self._trace_path is None:
            return
        with self._trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _task_trace(self, task: Task) -> dict[str, str | int]:
        return {
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "cable_type": task.cable_type,
            "cable_name": task.cable_name,
            "time_limit": int(task.time_limit),
        }

    def _pose_trace(self, pose: Pose) -> dict[str, list[float]]:
        return {
            "xyz": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "quat_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    def _wrench_trace(self, observation) -> list[float]:
        wrench = observation.wrist_wrench.wrench
        return [
            float(wrench.force.x),
            float(wrench.force.y),
            float(wrench.force.z),
            float(wrench.torque.x),
            float(wrench.torque.y),
            float(wrench.torque.z),
        ]

    def _joint_trace(self, observation) -> dict[str, list[float]]:
        pos_by_name = dict(
            zip(observation.joint_states.name, observation.joint_states.position)
        )
        vel_by_name = dict(
            zip(observation.joint_states.name, observation.joint_states.velocity)
        )
        return {
            "names": list(ARM_JOINT_NAMES),
            "positions": [
                float(pos_by_name.get(name, float("nan"))) for name in ARM_JOINT_NAMES
            ],
            "velocities": [
                float(vel_by_name.get(name, float("nan"))) for name in ARM_JOINT_NAMES
            ],
        }

    def _trace_actor_step(
        self,
        *,
        task: Task,
        task_kind: str,
        step: int,
        observation,
        actor_obs: np.ndarray,
        action: np.ndarray,
        motion_update: MotionUpdate,
    ) -> None:
        if self._trace_path is None:
            return
        slices = _actor_observation_slices(self._task_metadata_dim(task_kind))
        if self._trace_full_obs:
            self._trace_actor_obs.append(actor_obs.copy())
            self._trace_actions.append(action.copy())
        if step % self._trace_every_n != 0:
            return

        controller_state = observation.controller_state
        payload = {
            "event": "actor_step",
            "task_kind": task_kind,
            "step": int(step),
            "task": self._task_trace(task),
            "port_selection": {
                "sfp": self._sfp_port_name(task),
                "sfp_mount": self._sfp_mount_name(task),
                "sc": self._sc_port_name(task),
            },
            "actor_obs": {
                "task_metadata": _array_list(actor_obs[slices["task"]]),
                "joint_pos_rel_first6": _array_list(
                    actor_obs[slices["joint_pos"]][:6]
                ),
                "joint_vel_rel_first6": _array_list(
                    actor_obs[slices["joint_vel"]][:6]
                ),
                "eef_pose_xyz_wxyz": _array_list(actor_obs[slices["eef"]]),
                "body_forces_last6": _array_list(
                    actor_obs[slices["body_forces"]][-6:]
                ),
                "center_rgb_resnet18": _feature_summary(
                    actor_obs[slices["center_rgb"]]
                ),
                "left_rgb_resnet18": _feature_summary(actor_obs[slices["left_rgb"]]),
                "right_rgb_resnet18": _feature_summary(
                    actor_obs[slices["right_rgb"]]
                ),
                "last_action": _array_list(actor_obs[slices["last_action"]]),
            },
            "observation": {
                "tcp_pose": self._pose_trace(controller_state.tcp_pose),
                "reference_tcp_pose": self._pose_trace(
                    controller_state.reference_tcp_pose
                ),
                "tcp_error": [float(value) for value in controller_state.tcp_error],
                "target_mode": int(controller_state.target_mode.mode),
                "wrist_wrench": self._wrench_trace(observation),
                "arm_joints": self._joint_trace(observation),
            },
            "action": _array_list(action),
            "command": {
                "frame_id": motion_update.header.frame_id,
                "pose": self._pose_trace(motion_update.pose),
                "stiffness_diag": [
                    float(motion_update.target_stiffness[i * 6 + i])
                    for i in range(6)
                ],
                "damping_diag": [
                    float(motion_update.target_damping[i * 6 + i])
                    for i in range(6)
                ],
                "mode": int(motion_update.trajectory_generation_mode.mode),
            },
        }
        try:
            self._write_trace(payload)
        except Exception as exc:
            self.get_logger().error(f"Unable to write policy trace step: {exc}")
            self._trace_path = None

    def _load_resnet18(self):
        if self._resnet18 is not None or self._resnet18_failed:
            return self._resnet18
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - depends on runtime image
            self._resnet18_failed = True
            self.get_logger().error(f"Unable to import torchvision ResNet18: {exc}")
            return None

        self._torch = torch
        weights_path = os.environ.get("AIC_RSLRL_RESNET18_WEIGHTS", "")
        try:
            if weights_path:
                model = resnet18(weights=None)
                state_dict = torch.load(weights_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict)
            else:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.eval()
            self._resnet18 = model
            self.get_logger().info(
                "Loaded torchvision ResNet18 image encoder"
                + (
                    f" from {weights_path}"
                    if weights_path
                    else " with ImageNet V1 weights"
                )
            )
        except Exception as exc:  # pragma: no cover - depends on runtime image/cache
            self._resnet18_failed = True
            self.get_logger().error(
                "Unable to load ResNet18 weights. Camera features will be zeros: "
                f"{exc}"
            )
            return None
        return self._resnet18

    def _image_array(self, image_msg) -> np.ndarray:
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = image_msg.encoding.lower()
        if height <= 0 or width <= 0:
            raise ValueError("empty image")

        raw = np.frombuffer(image_msg.data, dtype=np.uint8)
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        row_bytes = width * channels
        step = int(image_msg.step) if int(image_msg.step) > 0 else row_bytes
        expected = height * step
        if raw.size < expected:
            raise ValueError(
                f"image buffer too small for {width}x{height} step={step}: {raw.size}"
            )
        image = raw[:expected].reshape(height, step)[:, :row_bytes]
        image = image.reshape(height, width, channels)
        if encoding in {"bgr8", "bgra8"}:
            image = image[..., [2, 1, 0]]
        else:
            image = image[..., :3]
        return np.ascontiguousarray(image)

    def _image_features(self, image_msg) -> np.ndarray:
        model = self._load_resnet18()
        torch = self._torch
        if model is None or torch is None:
            if self._require_resnet18:
                raise RuntimeError("ResNet18 image encoder is unavailable")
            return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
        try:
            image = self._image_array(image_msg)
            tensor = torch.from_numpy(image).to(dtype=torch.float32).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0) / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1
            )
            tensor = (tensor - mean) / std
            with torch.inference_mode():
                features = model(tensor).squeeze(0).detach().cpu().numpy()
            return features.astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - depends on live ROS images
            self.get_logger().error(f"Unable to extract image features: {exc}")
            if self._require_resnet18:
                raise
            return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)

    def _actor_observation(self, task: Task, observation, task_kind: str) -> np.ndarray:
        task_metadata = self._task_metadata(task, task_kind)
        joint_pos_rel, joint_vel_rel = self._joint_vectors(observation)
        pieces = [
            task_metadata,
            joint_pos_rel,
            joint_vel_rel,
            self._eef_pose(observation),
            self._body_forces(observation),
            self._image_features(observation.center_image),
            self._image_features(observation.left_image),
            self._image_features(observation.right_image),
            self._last_action,
        ]
        actor_obs = np.concatenate(pieces).astype(np.float32, copy=False)
        expected_shape = (self._expected_actor_observation_dim(task_kind),)
        if actor_obs.shape != expected_shape:
            raise ValueError(
                f"Unexpected {task_kind.upper()} actor observation shape {actor_obs.shape}; "
                f"expected {expected_shape}"
            )
        return actor_obs

    def _sfp_actor_observation(self, task: Task, observation) -> np.ndarray:
        return self._actor_observation(task, observation, "sfp")

    def _sc_actor_observation(self, task: Task, observation) -> np.ndarray:
        return self._actor_observation(task, observation, "sc")

    def _make_joint_position_update(
        self,
        target: np.ndarray,
        mirror_shoulder: bool = True,
    ) -> JointMotionUpdate:
        gazebo_target = target.astype(np.float64, copy=True)
        if mirror_shoulder:
            gazebo_target[0] *= -1.0
        return JointMotionUpdate(
            target_state=JointTrajectoryPoint(
                positions=[float(value) for value in gazebo_target]
            ),
            target_stiffness=[120.0, 120.0, 120.0, 50.0, 50.0, 50.0],
            target_damping=[45.0, 45.0, 45.0, 18.0, 18.0, 18.0],
            target_feedforward_torque=[0.0] * 6,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

    def _make_position_update(
        self,
        observation,
        action: np.ndarray,
        position_scale: float,
        rotation_scale: float,
        frame_id: str,
    ) -> MotionUpdate:
        current = observation.controller_state.tcp_pose
        delta_position = np.clip(action[:3], -1.0, 1.0) * position_scale
        delta_axis_angle = np.clip(action[3:6], -1.0, 1.0) * rotation_scale
        current_quat = np.array(
            [
                current.orientation.x,
                current.orientation.y,
                current.orientation.z,
                current.orientation.w,
            ],
            dtype=np.float64,
        )
        delta_quat = _axis_angle_to_quat_xyzw(delta_axis_angle)
        if frame_id == "gripper/tcp":
            target_position = delta_position
            target_quat = delta_quat
        elif frame_id == "base_link":
            target_position = np.array(
                [
                    current.position.x + delta_position[0],
                    current.position.y + delta_position[1],
                    current.position.z + delta_position[2],
                ],
                dtype=np.float64,
            )
            target_quat = _normalize_quat_xyzw(
                _quat_multiply_xyzw(delta_quat, current_quat)
            )
        else:
            raise ValueError(
                f"Unsupported MotionUpdate frame {frame_id!r}; expected 'base_link' "
                "or 'gripper/tcp'."
            )

        target = Pose()
        target.position.x = float(target_position[0])
        target.position.y = float(target_position[1])
        target.position.z = float(target_position[2])
        target.orientation.x = float(target_quat[0])
        target.orientation.y = float(target_quat[1])
        target.orientation.z = float(target_quat[2])
        target.orientation.w = float(target_quat[3])

        motion_update = MotionUpdate()
        motion_update.header.frame_id = frame_id
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.pose = target
        motion_update.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        return motion_update

    def _make_base_pose_update(
        self,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        feedforward_force_tip: np.ndarray | None = None,
        *,
        linear_stiffness: float = 85.0,
        angular_stiffness: float = 45.0,
        linear_damping: float = 38.0,
        angular_damping: float = 14.0,
    ) -> MotionUpdate:
        target = Pose()
        target.position.x = float(position[0])
        target.position.y = float(position[1])
        target.position.z = float(position[2])
        quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
        target.orientation.x = float(quat_xyzw[0])
        target.orientation.y = float(quat_xyzw[1])
        target.orientation.z = float(quat_xyzw[2])
        target.orientation.w = float(quat_xyzw[3])

        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.pose = target
        motion_update.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.target_stiffness = np.diag(
            [
                linear_stiffness,
                linear_stiffness,
                linear_stiffness,
                angular_stiffness,
                angular_stiffness,
                angular_stiffness,
            ]
        ).flatten()
        motion_update.target_damping = np.diag(
            [
                linear_damping,
                linear_damping,
                linear_damping,
                angular_damping,
                angular_damping,
                angular_damping,
            ]
        ).flatten()
        if feedforward_force_tip is None:
            feedforward_force_tip = np.zeros(3, dtype=np.float64)
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(
                x=float(feedforward_force_tip[0]),
                y=float(feedforward_force_tip[1]),
                z=float(feedforward_force_tip[2]),
            ),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        return motion_update

    def _make_tcp_delta_update(
        self,
        delta_position: np.ndarray,
        *,
        linear_stiffness: float = 35.0,
        angular_stiffness: float = 12.0,
        linear_damping: float = 18.0,
        angular_damping: float = 6.0,
    ) -> MotionUpdate:
        target = Pose()
        target.position.x = float(delta_position[0])
        target.position.y = float(delta_position[1])
        target.position.z = float(delta_position[2])
        target.orientation.x = 0.0
        target.orientation.y = 0.0
        target.orientation.z = 0.0
        target.orientation.w = 1.0

        motion_update = MotionUpdate()
        motion_update.header.frame_id = "gripper/tcp"
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.pose = target
        motion_update.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.target_stiffness = np.diag(
            [
                linear_stiffness,
                linear_stiffness,
                linear_stiffness,
                angular_stiffness,
                angular_stiffness,
                angular_stiffness,
            ]
        ).flatten()
        motion_update.target_damping = np.diag(
            [
                linear_damping,
                linear_damping,
                linear_damping,
                angular_damping,
                angular_damping,
                angular_damping,
            ]
        ).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.8, 0.8, 0.8, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION
        )
        return motion_update

    def _force_norm(self, observation) -> float:
        wrench = observation.wrist_wrench.wrench
        return float(
            np.linalg.norm([wrench.force.x, wrench.force.y, wrench.force.z])
        )

    def _sfp_local_search_offsets(self) -> list[tuple[float, float]]:
        radius = max(0.0, self._sfp_local_search_radius)
        step = max(1.0e-4, self._sfp_local_search_step)
        values = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float64)
        offsets = [
            (float(dx), float(dy))
            for dx in values
            for dy in values
            if (dx * dx + dy * dy) <= (radius * radius + 1.0e-9)
        ]
        offsets.sort(
            key=lambda item: (
                item[0] * item[0] + item[1] * item[1],
                item[0],
                item[1],
            )
        )
        return offsets

    def _run_joint_prepose(
        self,
        target_name: str | None,
        presets: dict[str, np.ndarray],
        label: str,
        duration_sec: float,
        mirror_shoulder: bool,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if target_name not in presets:
            return
        target = presets[target_name]
        send_feedback(f"moving to legal {label} warm-start preset for {target_name}")
        command = self._make_joint_position_update(
            target,
            mirror_shoulder=mirror_shoulder,
        )
        steps = max(1, int(duration_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        for _ in range(steps):
            move_robot(joint_motion_update=command)
            self.sleep_for(dt)

    def _run_sc_prepose(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sc_prepose_enabled:
            return
        self._run_joint_prepose(
            target_name=self._sc_port_name(task),
            presets=SC_NEAR_PORT_JOINT_PRESETS,
            label="SC",
            duration_sec=self._sc_prepose_sec,
            mirror_shoulder=self._sc_prepose_mirror_shoulder,
            move_robot=move_robot,
            send_feedback=send_feedback,
        )

    def _run_sfp_prepose(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_prepose_enabled:
            return
        self._run_joint_prepose(
            target_name=self._sfp_port_name(task),
            presets=SFP_NEAR_PORT_JOINT_PRESETS,
            label="SFP",
            duration_sec=self._sfp_prepose_sec,
            mirror_shoulder=True,
            move_robot=move_robot,
            send_feedback=send_feedback,
        )

    def _run_sc_terminal_target(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._sc_terminal_target_enabled:
            return None
        targets = SC_TERMINAL_TARGETS.get(self._sc_port_name(task) or "")
        port_name = self._sc_port_name(task) or ""
        targets = _env_target_sequence(
            f"AIC_RSLRL_SC_TERMINAL_TARGETS_{port_name.upper()}",
            targets,
        )
        if not targets:
            return None
        observation = get_observation()
        if observation is None:
            self.get_logger().error("Observation unavailable before SC terminal target.")
            return None

        pose = observation.controller_state.tcp_pose
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if (
            self._sc_terminal_orientation_enabled
            and port_name in SC_TERMINAL_QUATS_XYZW
        ):
            quat = np.array(SC_TERMINAL_QUATS_XYZW[port_name], dtype=np.float64)
        dwell = max(0.02, self._sc_terminal_target_dwell_sec)
        send_feedback("running legal SC terminal target sequence")
        path = _interpolated_targets(
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            targets,
            self._sc_terminal_target_max_step,
        )
        self.get_logger().info(
            "Running legal SC terminal target sequence: "
            f"target_count={len(targets)}, command_count={len(path)}, "
            f"dwell={dwell:.2f}s, max_step={self._sc_terminal_target_max_step}, "
            f"fixed_orientation={self._sc_terminal_orientation_enabled!r}"
        )
        for index, target_array in enumerate(path, start=1):
            command = self._make_base_pose_update(target_array, quat)
            move_robot(motion_update=command)
            if index == 1 or index == len(path) or index % self._log_every_n == 0:
                self.get_logger().info(
                    "SC terminal target "
                    f"{index}/{len(path)}: "
                    f"position={np.array2string(target_array, precision=4)}"
            )
            self.sleep_for(dwell)
        return np.array(targets[-1], dtype=np.float64), quat

    def _run_sc_tcp_z_recovery(
        self,
        base_target: np.ndarray | None,
        quat: np.ndarray | None,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sc_tcp_z_recovery_enabled or base_target is None or quat is None:
            return
        observation = get_observation()
        if observation is None:
            return
        tcp_z = float(observation.controller_state.tcp_pose.position.z)
        if tcp_z <= self._sc_tcp_z_recovery_threshold:
            self.get_logger().info(
                "Skipping SC TCP-z recovery: "
                f"tcp_z={tcp_z:.4f}, threshold={self._sc_tcp_z_recovery_threshold:.4f}"
            )
            return
        dwell = max(0.02, self._sc_tcp_z_recovery_dwell_sec)
        send_feedback("running legal SC TCP-z recovery")
        self.get_logger().info(
            "Running SC TCP-z recovery: "
            f"tcp_z={tcp_z:.4f}, threshold={self._sc_tcp_z_recovery_threshold:.4f}, "
            f"offsets={self._sc_tcp_z_recovery_offsets!r}, dwell={dwell:.2f}s"
        )
        for index, offset in enumerate(self._sc_tcp_z_recovery_offsets, start=1):
            target = base_target + np.array(offset, dtype=np.float64)
            move_robot(motion_update=self._make_base_pose_update(target, quat))
            self.sleep_for(dwell)
            observation = get_observation()
            tcp_z = (
                float(observation.controller_state.tcp_pose.position.z)
                if observation is not None
                else float("inf")
            )
            self.get_logger().info(
                "SC TCP-z recovery "
                f"{index}/{len(self._sc_tcp_z_recovery_offsets)}: "
                f"target={np.array2string(target, precision=4)}, tcp_z={tcp_z:.4f}"
            )
            if tcp_z <= self._sc_tcp_z_recovery_threshold:
                return

    def _run_sc_guarded_insert(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sc_guarded_insert_enabled:
            return
        steps = max(1, int(self._sc_guarded_insert_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        send_feedback("running legal SC guarded insertion")
        self.get_logger().info(
            "Running legal SC guarded insertion: "
            f"steps={steps}, down_step={self._sc_guarded_insert_down_step}, "
            f"retract_step={self._sc_guarded_insert_retract_step}, "
            f"force_limit={self._sc_guarded_insert_force_limit}, "
            f"retract_on_force={self._sc_guarded_insert_retract_on_force!r}, "
            f"linear_stiffness={self._sc_guarded_insert_linear_stiffness}, "
            f"linear_damping={self._sc_guarded_insert_linear_damping}"
        )
        for step in range(steps):
            observation = get_observation()
            if observation is None:
                self.get_logger().error(
                    "Observation became unavailable during SC guarded insert."
                )
                return
            force_norm = self._force_norm(observation)
            force_limited = force_norm > self._sc_guarded_insert_force_limit
            z_step = self._sc_guarded_insert_down_step
            if force_limited and self._sc_guarded_insert_retract_on_force:
                z_step = self._sc_guarded_insert_retract_step
            command = self._make_tcp_delta_update(
                np.array([0.0, 0.0, z_step], dtype=np.float64),
                linear_stiffness=self._sc_guarded_insert_linear_stiffness,
                angular_stiffness=self._sc_guarded_insert_angular_stiffness,
                linear_damping=self._sc_guarded_insert_linear_damping,
                angular_damping=self._sc_guarded_insert_angular_damping,
            )
            move_robot(motion_update=command)
            if step % self._log_every_n == 0:
                self.get_logger().info(
                    "SC guarded insert "
                    f"step={step}/{steps}, force_norm={force_norm:.2f}, "
                    f"z_step={z_step:.5f}"
                )
            self.sleep_for(dt)
        if not self._sc_guarded_insert_joint_hold_enabled:
            return
        observation = get_observation()
        if observation is None:
            self.get_logger().error(
                "Observation unavailable before SC guarded insert joint hold."
            )
            return
        target = self._gazebo_arm_joint_positions(observation)
        hold_steps = max(
            1,
            int(
                self._sc_guarded_insert_joint_hold_sec
                / max(self._sc_guarded_insert_joint_hold_dt, 1.0e-6)
            ),
        )
        hold_dt = max(0.01, self._sc_guarded_insert_joint_hold_dt)
        command = self._make_joint_position_update(target, mirror_shoulder=False)
        self.get_logger().info(
            "Running legal SC guarded insert joint hold: "
            f"steps={hold_steps}, dt={hold_dt:.3f}s, "
            f"target={np.array2string(target, precision=4)}"
        )
        for _ in range(hold_steps):
            move_robot(joint_motion_update=command)
            self.sleep_for(hold_dt)

    def _run_sfp_terminal_target(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._sfp_terminal_target_enabled:
            return None
        mount_name = self._sfp_mount_name(task) or ""
        targets = SFP_TERMINAL_TARGETS.get(mount_name)
        targets = _env_target_sequence(
            f"AIC_RSLRL_SFP_TERMINAL_TARGETS_{mount_name.upper()}",
            targets,
        )
        if not targets:
            return None
        observation = get_observation()
        if observation is None:
            self.get_logger().error("Observation unavailable before SFP terminal target.")
            return None

        pose = observation.controller_state.tcp_pose
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if (
            self._sfp_terminal_orientation_enabled
            and mount_name in SFP_TERMINAL_QUATS_XYZW
        ):
            quat = np.array(SFP_TERMINAL_QUATS_XYZW[mount_name], dtype=np.float64)
        dwell = max(0.02, self._sfp_terminal_target_dwell_sec)
        send_feedback("running legal SFP terminal target sequence")
        path = _interpolated_targets(
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            targets,
            self._sfp_terminal_target_max_step,
        )
        self.get_logger().info(
            "Running legal SFP terminal target sequence: "
            f"target_count={len(targets)}, command_count={len(path)}, "
            f"dwell={dwell:.2f}s, max_step={self._sfp_terminal_target_max_step}, "
            f"feedforward_base_z={self._sfp_terminal_feedforward_base_z:.2f}N"
        )
        for index, target_array in enumerate(path, start=1):
            feedforward_force_tip = None
            if (
                index == len(path)
                and abs(self._sfp_terminal_feedforward_base_z) > 1.0e-9
            ):
                base_force = np.array(
                    [0.0, 0.0, self._sfp_terminal_feedforward_base_z],
                    dtype=np.float64,
                )
                feedforward_force_tip = _quat_inverse_apply_xyzw(quat, base_force)
            command = self._make_base_pose_update(
                target_array,
                quat,
                feedforward_force_tip=feedforward_force_tip,
                linear_stiffness=self._sfp_terminal_target_linear_stiffness,
                angular_stiffness=self._sfp_terminal_target_angular_stiffness,
                linear_damping=self._sfp_terminal_target_linear_damping,
                angular_damping=self._sfp_terminal_target_angular_damping,
            )
            move_robot(motion_update=command)
            if index == 1 or index == len(path) or index % self._log_every_n == 0:
                self.get_logger().info(
                    "SFP terminal target "
                    f"{index}/{len(path)}: "
                    f"position={np.array2string(target_array, precision=4)}"
                )
            self.sleep_for(dwell)
        return np.array(targets[-1], dtype=np.float64), quat

    def _run_sfp_tcp_z_recovery(
        self,
        base_target: np.ndarray | None,
        quat: np.ndarray | None,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_tcp_z_recovery_enabled or base_target is None or quat is None:
            return
        observation = get_observation()
        if observation is None:
            return
        tcp_z = float(observation.controller_state.tcp_pose.position.z)
        if tcp_z <= self._sfp_tcp_z_recovery_threshold:
            self.get_logger().info(
                "Skipping SFP TCP-z recovery: "
                f"tcp_z={tcp_z:.4f}, threshold={self._sfp_tcp_z_recovery_threshold:.4f}"
            )
            return
        dwell = max(0.02, self._sfp_tcp_z_recovery_dwell_sec)
        send_feedback("running legal SFP TCP-z recovery")
        self.get_logger().info(
            "Running SFP TCP-z recovery: "
            f"tcp_z={tcp_z:.4f}, threshold={self._sfp_tcp_z_recovery_threshold:.4f}, "
            f"offsets={self._sfp_tcp_z_recovery_offsets!r}, dwell={dwell:.2f}s"
        )
        for index, offset in enumerate(self._sfp_tcp_z_recovery_offsets, start=1):
            target = base_target + np.array(offset, dtype=np.float64)
            move_robot(motion_update=self._make_base_pose_update(target, quat))
            self.sleep_for(dwell)
            observation = get_observation()
            tcp_z = (
                float(observation.controller_state.tcp_pose.position.z)
                if observation is not None
                else float("inf")
            )
            self.get_logger().info(
                "SFP TCP-z recovery "
                f"{index}/{len(self._sfp_tcp_z_recovery_offsets)}: "
                f"target={np.array2string(target, precision=4)}, tcp_z={tcp_z:.4f}"
            )
            if tcp_z <= self._sfp_tcp_z_recovery_threshold:
                return

    def _run_sfp_final_settle(
        self,
        observation,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if self._sfp_final_settle_sec <= 0.0:
            return
        steps = max(1, int(self._sfp_final_settle_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        action = np.array([0.0, 0.0, self._sfp_final_settle_step, 0.0, 0.0, 0.0])
        send_feedback("running optional SFP TCP-frame final settle")
        for _ in range(steps):
            command = self._make_position_update(
                observation=observation,
                action=action,
                position_scale=1.0,
                rotation_scale=1.0,
                frame_id="gripper/tcp",
            )
            move_robot(motion_update=command)
            self.sleep_for(dt)

    def _run_sfp_base_insert(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if self._sfp_base_insert_sec <= 0.0:
            return
        steps = max(1, int(self._sfp_base_insert_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        action = np.array(
            [0.0, 0.0, self._sfp_base_insert_step, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        send_feedback("running optional SFP base-frame final insert")
        self.get_logger().info(
            "Running optional SFP base-frame final insert: "
            f"steps={steps}, step={self._sfp_base_insert_step}"
        )
        for _ in range(steps):
            observation = get_observation()
            if observation is None:
                self.get_logger().error(
                    "Observation became unavailable during SFP base insert."
                )
                return
            command = self._make_position_update(
                observation=observation,
                action=action,
                position_scale=1.0,
                rotation_scale=0.0,
                frame_id="base_link",
            )
            move_robot(motion_update=command)
            self.sleep_for(dt)

    def _run_sfp_guarded_insert(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_guarded_insert_enabled:
            return
        mount_name = self._sfp_mount_name(task) or ""
        if (
            self._sfp_guarded_insert_targets is not None
            and mount_name not in self._sfp_guarded_insert_targets
        ):
            self.get_logger().info(
                f"Skipping legal SFP guarded insertion for {mount_name!r}"
            )
            return
        steps = max(1, int(self._sfp_guarded_insert_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        send_feedback("running legal SFP guarded insertion")
        self.get_logger().info(
            "Running legal SFP guarded insertion: "
            f"target={mount_name}, "
            f"steps={steps}, down_step={self._sfp_guarded_insert_down_step}, "
            f"lateral_step={self._sfp_guarded_insert_lateral_step}, "
            f"force_limit={self._sfp_guarded_insert_force_limit}, "
            f"linear_stiffness={self._sfp_guarded_insert_linear_stiffness}, "
            f"linear_damping={self._sfp_guarded_insert_linear_damping}"
        )
        golden_angle = 2.399963229728653
        for step in range(steps):
            observation = get_observation()
            if observation is None:
                self.get_logger().error(
                    "Observation became unavailable during SFP guarded insert."
                )
                return
            force_norm = self._force_norm(observation)
            angle = step * golden_angle
            lateral_step = self._sfp_guarded_insert_lateral_step
            if force_norm > self._sfp_guarded_insert_force_limit:
                delta = np.array(
                    [
                        -0.5 * lateral_step * np.cos(angle),
                        -0.5 * lateral_step * np.sin(angle),
                        self._sfp_guarded_insert_retract_step,
                    ],
                    dtype=np.float64,
                )
            else:
                delta = np.array(
                    [
                        lateral_step * np.cos(angle),
                        lateral_step * np.sin(angle),
                        self._sfp_guarded_insert_down_step,
                    ],
                    dtype=np.float64,
                )
            command = self._make_tcp_delta_update(
                delta,
                linear_stiffness=self._sfp_guarded_insert_linear_stiffness,
                angular_stiffness=self._sfp_guarded_insert_angular_stiffness,
                linear_damping=self._sfp_guarded_insert_linear_damping,
                angular_damping=self._sfp_guarded_insert_angular_damping,
            )
            move_robot(motion_update=command)
            if step % self._log_every_n == 0:
                self.get_logger().info(
                    "SFP guarded insert "
                    f"step={step}/{steps}, force_norm={force_norm:.2f}, "
                    f"delta={np.array2string(delta, precision=4)}"
                )
            self.sleep_for(dt)

    def _run_sfp_local_search(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_local_search_enabled:
            return
        mount_name = self._sfp_mount_name(task) or ""
        if (
            self._sfp_local_search_targets is not None
            and mount_name not in self._sfp_local_search_targets
        ):
            self.get_logger().info(
                f"Skipping legal SFP local search for {mount_name!r}"
            )
            return
        observation = get_observation()
        if observation is None:
            self.get_logger().error("Observation unavailable before SFP local search.")
            return

        pose = observation.controller_state.tcp_pose
        tcp_z = float(pose.position.z)
        if (
            self._sfp_local_search_tcp_z_threshold > 0.0
            and tcp_z <= self._sfp_local_search_tcp_z_threshold
        ):
            self.get_logger().info(
                "Skipping SFP local search: "
                f"tcp_z={tcp_z:.4f}, "
                f"threshold={self._sfp_local_search_tcp_z_threshold:.4f}"
            )
            return
        anchor = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        z_step = max(1.0e-4, self._sfp_local_search_z_step)
        z_down = max(0.0, self._sfp_local_search_z_down)
        z_offsets = -np.arange(0.0, z_down + 0.5 * z_step, z_step, dtype=np.float64)
        xy_offsets = self._sfp_local_search_offsets()
        dwell = max(0.02, self._sfp_local_search_dwell_sec)
        priority_dwell = max(0.02, self._sfp_local_search_priority_dwell_sec)
        priority_offsets = ()
        if self._sfp_local_search_priority_enabled:
            priority_offsets = _env_vector3_sequence(
                f"AIC_RSLRL_SFP_LOCAL_SEARCH_PRIORITY_OFFSETS_{mount_name.upper()}",
                SFP_LOCAL_SEARCH_PRIORITY_OFFSETS.get(mount_name, ()),
            )

        send_feedback("running legal SFP local search")
        self.get_logger().info(
            "Running legal SFP local search: "
            f"anchor={np.array2string(anchor, precision=4)}, "
            f"target={mount_name}, "
            f"radius={self._sfp_local_search_radius}, "
            f"step={self._sfp_local_search_step}, "
            f"tcp_z={tcp_z:.4f}, "
            f"tcp_z_threshold={self._sfp_local_search_tcp_z_threshold:.4f}, "
            f"z_down={z_down}, xy_points={len(xy_offsets)}, "
            f"z_levels={len(z_offsets)}, "
            f"priority_points={len(priority_offsets)}"
        )
        for index, offset in enumerate(priority_offsets, start=1):
            offset_array = np.array(offset, dtype=np.float64)
            command = self._make_base_pose_update(anchor + offset_array, quat)
            move_robot(motion_update=command)
            self.get_logger().info(
                "SFP local search priority "
                f"{index}/{len(priority_offsets)}: "
                f"dx={offset[0]:.4f}, dy={offset[1]:.4f}, dz={offset[2]:.4f}, "
                f"dwell={priority_dwell:.2f}s"
            )
            self.sleep_for(priority_dwell)

        if self._sfp_local_search_radius <= 0.0:
            self.sleep_for(1.0)
            return

        command_count = 0
        for z_offset in z_offsets:
            for dx, dy in xy_offsets:
                target = anchor + np.array([dx, dy, float(z_offset)], dtype=np.float64)
                command = self._make_base_pose_update(target, quat)
                move_robot(motion_update=command)
                command_count += 1
                if command_count % 25 == 0:
                    self.get_logger().info(
                        "SFP local search command "
                        f"{command_count}: dx={dx:.4f}, dy={dy:.4f}, dz={z_offset:.4f}"
                    )
                self.sleep_for(dwell)

        # Let the contact plugin observe a stable final pose if the search found
        # the socket.
        self.sleep_for(1.0)

    def _run_public_grid_search(
        self,
        task_kind: str,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if task_kind == "sc":
            enabled = self._sc_grid_search_enabled
            targets = self._sc_grid_search_targets
            x_radius = self._sc_grid_search_x_radius
            y_radius = self._sc_grid_search_y_radius
            step = self._sc_grid_search_step
            z_offsets = self._sc_grid_search_z_offsets
            dwell = self._sc_grid_search_dwell_sec
            probe_steps = self._sc_grid_search_probe_steps
            probe_step = self._sc_grid_search_probe_step
            probe_stiffness = self._sc_guarded_insert_linear_stiffness
            probe_damping = self._sc_guarded_insert_linear_damping
            angular_stiffness = self._sc_guarded_insert_angular_stiffness
            angular_damping = self._sc_guarded_insert_angular_damping
        elif task_kind == "sfp":
            enabled = self._sfp_grid_search_enabled
            targets = self._sfp_grid_search_targets
            x_radius = self._sfp_grid_search_x_radius
            y_radius = self._sfp_grid_search_y_radius
            step = self._sfp_grid_search_step
            z_offsets = self._sfp_grid_search_z_offsets
            dwell = self._sfp_grid_search_dwell_sec
            probe_steps = self._sfp_grid_search_probe_steps
            probe_step = self._sfp_grid_search_probe_step
            probe_stiffness = self._sfp_guarded_insert_linear_stiffness
            probe_damping = self._sfp_guarded_insert_linear_damping
            angular_stiffness = self._sfp_guarded_insert_angular_stiffness
            angular_damping = self._sfp_guarded_insert_angular_damping
        else:
            return

        if not enabled:
            return
        if targets is not None and target_key not in targets:
            self.get_logger().info(
                "Skipping public grid search: "
                f"task_kind={task_kind}, target_key={target_key!r}, "
                f"targets={sorted(targets)!r}"
            )
            return

        observation = get_observation()
        if observation is None:
            self.get_logger().error("Observation unavailable before public grid search.")
            return
        pose = observation.controller_state.tcp_pose
        anchor = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        xy_offsets = _grid_offsets(x_radius, y_radius, step)
        z_values = [float(offset[2]) for offset in z_offsets] or [0.0]
        dwell = max(0.01, dwell)
        probe_steps = max(0, probe_steps)
        send_feedback(f"running legal {task_kind.upper()} public grid search")
        self.get_logger().info(
            "Running legal public grid search: "
            f"task_kind={task_kind}, target_key={target_key}, "
            f"anchor={np.array2string(anchor, precision=4)}, "
            f"x_radius={x_radius:.4f}, y_radius={y_radius:.4f}, "
            f"step={step:.4f}, xy_points={len(xy_offsets)}, "
            f"z_offsets={z_values!r}, dwell={dwell:.3f}s, "
            f"probe_steps={probe_steps}, probe_step={probe_step:.5f}"
        )

        command_count = 0
        for dx, dy in xy_offsets:
            for dz in z_values:
                target = anchor + np.array([dx, dy, dz], dtype=np.float64)
                move_robot(motion_update=self._make_base_pose_update(target, quat))
                command_count += 1
                if command_count == 1 or command_count % self._log_every_n == 0:
                    observation = get_observation()
                    force_norm = (
                        self._force_norm(observation)
                        if observation is not None
                        else float("nan")
                    )
                    self.get_logger().info(
                        "Public grid search command "
                        f"{command_count}: dx={dx:.4f}, dy={dy:.4f}, "
                        f"dz={dz:.4f}, force_norm={force_norm:.2f}"
                    )
                self.sleep_for(dwell)

            for _ in range(probe_steps):
                command = self._make_tcp_delta_update(
                    np.array([0.0, 0.0, probe_step], dtype=np.float64),
                    linear_stiffness=probe_stiffness,
                    angular_stiffness=angular_stiffness,
                    linear_damping=probe_damping,
                    angular_damping=angular_damping,
                )
                move_robot(motion_update=command)
                self.sleep_for(dwell)

    def _public_scripted_target_key(self, task: Task, task_kind: str) -> str | None:
        if task_kind == "sfp":
            return self._sfp_mount_name(task)
        if task_kind == "sc":
            return self._sc_port_name(task)
        return None

    def _load_public_scripted_traces(self) -> dict[str, list[list[float]]]:
        if self._public_scripted_traces is not None:
            return self._public_scripted_traces
        with self._public_scripted_trace_path.open("r", encoding="utf-8") as infile:
            traces = json.load(infile)
        if not isinstance(traces, dict):
            raise ValueError(
                f"{self._public_scripted_trace_path} must contain a JSON object"
            )
        for target_key, samples in traces.items():
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"Trace {target_key!r} must be a non-empty list")
            for index, sample in enumerate(samples):
                if not isinstance(sample, list) or len(sample) != 7:
                    raise ValueError(
                        f"Trace {target_key!r}[{index}] must be "
                        "[x, y, z, qx, qy, qz, qw]"
                    )
        self._public_scripted_traces = traces
        return traces

    def _load_public_scripted_joint_traces(self) -> dict[str, list[list[float]]]:
        if self._public_scripted_joint_traces is not None:
            return self._public_scripted_joint_traces
        with self._public_scripted_joint_trace_path.open("r", encoding="utf-8") as infile:
            traces = json.load(infile)
        if not isinstance(traces, dict):
            raise ValueError(
                f"{self._public_scripted_joint_trace_path} must contain a JSON object"
            )
        for target_key, samples in traces.items():
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"Joint trace {target_key!r} must be non-empty")
            for index, sample in enumerate(samples):
                if not isinstance(sample, list) or len(sample) != len(ARM_JOINT_NAMES):
                    raise ValueError(
                        f"Joint trace {target_key!r}[{index}] must contain "
                        f"{len(ARM_JOINT_NAMES)} arm joint positions"
                    )
        self._public_scripted_joint_traces = traces
        return traces

    def _run_public_scripted_post_trace_refinements(
        self,
        task: Task,
        task_kind: str,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if task_kind == "sfp":
            self._run_sfp_guarded_insert(
                task, get_observation, move_robot, send_feedback
            )
            terminal = self._run_sfp_terminal_target(
                task, get_observation, move_robot, send_feedback
            )
            if terminal is not None:
                self._run_sfp_tcp_z_recovery(
                    terminal[0], terminal[1], get_observation, move_robot, send_feedback
                )
            self._run_public_grid_search(
                task_kind, target_key, get_observation, move_robot, send_feedback
            )
            self._run_sfp_terminal_fallback_trace_suffix(
                target_key, get_observation, move_robot, send_feedback
            )
            self._run_sfp_terminal_fallback_joint_finish(
                target_key, get_observation, move_robot, send_feedback
            )
            observation = get_observation()
            if observation is not None:
                self._run_sfp_final_settle(observation, move_robot, send_feedback)
            self._run_sfp_base_insert(get_observation, move_robot, send_feedback)
            self._run_sfp_local_search(task, get_observation, move_robot, send_feedback)
        elif task_kind == "sc":
            terminal = self._run_sc_terminal_target(
                task, get_observation, move_robot, send_feedback
            )
            if terminal is not None:
                self._run_sc_tcp_z_recovery(
                    terminal[0], terminal[1], get_observation, move_robot, send_feedback
                )
            self._run_public_grid_search(
                task_kind, target_key, get_observation, move_robot, send_feedback
            )

    def _sfp_terminal_fallback_needed(
        self,
        target_key: str,
        get_observation: GetObservationCallback,
        label: str,
    ) -> bool:
        observation = get_observation()
        if observation is None:
            self.get_logger().error(
                f"Observation unavailable before SFP terminal fallback {label}."
            )
            return False
        tcp_z = float(observation.controller_state.tcp_pose.position.z)
        threshold = self._sfp_terminal_fallback_tcp_z_threshold
        if tcp_z <= threshold:
            self.get_logger().info(
                f"Skipping SFP terminal fallback {label}: "
                f"target_key={target_key}, tcp_z={tcp_z:.4f}, "
                f"threshold={threshold:.4f}"
            )
            return False
        self.get_logger().info(
            f"SFP terminal fallback {label} needed: "
            f"target_key={target_key}, tcp_z={tcp_z:.4f}, threshold={threshold:.4f}"
        )
        return True

    def _run_sfp_terminal_fallback_trace_suffix(
        self,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_terminal_fallback_trace_suffix_enabled:
            return
        if (
            self._sfp_terminal_fallback_trace_targets is not None
            and target_key not in self._sfp_terminal_fallback_trace_targets
        ):
            self.get_logger().info(
                "Skipping SFP terminal fallback trace suffix: "
                f"target_key={target_key!r} not in "
                f"{sorted(self._sfp_terminal_fallback_trace_targets)!r}"
            )
            return
        if not self._sfp_terminal_fallback_needed(
            target_key, get_observation, "trace suffix"
        ):
            return
        try:
            traces = self._load_public_scripted_traces()
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load public scripted traces for fallback: {exc}"
            )
            return
        trace = traces.get(target_key)
        if not trace:
            self.get_logger().error(f"No public scripted fallback trace for {target_key!r}")
            return
        start_index = min(
            max(0, self._sfp_terminal_fallback_trace_start_index),
            len(trace) - 1,
        )
        suffix = trace[start_index:]
        dt = max(0.01, self._sfp_terminal_fallback_trace_dt)
        send_feedback("running legal SFP terminal fallback trace suffix")
        self.get_logger().info(
            "Running SFP terminal fallback trace suffix: "
            f"target_key={target_key}, start_index={start_index}, "
            f"commands={len(suffix)}, dt={dt:.3f}s"
        )
        for index, sample in enumerate(suffix, start=1):
            target = np.array(sample[:3], dtype=np.float64)
            quat = np.array(sample[3:], dtype=np.float64)
            move_robot(motion_update=self._make_base_pose_update(target, quat))
            if index == 1 or index == len(suffix) or index % self._log_every_n == 0:
                self.get_logger().info(
                    "SFP fallback trace suffix "
                    f"{index}/{len(suffix)}: "
                    f"position={np.array2string(target, precision=4)}"
                )
            self.sleep_for(dt)

    def _run_sfp_terminal_fallback_joint_finish(
        self,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> None:
        if not self._sfp_terminal_fallback_joints_enabled:
            return
        if target_key not in PUBLIC_SCRIPTED_FINAL_JOINTS:
            return
        if (
            self._sfp_terminal_fallback_joint_targets is not None
            and target_key not in self._sfp_terminal_fallback_joint_targets
        ):
            self.get_logger().info(
                "Skipping SFP terminal fallback joint finish: "
                f"target_key={target_key!r} not in "
                f"{sorted(self._sfp_terminal_fallback_joint_targets)!r}"
            )
            return
        if not self._sfp_terminal_fallback_needed(
            target_key, get_observation, "joint finish"
        ):
            return

        steps_override = (
            self._sfp_terminal_fallback_joint_steps
            if self._sfp_terminal_fallback_joint_steps > 0
            else None
        )
        send_feedback("running legal SFP terminal fallback joint finish")
        self.get_logger().info(
            "Running SFP terminal fallback joint finish: "
            f"target_key={target_key}, steps_override={steps_override}"
        )
        self._run_public_scripted_joint_finish(
            target_key,
            move_robot,
            send_feedback,
            force=True,
            steps_override=steps_override,
        )

    def _run_public_scripted_joint_trace_replay(
        self,
        task: Task,
        task_kind: str,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        try:
            traces = self._load_public_scripted_joint_traces()
        except Exception as exc:
            self.get_logger().error(f"Failed to load public scripted joint traces: {exc}")
            return False
        trace = traces.get(target_key)
        if trace is None:
            self.get_logger().error(
                "No public scripted joint trace for "
                f"{target_key!r} in {self._public_scripted_joint_trace_path}"
            )
            return False

        dt = max(0.01, self._public_scripted_joint_trace_dt)
        send_feedback("running public-sample scripted joint trace replay")
        self.get_logger().info(
            "Running public-sample scripted joint trace replay: "
            f"target_key={target_key}, samples={len(trace)}, dt={dt:.3f}s, "
            f"path={self._public_scripted_joint_trace_path}"
        )
        for index, sample in enumerate(trace, start=1):
            target = np.array(sample, dtype=np.float64)
            move_robot(
                joint_motion_update=self._make_joint_position_update(
                    target, mirror_shoulder=False
                )
            )
            if index == 1 or index == len(trace) or index % self._log_every_n == 0:
                self.get_logger().info(
                    "Public scripted joint trace "
                    f"{index}/{len(trace)}: "
                    f"target={np.array2string(target, precision=4)}"
                )
            self.sleep_for(dt)

        self._run_public_scripted_post_trace_refinements(
            task, task_kind, target_key, get_observation, move_robot, send_feedback
        )
        if self._public_scripted_dwell_sec > 0.0:
            self.sleep_for(self._public_scripted_dwell_sec)
        return True

    def _run_public_scripted_trace_replay(
        self,
        task: Task,
        task_kind: str,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        try:
            traces = self._load_public_scripted_traces()
        except Exception as exc:
            self.get_logger().error(f"Failed to load public scripted traces: {exc}")
            return False
        trace = traces.get(target_key)
        if trace is None:
            self.get_logger().error(
                "No public scripted trace for "
                f"{target_key!r} in {self._public_scripted_trace_path}"
            )
            return False

        dt = max(0.01, self._public_scripted_trace_dt)
        default_delta = _env_vector3(
            "AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA", (0.0, 0.0, 0.0)
        )
        target_delta = _env_vector3(
            f"AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA_{target_key.upper()}",
            tuple(float(item) for item in default_delta),
        )
        prepend_steps = max(
            0,
            _env_int(
                f"AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PREPEND_STEPS_{target_key.upper()}",
                self._public_scripted_trace_prepend_steps,
            ),
        )
        send_feedback("running public-sample scripted trace replay")
        self.get_logger().info(
            "Running public-sample scripted trace replay: "
            f"target_key={target_key}, commands={len(trace)}, dt={dt:.3f}s, "
            f"trace_delta={np.array2string(target_delta, precision=4)}, "
            f"prepend_steps={prepend_steps}, "
            f"path={self._public_scripted_trace_path}"
        )
        if prepend_steps > 0:
            observation = get_observation()
            if observation is None:
                self.get_logger().error(
                    "Observation unavailable before public scripted trace prepend."
                )
                return False
            pose = observation.controller_state.tcp_pose
            start_position = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float64,
            )
            start_quat = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64,
            )
            first_sample = trace[0]
            first_position = np.array(first_sample[:3], dtype=np.float64) + target_delta
            first_quat = np.array(first_sample[3:], dtype=np.float64)
            for step in range(prepend_steps):
                alpha = (step + 1) / prepend_steps
                target = (1.0 - alpha) * start_position + alpha * first_position
                quat = _slerp_quat_xyzw(start_quat, first_quat, alpha)
                move_robot(motion_update=self._make_base_pose_update(target, quat))
                if (
                    step == 0
                    or step + 1 == prepend_steps
                    or step % self._log_every_n == 0
                ):
                    self.get_logger().info(
                        "Public scripted trace prepend "
                        f"{step + 1}/{prepend_steps}: "
                        f"position={np.array2string(target, precision=4)}"
                    )
                self.sleep_for(dt)

        for index, sample in enumerate(trace, start=1):
            target = np.array(sample[:3], dtype=np.float64) + target_delta
            quat = np.array(sample[3:], dtype=np.float64)
            move_robot(motion_update=self._make_base_pose_update(target, quat))
            if index == 1 or index == len(trace) or index % self._log_every_n == 0:
                self.get_logger().info(
                    "Public scripted trace command "
                    f"{index}/{len(trace)}: "
                    f"position={np.array2string(target, precision=4)}"
                )
            self.sleep_for(dt)

        if not self._run_public_scripted_joint_finish(
            target_key, move_robot, send_feedback
        ):
            return False

        self._run_public_scripted_final_pose_hold(
            target_key,
            get_observation,
            move_robot,
            send_feedback,
            delta_env_prefix="TRACE",
        )

        self._run_public_scripted_post_trace_refinements(
            task, task_kind, target_key, get_observation, move_robot, send_feedback
        )

        if self._public_scripted_dwell_sec > 0.0:
            self.sleep_for(self._public_scripted_dwell_sec)
        return True

    def _run_public_scripted_joint_finish(
        self,
        target_key: str,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        *,
        force: bool = False,
        steps_override: int | None = None,
    ) -> bool:
        if not self._public_scripted_final_joints_enabled and not force:
            return True
        if (
            not force
            and
            self._public_scripted_final_joint_targets is not None
            and target_key not in self._public_scripted_final_joint_targets
        ):
            self.get_logger().info(
                f"Skipping public scripted joint finish for {target_key!r}"
            )
            return True

        final_joint_raw = PUBLIC_SCRIPTED_FINAL_JOINTS.get(target_key)
        if final_joint_raw is None:
            self.get_logger().error(
                f"No public scripted final joint target for {target_key!r}"
            )
            return False
        final_joint_target = np.array(final_joint_raw, dtype=np.float64)
        final_joint_steps = max(
            1,
            steps_override
            if steps_override is not None
            else self._public_scripted_final_joint_steps,
        )
        final_joint_dt = max(0.01, self._public_scripted_final_joint_dt)
        joint_command = self._make_joint_position_update(
            final_joint_target,
            mirror_shoulder=False,
        )
        send_feedback("running public-geometry scripted joint finish")
        self.get_logger().info(
            "Running public-geometry scripted joint finish: "
            f"target_key={target_key}, steps={final_joint_steps}, "
            f"dt={final_joint_dt:.3f}s, "
            f"force={force!r}, "
            f"target={np.array2string(final_joint_target, precision=4)}"
        )
        for step in range(final_joint_steps):
            move_robot(joint_motion_update=joint_command)
            if (
                step == 0
                or step + 1 == final_joint_steps
                or step % self._log_every_n == 0
            ):
                self.get_logger().info(
                    "Public scripted joint finish "
                    f"{step + 1}/{final_joint_steps}"
                )
            self.sleep_for(final_joint_dt)
        return True

    def _run_public_scripted_final_pose_hold(
        self,
        target_key: str,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        delta_env_prefix: str,
    ) -> None:
        if not self._public_scripted_final_pose_hold_enabled:
            return
        final_pose = PUBLIC_SCRIPTED_FINAL_POSES.get(target_key)
        if final_pose is None:
            return
        position_raw, quat_raw = final_pose
        default_delta = _env_vector3(
            f"AIC_RSLRL_PUBLIC_SCRIPTED_{delta_env_prefix}_DELTA",
            (0.0, 0.0, 0.0),
        )
        target_delta = _env_vector3(
            f"AIC_RSLRL_PUBLIC_SCRIPTED_{delta_env_prefix}_DELTA_{target_key.upper()}",
            tuple(float(item) for item in default_delta),
        )
        position = np.array(position_raw, dtype=np.float64) + target_delta
        quat = np.array(quat_raw, dtype=np.float64)
        steps = max(1, self._public_scripted_final_pose_hold_steps)
        min_steps = max(0, min(self._public_scripted_final_pose_hold_min_steps, steps))
        dt = max(0.01, self._public_scripted_final_pose_hold_dt)
        z_threshold = self._public_scripted_final_pose_hold_z_threshold

        send_feedback("holding public final pose with official TCP feedback")
        self.get_logger().info(
            "Running public final pose hold: "
            f"target_key={target_key}, steps={steps}, min_steps={min_steps}, "
            f"dt={dt:.3f}s, z_threshold={z_threshold:.4f}, "
            f"delta_env_prefix={delta_env_prefix!r}, "
            f"position={np.array2string(position, precision=4)}"
        )
        for step in range(steps):
            move_robot(motion_update=self._make_base_pose_update(position, quat))
            observation = get_observation()
            tcp_z = None
            if observation is not None:
                tcp_z = float(observation.controller_state.tcp_pose.position.z)
            if (
                tcp_z is not None
                and z_threshold > 0.0
                and step + 1 >= min_steps
                and tcp_z <= z_threshold
            ):
                self.get_logger().info(
                    "Public final pose hold reached TCP z threshold: "
                    f"step={step + 1}/{steps}, tcp_z={tcp_z:.4f}"
                )
                self.sleep_for(dt)
                return
            if step == 0 or step + 1 == steps or step % self._log_every_n == 0:
                suffix = "" if tcp_z is None else f", tcp_z={tcp_z:.4f}"
                self.get_logger().info(
                    f"Public final pose hold {step + 1}/{steps}{suffix}"
                )
            self.sleep_for(dt)

    def _run_public_scripted_insert(
        self,
        task: Task,
        task_kind: str,
        observation,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        get_observation: GetObservationCallback | None = None,
    ) -> bool:
        target_key = self._public_scripted_target_key(task, task_kind)
        if not target_key or target_key not in PUBLIC_SCRIPTED_FINAL_POSES:
            self.get_logger().error(
                "No public scripted final pose for "
                f"task_kind={task_kind!r}, target_key={target_key!r}"
            )
            return False

        trace_target_enabled = (
            self._public_scripted_trace_targets is None
            or target_key in self._public_scripted_trace_targets
        )
        if self._public_scripted_trace_replay_enabled and trace_target_enabled:
            if get_observation is None:
                self.get_logger().error(
                    "Observation callback unavailable for public scripted trace replay."
                )
                return False
            if self._public_scripted_joint_trace_replay_enabled:
                return self._run_public_scripted_joint_trace_replay(
                    task,
                    task_kind,
                    target_key,
                    get_observation,
                    move_robot,
                    send_feedback,
                )
            return self._run_public_scripted_trace_replay(
                task, task_kind, target_key, get_observation, move_robot, send_feedback
            )
        if self._public_scripted_trace_replay_enabled and not trace_target_enabled:
            self.get_logger().info(
                f"Skipping public scripted trace replay for target_key={target_key!r}"
            )

        final_position_raw, final_quat_raw = PUBLIC_SCRIPTED_FINAL_POSES[target_key]
        default_delta = _env_vector3(
            "AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_DELTA", (0.0, 0.0, 0.0)
        )
        target_delta = _env_vector3(
            f"AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_DELTA_{target_key.upper()}",
            tuple(float(item) for item in default_delta),
        )
        final_position = np.array(final_position_raw, dtype=np.float64) + target_delta
        final_quat = np.array(final_quat_raw, dtype=np.float64)
        start_pose = observation.controller_state.tcp_pose
        start_position = np.array(
            [start_pose.position.x, start_pose.position.y, start_pose.position.z],
            dtype=np.float64,
        )
        start_quat = np.array(
            [
                start_pose.orientation.x,
                start_pose.orientation.y,
                start_pose.orientation.z,
                start_pose.orientation.w,
            ],
            dtype=np.float64,
        )
        approach_position = final_position + np.array(
            [0.0, 0.0, self._public_scripted_above_z], dtype=np.float64
        )
        approach_steps = max(1, self._public_scripted_approach_steps)
        descent_steps = max(1, self._public_scripted_descent_steps)
        dt = max(0.01, self._public_scripted_dt)

        send_feedback("running public-geometry scripted insert")
        self.get_logger().info(
            "Running public-geometry scripted insert: "
            f"task_kind={task_kind}, target_key={target_key}, "
            f"approach_steps={approach_steps}, descent_steps={descent_steps}, "
            f"dt={dt:.3f}s, "
            f"target_delta={np.array2string(target_delta, precision=4)}, "
            f"final_position={np.array2string(final_position, precision=4)}"
        )

        for step in range(approach_steps):
            alpha = (step + 1) / approach_steps
            position = (1.0 - alpha) * start_position + alpha * approach_position
            quat = _slerp_quat_xyzw(start_quat, final_quat, alpha)
            move_robot(motion_update=self._make_base_pose_update(position, quat))
            if step == 0 or step + 1 == approach_steps or step % self._log_every_n == 0:
                self.get_logger().info(
                    "Public scripted approach "
                    f"{step + 1}/{approach_steps}: "
                    f"position={np.array2string(position, precision=4)}"
                )
            self.sleep_for(dt)

        for step in range(descent_steps):
            alpha = (step + 1) / descent_steps
            position = (1.0 - alpha) * approach_position + alpha * final_position
            move_robot(motion_update=self._make_base_pose_update(position, final_quat))
            if step == 0 or step + 1 == descent_steps or step % self._log_every_n == 0:
                self.get_logger().info(
                    "Public scripted descent "
                    f"{step + 1}/{descent_steps}: "
                    f"position={np.array2string(position, precision=4)}"
                )
            self.sleep_for(dt)

        if not self._run_public_scripted_joint_finish(
            target_key, move_robot, send_feedback
        ):
            return False

        if get_observation is not None:
            self._run_public_scripted_final_pose_hold(
                target_key,
                get_observation,
                move_robot,
                send_feedback,
                delta_env_prefix="FINAL",
            )

        if task_kind == "sc" and get_observation is not None:
            terminal = self._run_sc_terminal_target(
                task, get_observation, move_robot, send_feedback
            )
            if terminal is not None:
                self._run_sc_tcp_z_recovery(
                    terminal[0], terminal[1], get_observation, move_robot, send_feedback
                )
            self._run_public_grid_search(
                task_kind, target_key, get_observation, move_robot, send_feedback
            )
            self._run_sc_guarded_insert(get_observation, move_robot, send_feedback)

        if self._public_scripted_dwell_sec > 0.0:
            self.sleep_for(self._public_scripted_dwell_sec)
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Run a task-specific exported actor using only official observations."""
        observation = get_observation()
        if observation is None:
            reason = "No Observation message available when insert_cable() started."
            self.get_logger().error(reason)
            send_feedback(reason)
            return False

        task_kind = self._task_kind_from_task(task)
        self.get_logger().info(
            "Task metadata: "
            f"task_kind={task_kind!r}, "
            f"plug_type={task.plug_type!r}, port_type={task.port_type!r}, "
            f"plug_name={task.plug_name!r}, port_name={task.port_name!r}, "
            f"target_module_name={task.target_module_name!r}, "
            f"time_limit={task.time_limit}"
        )
        self.get_logger().info(
            "Observation snapshot received: "
            f"joint_names={list(observation.joint_states.name)}, "
            f"left_image={observation.left_image.width}x{observation.left_image.height}, "
            f"center_image={observation.center_image.width}x{observation.center_image.height}, "
            f"right_image={observation.right_image.width}x{observation.right_image.height}"
        )

        if task_kind not in {"sc", "sfp"}:
            reason = (
                "No functional exported actor adapter is implemented for task kind "
                f"{task_kind!r}."
            )
            self.get_logger().error(reason)
            send_feedback(reason)
            return False
        if self._public_scripted_insert_enabled:
            self._start_trace(task, task_kind)
            ok = self._run_public_scripted_insert(
                task,
                task_kind,
                observation,
                move_robot,
                send_feedback,
                get_observation,
            )
            self._finish_trace("public_scripted_completed" if ok else "public_scripted_failed")
            return ok

        actor = self._actor_for_task(task_kind)
        if actor is None:
            reason = (
                f"No {task_kind.upper()}/default TorchScript actor loaded. Export the "
                "Isaac actor with play.py and pass the matching policy artifact."
            )
            self.get_logger().error(reason)
            send_feedback(reason)
            return False
        if self._torch is None:
            reason = "Torch is unavailable after actor load."
            self.get_logger().error(reason)
            send_feedback(reason)
            return False

        self._start_trace(task, task_kind)
        self._last_action[:] = 0.0
        if task_kind == "sc":
            self._run_sc_prepose(task, move_robot, send_feedback)
        elif task_kind == "sfp":
            self._run_sfp_prepose(task, move_robot, send_feedback)
        if task_kind == "sc" and not self._sc_actor_enabled:
            send_feedback("SC actor disabled after legal prepose")
            self.get_logger().info("SC actor disabled after legal prepose.")
            self._finish_trace("sc_actor_disabled")
            return True

        actor_input_dim = None
        self._load_resnet18()
        start_time = time.monotonic()
        task_limit_sec = (
            float(task.time_limit)
            if int(task.time_limit) > 0
            else (
                self._sfp_max_control_sec if task_kind == "sfp" else self._sc_max_control_sec
            )
        )
        prepose_budget = 0.0
        if task_kind == "sc" and self._sc_prepose_enabled:
            prepose_budget = self._sc_prepose_sec
        elif task_kind == "sfp" and self._sfp_prepose_enabled:
            prepose_budget = self._sfp_prepose_sec
        max_control_sec = (
            self._sfp_max_control_sec if task_kind == "sfp" else self._sc_max_control_sec
        )
        position_scale = (
            self._sfp_position_scale if task_kind == "sfp" else self._sc_position_scale
        )
        rotation_scale = (
            self._sfp_rotation_scale if task_kind == "sfp" else self._sc_rotation_scale
        )
        command_frame = (
            self._sfp_command_frame if task_kind == "sfp" else self._sc_command_frame
        )
        control_sec = min(
            max_control_sec,
            max(1.0, task_limit_sec - prepose_budget - 1.0),
        )
        steps = max(1, int(control_sec * self._control_hz))
        dt = 1.0 / max(self._control_hz, 1e-6)
        send_feedback(f"running {task_kind.upper()} exported actor")

        for step in range(steps):
            observation = get_observation()
            if observation is None:
                self.get_logger().error(
                    f"Observation became unavailable during {task_kind.upper()} control loop."
                )
                self._finish_trace("missing_observation")
                return False
            try:
                actor_obs = self._actor_observation(task, observation, task_kind)
                obs_tensor = self._torch.from_numpy(actor_obs).unsqueeze(0)
                if actor_input_dim is None:
                    actor_input_dim = int(obs_tensor.shape[-1])
                    self.get_logger().info(
                        f"{task_kind.upper()} actor input dimension: {actor_input_dim}"
                    )
                    self.get_logger().info(
                        f"{task_kind.upper()} actor replay: steps={steps}, "
                        f"dt={dt:.4f}s, fixed_step={self._fixed_step_replay}"
                    )
                with self._torch.inference_mode():
                    action_tensor = actor(obs_tensor)
                if isinstance(action_tensor, (tuple, list)):
                    action_tensor = action_tensor[0]
                action = (
                    action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                if action.shape != (6,):
                    raise ValueError(f"unexpected actor action shape {action.shape}")
                action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                action = np.clip(action, -1.0, 1.0)
                motion_update = self._make_position_update(
                    observation=observation,
                    action=action,
                    position_scale=position_scale,
                    rotation_scale=rotation_scale,
                    frame_id=command_frame,
                )
                self._trace_actor_step(
                    task=task,
                    task_kind=task_kind,
                    step=step,
                    observation=observation,
                    actor_obs=actor_obs,
                    action=action,
                    motion_update=motion_update,
                )
                move_robot(motion_update=motion_update)
                self._last_action = action.astype(np.float32, copy=True)
                if step % self._log_every_n == 0:
                    self.get_logger().info(
                        f"{task_kind.upper()} actor step {step}/{steps}: "
                        f"action={np.array2string(action, precision=4)}"
                    )
                    send_feedback(f"{task_kind.upper()} actor step {step}/{steps}")
            except Exception as exc:
                self.get_logger().error(
                    f"{task_kind.upper()} actor control failed at step {step}: {exc}"
                )
                send_feedback(f"{task_kind.upper()} actor control failed: {exc}")
                self._finish_trace("actor_control_failed")
                return False
            if self._fixed_step_replay:
                self.sleep_for(dt)
            else:
                elapsed = time.monotonic() - start_time
                self.sleep_for(max(0.0, min(dt, control_sec - elapsed)))
                if elapsed >= control_sec:
                    break

        if task_kind == "sfp":
            observation = get_observation()
            self._run_sfp_guarded_insert(
                task, get_observation, move_robot, send_feedback
            )
            terminal = self._run_sfp_terminal_target(
                task, get_observation, move_robot, send_feedback
            )
            if terminal is not None:
                self._run_sfp_tcp_z_recovery(
                    terminal[0], terminal[1], get_observation, move_robot, send_feedback
                )
            target_key = self._public_scripted_target_key(task, task_kind)
            if target_key:
                self._run_public_grid_search(
                    task_kind, target_key, get_observation, move_robot, send_feedback
                )
            observation = get_observation()
            if observation is not None:
                self._run_sfp_final_settle(observation, move_robot, send_feedback)
            self._run_sfp_base_insert(get_observation, move_robot, send_feedback)
            self._run_sfp_local_search(task, get_observation, move_robot, send_feedback)
        elif task_kind == "sc":
            terminal = self._run_sc_terminal_target(
                task, get_observation, move_robot, send_feedback
            )
            if terminal is not None:
                self._run_sc_tcp_z_recovery(
                    terminal[0], terminal[1], get_observation, move_robot, send_feedback
                )

        self.get_logger().info(
            f"RslRlCheckpointPolicy {task_kind.upper()} control loop completed."
        )
        self._finish_trace("completed")
        return True
