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

SC_PORT_ONE_HOT = {
    "sc_port": np.array([1.0, 0.0], dtype=np.float32),
    "sc_port_2": np.array([0.0, 1.0], dtype=np.float32),
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
        self._sc_actor_enabled = _env_bool("AIC_RSLRL_SC_ACTOR_ENABLED", True)
        self._fixed_step_replay = _env_bool("AIC_RSLRL_FIXED_STEP_REPLAY", False)
        self._zero_joint_obs = _env_bool("AIC_RSLRL_ZERO_JOINT_OBS", False)
        self._zero_body_forces = _env_bool("AIC_RSLRL_ZERO_BODY_FORCES", False)
        self._sfp_include_mount_metadata = _env_bool(
            "AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA", False
        )
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
            f"AIC_RSLRL_SC_ACTOR_ENABLED={self._sc_actor_enabled!r}, "
            f"AIC_RSLRL_FIXED_STEP_REPLAY={self._fixed_step_replay!r}, "
            f"AIC_RSLRL_ZERO_JOINT_OBS={self._zero_joint_obs!r}, "
            f"AIC_RSLRL_ZERO_BODY_FORCES={self._zero_body_forces!r}, "
            "AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA="
            f"{self._sfp_include_mount_metadata!r}, "
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
        body_forces[-6:] = 0.1 * wrist_wrench
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
        actor = self._actor_for_task(task_kind)
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
            if observation is not None:
                self._run_sfp_final_settle(observation, move_robot, send_feedback)
            self._run_sfp_base_insert(get_observation, move_robot, send_feedback)

        self.get_logger().info(
            f"RslRlCheckpointPolicy {task_kind.upper()} control loop completed."
        )
        self._finish_trace("completed")
        return True
