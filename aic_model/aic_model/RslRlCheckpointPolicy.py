"""ROS policy for replaying exported Isaac RSL-RL actors in Gazebo eval.

This class uses only the official ``aic_model`` policy API: task metadata,
``Observation`` messages, and ``move_robot`` callbacks. It does not subscribe to
ground-truth transforms, scoring topics, Gazebo internals, or hidden simulator
state.
"""

from __future__ import annotations

import os
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
ACTOR_OBSERVATION_DIM = 3149


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
        self._require_resnet18 = _env_bool("AIC_RSLRL_REQUIRE_RESNET18", False)
        self._log_every_n = max(1, _env_int("AIC_RSLRL_LOG_EVERY_N", 20))

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
            f"AIC_RSLRL_REQUIRE_RESNET18={self._require_resnet18!r}"
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
            for name in SFP_PORT_ONE_HOT:
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

    def _joint_vectors(self, observation) -> tuple[np.ndarray, np.ndarray]:
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
        if task_kind == "sfp":
            port_name = self._sfp_port_name(task)
            one_hot_by_port = SFP_PORT_ONE_HOT
        elif task_kind == "sc":
            port_name = self._sc_port_name(task)
            one_hot_by_port = SC_PORT_ONE_HOT
        else:
            raise ValueError(f"Unsupported task kind for actor observation: {task_kind!r}")
        if port_name is None:
            raise ValueError(
                f"Unable to infer {task_kind.upper()} target port from official task metadata: "
                f"port_name={task.port_name!r}, target_module_name={task.target_module_name!r}"
            )
        joint_pos_rel, joint_vel_rel = self._joint_vectors(observation)
        pieces = [
            one_hot_by_port[port_name],
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
        if actor_obs.shape != (ACTOR_OBSERVATION_DIM,):
            raise ValueError(
                f"Unexpected {task_kind.upper()} actor observation shape {actor_obs.shape}; "
                f"expected {(ACTOR_OBSERVATION_DIM,)}"
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

        self._last_action[:] = 0.0
        if task_kind == "sc":
            self._run_sc_prepose(task, move_robot, send_feedback)
        elif task_kind == "sfp":
            self._run_sfp_prepose(task, move_robot, send_feedback)
        if task_kind == "sc" and not self._sc_actor_enabled:
            send_feedback("SC actor disabled after legal prepose")
            self.get_logger().info("SC actor disabled after legal prepose.")
            return True

        actor_input_dim = None
        self._load_resnet18()
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
                        f"{task_kind.upper()} actor fixed-step replay: "
                        f"steps={steps}, dt={dt:.4f}s"
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
                return False
            self.sleep_for(dt)

        if task_kind == "sfp":
            observation = get_observation()
            if observation is not None:
                self._run_sfp_final_settle(observation, move_robot, send_feedback)
            self._run_sfp_base_insert(get_observation, move_robot, send_feedback)

        self.get_logger().info(
            f"RslRlCheckpointPolicy {task_kind.upper()} control loop completed."
        )
        return True
