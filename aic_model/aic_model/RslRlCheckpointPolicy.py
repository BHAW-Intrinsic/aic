"""ROS policy scaffold for replaying Isaac RSL-RL checkpoints in Gazebo eval.

This class is intentionally conservative. It wires the official ``aic_model``
policy API to checkpoint-related environment variables, but it does not pretend
that a raw Isaac Lab RSL-RL checkpoint can be used directly in Gazebo. The
missing piece is the observation/action adapter documented in
``docs/bahw_docs/eval_wrapper/README.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


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


class RslRlCheckpointPolicy(Policy):
    """Placeholder policy for official Gazebo eval of Isaac RSL-RL actors.

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

    The class loads no privileged state and does not use hidden Gazebo
    transforms. A future implementation must reconstruct the actor observation
    from the official ``Observation`` and ``Task`` messages only.
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

        self.get_logger().info(
            "RslRlCheckpointPolicy configured with "
            f"AIC_RSLRL_CHECKPOINT={self.checkpoint_path!r}, "
            f"AIC_RSLRL_SC_CHECKPOINT={self.sc_checkpoint_path!r}, "
            f"AIC_RSLRL_SFP_CHECKPOINT={self.sfp_checkpoint_path!r}, "
            f"AIC_RSLRL_POLICY_ARTIFACT={self.policy_artifact_path!r}, "
            f"AIC_RSLRL_SC_POLICY_ARTIFACT={self.sc_policy_artifact_path!r}, "
            f"AIC_RSLRL_SFP_POLICY_ARTIFACT={self.sfp_policy_artifact_path!r}, "
            f"AIC_RSLRL_TASK_KIND={self.task_kind!r}"
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

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Return ``False`` until the Gazebo adapter is implemented.

        The official eval process can still use this class to verify lifecycle,
        task routing, checkpoint path plumbing, and scoring-output collection.
        It should not be interpreted as a functional checkpoint evaluation.
        """
        del move_robot
        observation = get_observation()
        reason = (
            "RslRlCheckpointPolicy scaffold reached insert_cable(), but the "
            "Gazebo Observation -> Isaac actor observation adapter and the "
            "actor action -> MotionUpdate adapter are not implemented yet. "
            "This run verifies official eval orchestration only."
        )
        self.get_logger().error(reason)
        self.get_logger().info(
            "Task metadata: "
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
        send_feedback(reason)
        return False
