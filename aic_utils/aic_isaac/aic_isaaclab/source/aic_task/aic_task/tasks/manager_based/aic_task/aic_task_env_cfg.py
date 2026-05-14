# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import TiledCameraCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.gamepad import Se3GamepadCfg

from . import mdp
from .mdp.events import (
    randomize_board_and_parts,
    randomize_dome_light,
    reset_robot_near_sc_port,
    reset_robot_near_sfp_port,
)

# Resolve asset directory relative to this file (portable across machines)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AIC_ASSET_DIR = os.path.join(_THIS_DIR, "Intrinsic_assets")
AIC_SCENE_DIR = AIC_ASSET_DIR
AIC_PARTS_DIR = os.path.join(AIC_ASSET_DIR, "assets")

EXTENSION_PATH = os.path.dirname(os.path.abspath(__file__))

##
# Scene definition
##


@configclass
class AICTaskSceneCfg(InteractiveSceneCfg):
    """Scene for aic task: UR5e robot, aic_scene, task_board."""

    # UR5e + gripper (fully defined here using local asset)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_ASSET_DIR, "aic_unified_robot_cable_sdf.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.18, -0.122, 0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": 0.1597,
                "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648,
                "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710,
                "wrist_3_joint": 1.4110,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                effort_limit_sim=87.0,
                stiffness=2000.0,
                damping=100.0,
            ),
            # "gripper": ImplicitActuatorCfg(
            #     joint_names_expr=[
            #         "gripper_left_finger_joint",
            #         "gripper_right_finger_joint",
            #     ],
            #     effort_limit_sim=20.0,
            #     stiffness=800.0,
            #     damping=40.0,
            # ),
        },
    )

    # cable = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/cable",
    #     spawn=None,
    #     init_state=ArticulationCfg.InitialStateCfg(),
    #     actuators={},
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         solver_position_iteration_count=64,
    #         solver_velocity_iteration_count=32,
    #     ),
    # )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    aic_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/aic_scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_SCENE_DIR, "scene", "aic.usd"),
            # usd_path=f"/home/nvidia/Downloads/aic_world.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    task_board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/task_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                AIC_PARTS_DIR, "Task Board Base", "task_board_rigid.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2837, 0.229, 0.0),
            # rot=(0.70686, -0.01851, 0.70686, 0.01851),
        ),
    )

    sc_port = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sc_port",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_PARTS_DIR, "SC Port", "sc_port.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2904, 0.1928, 0.005),
            rot=(0.73136, 0.0, 0.0, -0.682),
        ),
    )

    sc_port_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sc_port_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_PARTS_DIR, "SC Port", "sc_port.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2913, 0.1507, 0.005),
            rot=(0.73136, 0.0, 0.0, -0.682),
        ),
    )

    nic_card = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/nic_card",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_PARTS_DIR, "NIC Card", "nic_card.usd"),
            # scale=(0.009, 0.009, 0.009),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25135, 0.25229, 0.0743),
            rot=(0.0, 0.0, -0.7068252, 0.7073883),
        ),
    )

    # nic_card_mount = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/nic_card_mount",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(AIC_PARTS_DIR, "NIC Card Mount", "nic_card_mount_visual.usd"),
    #         scale=(0.00001, 0.00001, 0.00001),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(1.02, -0.010, 0.080),
    #         rot=(0.7073, 0.7073, 0.7073, -0.7073),
    #     ),
    # )

    def __post_init__(self):
        super().__post_init__()

        _cam_spawn = sim_utils.PinholeCameraCfg(
            focal_length=22.48,
            focus_distance=0.0,
            horizontal_aperture=20.955,
            vertical_aperture=18.627,
            clipping_range=(0.07, 20.0),
        )

        self.center_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/aic_unified_robot/center_camera_optical/center_camera",
            spawn=_cam_spawn,
            height=224,
            width=224,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )
        self.left_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/aic_unified_robot/left_camera_optical/left_camera",
            spawn=_cam_spawn,
            height=224,
            width=224,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )
        self.right_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/aic_unified_robot/right_camera_optical/right_camera",
            spawn=_cam_spawn,
            height=224,
            width=224,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.55, 0.75),
            pos_y=(-0.10, 0.02),
            pos_z=(0.01, 0.15),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # randomize_robot_pose = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pose_range": {
    #             "x": (-0.1, 0.1),   # random offset around init_state
    #             "y": (-0.1, 0.1),
    #             "z": (0.0, 0.0),
    #         },
    #         "velocity_range": {},
    #     },
    # )

    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        },
    )

    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            "parts": [
                {
                    "scene_name": "sc_port",
                    "offset": (0.0067, -0.0362, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "sc_port_2",
                    "offset": (0.0076, -0.0783, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "nic_card",
                    "offset": (-0.03235, 0.02329, 0.0743),
                    "pose_range": {"y": (0.0, 0.12)},
                    "snap_step": {"y": 0.04},
                },
            ],
        },
    )

    sample_active_sc_target = EventTerm(
        func=mdp.sample_active_sc_target,
        mode="reset",
    )

    reset_robot_near_sc_port = EventTerm(
        func=reset_robot_near_sc_port,
        mode="reset",
        params={
            "probability": 1.0,
            "blend": 0.95,
            "position_noise": 0.01,
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_sc_progress_buffers = EventTerm(
        func=mdp.reset_sc_progress_buffers,
        mode="reset",
    )
    reset_sc_scripted_action_prior_buffer = EventTerm(
        func=mdp.reset_sc_scripted_action_prior_buffer,
        mode="reset",
        params={
            "asset_name": "robot",
            "action_body_name": "gripper_tcp",
            "action_scale": 0.05,
            "action_clip": 1.0,
            "approach_depth": 0.0,
            "target_depth": 0.02,
            "max_translation_step": 0.025,
            "max_rotation_step": 0.10,
            "align_lateral_threshold": 0.05,
            "align_orientation_threshold": 0.50,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    sc_insertion_success = DoneTerm(
        func=mdp.sc_insertion_success,
        params={
            "lateral_threshold": 0.005,
            "orientation_threshold": 0.20,
            "depth_threshold": 0.012,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for eval-compatible actor and privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Eval-compatible observations for the deployed actor."""

        # Task metadata from eval/task selection, not privileged geometry.
        task_metadata = ObsTerm(func=mdp.active_sc_target_one_hot)

        # Robot state (joint space)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # End-effector pose in env frame (pos xyz + quat wxyz = 7 dims)
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        # Body forces
        body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "base_link",
                        "shoulder_link",
                        "upper_arm_link",
                        "forearm_link",
                        "wrist_1_link",
                        "wrist_2_link",
                        "wrist_3_link",
                    ],
                )
            },
        )

        center_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("center_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        left_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("left_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        right_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("right_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )

        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Training-only plug-to-port geometry observations for the critic."""

        plug_to_port_vec = ObsTerm(func=mdp.sc_plug_to_port_vec)
        lateral_error = ObsTerm(func=mdp.sc_lateral_error_obs)
        orientation_error = ObsTerm(func=mdp.sc_orientation_error_obs)
        insertion_depth = ObsTerm(func=mdp.sc_insertion_depth_obs)
        active_port_pose = ObsTerm(func=mdp.sc_active_port_pose)
        plug_tip_pose = ObsTerm(func=mdp.sc_plug_tip_pose_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- SC insertion shaping --
    sc_approach = RewTerm(
        func=mdp.sc_approach_reward,
        weight=3.0,
        params={"std": 1.00},
    )
    sc_distance_progress = RewTerm(
        func=mdp.sc_distance_progress_reward,
        weight=2.0,
        params={"scale": 0.02, "clip": 1.0},
    )
    sc_lateral_progress = RewTerm(
        func=mdp.sc_lateral_progress_reward,
        weight=1.0,
        params={"scale": 0.005, "clip": 1.0},
    )
    sc_orientation_progress = RewTerm(
        func=mdp.sc_orientation_progress_reward,
        weight=0.5,
        params={"scale": 0.10, "clip": 1.0},
    )
    sc_depth_progress = RewTerm(
        func=mdp.sc_depth_progress_reward,
        weight=1.0,
        params={"scale": 0.01, "clip": 1.0},
    )
    sc_coarse_lateral_alignment = RewTerm(
        func=mdp.sc_lateral_alignment_reward,
        weight=10.0,
        params={"std": 0.30},
    )
    sc_coarse_orientation_alignment = RewTerm(
        func=mdp.sc_orientation_alignment_reward,
        weight=2.0,
        params={"std": 2.00},
    )
    sc_lateral_alignment = RewTerm(
        func=mdp.sc_lateral_alignment_reward,
        weight=1.0,
        params={"std": 0.02},
    )
    sc_orientation_alignment = RewTerm(
        func=mdp.sc_orientation_alignment_reward,
        weight=0.5,
        params={"std": 0.35},
    )
    sc_insertion_depth = RewTerm(
        func=mdp.sc_insertion_depth_reward,
        weight=4.0,
        params={
            "depth_scale": 0.02,
            "max_depth": 0.03,
            "lateral_threshold": 0.01,
            "orientation_threshold": 0.35,
        },
    )
    sc_insertion_success = RewTerm(
        func=mdp.sc_insertion_success_bonus,
        weight=10.0,
        params={
            "lateral_threshold": 0.005,
            "orientation_threshold": 0.20,
            "depth_threshold": 0.012,
        },
    )
    sc_scripted_action_prior = RewTerm(
        func=mdp.sc_scripted_action_prior_reward,
        weight=5.0,
        params={
            "action_name": "arm_action",
            "asset_name": "robot",
            "action_body_name": "gripper_tcp",
            "action_scale": 0.05,
            "action_clip": 1.0,
            "approach_depth": 0.0,
            "target_depth": 0.02,
            "max_translation_step": 0.025,
            "max_rotation_step": 0.10,
            "align_lateral_threshold": 0.05,
            "align_orientation_threshold": 0.50,
            "std": 1.00,
        },
    )

    # -- Smoothness penalties --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Safety: penalize joints approaching their limits --
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class SfpEventCfg:
    """Reset events for the SFP insertion task variant."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        },
    )
    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            "parts": [
                {
                    "scene_name": "sc_port",
                    "offset": (0.0067, -0.0362, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "sc_port_2",
                    "offset": (0.0076, -0.0783, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "nic_card",
                    "offset": (-0.03235, 0.02329, 0.0743),
                    # Step 9 first randomized SFP stage: small continuous NIC
                    # y variation. Do not snap; a 0.04 m grid would collapse
                    # this +/-0.002 m curriculum back to the fixed pose.
                    "pose_range": {"y": (-0.002, 0.002)},
                    "snap_step": {"y": 0.0},
                },
            ],
        },
    )
    sample_active_sfp_target = EventTerm(
        func=mdp.sample_active_sfp_target,
        mode="reset",
    )
    reset_robot_near_sfp_port = EventTerm(
        func=reset_robot_near_sfp_port,
        mode="reset",
        params={
            "probability": 1.0,
            "blend": 1.0,
            "position_noise": 0.002,
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_sfp_progress_buffers = EventTerm(
        func=mdp.reset_sfp_progress_buffers,
        mode="reset",
    )


@configclass
class SfpGazeboTransferEventCfg(SfpEventCfg):
    """SFP reset events that match the official Gazebo SFP target structure."""

    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            "sample_sfp_mount": True,
            "parts": [
                {
                    "scene_name": "sc_port",
                    "offset": (0.0067, -0.0362, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "sc_port_2",
                    "offset": (0.0076, -0.0783, 0.005),
                    "pose_range": {"x": (0.0, 0.0)},
                },
                {
                    "scene_name": "nic_card",
                    "offset": (-0.03235, 0.02329, 0.0743),
                    # Official Gazebo SFP trials use sfp_port_0 on different
                    # NIC mounts separated by roughly 0.04 m in y. Train over
                    # that mount-scale shift instead of only the previous
                    # +/-0.002 m perturbation.
                    "sfp_mount_y_offsets": (0.0, -0.04),
                    "sfp_mount_y_jitter": (-0.005, 0.005),
                },
            ],
        },
    )
    sample_active_sfp_target = EventTerm(
        func=mdp.sample_active_sfp_target,
        mode="reset",
        params={"target_id": 0},
    )


@configclass
class SfpTerminationsCfg:
    """Termination terms for the SFP insertion task variant."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    sfp_insertion_success = DoneTerm(
        func=mdp.sfp_insertion_success,
        params={
            # Intermediate SFP curriculum gate. The first coarse gate
            # (0.020, 0.50, 0.005) is solved from the pre-corrected reset; this
            # stage asks PPO to keep lateral centering while inserting deeper.
            "lateral_threshold": 0.015,
            "orientation_threshold": 0.25,
            "depth_threshold": 0.015,
        },
    )
    # First SFP PPO curriculum: do not immediately terminate lateral misses.
    # The reward penalty below makes lateral escape costly across a short
    # timeout horizon; immediate reset made corridor exit an escape behavior.
    sfp_corridor_lateral_violation = None
    sfp_corridor_orientation_violation = None
    sfp_corridor_min_depth_violation = None
    sfp_corridor_max_depth_violation = None


@configclass
class SfpObservationsCfg:
    """Observation groups for the SFP insertion task variant."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Eval-compatible observations for the deployed SFP actor."""

        task_metadata = ObsTerm(func=mdp.active_sfp_target_one_hot)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "base_link",
                        "shoulder_link",
                        "upper_arm_link",
                        "forearm_link",
                        "wrist_1_link",
                        "wrist_2_link",
                        "wrist_3_link",
                    ],
                )
            },
        )
        center_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("center_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        left_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("left_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        right_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("right_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Training-only SFP plug-to-port geometry observations for the critic."""

        plug_to_port_vec = ObsTerm(func=mdp.sfp_plug_to_port_vec)
        lateral_error = ObsTerm(func=mdp.sfp_lateral_error_obs)
        orientation_error = ObsTerm(func=mdp.sfp_orientation_error_obs)
        insertion_depth = ObsTerm(func=mdp.sfp_insertion_depth_obs)
        active_port_pose = ObsTerm(func=mdp.sfp_active_port_pose)
        plug_tip_pose = ObsTerm(func=mdp.sfp_plug_tip_pose_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class SfpGazeboTransferObservationsCfg(SfpObservationsCfg):
    """SFP observations with official Gazebo port and module metadata."""

    @configclass
    class PolicyCfg(SfpObservationsCfg.PolicyCfg):
        """Eval-compatible SFP actor obs with target module one-hot."""

        task_metadata = ObsTerm(func=mdp.active_sfp_gazebo_task_one_hot)

    policy: PolicyCfg = PolicyCfg()


@configclass
class SfpRewardsCfg:
    """SFP insertion reward terms."""

    sfp_approach = RewTerm(
        func=mdp.sfp_approach_reward,
        weight=3.0,
        params={"std": 1.00, "active_until_depth": -0.002},
    )
    sfp_distance_progress = RewTerm(
        func=mdp.sfp_distance_progress_reward,
        weight=2.0,
        params={"scale": 0.02, "clip": 1.0, "active_until_depth": -0.002},
    )
    sfp_lateral_progress = RewTerm(
        func=mdp.sfp_lateral_progress_reward,
        weight=30.0,
        params={"scale": 0.005, "clip": 1.0},
    )
    sfp_lateral_error = RewTerm(
        func=mdp.sfp_lateral_error_penalty,
        weight=-60.0,
        params={"scale": 0.060, "clip": 1.0},
    )
    sfp_lateral_corridor = RewTerm(
        func=mdp.sfp_lateral_corridor_penalty,
        weight=-80.0,
        params={
            "soft_limit": 0.020,
            "hard_limit": 0.060,
            "clip": 1.0,
            "violation_cost": 1.0,
        },
    )
    sfp_depth_backout = RewTerm(
        func=mdp.sfp_depth_backout_penalty,
        weight=-80.0,
        params={
            "soft_min_depth": -0.010,
            "hard_min_depth": -0.080,
            "clip": 1.0,
            "violation_cost": 1.0,
        },
    )
    sfp_port_frame_lateral_action = RewTerm(
        func=mdp.sfp_port_frame_lateral_action_reward,
        weight=20.0,
        params={
            "action_name": "arm_action",
            "command_scale": 0.02,
            "realized_lateral_scale": 3.0e-5,
            "min_lateral_error": 0.002,
            "lateral_scale": 0.006,
            "lateral_threshold": 0.060,
            "orientation_threshold": 0.80,
            "min_depth": -0.080,
            "max_depth": 0.060,
        },
    )
    sfp_port_frame_depth_action = RewTerm(
        func=mdp.sfp_port_frame_depth_action_reward,
        weight=80.0,
        params={
            "action_name": "arm_action",
            "command_scale": 0.25,
            "realized_depth_scale": 3.0e-5,
            "min_depth": -0.080,
            "target_depth": 0.018,
            "lateral_threshold": 0.030,
            "orientation_threshold": 0.25,
        },
    )
    sfp_lateral_correction_action = RewTerm(
        func=mdp.sfp_lateral_correction_action_reward,
        weight=0.0,
        params={
            "action_name": "arm_action",
            "asset_name": "robot",
            "action_scale": 0.001,
            "command_scale": 0.0002,
            "min_lateral_error": 0.002,
            "lateral_scale": 0.006,
            "lateral_threshold": 0.060,
            "orientation_threshold": 0.80,
            "min_depth": -0.080,
            "max_depth": 0.060,
        },
    )
    sfp_orientation_progress = RewTerm(
        func=mdp.sfp_orientation_progress_reward,
        weight=0.5,
        params={"scale": 0.10, "clip": 1.0},
    )
    sfp_depth_progress = RewTerm(
        func=mdp.sfp_depth_progress_reward,
        weight=40.0,
        params={"scale": 0.01, "clip": 1.0},
    )
    sfp_coarse_lateral_alignment = RewTerm(
        func=mdp.sfp_lateral_alignment_reward,
        weight=10.0,
        params={"std": 0.30},
    )
    sfp_coarse_orientation_alignment = RewTerm(
        func=mdp.sfp_orientation_alignment_reward,
        weight=2.0,
        params={"std": 2.00},
    )
    sfp_lateral_alignment = RewTerm(
        func=mdp.sfp_lateral_alignment_reward,
        weight=4.0,
        params={"std": 0.015},
    )
    sfp_orientation_alignment = RewTerm(
        func=mdp.sfp_orientation_alignment_reward,
        weight=0.5,
        params={"std": 0.35},
    )
    sfp_insertion_depth = RewTerm(
        func=mdp.sfp_insertion_depth_reward,
        weight=160.0,
        params={
            "depth_scale": 0.006,
            "max_depth": 0.045,
            "min_depth": 0.0,
            "lateral_threshold": 0.030,
            "orientation_threshold": 0.25,
        },
    )
    sfp_insertion_action = RewTerm(
        func=mdp.sfp_insertion_action_reward,
        weight=60.0,
        params={
            "action_name": "arm_action",
            "asset_name": "robot",
            "action_scale": 0.003,
            "command_scale": 0.0006,
            "realized_depth_scale": 3.0e-5,
            "lateral_threshold": 0.015,
            "orientation_threshold": 0.30,
            "lateral_std": 0.008,
        },
    )
    sfp_insertion_success = RewTerm(
        func=mdp.sfp_insertion_success_bonus,
        weight=100.0,
        params={
            "lateral_threshold": 0.015,
            "orientation_threshold": 0.25,
            "depth_threshold": 0.015,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class AICTaskEnvCfg(ManagerBasedRLEnvCfg):
    """AIC task env: UR5e robot and custom scene."""

    # Scene settings
    scene: AICTaskSceneCfg = AICTaskSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # General settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 200.0
        self.sim.dt = 1.0 / 120.0
        # self.sim.gravity = (0.0, 0.0, 3)
        self.viewer.eye = (8.0, 0.0, 5.0)

        # # Arm action: joint position control
        # self.actions.arm_action = JointPositionActionCfg(
        #     asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        # )

        # Arm action: differential IK (for teleoperation)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="gripper_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="svd",
                ik_params={"k_val": 1.0, "min_singular_value": 1e-5},
            ),
            scale=0.05,
        )

        # Command generator: end-effector body and pitch.
        self.commands.ee_pose.body_name = "gripper_tcp"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Teleop device configuration
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.08,
                    rot_sensitivity=0.05,
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
            },
        )


@configclass
class AICTaskSfpEnvCfg(AICTaskEnvCfg):
    """AIC task variant for SFP insertion."""

    observations: SfpObservationsCfg = SfpObservationsCfg()
    rewards: SfpRewardsCfg = SfpRewardsCfg()
    terminations: SfpTerminationsCfg = SfpTerminationsCfg()
    events: SfpEventCfg = SfpEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # SFP insertion is a millimeter-scale task; use smaller relative-IK
        # deltas than SC so near-port PPO does not leave the insertion corridor.
        self.actions.arm_action.scale = 0.003
        # Final insertion should resolve quickly from the near-port curriculum.
        # Short episodes keep failed non-terminated attempts from drifting far.
        self.episode_length_s = 5.0


@configclass
class AICTaskSfpGazeboTransferEnvCfg(AICTaskSfpEnvCfg):
    """SFP variant for official Gazebo transfer training."""

    observations: SfpGazeboTransferObservationsCfg = SfpGazeboTransferObservationsCfg()
    events: SfpGazeboTransferEventCfg = SfpGazeboTransferEventCfg()
