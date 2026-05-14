# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="AIC-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_task_env_cfg:AICTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_sc_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_sc_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="AIC-SFP-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_task_env_cfg:AICTaskSfpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_sfp_cfg:PPORunnerCfg",
        "rsl_rl_sfp_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_sfp_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="AIC-SFP-Gazebo-Transfer-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.aic_task_env_cfg:AICTaskSfpGazeboTransferEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_sfp_gazebo_transfer_cfg:PPORunnerCfg"
        ),
        "rsl_rl_sfp_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_sfp_gazebo_transfer_cfg:PPORunnerCfg"
        ),
    },
)
