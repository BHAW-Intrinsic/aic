# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SFP insertion teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Asymmetric actor-critic PPO config for the SFP insertion teacher."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "aic_sfp_insert"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    # Training-only PPO initialization: action-frame diagnostics showed raw
    # z-negative is the inward SFP direction. PPO can still update this normally.
    aic_actor_output_bias = (0.0, 0.0, -0.20, 0.0, 0.0, 0.0)
    aic_actor_output_zero_weights = True
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.005),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
