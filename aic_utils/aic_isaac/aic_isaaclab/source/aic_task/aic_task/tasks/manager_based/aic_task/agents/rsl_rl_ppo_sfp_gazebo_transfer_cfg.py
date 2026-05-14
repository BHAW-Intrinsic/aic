"""RSL-RL PPO config for the Gazebo-transfer SFP insertion task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_sfp_cfg import PPORunnerCfg as SfpPPORunnerCfg


@configclass
class PPORunnerCfg(SfpPPORunnerCfg):
    """PPO config for SFP port-0 insertion across Gazebo-style NIC shifts."""

    experiment_name = "aic_sfp_gazebo_transfer"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.25),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
