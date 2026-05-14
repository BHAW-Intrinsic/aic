"""RSL-RL PPO config for the Gazebo-transfer SFP insertion task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_sfp_cfg import PPORunnerCfg as SfpPPORunnerCfg


@configclass
class PPORunnerCfg(SfpPPORunnerCfg):
    """PPO config for SFP port-0 insertion across Gazebo-style NIC shifts."""

    experiment_name = "aic_sfp_gazebo_transfer"
