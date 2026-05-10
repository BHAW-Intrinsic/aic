# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior-clone the SC actor from the scripted insertion controller."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Pretrain the SC RSL-RL actor with scripted actions."
)
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_sc_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
parser.add_argument(
    "--max_updates", type=int, default=1000, help="Number of BC gradient updates."
)
parser.add_argument(
    "--learning_rate", type=float, default=1.0e-3, help="BC optimizer learning rate."
)
parser.add_argument(
    "--max_grad_norm", type=float, default=1.0, help="Actor gradient clipping norm."
)
parser.add_argument(
    "--report_every", type=int, default=50, help="Print progress every N updates."
)
parser.add_argument(
    "--save_every",
    type=int,
    default=250,
    help="Save checkpoint every N updates; <=0 disables.",
)
parser.add_argument("--action_body_name", type=str, default="gripper_tcp")
parser.add_argument("--action_scale", type=float, default=0.05)
parser.add_argument("--action_clip", type=float, default=1.0)
parser.add_argument("--approach_depth", type=float, default=0.0)
parser.add_argument("--target_depth", type=float, default=0.02)
parser.add_argument("--max_translation_step", type=float, default=0.025)
parser.add_argument("--max_rotation_step", type=float, default=0.10)
parser.add_argument("--align_lateral_threshold", type=float, default=0.05)
parser.add_argument("--align_orientation_threshold", type=float, default=0.50)
parser.add_argument("--success_lateral_threshold", type=float, default=0.005)
parser.add_argument("--success_orientation_threshold", type=float, default=0.20)
parser.add_argument("--success_depth_threshold", type=float, default=0.012)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.envs import multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task import mdp


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Pretrain the actor with supervised scripted action labels."""
    if args_cli.task is None:
        raise ValueError("--task is required.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    if hasattr(env_cfg, "events") and hasattr(
        env_cfg.events, "randomize_board_and_parts"
    ):
        env_cfg.events.randomize_board_and_parts.params["sync_usd_xforms"] = False
    env_cfg.log_dir = log_dir

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_cfg_dict = cli_args.runner_cfg_to_dict(agent_cfg)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            f"Unsupported runner class for BC pretrain: {agent_cfg.class_name}"
        )
    runner = OnPolicyRunner(
        wrapped_env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    if not hasattr(runner.logger, "writer"):
        runner.logger.writer = None
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    actor = runner.alg.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.learning_rate)
    loss_fn = torch.nn.MSELoss()
    obs = wrapped_env.get_observations().to(agent_cfg.device)

    print("== AIC SC Scripted BC Pretrain ==")
    print(f"task: {args_cli.task}")
    print(f"agent: {args_cli.agent}")
    print(f"log_dir: {log_dir}")
    print(f"num_envs: {wrapped_env.num_envs}")
    print(f"max_updates: {args_cli.max_updates}")
    print(f"learning_rate: {args_cli.learning_rate}")
    print(f"resume_path: {resume_path}")

    start_time = time.time()
    last_loss = float("nan")
    for update in range(1, args_cli.max_updates + 1):
        with torch.no_grad():
            labels = mdp.sc_scripted_raw_action(
                base_env,
                action_body_name=args_cli.action_body_name,
                action_scale=args_cli.action_scale,
                action_clip=args_cli.action_clip,
                approach_depth=args_cli.approach_depth,
                target_depth=args_cli.target_depth,
                max_translation_step=args_cli.max_translation_step,
                max_rotation_step=args_cli.max_rotation_step,
                align_lateral_threshold=args_cli.align_lateral_threshold,
                align_orientation_threshold=args_cli.align_orientation_threshold,
            ).to(agent_cfg.device)

        pred = actor(obs, stochastic_output=False)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args_cli.max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

        with torch.inference_mode():
            obs, _, dones, _ = wrapped_env.step(labels.to(wrapped_env.device))
            obs = obs.to(agent_cfg.device)
            actor.reset(dones)

        if (
            update == 1
            or update % args_cli.report_every == 0
            or update == args_cli.max_updates
        ):
            with torch.inference_mode():
                lateral = mdp.sc_lateral_error(base_env)
                orientation = mdp.sc_orientation_error(base_env)
                depth = mdp.sc_insertion_depth(base_env)
                success = mdp.sc_insertion_success(
                    base_env,
                    lateral_threshold=args_cli.success_lateral_threshold,
                    orientation_threshold=args_cli.success_orientation_threshold,
                    depth_threshold=args_cli.success_depth_threshold,
                )
                pred_error = torch.norm(pred.detach() - labels, dim=-1)
            print(
                f"update={update} loss={last_loss:.6f} "
                f"pred_error_mean={pred_error.mean().item():.6f} "
                f"successes={int(success.sum().item())}/{base_env.num_envs} "
                f"lateral_mean={lateral.mean().item():.6f} "
                f"orientation_mean={orientation.mean().item():.6f} "
                f"depth_mean={depth.mean().item():.6f}",
                flush=True,
            )

        if args_cli.save_every > 0 and update % args_cli.save_every == 0:
            runner.current_learning_iteration = update
            checkpoint_path = os.path.join(log_dir, f"model_{update}.pt")
            runner.save(checkpoint_path, infos={"bc_loss": last_loss})
            print(f"saved_checkpoint: {checkpoint_path}", flush=True)

    runner.current_learning_iteration = args_cli.max_updates
    final_checkpoint = os.path.join(log_dir, f"model_{args_cli.max_updates}.pt")
    runner.save(final_checkpoint, infos={"bc_loss": last_loss})
    print(f"final_checkpoint: {final_checkpoint}")
    print(f"elapsed_seconds: {time.time() - start_time:.2f}")

    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
