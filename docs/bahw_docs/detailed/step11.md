# Step 11: Gazebo-Compatible SFP Retraining

## Goal

Train a deployment-compatible SFP actor on the target structure used by the
official Gazebo qualification eval.

Step 10 showed that the old SFP Isaac task and the official Gazebo SFP task do
not ask quite the same problem:

- old Isaac SFP: choose between `sfp_port_0` and `sfp_port_1` on one NIC
- official Gazebo SFP: insert into `sfp_port_0` on different
  `nic_card_mount_*` modules

The new task must keep the actor observation legal for eval, so it does not add
privileged port geometry to the actor. It keeps the existing 3149D observation
shape and forces the policy to use camera/proprioception for the mount-scale
shift.

## Initial Implementation

Added a sibling Isaac Lab task:

```text
AIC-SFP-Gazebo-Transfer-Task-v0
```

Files changed:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_gazebo_transfer_cfg.py`

Behavior:

- existing `AIC-SFP-Task-v0` remains unchanged
- active SFP target is fixed to `sfp_port_0`
- NIC/card y randomization is widened to `[-0.045, 0.005] m`
- PPO logs go under experiment `aic_sfp_gazebo_transfer`

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_gazebo_transfer_cfg.py
```

```text
pass
```

```bash
git diff --check
```

```text
pass
```

Commit:

```text
d9ff95e Add Gazebo-transfer SFP task
```

## Remote Plan

After commit/push and host sync:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

If the container copy needs updating:

```bash
docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/
```

Task discovery check:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
```

Training command:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-SFP-Gazebo-Transfer-Task-v0 \
  --agent rsl_rl_sfp_cfg_entry_point \
  --num_envs 64 \
  --max_iterations 1500 \
  --run_name step11_sfp_gazebo_transfer \
  --headless \
  --enable_cameras
```

## Remote Sync And Smoke Test

Pushed branch:

```bash
git push origin aloy
```

Remote host sync:

```bash
cd ~/IsaacLab/aic
git fetch origin
git switch aloy
git pull --ff-only
```

Result:

```text
0e12ccd..d9ff95e  aloy -> origin/aloy
Fast-forward
ISAAC_STEP11_SYNC_EXIT:0
```

Container sync:

```bash
cd ~/IsaacLab
docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/
docker cp ~/IsaacLab/aic/docs/bahw_docs \
  isaac-lab-base:/workspace/isaaclab/aic/docs/
```

Result:

```text
ISAAC_STEP11_CONTAINER_SYNC_EXIT:0
```

Smoke command:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 1 \
   --max_iterations 0 \
   --headless \
   --enable_cameras"
```

Important output:

```text
Parsing configuration from:
  aic_task.tasks.manager_based.aic_task.aic_task_env_cfg:AICTaskSfpGazeboTransferEnvCfg
Parsing configuration from:
  aic_task.tasks.manager_based.aic_task.agents.rsl_rl_ppo_sfp_gazebo_transfer_cfg:PPORunnerCfg
Logging experiment in directory:
  /workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer
Active Observation Terms in Group: 'policy' (shape: (3149,))
Actor Model: Linear(in_features=3149, out_features=512, ...)
Critic Model: Linear(in_features=3169, out_features=512, ...)
ISAAC_STEP11_SMOKE_EXIT:0
```

## Live Training Run

Started in tmux:

```text
isaac-step11-sfp-gazebo-train-d9ff95e
```

Command:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 64 \
   --max_iterations 1500 \
   --run_name step11_sfp_gazebo_transfer_d9ff95e \
   --headless \
   --enable_cameras"
```

Log path:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_11-50-22_step11_sfp_gazebo_transfer_d9ff95e
```

First iteration:

```text
Learning iteration 0/1500
Total steps: 9600
Steps per second: 296
Mean reward: 209.13
Mean episode length: 67.79
Episode_Reward/sfp_insertion_depth: 53.3603
Episode_Reward/sfp_insertion_success: 0.0911
Episode_Termination/time_out: 0.4899
Episode_Termination/sfp_insertion_success: 0.0692
ETA: 13:28:26
```

Interpretation:

- The widened port-0 mount-shift task is harder than the previous narrow
  randomization task, but not fully sparse.
- Training should be allowed to continue before deciding whether to warm-start
  from the previous SFP checkpoint or adjust the curriculum.
