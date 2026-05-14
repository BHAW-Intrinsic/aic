# Step 11: Gazebo-Compatible SFP Retraining

## Goal

Train a deployment-compatible SFP actor on the target structure used by the
official Gazebo qualification eval.

Step 10 showed that the old SFP Isaac task and the official Gazebo SFP task do
not ask quite the same problem:

- old Isaac SFP: choose between `sfp_port_0` and `sfp_port_1` on one NIC
- official Gazebo SFP: insert into `sfp_port_0` on different
  `nic_card_mount_*` modules

The task must keep the actor observation legal for eval, so it does not add
privileged port geometry to the actor. The first implementation kept the
existing 3149D observation shape and forced the policy to infer the mount-scale
shift from camera/proprioception; the later revision adds only official
`Task.target_module_name` metadata and expands the Gazebo-transfer SFP actor to
3151D.

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

## First Scratch Run Plateau

The first `d9ff95e` scratch run was monitored through iteration 52 and then
stopped cleanly with Ctrl-C after `model_50.pt` was saved. The observed SFP
insertion success stayed roughly in the `0.14-0.26` band and did not show a
clear upward trend:

```text
iteration 0:  sfp_insertion_success=0.0692
iteration 1:  sfp_insertion_success=0.2655
iteration 50: sfp_insertion_success=0.1714
iteration 52: sfp_insertion_success=0.2568
```

Conclusion:

- The task was legal, but too much of the 4 cm Gazebo mount switch had to be
  inferred indirectly from vision/proprioception.
- PPO exploration was also too small for the 0.003 m SFP action scale, so the
  initial policy mostly pushed along the insertion axis and did not explore
  enough lateral correction.

## Legal Target-Module Metadata Revision

The official Gazebo `Task` includes both `port_name` and `target_module_name`.
Using those fields is legal because they are passed directly to the submitted
policy. No scoring TF, Gazebo model state, or hidden geometry is used.

Implementation changes:

- Added Gazebo-style SFP mount IDs in `mdp/geometry.py`:
  `nic_card_mount_0` and `nic_card_mount_1`.
- Added `mdp.active_sfp_gazebo_task_one_hot`, returning:

```text
[sfp_port_0, sfp_port_1, nic_card_mount_0, nic_card_mount_1]
```

- Updated `AIC-SFP-Gazebo-Transfer-Task-v0` to use the 4D metadata term.
  Actor observation shape becomes `3151`; critic shape becomes `3171`.
- Changed NIC randomization from one continuous y range to sampled mount IDs:
  `0.0 m` for mount 0 and `-0.04 m` for mount 1, with `+/-0.005 m` jitter.
- Kept the original `AIC-SFP-Task-v0` and older 3149D Gazebo wrapper path
  unchanged.
- Added `AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA=true` to opt the Gazebo wrapper
  into the 3151D SFP actor input. The default remains 3149D for old artifacts.
- Increased the Gazebo-transfer PPO actor initial std to `0.25` and entropy
  coefficient to `0.002` so lateral correction is actually explored.

Local syntax check:

```bash
python3 -m py_compile \
  aic_model/aic_model/RslRlCheckpointPolicy.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_gazebo_transfer_cfg.py
```

Result:

```text
no output; exit 0
```

Verification command on the host:

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

Run session:

```text
isaac-step11-meta-smoke-f00eb1e
```

Important output:

```text
Active Observation Terms in Group: 'policy' (shape: (3151,))
task_metadata shape: (4,)
Actor Model: Linear(in_features=3151, out_features=512, ...)
Critic Model: Linear(in_features=3171, out_features=512, ...)
ISAAC_META_SMOKE_EXIT:0
```

## Live Metadata Training Run

Started in tmux:

```text
isaac-step11-sfp-gazebo-meta-train-3bd2119
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
   --run_name step11_sfp_gazebo_meta_3bd2119 \
   --headless \
   --enable_cameras"
```

Log path:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_12-42-06_step11_sfp_gazebo_meta_3bd2119
```

Early metrics:

```text
iteration 0:  sfp_insertion_success=0.1584
iteration 5:  sfp_insertion_success=0.4409
iteration 8:  sfp_insertion_success=0.3876
iteration 32: sfp_insertion_success=0.4208
iteration 36: sfp_insertion_success=0.4110
```

Interpretation:

- This is better than the first Gazebo-transfer scratch run, which stayed near
  `0.14-0.26` through iteration 52.
- It is not yet qualification-ready; let it continue to at least the
  iteration-100 range before deciding whether to export/evaluate or change the
  curriculum again.

## Metadata Run Failure Mode

The metadata run regressed by the iteration-100 range:

```text
iteration 87:  sfp_insertion_success=0.1236
iteration 90:  sfp_insertion_success=0.0712
iteration 101: sfp_insertion_success=0.0684
iteration 104: sfp_insertion_success=0.0757
```

At the same time, mean reward kept increasing to about `700+`, and
`sfp_insertion_depth` reward stayed high. This means PPO found a reward shortcut:
push deep and collect depth/action/alignment shaping while missing the final
strict lateral success threshold.

The run was stopped after `model_100.pt` was saved:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_12-42-06_step11_sfp_gazebo_meta_3bd2119/model_100.pt
```

## Reward Tightening Revision

Changed the Gazebo-transfer SFP curriculum so success dominates depth:

- `sfp_depth_progress_reward` now optionally gates depth progress by lateral and
  orientation thresholds.
- Gazebo-transfer config gates depth progress at lateral `<0.018 m` and
  orientation `<0.25 rad`.
- `sfp_port_frame_depth_action` weight reduced from `80` to `30` and gated at
  lateral `<0.018 m`.
- `sfp_insertion_depth` gate tightened from lateral `<0.030 m` to
  `<0.015 m`.
- sparse `sfp_insertion_success` bonus increased from `100` to `1000`.
- lateral pressure increased:
  `sfp_lateral_progress=60`, `sfp_lateral_error=-120`, and
  `sfp_lateral_corridor=-160` with a `0.012-0.030 m` corridor.
- Added Gazebo-transfer max-depth termination at `0.055 m` so overshooting while
  misaligned cannot remain a long high-reward timeout trajectory.
