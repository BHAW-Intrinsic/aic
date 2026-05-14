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
