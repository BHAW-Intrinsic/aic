# Step 3: Replace Command-Pose Rewards With Insertion Rewards

This note records the first reward pass that stops optimizing sampled
`ee_pose` commands and starts optimizing SC plug-to-port insertion geometry.

## Step 2 Input

Step 2 split observations into:

- eval-compatible `policy`
- training-only `privileged`

The actor no longer observes `pose_command`, and the critic receives plug-to-port
geometry. Step 3 now makes the reward objective match that insertion geometry.

## Remaining Plan Check

No direction change is needed for later steps. Step 3 feeds directly into:

- Step 4 success termination, using the same lateral/orientation/depth
  thresholds.
- Step 5 SC PPO teacher config, using the asymmetric actor-critic observation
  groups from Step 2.

The command generator remains configured for now because it is still part of the
existing environment setup, but no nonzero reward term should depend on
`ee_pose`.

## Implemented Locally

Changed `mdp/rewards.py`:

- Added `sc_approach_reward`.
- Added `sc_lateral_alignment_reward`.
- Added `sc_orientation_alignment_reward`.
- Added `sc_insertion_depth_reward`.
- Added `sc_insertion_success_bonus`.

Reward conventions:

- approach reward: high when the SC plug tip is close to the active port
  entrance.
- lateral reward: high when the plug tip is centered on the active port
  insertion axis.
- orientation reward: high when the plug axis aligns with the active port
  insertion axis.
- depth reward: only pays positive insertion depth when lateral and angular
  alignment pass thresholds.
- success bonus: sparse bonus when lateral error, orientation error, and depth
  all pass conservative thresholds.

Changed `aic_task_env_cfg.py`:

- Removed the nonzero command-pose reward terms:
  - `end_effector_position_tracking`
  - `end_effector_position_tracking_fine_grained`
  - `end_effector_position_tracking_exp`
  - `end_effector_orientation_tracking`
  - `end_effector_orientation_tracking_fine_grained`
  - `reaching_bonus`
- Added insertion reward terms:
  - `sc_approach`, weight `0.5`
  - `sc_lateral_alignment`, weight `1.0`
  - `sc_orientation_alignment`, weight `0.5`
  - `sc_insertion_depth`, weight `4.0`
  - `sc_insertion_success`, weight `10.0`
- Kept smoothness and safety penalties:
  - `action_rate`
  - `joint_vel`
  - `joint_acc`
  - `joint_torques`
  - `joint_pos_limits`

Changed `mdp/__init__.py`:

- Exported the new reward functions for config use.

Added `scripts/check_aic_rewards.py`:

- Builds `AIC-Task-v0`.
- Steps random actions.
- Checks direct insertion reward tensors for finite values.
- Writes logs under `logs/aic_rewards/`.

## Local Verification

Syntax checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
```

Status: passed locally.

Whitespace check:

```bash
git diff --check
```

Status: passed locally.

## Remote Verification Still Needed

Run inside the Isaac Lab container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  --task AIC-Task-v0 --num_envs 16 --num_steps 8 --headless --enable_cameras
```

Expected checks:

- Reward Manager contains insertion reward terms, not command-pose reward terms.
- Total random-policy rewards are finite.
- Direct insertion reward tensors are finite after reset and after random
  actions.
- Reward log is copied from `logs/aic_rewards/`.
