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

## Remote Verification

Implementation commits:

- `b2fd215 Replace command rewards with insertion rewards`
- `ca3456f Add reward shape checks`

Pulled on the remote host:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

Copied the changed checker into the running Isaac Lab container after the final
shape-check update:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
```

Earlier in the same step, the changed reward/config files were also copied into
the container:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
```

Ran inside the Isaac Lab container through host `tmux` session
`isaac-step3-rewards-ca3456f`:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  --task AIC-Task-v0 --num_envs 16 --num_steps 8 --headless --enable_cameras
```

Key output:

```text
Reward Manager contains 10 active terms:
sc_approach, sc_lateral_alignment, sc_orientation_alignment,
sc_insertion_depth, sc_insertion_success, action_rate, joint_vel, joint_acc,
joint_torques, joint_pos_limits
```

The command manager still has `ee_pose`, but no active reward term depends on it.

```text
== Analytic Reward Shape Checks ==
approach_reward ... monotonic=True
lateral_reward ... monotonic=True
orientation_reward ... monotonic=True
depth_reward aligned ... monotonic=True
depth_reward misaligned ... zeroed=True
analytic_shape_checks_ok: True
```

Random-policy tensor checks stayed finite:

```text
== Direct Reward Tensor Checks: after reset ==
sc_approach: finite=True mean=0.012793 min=0.010049 max=0.016419
sc_lateral_alignment: finite=True mean=0.002607 min=0.000000 max=0.036934
sc_orientation_alignment: finite=True mean=0.000000 min=0.000000 max=0.000001
sc_insertion_depth: finite=True mean=0.000000 min=0.000000 max=0.000000
sc_insertion_success: finite=True mean=0.000000 min=0.000000 max=0.000000
...
step 07 total_reward: finite=True mean=-0.002038 min=-0.009877 max=-0.000344
...
overall_finite: True
STEP3_REWARD_EXIT:0
```

Copied the reward log back to the host:

```bash
mkdir -p ~/IsaacLab/aic/logs
docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_rewards \
  ~/IsaacLab/aic/logs/
```

Result:

```text
COPY_EXIT:0
~/IsaacLab/aic/logs/aic_rewards/20260510_103443_AIC-Task-v0.log
```
