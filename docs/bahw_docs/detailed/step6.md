# Step 6: Train And Evaluate SC Teacher

This note tracks SC PPO teacher training and evaluation.

## Step 5 Input

Step 5 added the SC-specific RSL-RL config and verified a one-iteration smoke
run:

- agent entry point: `rsl_rl_sc_cfg_entry_point`
- experiment name: `aic_sc_insert`
- actor inputs: policy group, `3149` dims
- critic inputs: policy plus privileged groups, `3169` dims

## Remaining Plan Check

Step 6 is training-gated. Steps 7 to 9 should not be marked complete until the
SC teacher is useful enough to justify extending the same interface to SFP and
later distilling specialist policies.

The Step 6 commands now explicitly pass `--agent rsl_rl_sc_cfg_entry_point`.

## Initial Training Command

Run inside the Isaac Lab container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 64 --headless --enable_cameras \
  --run_name step6_sc_teacher
```

Expected output location:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/<timestamp>_step6_sc_teacher
```

Host copy target:

```text
~/IsaacLab/aic/logs/rsl_rl/aic_sc_insert/<timestamp>_step6_sc_teacher
```

## Baseline Training Run

Implementation before training:

- Step 5 SC teacher config commit: `c4e98ae`
- Step 6 training-start docs commit: `26b63a7`
- evaluation script commit: `4ba304f`

Host/tmux command:

```bash
tmux new-session -d -s isaac-step6-sc-train-26b63a7 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --headless --enable_cameras --run_name step6_sc_teacher'; echo STEP6_TRAIN_EXIT:\$?"
```

Container log/checkpoint directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_11-06-52_step6_sc_teacher
```

Checkpoints were written at `model_0.pt`, then every 50 iterations through
`model_500.pt`.

Observed training result:

- The run was infrastructure-stable with `64` envs and cameras enabled.
- The run was stopped at iteration `523` because it was not learning insertion.
- Around iterations `367` to `500`, the insertion terms stayed flat:
  - `Episode_Reward/sc_insertion_depth: 0.0000`
  - `Episode_Reward/sc_insertion_success: 0.0000`
  - `Episode_Termination/sc_insertion_success: 0.0000`
  - `Episode_Termination/time_out: 1.0000`
- Interpretation: the actor never entered the narrow lateral/orientation corridor
  where the depth reward and success reward turn on.

Cleanup note:

- The first interrupt left a stale training Python process in the container.
- It was removed with a named host tmux cleanup session:

```bash
tmux new-session -d -s isaac-step6-kill-stale-train \
  "docker exec isaac-lab-base bash -lc 'pkill -KILL -f \"scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64\"; sleep 5; ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E \"rsl_rl/train.py|rsl_rl/evaluate.py\" | grep -v grep; echo KILL_EXIT:\$?'; sleep 30"
```

## Metrics To Watch

- `Episode_Termination/sc_insertion_success`
- `Episode_Termination/time_out`
- `Episode_Reward/sc_insertion_depth`
- `Episode_Reward/sc_insertion_success`
- mean reward and mean episode length

Success metrics still need an explicit evaluation pass after a checkpoint exists:

- success rate
- mean episode length on success
- mean lateral error at termination
- mean insertion depth at termination
- failure breakdown
- video review for real insertion rather than hovering

## Evaluation Script

Added `scripts/rsl_rl/evaluate.py` to evaluate a checkpoint without relying on
the free-running `play.py` loop.

The evaluator:

- loads the checkpoint using the same RSL-RL runner path as `play.py`
- disables environment auto-termination before env creation
- runs manual episode accounting
- records terminal lateral error, orientation error, and insertion depth before
  reset
- reports success rate and per-target success rates for `sc_port` and
  `sc_port_2`
- reports failure diagnostics for timeout, lateral miss, orientation miss, and
  depth shortfall
- writes timestamped logs under `logs/aic_eval/`

Command:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 16 --headless --enable_cameras \
  --checkpoint <checkpoint_path> \
  --num_eval_episodes 256
```

Baseline model-500 evaluation command:

```bash
tmux new-session -d -s isaac-step6-eval-model500 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 16 --headless --enable_cameras --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_11-06-52_step6_sc_teacher/model_500.pt --num_eval_episodes 16'; echo STEP6_EVAL_EXIT:\$?"
```

Status:

- The evaluator created the env, loaded the old reward config, disabled
  terminations, and set `max_episode_steps: 6000`.
- Result: `STEP6_EVAL_EXIT:0`.
- Metrics:
  - episodes: `16`
  - successes: `0`
  - success rate: `0.000000`
  - mean episode length: `6000.000`
  - mean episode length on success: `nan`
  - mean lateral error at termination: `1.228034`
  - mean orientation error at termination: `0.346956`
  - mean insertion depth at termination: `0.720373`
  - timeout failures: `16`
  - lateral misses: `16`
  - orientation misses: `5`
  - depth shortfalls: `0`
  - `sc_port`: `8` episodes, `0` successes
  - `sc_port_2`: `8` episodes, `0` successes
- Interpretation: the policy learned to move along/in the insertion direction
  but missed the port laterally by a large margin, so Step 7 remains blocked.

Log copied on the host:

```text
~/IsaacLab/aic/logs/aic_eval/20260510_114251_AIC-Task-v0.log
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py
git diff --check
```

Status: passed locally.

## First Remediation

Commit `741a69d` adds progress reward shaping for Step 6 retraining.

Reason:

- Baseline rewards only gave depth/success signal inside a narrow aligned
  corridor.
- The policy never discovered that corridor, so the sparse and gated insertion
  terms stayed at zero.

Code changes:

- `mdp/geometry.py`
  - added per-env previous-metric buffer names
  - added `reset_sc_progress_buffers`
- `aic_task_env_cfg.py`
  - added a reset event to initialize progress buffers after active SC target
    sampling
  - increased `sc_approach` from weight `0.5`, std `0.50` to weight `1.0`,
    std `1.00`
  - added `sc_distance_progress`
  - added `sc_lateral_progress`
  - added `sc_orientation_progress`
  - added `sc_depth_progress`
- `mdp/rewards.py`
  - added stateful progress reward helpers
- `check_aic_rewards.py`
  - added direct finite checks for the progress rewards

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
git diff --check
```

Result: passed locally.

Remote sync:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

Result:

```text
Updating 4ba304f..741a69d
Fast-forward
...
5 files changed, 185 insertions(+), 2 deletions(-)
PULL_EXIT:0
```

Container sync:

```bash
cd ~/IsaacLab/aic
docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py
docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py
```

Result: `COPY_EXIT:0`.

Next checks:

- run `check_aic_rewards.py` remotely and confirm the new progress terms are
  active and finite
- retrain with `--run_name step6_sc_progress`
- do not start SFP work until the remediated SC teacher reaches depth/success

Remote reward check command:

```bash
tmux new-session -d -s isaac-step6-remed-reward-741a69d \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 8 --num_steps 4 --headless --enable_cameras'; echo STEP6_REWARD_CHECK_EXIT:\$?; sleep 30"
```

Remote reward check result:

- Event manager now includes `reset_sc_progress_buffers`.
- Reward manager now contains `14` active terms:
  - `sc_approach`
  - `sc_distance_progress`
  - `sc_lateral_progress`
  - `sc_orientation_progress`
  - `sc_depth_progress`
  - existing fine alignment/depth/success and smoothness terms
- All direct reward tensors were finite.
- Random-policy total rewards were finite for `4` steps.
- Exit: `STEP6_REWARD_CHECK_EXIT:0`.

Reward log copied on the host:

```text
~/IsaacLab/aic/logs/aic_rewards/20260510_115702_AIC-Task-v0.log
```

## Remediated Training Run

Command:

```bash
tmux new-session -d -s isaac-step6-sc-progress-train-741a69d \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --headless --enable_cameras --run_name step6_sc_progress --max_iterations 500'; echo STEP6_PROGRESS_TRAIN_EXIT:\$?; sleep 30"
```

Result:

- Stopped at iteration `106` because the first remediation still did not reach
  the fine insertion corridor.
- Last observed values:
  - `Episode_Reward/sc_approach: 0.2885`
  - `Episode_Reward/sc_distance_progress: 0.0086`
  - `Episode_Reward/sc_lateral_progress: 0.0020`
  - `Episode_Reward/sc_depth_progress: 0.0057`
  - `Episode_Reward/sc_lateral_alignment: 0.0000`
  - `Episode_Reward/sc_insertion_depth: 0.0000`
  - `Episode_Termination/sc_insertion_success: 0.0000`
- Interpretation: progress-only shaping moved the policy closer in a coarse
  sense, but it still lacked a strong absolute reward for being near the port
  axis.

## Second Remediation

Commit `d1eaea9` adds two coarse absolute alignment terms:

- `sc_coarse_lateral_alignment`
  - function: `sc_lateral_alignment_reward`
  - weight: `1.0`
  - std: `0.30`
- `sc_coarse_orientation_alignment`
  - function: `sc_orientation_alignment_reward`
  - weight: `0.5`
  - std: `2.00`

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
git diff --check
```

Result: passed locally.

Remote sync:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
```

Result: `PULL_EXIT:0`, `COPY_EXIT:0`.

Remote reward check command:

```bash
tmux new-session -d -s isaac-step6-coarse-reward-d1eaea9 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 8 --num_steps 3 --headless --enable_cameras'; echo STEP6_COARSE_REWARD_EXIT:\$?; sleep 30"
```

Remote reward check result:

- Reward manager now contains `16` active terms.
- At reset, the new coarse terms produced useful nonzero signal:
  - `sc_coarse_lateral_alignment` mean `0.463989`
  - `sc_coarse_orientation_alignment` mean `0.118190`
- All direct reward tensors were finite.
- Random-policy total rewards were finite for `3` steps.
- Exit: `STEP6_COARSE_REWARD_EXIT:0`.

Reward log copied on the host:

```text
~/IsaacLab/aic/logs/aic_rewards/20260510_120805_AIC-Task-v0.log
```

## Second Remediated Training Run

Command:

```bash
tmux new-session -d -s isaac-step6-sc-coarse-train-d1eaea9 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --headless --enable_cameras --run_name step6_sc_coarse --max_iterations 500'; echo STEP6_COARSE_TRAIN_EXIT:\$?; sleep 30"
```

Result:

- Stopped at iteration `45`.
- Last observed values:
  - `Episode_Reward/sc_approach: 0.0252`
  - `Episode_Reward/sc_distance_progress: -0.0027`
  - `Episode_Reward/sc_lateral_progress: -0.0083`
  - `Episode_Reward/sc_coarse_lateral_alignment: 0.0018`
  - `Episode_Reward/sc_coarse_orientation_alignment: 0.0737`
  - `Episode_Reward/sc_lateral_alignment: 0.0000`
  - `Episode_Reward/sc_insertion_depth: 0.0000`
  - `Episode_Reward/sc_insertion_success: 0.0000`
  - `Episode_Reward/joint_acc: -0.1889`
- Interpretation: coarse terms were active but too small relative to smoothness
  penalties, especially joint acceleration.

Process cleanup note:

- Interrupted Isaac runs can leave their Python process alive inside the
  container even after tmux prints an exit line.
- Before launching another run, stale `rsl_rl/train.py` and
  `check_aic_rewards.py` processes were killed and verified absent.

## Third Remediation

Commit `2ce4c3e` rebalances rewards for exploration:

- increased task reward weights:
  - `sc_approach`: `1.0` to `3.0`
  - `sc_distance_progress`: `1.0` to `2.0`
  - `sc_lateral_progress`: `0.5` to `1.0`
  - `sc_orientation_progress`: `0.25` to `0.5`
  - `sc_depth_progress`: `0.5` to `1.0`
  - `sc_coarse_lateral_alignment`: `1.0` to `10.0`
  - `sc_coarse_orientation_alignment`: `0.5` to `2.0`
- reduced smoothness penalties:
  - `joint_vel`: `-0.0001` to `-1.0e-5`
  - `joint_acc`: `-1.0e-7` to `-1.0e-8`
  - `joint_torques`: `-1.0e-6` to `-1.0e-7`

Remote reward check:

```bash
tmux new-session -d -s isaac-step6-rebalanced-reward-clean-2ce4c3e \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 8 --num_steps 2 --headless --enable_cameras'; echo STEP6_REBALANCED_REWARD_EXIT:\$?; sleep 30"
```

Result:

- Reward manager loaded `16` active terms with the rebalanced weights.
- Random-policy total reward was positive and finite:
  - step 0 mean `0.213056`
  - step 1 mean `0.195096`
- Exit: `STEP6_REBALANCED_REWARD_EXIT:0`.

Reward log copied on the host:

```text
~/IsaacLab/aic/logs/aic_rewards/20260510_122103_AIC-Task-v0.log
```

## Third Remediated Training Run

Command:

```bash
tmux new-session -d -s isaac-step6-sc-rebalanced-train-2ce4c3e \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --headless --enable_cameras --run_name step6_sc_rebalanced --max_iterations 500'; echo STEP6_REBALANCED_TRAIN_EXIT:\$?; sleep 30"
```

Result:

- Stopped at iteration `110`.
- Last observed values:
  - iteration `100` mean reward: `97.78`
  - iteration `110` mean reward: `101.52`
  - `Episode_Reward/sc_approach: 0.3688`
  - `Episode_Reward/sc_distance_progress: 0.0042`
  - `Episode_Reward/sc_lateral_progress: -0.0053`
  - `Episode_Reward/sc_depth_progress: 0.0190`
  - `Episode_Reward/sc_coarse_lateral_alignment: 0.0827`
  - `Episode_Reward/sc_coarse_orientation_alignment: 0.3728`
  - `Episode_Reward/sc_lateral_alignment: 0.0000`
  - `Episode_Reward/sc_orientation_alignment: 0.0155`
  - `Episode_Reward/sc_insertion_depth: 0.0000`
  - `Episode_Reward/sc_insertion_success: 0.0000`
- Interpretation: reward scaling made the rollout reward positive, but PPO still
  did not discover the narrow fine-alignment/insertion corridor from the default
  reset distribution.

Conclusion from the three remediation attempts:

- Do not keep adding scalar reward weight changes blindly.
- The next Step 6 work is to validate the geometry/action path with a scripted
  privileged controller.
- If the scripted controller can insert, use it to derive a near-port
  reset/curriculum or demonstration seed.
- If the scripted controller cannot insert, fix the action convention, plug-tip
  transform, or port-entry transform before more PPO.

## Scripted SC Insertion Check

Added `scripts/check_aic_scripted_insert.py`.

Purpose:

- Use the same Step 1 geometry helpers as rewards and terminations.
- Compute small relative differential-IK actions from privileged SC geometry.
- Report lateral error, orientation error, insertion depth, first-success steps,
  and per-target success rates.
- Run headless so it does not require remote desktop or X11.

Local check:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
```

Result: passed locally.

Remote sync:

```bash
tmux new-session -d -s isaac-step6-pull-be9cfe4 \
  "cd ~/IsaacLab/aic && git pull --ff-only && docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py; echo STEP6_PULL_COPY_EXIT:\$?; sleep 60"
```

Result:

- Host repo updated from `2ce4c3e` to `be9cfe4`.
- New script copied into
  `/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/`.
- Exit: `STEP6_PULL_COPY_EXIT:0`.

Stale-process check before launch:

```bash
tmux new-session -d -s isaac-step6-check-stale-before-scripted \
  "docker exec isaac-lab-base bash -lc 'ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E \"rsl_rl/train.py|rsl_rl/evaluate.py|check_aic_rewards.py|check_aic_scripted_insert.py\" | grep -v grep || true'; echo STEP6_STALE_CHECK_EXIT:\$?; sleep 60"
```

Result: no stale matching processes; exit `STEP6_STALE_CHECK_EXIT:0`.

Remote command:

```bash
tmux new-session -d -s isaac-step6-scripted-insert-be9cfe4 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 8 --max_steps 1500 --report_every 50 --headless --enable_cameras'; echo STEP6_SCRIPTED_INSERT_EXIT:\$?; sleep 30"
```

Result:

- Action space loaded as `Box(-inf, inf, (8, 6), float32)`.
- Terminations were disabled in the check script so success could be measured
  manually.
- Initial geometry:
  - successes: `0/8`
  - lateral mean/min/max: `0.215460 / 0.033184 / 0.362503`
  - orientation mean/min/max: `2.716869 / 2.512053 / 2.872937`
  - depth mean/min/max: `-1.225927 / -1.247427 / -1.191501`
- The script could move the plug toward and sometimes past the entrance plane:
  - step `100` depth mean: `-0.010867`
  - step `250` depth mean: `0.016971`
  - step `1500` depth mean/min/max: `0.020453 / -0.038301 / 0.141677`
- It did not solve fine alignment:
  - step `1500` lateral mean/min/max: `0.547527 / 0.112848 / 1.191187`
  - step `1500` orientation mean/min/max: `1.040751 / 0.192459 / 1.598559`
- Summary:
  - successes: `0/8`
  - first success steps: all `-1`
  - `sc_port`: `5` episodes, `0` successes
  - `sc_port_2`: `3` episodes, `0` successes
- The shell printed `STEP6_SCRIPTED_INSERT_EXIT:0`; use the explicit success
  summary above as the run result.

Log copied on the host:

```text
~/IsaacLab/aic/logs/aic_scripted_insert/20260510_124425_AIC-Task-v0.log
```

Copy command:

```bash
tmux new-session -d -s isaac-step6-copy-scripted-log \
  "mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert && docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_124425_AIC-Task-v0.log ~/IsaacLab/aic/logs/aic_scripted_insert/20260510_124425_AIC-Task-v0.log && docker exec isaac-lab-base bash -lc 'ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E \"rsl_rl/train.py|rsl_rl/evaluate.py|check_aic_rewards.py|check_aic_scripted_insert.py\" | grep -v grep || true'; echo STEP6_SCRIPTED_LOG_COPY_EXIT:\$?; sleep 60"
```

Result:

- Log copied successfully.
- No stale matching process remained.
- Exit: `STEP6_SCRIPTED_LOG_COPY_EXIT:0`.

Interpretation:

- This is no longer just a sparse-reward problem.
- The current controller can move the plug tip along the port axis enough to
  satisfy depth, but it cannot keep the plug tip laterally centered or
  orientation-aligned.
- Do not start Step 7 and do not run more PPO yet.
- Next Step 6 work is to diagnose the control-frame mapping:
  - IK action target is `wrist_3_link`
  - measured insertion point is `robot.sc_tip_link`
  - the scripted controller currently commands wrist-frame motion from plug-tip
    geometry without an explicit plug-tip `body_offset`
  - likely fixes are adding the correct action `body_offset`, commanding the
    actual plug-tip frame if Isaac Lab supports it, or correcting the plug/port
    axis convention before curriculum design.

## Tip-Frame Scripted Check

Commit `f7af540` updated `scripts/check_aic_scripted_insert.py` to add
`--control_frame tip` as the default mode.

The new mode:

- computes a desired SC tip pose from privileged plug-to-port geometry
- estimates the current transform from `wrist_3_link` to `sc_tip_link`
- solves the equivalent desired `wrist_3_link` pose
- sends the relative IK action for that wrist pose
- logs the initial/final `wrist_to_sc_tip` transform and drift

Reason:

- The previous scripted controller used plug-tip errors but sent them directly
  as wrist-frame relative IK deltas.
- The environment action controls `wrist_3_link`, while the insertion geometry
  and success condition measure `sc_tip_link`.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
git diff --check
```

Result: passed locally.

Remote sync:

```bash
tmux new-session -d -s isaac-step6-pull-f7af540 \
  "cd ~/IsaacLab/aic && git pull --ff-only && docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py; echo STEP6_PULL_COPY_F7AF540_EXIT:\$?; sleep 60"
```

Result:

- Host repo updated from `be9cfe4` to `f7af540`.
- Updated script copied into the Isaac Lab container.
- Exit: `STEP6_PULL_COPY_F7AF540_EXIT:0`.

Stale-process check before launch:

```bash
tmux new-session -d -s isaac-step6-check-stale-before-tip \
  "docker exec isaac-lab-base bash -lc 'ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E \"rsl_rl/train.py|rsl_rl/evaluate.py|check_aic_rewards.py|check_aic_scripted_insert.py\" | grep -v grep || true'; echo STEP6_TIP_STALE_CHECK_EXIT:\$?; sleep 60"
```

Result: no stale matching processes; exit `STEP6_TIP_STALE_CHECK_EXIT:0`.

Remote command:

```bash
tmux new-session -d -s isaac-step6-scripted-tip-f7af540 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 8 --max_steps 1500 --report_every 50 --control_frame tip --headless --enable_cameras'; echo STEP6_SCRIPTED_TIP_EXIT:\$?; sleep 60"
```

Result:

- Action space loaded as `Box(-inf, inf, (8, 6), float32)`.
- Initial `wrist_to_sc_tip_pos` for env 0:
  `[0.09874817728996277, 0.19324107468128204, -0.7596471309661865]`
- Initial `wrist_to_sc_tip_quat` for env 0:
  `[-0.037868522107601166, -0.003483319655060768, -0.9961195588111877, -0.07937105745077133]`
- The initial offset is about `0.79 m`, which is far too large for a small TCP
  helper offset between the wrist and a rigidly held SC plug tip.
- Step `1500` metrics:
  - successes: `0/8`
  - lateral mean/min/max: `0.749406 / 0.302526 / 1.156759`
  - orientation mean/min/max: `1.420358 / 0.261339 / 1.624103`
  - depth mean/min/max: `0.023790 / 0.014668 / 0.042206`
- Summary:
  - first success steps: all `-1`
  - `sc_port`: `4` episodes, `0` successes
  - `sc_port_2`: `4` episodes, `0` successes
  - `wrist_to_sc_tip_pos_drift`: mean/min/max
    `1.185182 / 0.758866 / 1.513464`
  - final `wrist_to_sc_tip_pos` for env 0:
    `[0.09240195155143738, -0.3479488492012024, 0.40017855167388916]`
  - final `wrist_to_sc_tip_quat` for env 0:
    `[0.15916739404201508, -0.3354725241661072, 0.9281059503555298, -0.0272614024579525]`
- Exit: `STEP6_SCRIPTED_TIP_EXIT:0`.

Log copied on the host:

```text
~/IsaacLab/aic/logs/aic_scripted_insert/20260510_125349_AIC-Task-v0.log
```

Copy commands used from a named host tmux session:

```bash
tmux new-session -d -s isaac-step6-copy-tip-log-f7af540b
tmux send-keys -t isaac-step6-copy-tip-log-f7af540b \
  'mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert' C-m
tmux send-keys -t isaac-step6-copy-tip-log-f7af540b \
  'latest=$(docker exec isaac-lab-base bash -lc "ls -t /workspace/isaaclab/aic/logs/aic_scripted_insert/*_AIC-Task-v0.log | head -1"); echo LATEST:$latest; docker cp isaac-lab-base:$latest ~/IsaacLab/aic/logs/aic_scripted_insert/; echo STEP6_TIP_LOG_COPY_EXIT:$?' C-m
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_125349_AIC-Task-v0.log
Successfully copied 6.5kB (transferred 8.19kB) to /var/home/bahw/IsaacLab/aic/logs/aic_scripted_insert/
STEP6_TIP_LOG_COPY_EXIT:0
```

Post-run stale-process check:

```bash
tmux new-session -d -s isaac-step6-stale-check-after-tip-f7af540
tmux send-keys -t isaac-step6-stale-check-after-tip-f7af540 \
  'pgrep -af "check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh"; echo STEP6_TIP_STALE_CHECK_EXIT:$?' C-m
```

Result:

```text
STEP6_TIP_STALE_CHECK_EXIT:1
```

No matching stale process was listed; exit `1` is the expected `pgrep` result
when no process matches.

Interpretation:

- Solving a desired wrist pose from the current tip pose did not fix scripted
  insertion.
- The measured transform between `wrist_3_link` and `sc_tip_link` changes by
  roughly meter scale during the run, so `sc_tip_link` is not behaving like a
  fixed helper frame rigidly attached to the wrist action target.
- The failure is now primarily an action-frame/asset-attachment problem, not a
  reward-weight problem.
- Do not start Step 7, PPO curriculum design, or more reward-only training yet.
- Next Step 6 work is to inspect and repair the control path: identify the
  actual robot/gripper/TCP frame that controls the held SC plug, determine
  whether the SC plug is rigidly attached to the gripper in Isaac, and then set
  the IK action target/body offset or asset attachment so `sc_tip_link` motion
  is controllable.

## Attachment Diagnosis

Read-only codebase exploration after the tip-frame run found the likely root
cause:

- Isaac loads a unified robot/cable USD from `aic_unified_robot_cable_sdf.usd`.
- Runtime robot bodies include `wrist_3_link`, `tool0`, `ati_tool_link`,
  `gripper_tcp`, cable segment bodies, `sc_plug_link`, and `sc_tip_link`.
- The SC plug asset itself has `sc_tip_link` fixed to `sc_plug_link`, but the
  plug/cable chain is not rigidly fixed to `wrist_3_link`.
- Gazebo/eval control uses the TCP (`gripper/tcp`) rather than the wrist frame;
  the Isaac body naming exposes the analogous `gripper_tcp`.
- Therefore the first question is not another reward-weight question. It is
  whether Isaac's currently loaded SC plug is the gripped plug, and whether it
  is attached to the gripper/TCP in a controllable way.

Commit `4221cf0` updated `scripts/check_aic_scripted_insert.py` again for this
diagnosis:

- adds `--action_body_name`, defaulting to `wrist_3_link`
- sets the Isaac Lab differential IK action body to that value before env
  creation
- logs relative transforms from candidate bodies to `sc_tip_link`
- default diagnostic bodies:
  `wrist_3_link,gripper_tcp,ati_tool_link,tool0,sc_plug_link`
- keeps `--control_frame tip`, but now solves the desired action-body pose rather
  than hard-coding `wrist_3_link`

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
git diff --check
```

Result: passed locally.

Remote sync:

```bash
tmux new-session -d -s isaac-step6-pull-4221cf0
tmux send-keys -t isaac-step6-pull-4221cf0 \
  'cd ~/IsaacLab/aic && git pull --ff-only && docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py && docker cp docs/bahw_docs/detailed/step6.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/detailed/step6.md && docker cp docs/bahw_docs/plan.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/plan.md; echo STEP6_PULL_COPY_4221CF0_EXIT:$?' C-m
```

Result:

- Host repo updated from `f7af540` to `4221cf0`.
- Updated scripted checker and docs copied into the Isaac Lab container.
- Exit: `STEP6_PULL_COPY_4221CF0_EXIT:0`.

Stale-process check before launch:

```bash
tmux new-session -d -s isaac-step6-stale-before-gripper-4221cf0
tmux send-keys -t isaac-step6-stale-before-gripper-4221cf0 \
  'pgrep -af "check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh"; echo STEP6_GRIPPER_STALE_BEFORE_EXIT:$?' C-m
```

Result: no matching process; `STEP6_GRIPPER_STALE_BEFORE_EXIT:1`.

Remote diagnostic:

```bash
tmux new-session -d -s isaac-step6-scripted-gripper-4221cf0 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 8 --max_steps 300 --report_every 50 --control_frame tip --action_body_name gripper_tcp --headless --enable_cameras'; echo STEP6_GRIPPER_DIAG_EXIT:\$?; sleep 60"
```

Result:

- Action body was set to `gripper_tcp`.
- Initial env 0 relative offsets:
  - `wrist_3_link_to_sc_tip_pos`:
    `[0.08932173252105713, 0.42364028096199036, -0.6812333464622498]`
  - `gripper_tcp_to_sc_tip_pos`:
    `[0.08932200074195862, 0.4236403703689575, -0.8777335286140442]`
  - `sc_plug_link_to_sc_tip_pos`:
    `[0.0116500835865736, 1.9744038581848145e-07, -3.725290298461914e-09]`
- Step `300` metrics:
  - successes: `0/8`
  - lateral mean/min/max: `0.297135 / 0.019377 / 0.666167`
  - orientation mean/min/max: `1.330962 / 0.202430 / 1.571628`
  - depth mean/min/max: `-0.001213 / -0.121757 / 0.024458`
- Summary:
  - first success steps: all `-1`
  - `sc_port`: `5` episodes, `0` successes
  - `sc_port_2`: `3` episodes, `0` successes
  - `wrist_3_link_to_sc_tip_pos_drift`: mean `0.953258`
  - `gripper_tcp_to_sc_tip_pos_drift`: mean `0.953258`
  - `ati_tool_link_to_sc_tip_pos_drift`: mean `0.953258`
  - `tool0_to_sc_tip_pos_drift`: mean `0.953258`
  - `sc_plug_link_to_sc_tip_pos_drift`: mean `0.000000`
- Exit: `STEP6_GRIPPER_DIAG_EXIT:0`.

Log copied on the host:

```text
~/IsaacLab/aic/logs/aic_scripted_insert/20260510_130823_AIC-Task-v0.log
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_130823_AIC-Task-v0.log
Successfully copied 4.49kB (transferred 6.14kB) to /var/home/bahw/IsaacLab/aic/logs/aic_scripted_insert/
STEP6_GRIPPER_LOG_COPY_EXIT:0
```

Post-run stale-process check:

```text
STEP6_GRIPPER_STALE_AFTER_EXIT:1
```

No matching stale process was listed.

Interpretation:

- The SC tip is rigidly fixed to `sc_plug_link`.
- The SC plug/tip is not rigidly fixed to `wrist_3_link`, `tool0`,
  `ati_tool_link`, or `gripper_tcp`.
- The loaded Isaac asset behaves like an SFP-gripped cable where the SC plug is
  on the free end. That is the wrong attachment for SC insertion training.

## Virtual Gripped SC Tip Remediation

Because the downloaded Isaac asset pack is ignored locally and currently exposes
only `aic_unified_robot_cable_sdf.usd`, Step 6 now uses a temporary virtual SC
tip helper for the SC teacher:

- `mdp.geometry.sc_plug_tip_pose()` returns a helper pose composed from
  `robot.gripper_tcp` when called with the default SC plug tip body.
- The helper offset places the virtual tip ahead of the TCP along local `+Z`:
  - `SC_GRIPPED_TIP_BODY = "gripper_tcp"`
  - `SC_GRIPPED_TIP_POS_LOCAL = (0.0, 0.0, 0.07)`
  - `SC_GRIPPED_TIP_QUAT_LOCAL = (1.0, 0.0, 0.0, 0.0)`
- `AIC-Task-v0` now targets `gripper_tcp` for differential IK.
- The policy `eef_pose` observation now reports `gripper_tcp`.
- `check_aic_scripted_insert.py` defaults to `--action_body_name gripper_tcp`.

This is an explicit training-side workaround. It makes the insertion geometry
controllable by the robot and is closer to Gazebo's TCP control path, but it does
not repair the cable USD visual/physics attachment. A later asset fix should
replace this helper with a real gripped SC plug frame or a fixed TCP-to-tip
offset derived from Gazebo.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
git diff --check
```

Result: passed locally.

Remote scripted checks after `9e9bc56`:

- Default helper run:
  - command used `--num_envs 8 --max_steps 1500 --control_frame tip`
  - interrupted at step `1050` after it stabilized near the approach plane
  - best observed lateral stayed near `0.016` to `0.020` m
  - depth stayed negative, around `-0.037` m
- Relaxed helper run:
  - command used `--num_envs 4 --max_steps 500 --align_lateral_threshold 0.05`
  - final successes: `0/4`
  - final lateral mean/min/max: `0.038145 / 0.035735 / 0.042264`
  - final orientation mean/min/max: `0.041006 / 0.039839 / 0.041785`
  - final depth mean/min/max: `-0.036927 / -0.044303 / -0.030177`
  - helper drift from gripper/TCP stayed `0.0`, confirming the virtual helper is
    fixed to the action body

Interpretation:

- The first TCP helper was controllable and stable, but it asked the gripper
  TCP itself to reach the port interior.
- That stalls at a near-port hover around `3` to `4` cm outside the entrance.
- The next adjustment is to place the virtual tip farther ahead of the TCP along
  local `+Z` so the TCP can remain outside the port while the helper tip
  inserts. The next offset to test is `0.07` m.

## Virtual Tip Offset 0.07 Scripted Check

Commit `a7144d2` extended the virtual helper offset:

```python
SC_GRIPPED_TIP_POS_LOCAL = (0.0, 0.0, 0.07)
```

Local checks before the remote run:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py
git diff --check
```

Result: passed locally.

Host-side sync command:

```bash
tmux new-session -d -s isaac-step6-pull-a7144d2
tmux send-keys -t isaac-step6-pull-a7144d2 \
  'cd ~/IsaacLab/aic && git pull --ff-only && docker cp aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py && docker cp docs/bahw_docs/detailed/step6.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/detailed/step6.md && docker cp docs/bahw_docs/plan.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/plan.md; echo STEP6_PULL_COPY_A7144D2_EXIT:$?; sleep 60' C-m
```

Sync result:

```text
STEP6_PULL_COPY_A7144D2_EXIT:0
```

Host-side scripted check command:

```bash
tmux new-session -d -s isaac-step6-scripted-offset7-a7144d2
tmux send-keys -t isaac-step6-scripted-offset7-a7144d2 \
  'pgrep -af "check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh"; echo STEP6_OFFSET7_STALE_BEFORE_EXIT:$?; docker exec isaac-lab-base bash -lc "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 4 --max_steps 500 --report_every 25 --control_frame tip --align_lateral_threshold 0.05 --approach_depth 0.0 --target_depth 0.020 --headless --enable_cameras"; echo STEP6_OFFSET7_SCRIPTED_EXIT:$?; sleep 60' C-m
```

The run wrote this container log:

```text
/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_133301_AIC-Task-v0.log
```

Host-side log copy command:

```bash
tmux new-session -d -s isaac-step6-copy-offset7-log-a7144d2 \
  "mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert; latest=\$(docker exec isaac-lab-base bash -lc \"ls -t /workspace/isaaclab/aic/logs/aic_scripted_insert/*_AIC-Task-v0.log | head -1\"); echo LATEST:\$latest; docker cp isaac-lab-base:\$latest ~/IsaacLab/aic/logs/aic_scripted_insert/; pgrep -af \"check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh\"; echo STEP6_OFFSET7_LOG_COPY_EXIT:\$?; sleep 60"
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_133301_AIC-Task-v0.log
Successfully copied 6.75kB (transferred 8.7kB) to /var/home/bahw/IsaacLab/aic/logs/aic_scripted_insert/
STEP6_OFFSET7_LOG_COPY_EXIT:0
```

Key output:

```text
initial_gripper_tcp_to_sc_tip_pos env0: [-4.0978193283081055e-08, 8.707866072654724e-08, 0.07000000774860382]
step=50 successes=2/4 lateral(mean=0.004040 min=0.001755 max=0.005724) orientation(mean=0.011413 min=0.000488 max=0.017592) depth(mean=0.012038 min=0.006995 max=0.016739)
step=75 successes=3/4 lateral(mean=0.003664 min=0.002409 max=0.005399) orientation(mean=0.008536 min=0.001292 max=0.016702) depth(mean=0.015666 min=0.011833 max=0.018105)
step=500 successes=3/4 lateral(mean=0.005460 min=0.004821 max=0.007313) orientation(mean=0.013375 min=0.009911 max=0.019822) depth(mean=0.020952 min=0.011679 max=0.025222)

== Summary ==
successes: 3/4
first_success_steps: [45, 46, 52, -1]
per_target:
  sc_port: episodes=1 successes=1 success_rate=1.000000
  sc_port_2: episodes=3 successes=2 success_rate=0.666667
final_lateral: mean=0.005460 min=0.004821 max=0.007313
final_orientation: mean=0.013375 min=0.009911 max=0.019822
final_depth: mean=0.020952 min=0.011679 max=0.025222
wrist_3_link_to_sc_tip_pos_drift: mean=0.000000 min=0.000000 max=0.000000
gripper_tcp_to_sc_tip_pos_drift: mean=0.000000 min=0.000000 max=0.000000
ati_tool_link_to_sc_tip_pos_drift: mean=0.000000 min=0.000000 max=0.000000
tool0_to_sc_tip_pos_drift: mean=0.000000 min=0.000000 max=0.000000
sc_plug_link_to_sc_tip_pos_drift: mean=0.810037 min=0.727583 max=0.868269
STEP6_OFFSET7_SCRIPTED_EXIT:0
```

Interpretation:

- The virtual helper is now rigidly attached to the controlled TCP path and can
  be driven into the SC success condition.
- The scripted check reached `3/4` successes quickly, so the old failure was not
  mainly the reward formula; it was the uncontrolled SC plug/tip attachment.
- The remaining miss is a strict lateral-threshold miss on one `sc_port_2`
  episode. This is good enough to proceed to a short PPO smoke/retrain, but not
  enough to mark Step 6 complete.
- `sc_plug_link_to_sc_tip_pos_drift` is expected to be nonzero now because the
  default SC tip pose is the virtual helper, not the physical free-end USD SC
  plug. This is a deliberate training workaround.

## Reward Smoke Check After Virtual Tip

Host-side command:

```bash
tmux new-session -d -s isaac-step6-reward-smoke-a7144d2 \
  "docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 8 --num_steps 2 --headless --enable_cameras\"; echo STEP6_REWARD_SMOKE_EXIT:\$?; sleep 60"
```

The run wrote this container log:

```text
/workspace/isaaclab/aic/logs/aic_rewards/20260510_133629_AIC-Task-v0.log
```

Host-side log copy command:

```bash
tmux new-session -d -s isaac-step6-copy-reward-smoke-a7144d2 \
  "mkdir -p ~/IsaacLab/aic/logs/aic_rewards; latest=\$(docker exec isaac-lab-base bash -lc \"ls -t /workspace/isaaclab/aic/logs/aic_rewards/*_AIC-Task-v0.log | head -1\"); echo LATEST:\$latest; docker cp isaac-lab-base:\$latest ~/IsaacLab/aic/logs/aic_rewards/; echo STEP6_REWARD_LOG_COPY_EXIT:\$?; sleep 60"
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_rewards/20260510_133629_AIC-Task-v0.log
Successfully copied 5.4kB (transferred 7.17kB) to /var/home/bahw/IsaacLab/aic/logs/aic_rewards/
STEP6_REWARD_LOG_COPY_EXIT:0
```

Key output:

```text
Active Termination Terms:
  time_out: True
  sc_insertion_success: False

Active Reward Terms: 16
analytic_shape_checks_ok: True
overall_finite: True
STEP6_REWARD_SMOKE_EXIT:0
```

Interpretation:

- The current reward/termination configuration is still numerically valid with
  the virtual helper.
- Random actions after reset do not reach insertion rewards, which is expected.
  The scripted check is the evidence that the geometry is controllable.

## Expanded Virtual Tip Scripted Validation

The first `0.07` m helper check used only `4` envs, so the next check expanded
to `16` envs and a longer horizon before restarting PPO.

Host-side command:

```bash
tmux new-session -d -s isaac-step6-scripted-offset7-expanded-a7144d2 \
  "pgrep -af \"check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh\"; echo STEP6_OFFSET7_EXPANDED_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 16 --max_steps 750 --report_every 50 --control_frame tip --align_lateral_threshold 0.05 --approach_depth 0.0 --target_depth 0.020 --headless --enable_cameras\"; echo STEP6_OFFSET7_EXPANDED_SCRIPTED_EXIT:\$?; sleep 60"
```

The run wrote this container log:

```text
/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_133909_AIC-Task-v0.log
```

Host-side log copy command:

```bash
tmux new-session -d -s isaac-step6-copy-expanded-log-a7144d2 \
  "mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert; latest=\$(docker exec isaac-lab-base bash -lc \"ls -t /workspace/isaaclab/aic/logs/aic_scripted_insert/*_AIC-Task-v0.log | head -1\"); echo LATEST:\$latest; docker cp isaac-lab-base:\$latest ~/IsaacLab/aic/logs/aic_scripted_insert/; pgrep -af \"check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh\"; echo STEP6_EXPANDED_LOG_COPY_EXIT:\$?; sleep 60"
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_133909_AIC-Task-v0.log
Successfully copied 5.97kB (transferred 7.68kB) to /var/home/bahw/IsaacLab/aic/logs/aic_scripted_insert/
STEP6_EXPANDED_LOG_COPY_EXIT:0
```

Key output:

```text
step=50 successes=10/16 lateral(mean=0.007478 min=0.001788 max=0.061460) orientation(mean=0.005886 min=0.000000 max=0.023861) depth(mean=0.006869 min=-0.090911 max=0.019767)
step=100 successes=14/16 lateral(mean=0.004173 min=0.002690 max=0.005685) orientation(mean=0.008887 min=0.003906 max=0.017069) depth(mean=0.019778 min=0.011725 max=0.023536)
step=750 successes=14/16 lateral(mean=0.004886 min=0.003990 max=0.007156) orientation(mean=0.023637 min=0.006329 max=0.116680) depth(mean=0.022083 min=0.011652 max=0.026422)

== Summary ==
successes: 14/16
first_success_steps: [50, 37, 51, 51, 50, 52, 48, 45, 42, 51, 45, 47, -1, 47, 48, -1]
per_target:
  sc_port: episodes=8 successes=7 success_rate=0.875000
  sc_port_2: episodes=8 successes=7 success_rate=0.875000
final_lateral: mean=0.004886 min=0.003990 max=0.007156
final_orientation: mean=0.023637 min=0.006329 max=0.116680
final_depth: mean=0.022083 min=0.011652 max=0.026422
gripper_tcp_to_sc_tip_pos_drift: mean=0.000000 min=0.000000 max=0.000000
```

Interpretation:

- Scripted insertion is no longer target-specific broken: both `sc_port` and
  `sc_port_2` reached `7/8` successes.
- The two failed envs were near-threshold cases, not gross misses. Final
  orientation remained comfortably below `0.20` rad; the main miss was lateral
  error slightly above the strict `0.005` m threshold, with minimum depth also
  close to the `0.012` m threshold.
- PPO retraining is justified now because the geometry/action path can generate
  success under a privileged controller. If PPO still fails, the next likely
  work is curriculum or lateral convergence shaping, not another blind reward
  rebalance.

## PPO Smoke After Virtual Tip

Host-side command:

```bash
tmux new-session -d -s isaac-step6-train-virtual-tip-a7144d2 \
  "pgrep -af \"rsl_rl/train.py|check_aic_scripted_insert|isaaclab.sh\"; echo STEP6_TRAIN_VIRTUAL_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 500 --run_name step6_sc_virtual_tip_a7144d2 --headless --enable_cameras\"; echo STEP6_TRAIN_VIRTUAL_EXIT:\$?; sleep 120"
```

Training log directory inside the container:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_13-43-10_step6_sc_virtual_tip_a7144d2
```

The run was manually stopped at iteration `65/500` because the insertion terms
remained zero.

Key output at stop:

```text
Learning iteration 65/500
Episode_Reward/sc_approach: 0.4989
Episode_Reward/sc_coarse_lateral_alignment: 0.7896
Episode_Reward/sc_lateral_alignment: 0.0000
Episode_Reward/sc_insertion_depth: 0.0000
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.0000
STEP6_TRAIN_VIRTUAL_EXIT:0
```

Interpretation:

- The virtual tip fixed controllability for a scripted privileged controller,
  but PPO still did not sample the fine insertion corridor from the normal reset
  distribution.
- The next remediation is a near-port curriculum reset, not another reward-only
  rebalance.

## First-Success Joint Seed Extraction

Commit `8165aee` extended `check_aic_scripted_insert.py` to log the six UR arm
joint positions at each environment's first scripted success and the per-target
mean joint seed.

Host-side sync command:

```bash
tmux new-session -d -s isaac-step6-pull-8165aee \
  "cd ~/IsaacLab/aic && git pull --ff-only && docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py && docker cp docs/bahw_docs/detailed/step6.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/detailed/step6.md && docker cp docs/bahw_docs/plan.md isaac-lab-base:/workspace/isaaclab/aic/docs/bahw_docs/plan.md; echo STEP6_PULL_COPY_8165AEE_EXIT:\$?; sleep 60"
```

Sync result:

```text
STEP6_PULL_COPY_8165AEE_EXIT:0
```

Host-side seed extraction command:

```bash
tmux new-session -d -s isaac-step6-joint-seeds-8165aee \
  "pgrep -af \"check_aic_scripted_insert|rsl_rl/train.py|isaaclab.sh\"; echo STEP6_JOINT_SEEDS_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 16 --max_steps 300 --report_every 50 --control_frame tip --align_lateral_threshold 0.05 --approach_depth 0.0 --target_depth 0.020 --headless --enable_cameras\"; echo STEP6_JOINT_SEEDS_EXIT:\$?; sleep 60"
```

The run wrote this container log:

```text
/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_135208_AIC-Task-v0.log
```

Host-side log copy command:

```bash
tmux new-session -d -s isaac-step6-copy-joint-seeds-log-8165aee \
  "mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert; latest=\$(docker exec isaac-lab-base bash -lc \"ls -t /workspace/isaaclab/aic/logs/aic_scripted_insert/*_AIC-Task-v0.log | head -1\"); echo LATEST:\$latest; docker cp isaac-lab-base:\$latest ~/IsaacLab/aic/logs/aic_scripted_insert/; echo STEP6_JOINT_SEEDS_LOG_COPY_EXIT:\$?; sleep 60"
```

Copy result:

```text
LATEST:/workspace/isaaclab/aic/logs/aic_scripted_insert/20260510_135208_AIC-Task-v0.log
Successfully copied 7.52kB (transferred 9.22kB) to /var/home/bahw/IsaacLab/aic/logs/aic_scripted_insert/
STEP6_JOINT_SEEDS_LOG_COPY_EXIT:0
```

Key output:

```text
successes: 15/16
per_target:
  sc_port: episodes=10 successes=9 success_rate=0.900000
  sc_port_2: episodes=6 successes=6 success_rate=1.000000
arm_joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
first_success_joint_pos_mean_per_target:
  sc_port: [0.8141875863075256, -1.8485052585601807, -1.8315728902816772, -1.0275382995605469, 1.5704457759857178, 2.171452760696411]
  sc_port_2: [0.7603225708007812, -1.8013938665390015, -1.8958141803741455, -1.0111992359161377, 1.570515513420105, 2.1116960048675537]
```

## Near-Port Reset Curriculum

Added `mdp.events.reset_robot_near_sc_port` and wired it into `EventCfg` after
`sample_active_sc_target` and before `reset_sc_progress_buffers`.

Current curriculum parameters:

```python
probability = 1.0
blend = 0.95
position_noise = 0.01
velocity_range = (0.0, 0.0)
```

The event selects the seed for the active SC target, blends from the normal
default arm joints toward the scripted first-success seed, adds small joint
noise, clamps to joint limits, and writes only the six UR arm joints to sim.

This is intentionally a Step 6 curriculum. It is not the final deployment reset
distribution. Once the policy learns final insertion, reduce curriculum strength
or stage back toward the normal reset distribution.

First reset smoke with `blend=0.85`, `position_noise=0.015`:

```text
step=0 successes=0/16
final_lateral: mean=0.058786 min=0.048729 max=0.071984
final_orientation: mean=0.016874 min=0.007143 max=0.030467
final_depth: mean=-0.025474 min=-0.036422 max=-0.013701
```

Interpretation: the event was wired correctly, but the reset was still too far
outside the fine insertion corridor. The curriculum was tightened to
`blend=0.95`, `position_noise=0.01` before the next smoke test.

Second reset smoke with `blend=0.95`, `position_noise=0.01`:

```text
step=0 successes=0/16
final_lateral: mean=0.023949 min=0.015067 max=0.034375
final_orientation: mean=0.009376 min=0.002013 max=0.019501
final_depth: mean=0.001344 min=-0.004707 max=0.005534
```

Scripted insertion from that reset:

```text
step=0 successes=0/16 lateral(mean=0.021286 min=0.014807 max=0.030629) orientation(mean=0.009376 min=0.000000 max=0.023996) depth(mean=0.001533 min=-0.009456 max=0.005629)
step=6 successes=16/16 lateral(mean=0.003078 min=0.002238 max=0.004462) orientation(mean=0.004618 min=0.000000 max=0.012563) depth(mean=0.016588 min=0.013244 max=0.018757)
```

The curriculum reset is therefore close enough for a short final insertion
controller, but it does not start in the success state.

PPO retry from this reset:

```bash
tmux new-session -d -s isaac-step6-train-near-reset-66e3aac \
  "pgrep -af \"rsl_rl/train.py|check_aic_scripted_insert|isaaclab.sh\"; echo STEP6_TRAIN_NEAR_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 500 --run_name step6_sc_near_reset_66e3aac --headless --enable_cameras\"; echo STEP6_TRAIN_NEAR_EXIT:\$?; sleep 120"
```

The run was manually stopped at iteration `25/500`. It had nonzero insertion
samples at the start, but the signal did not improve:

```text
Learning iteration 0/500
Episode_Reward/sc_insertion_depth: 0.0005
Episode_Reward/sc_insertion_success: 0.0016
Episode_Termination/sc_insertion_success: 0.0150

Learning iteration 25/500
Mean action std: 1.00
Episode_Reward/sc_insertion_depth: 0.0000
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.0156
```

Interpretation: near-port samples exist, but the initial PPO action standard
deviation is too large for a millimeter-scale final insertion problem. The next
SC PPO config change reduces actor `init_std` from `1.0` to `0.2` and entropy
coefficient from `0.006` to `0.001`.

Reduced-std PPO retry:

```bash
tmux new-session -d -s isaac-step6-train-near-std02-clean-524e548 \
  "pgrep -af \"rsl_rl/train.py|isaaclab.sh\"; echo STEP6_TRAIN_STD02_CLEAN_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 250 --run_name step6_sc_near_reset_std02_clean_524e548 --headless --enable_cameras\"; echo STEP6_TRAIN_STD02_CLEAN_EXIT:\$?; sleep 120"
```

Before this clean run, stale Isaac processes from earlier interrupted runs were
holding about `20` GB of GPU memory and caused the first reduced-std launch to
fail with PhysX/Vulkan out-of-memory errors. They were removed from inside the
container with:

```bash
docker exec isaac-lab-base bash -lc \
  "pkill -INT -f rsl_rl/train.py || true; sleep 5; pkill -TERM -f rsl_rl/train.py || true"
```

The clean reduced-std run was manually stopped at iteration `70/250`. It
improved the initial signal but still plateaued:

```text
Learning iteration 0/250
Mean action std: 0.20
Episode_Reward/sc_insertion_depth: 0.0014
Episode_Reward/sc_insertion_success: 0.0014
Episode_Termination/sc_insertion_success: 0.0495

Learning iteration 1/250
Episode_Termination/sc_insertion_success: 0.1250

Learning iteration 70/250
Episode_Reward/sc_insertion_depth: 0.0015
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.1250
STEP6_TRAIN_STD02_CLEAN_EXIT:0
```

Interpretation:

- Reduced exploration helped: success termination rose from about `0.05` to
  `0.125`.
- It did not improve beyond the initial curriculum success rate.
- The actor is still trying to solve final insertion under randomized board/SC
  port positions from eval-compatible observations. Next curriculum stage freezes
  board and SC port randomization so the policy can first learn a fixed final
  insertion behavior for the two active SC targets.

## Fixed-Port Curriculum Stage

Temporary Step 6 curriculum change:

```python
board_range = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
sc_port.pose_range = {"x": (0.0, 0.0)}
sc_port_2.pose_range = {"x": (0.0, 0.0)}
```

This intentionally freezes SC port randomization while keeping active target
sampling between `sc_port` and `sc_port_2`. After the policy learns the final
fixed-port insertion, reintroduce board/SC port randomization gradually.

Fixed-port reset smoke command:

```bash
tmux new-session -d -s isaac-step6-fixed-reset-smoke-a219974 \
  "docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-Task-v0 --num_envs 16 --max_steps 0 --report_every 1 --control_frame tip --align_lateral_threshold 0.05 --approach_depth 0.0 --target_depth 0.020 --headless --enable_cameras\"; echo STEP6_FIXED_RESET_SMOKE_EXIT:\$?; sleep 60"
```

Reset smoke output:

```text
step=0 successes=0/16
final_lateral: mean=0.024799 min=0.018159 max=0.037821
final_orientation: mean=0.010348 min=0.002013 max=0.017438
final_depth: mean=0.000837 min=-0.008100 max=0.005816
```

Fixed-port PPO command:

```bash
tmux new-session -d -s isaac-step6-train-fixed-std02-a219974 \
  "pgrep -af \"rsl_rl/train.py|isaaclab.sh\"; echo STEP6_TRAIN_FIXED_STALE_BEFORE_EXIT:\$?; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 250 --run_name step6_sc_fixed_std02_a219974 --headless --enable_cameras\"; echo STEP6_TRAIN_FIXED_STD02_EXIT:\$?; sleep 120"
```

The fixed-port PPO run was manually stopped at iteration `110/250` because it
plateaued:

```text
Learning iteration 0/250
Episode_Termination/sc_insertion_success: 0.0872

Learning iteration 28/250
Episode_Termination/sc_insertion_success: 0.2656

Learning iteration 110/250
Mean action std: 0.20
Episode_Reward/sc_insertion_depth: 0.0000
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.2656
STEP6_TRAIN_FIXED_STD02_EXIT:0
```

After stopping the tmux session, the Isaac Kit process remained alive and held
GPU memory. It was cleaned up with a targeted stale-process kill:

```bash
tmux new-session -d -s isaac-step6-stale-kill-a219974 \
  "docker exec isaac-lab-base bash -lc \"kill -TERM 12706 || true; sleep 5; kill -KILL 12706 2>/dev/null || true; ps -eo pid,ppid,stat,cmd | grep -E \\\"step6_sc_fixed_std02_a219974|rsl_rl/train.py|kit/python/bin/python3\\\" | grep -v grep || true\"; nvidia-smi; echo STEP6_STALE_KILL_EXIT:\$?; sleep 60"
```

Cleanup result:

```text
GPU memory after cleanup: 491MiB / 24564MiB
Processes: gnome-shell and sunshine-kms only
STEP6_STALE_KILL_EXIT:0
```

Interpretation:

- Freezing SC port randomization improved the early success sample rate from the
  reduced-std near-port curriculum, but PPO still did not learn reliable final
  insertion.
- The final step from near-port pose to successful insertion is too narrow for
  reward-only PPO under the current actor exploration.
- The next remediation is a privileged scripted-action-prior reward that teaches
  the final relative-IK action without adding privileged geometry to actor
  observations.

## Privileged Scripted-Action Prior

Added a Step 6 teacher/curriculum reward:

```python
sc_scripted_action_prior = RewTerm(
    func=mdp.sc_scripted_action_prior_reward,
    weight=5.0,
)
```

What it does:

- Computes the same raw relative-IK `arm_action` used by the successful
  `check_aic_scripted_insert.py --control_frame tip` controller.
- Uses privileged SC geometry only inside the reward:
  - virtual/gripped SC tip pose
  - active SC port entrance pose
  - active SC port insertion axis
  - lateral/orientation thresholds for switching from entrance approach to
    positive insertion depth
- Compares that desired raw action against
  `env.action_manager.get_term("arm_action").raw_actions`.
- Uses a dense `1 - tanh(action_error / 1.0)` kernel instead of a sparse
  success-only signal.
- Caches the desired scripted action at reset and after each reward call so the
  reward compares the action just taken against the prior for the state it was
  taken from, not the post-step state.

This is intentionally train-only shaping for the Step 6 teacher. The actor
observation group remains eval-compatible and still does not receive hidden
plug-to-port geometry.

Local checks before the remote Isaac smoke:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
git diff --check
```

Local result:

```text
py_compile: passed
git diff --check: passed
```

Next remote checks:

```bash
tmux new-session -d -s isaac-step6-reward-prior-smoke-<commit> \
  "docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 16 --num_steps 4 --headless --enable_cameras\"; echo STEP6_REWARD_PRIOR_SMOKE_EXIT:\$?; sleep 60"
```

If the reward smoke passes, retry PPO from the fixed-port near-reset curriculum:

```bash
tmux new-session -d -s isaac-step6-train-action-prior-<commit> \
  "docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 250 --run_name step6_sc_action_prior_<commit> --headless --enable_cameras\"; echo STEP6_TRAIN_ACTION_PRIOR_EXIT:\$?; sleep 120"
```

Reward-prior implementation commit:

```text
12462ab Add SC scripted action prior
```

Host pull:

```bash
tmux new-session -d -s isaac-step6-pull-12462ab \
  "cd ~/IsaacLab/aic && git status --short && git pull --ff-only && git rev-parse --short HEAD; echo STEP6_PULL_12462AB_EXIT:\$?; sleep 60"
```

Pull result:

```text
Updating a219974..12462ab
Fast-forward
12462ab
STEP6_PULL_12462AB_EXIT:0
```

Container copy:

```bash
tmux new-session -d -s isaac-step6-copy-12462ab \
  "cd ~/IsaacLab/aic && for p in aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py docs/bahw_docs/README.md docs/bahw_docs/detailed/step6.md docs/bahw_docs/plan.md; do docker cp \"\$p\" isaac-lab-base:/workspace/isaaclab/aic/\"\$p\" || exit 1; echo COPIED:\$p; done; echo STEP6_COPY_12462AB_EXIT:\$?; sleep 60"
```

Copy result:

```text
STEP6_COPY_12462AB_EXIT:0
```

Reward smoke command:

```bash
tmux new-session -d -s isaac-step6-reward-prior-smoke-12462ab \
  "pgrep -af \"rsl_rl/train.py|check_aic_rewards.py|isaaclab.sh\"; echo STEP6_REWARD_PRIOR_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-Task-v0 --num_envs 16 --num_steps 4 --headless --enable_cameras\"; echo STEP6_REWARD_PRIOR_SMOKE_EXIT:\$?; sleep 120"
```

Reward smoke output:

```text
Active Event Terms:
  reset_sc_scripted_action_prior_buffer

Active Reward Terms:
  sc_scripted_action_prior weight=5.0

gym observation space:
  policy shape: (16, 3149)
  privileged shape: (16, 20)

sc_scripted_action_prior after reset:
  finite=True mean=0.512493 min=0.477474 max=0.533042

sc_scripted_action_prior after random steps:
  step 00 finite=True mean=0.425547
  step 01 finite=True mean=0.395623
  step 02 finite=True mean=0.416532
  step 03 finite=True mean=0.419703

overall_finite: True
STEP6_REWARD_PRIOR_SMOKE_EXIT:0
```

Action-prior PPO command:

```bash
tmux new-session -d -s isaac-step6-train-action-prior-12462ab \
  "pgrep -af \"rsl_rl/train.py|check_aic_rewards.py|isaaclab.sh\"; echo STEP6_TRAIN_ACTION_PRIOR_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_iterations 250 --run_name step6_sc_action_prior_12462ab --headless --enable_cameras\"; echo STEP6_TRAIN_ACTION_PRIOR_EXIT:\$?; sleep 120"
```

Action-prior PPO output:

```text
Learning iteration 0/250
Episode_Reward/sc_scripted_action_prior: 0.0019
Episode_Reward/sc_insertion_depth: 0.0013
Episode_Reward/sc_insertion_success: 0.0013
Episode_Termination/sc_insertion_success: 0.0716

Learning iteration 1/250
Episode_Reward/sc_scripted_action_prior: 0.0026
Episode_Termination/sc_insertion_success: 0.1406

Learning iteration 25/250
Episode_Reward/sc_scripted_action_prior: 0.0021
Episode_Reward/sc_insertion_depth: 0.0000
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.1406

Learning iteration 30/250
Episode_Reward/sc_scripted_action_prior: 0.0025
Episode_Reward/sc_insertion_depth: 0.0000
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/sc_insertion_success: 0.1406
STEP6_TRAIN_ACTION_PRIOR_EXIT:0
```

Interpretation:

- The action-prior reward is implemented correctly enough to load, stay finite,
  and appear in RSL-RL logs.
- As a scalar PPO reward, it did not teach the final insertion action. It
  plateaued below the previous fixed-port run's `0.2656` success termination.
- More reward-only tuning is unlikely to be the highest-leverage next step.
- Next Step 6 work should inspect whether we can directly bootstrap the actor
  from scripted `(policy observation, action)` pairs, then resume PPO. This keeps
  the actor eval-compatible while using privileged geometry only as the
  offline/training label generator.

Cleanup after the interrupted run:

```bash
tmux new-session -d -s isaac-step6-action-prior-clean-final-12462ab \
  "sleep 10; docker exec isaac-lab-base bash -lc \"pkill -TERM -f step6_sc_action_prior_12462ab || true; sleep 2; ps -eo pid,ppid,stat,cmd | grep -E \\\"step6_sc_action_prior_12462ab|rsl_rl/train.py|kit/python/bin/python3\\\" | grep -v grep || true\"; nvidia-smi; echo STEP6_ACTION_PRIOR_CLEAN_FINAL_EXIT:\$?; sleep 60"
```

Cleanup result:

```text
GPU memory after cleanup: 491MiB / 24564MiB
Processes: gnome-shell and sunshine-kms only
STEP6_ACTION_PRIOR_CLEAN_FINAL_EXIT:0
```

## Scripted Actor Bootstrap

Reason for the next remediation:

- Scripted IK can solve the final insertion from the near-port curriculum.
- PPO reward-only attempts, including a scalar scripted-action-prior reward,
  still do not reliably learn that narrow action sequence.
- RSL-RL has normal checkpoint resume, but this AIC workspace does not have a
  built-in behavior-cloning or imitation-learning hook.

Added:

```text
aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py
```

The script builds the same `OnPolicyRunner` and actor architecture used by
`rsl_rl/train.py`, trains only the actor with supervised MSE against scripted raw
`arm_action` labels, and saves a normal RSL-RL checkpoint. PPO can then resume
from that checkpoint with:

```bash
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --resume --checkpoint <bc_checkpoint> \
  --num_envs 64 --max_iterations 250 --run_name step6_sc_bc_resume_<commit> \
  --headless --enable_cameras
```

The actor inputs remain eval-compatible because the BC loss uses the normal
RSL-RL actor observation group, which is still `["policy"]`. Privileged geometry
is used only to generate the supervised action labels through:

```text
mdp.sc_scripted_raw_action(...)
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
git diff --check
```

Local result:

```text
py_compile: passed
git diff --check: passed
```

Planned smoke command:

```bash
tmux new-session -d -s isaac-step6-bc-smoke-<commit> \
  "docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 16 --max_updates 10 --report_every 1 --save_every 0 --run_name step6_sc_bc_smoke_<commit> --headless --enable_cameras\"; echo STEP6_BC_SMOKE_EXIT:\$?; sleep 120"
```

If the smoke passes, run a longer BC pass, evaluate the saved checkpoint, then
resume PPO from it.

## BC Pretrain Results

Remote pull/copy for the save fix:

```bash
tmux new-session -d -s isaac-step6-pull-copy-8a19d2a \
  "cd ~/IsaacLab/aic && git status --short && git pull --ff-only && git rev-parse --short HEAD && docker cp aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py; echo STEP6_PULL_COPY_8A19D2A_EXIT:\$?; sleep 60"
```

Smoke command:

```bash
tmux new-session -d -s isaac-step6-bc-smoke-8a19d2a-rerun \
  "pgrep -af \"rsl_rl/train.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_SMOKE_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 16 --max_updates 10 --report_every 1 --save_every 0 --run_name step6_sc_bc_smoke_8a19d2a --headless --enable_cameras\"; echo STEP6_BC_SMOKE_EXIT:\$?; sleep 120"
```

Smoke result:

```text
update=10 loss=0.110739 pred_error_mean=0.804083 successes=0/16 lateral_mean=0.014871 orientation_mean=0.013553 depth_mean=0.007266
final_checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-02-23_step6_sc_bc_smoke_8a19d2a/model_10.pt
STEP6_BC_SMOKE_EXIT:0
```

Long expert-rollout BC command:

```bash
tmux new-session -d -s isaac-step6-bc-train-8a19d2a \
  "pgrep -af \"rsl_rl/train.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_TRAIN_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --max_updates 1000 --report_every 50 --save_every 250 --run_name step6_sc_bc_8a19d2a --headless --enable_cameras\"; echo STEP6_BC_TRAIN_EXIT:\$?; sleep 120"
```

Long expert-rollout BC result:

```text
update=1000 loss=0.003010 pred_error_mean=0.099506 successes=0/64 lateral_mean=0.014043 orientation_mean=0.014401 depth_mean=0.007004
final_checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-03-31_step6_sc_bc_8a19d2a/model_1000.pt
elapsed_seconds: 287.72
STEP6_BC_TRAIN_EXIT:0
```

Note: the inline `successes=...` in `pretrain_sc_bc.py` is not an episode-level
success metric. The script steps the env and then samples the geometry state, so
success terminations and resets can make this print misleading. Use
`scripts/rsl_rl/evaluate.py` for policy metrics.

BC checkpoint evaluation command:

```bash
tmux new-session -d -s isaac-step6-bc-eval-300-8a19d2a \
  "pgrep -af \"rsl_rl/evaluate.py|rsl_rl/train.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_EVAL_300_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --num_eval_episodes 256 --max_episode_steps 300 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-03-31_step6_sc_bc_8a19d2a/model_1000.pt --headless --enable_cameras\"; echo STEP6_BC_EVAL_300_EXIT:\$?; sleep 120"
```

BC checkpoint evaluation result:

```text
progress: 64/256 successes=49
progress: 128/256 successes=97
progress: 192/256 successes=145
progress: 256/256 successes=193

episodes: 256
successes: 193
success_rate: 0.753906
mean_episode_length: 79.527
mean_episode_length_on_success: 7.560
mean_lateral_error_at_termination: 0.009816
mean_orientation_error_at_termination: 0.033182
mean_insertion_depth_at_termination: 0.018228
failure_breakdown:
  timeout: 63
  lateral_miss: 63
  orientation_miss: 2
  depth_shortfall: 9
per_target:
  sc_port: episodes=134 successes=95 success_rate=0.708955
  sc_port_2: episodes=122 successes=98 success_rate=0.803279
STEP6_BC_EVAL_300_EXIT:0
```

PPO resume attempt:

```bash
tmux new-session -d -s isaac-step6-bc-resume-ppo-8a19d2a-v2 \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_RESUME_PPO_V2_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_15-03-31_step6_sc_bc_8a19d2a --checkpoint model_1000.pt --num_envs 64 --max_iterations 250 --run_name step6_sc_bc_resume_8a19d2a_v2 --headless --enable_cameras\"; echo STEP6_BC_RESUME_PPO_V2_EXIT:\$?; sleep 120"
```

Important resume detail: `train.py` uses RSL-RL `get_checkpoint_path`, so resume
must pass `--load_run <run_dir> --checkpoint <file_name>`. Passing the absolute
checkpoint path to `--checkpoint` failed.

PPO result at the first saved checkpoint:

```text
model_1050.pt saved under:
/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-18-14_step6_sc_bc_resume_8a19d2a_v2/model_1050.pt

training at iteration 1050:
Episode_Reward/sc_insertion_depth: 0.0022
Episode_Reward/sc_insertion_success: 0.0000
Episode_Termination/time_out: 0.0814
Episode_Termination/sc_insertion_success: 0.2500
```

Evaluation of `model_1050.pt` with the same 300-step gate started at `0/128`
successes, so the PPO-resumed checkpoint is worse than the BC checkpoint and
should not be preferred.

## Actor-Rollout BC Remediation

Reason:

- Expert-rollout BC produced a useful deterministic policy, but still missed
  laterally in `63/256` evaluation episodes.
- PPO resume from the useful BC checkpoint quickly regressed.
- The likely remaining BC failure is distribution shift: the actor is trained on
  states reached by perfect scripted actions, then evaluation uses states reached
  by the actor's own imperfect actions.

Added to `pretrain_sc_bc.py`:

```text
--rollout_policy expert|actor|blend
--rollout_actor_weight <float>
```

`expert` preserves the previous behavior. `actor` steps the environment with the
actor's deterministic action while still labeling each visited state with the
scripted privileged controller. `blend` linearly mixes scripted and actor
actions. The next run should resume from the useful BC checkpoint and use
`--rollout_policy actor`.

Actor-rollout smoke command:

```bash
tmux new-session -d -s isaac-step6-bc-actor-smoke-e88ad49 \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_ACTOR_SMOKE_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_15-03-31_step6_sc_bc_8a19d2a --checkpoint model_1000.pt --num_envs 16 --max_updates 20 --report_every 5 --save_every 0 --learning_rate 3e-4 --rollout_policy actor --run_name step6_sc_bc_actor_smoke_e88ad49 --headless --enable_cameras\"; echo STEP6_BC_ACTOR_SMOKE_EXIT:\$?; sleep 120"
```

Actor-rollout smoke result:

```text
rollout_policy: actor
resume_path: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-03-31_step6_sc_bc_8a19d2a/model_1000.pt
update=20 loss=0.039893 pred_error_mean=0.470724 successes=0/16 lateral_mean=0.013944 orientation_mean=0.018623 depth_mean=0.017231
final_checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-32-05_step6_sc_bc_actor_smoke_e88ad49/model_20.pt
STEP6_BC_ACTOR_SMOKE_EXIT:0
```

Long actor-rollout BC command:

```bash
tmux new-session -d -s isaac-step6-bc-actor-train-e88ad49 \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_ACTOR_TRAIN_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_15-03-31_step6_sc_bc_8a19d2a --checkpoint model_1000.pt --num_envs 64 --max_updates 1000 --report_every 50 --save_every 250 --learning_rate 3e-4 --rollout_policy actor --run_name step6_sc_bc_actor_e88ad49 --headless --enable_cameras\"; echo STEP6_BC_ACTOR_TRAIN_EXIT:\$?; sleep 120"
```

Long actor-rollout BC result:

```text
log_dir: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-34-12_step6_sc_bc_actor_e88ad49
rollout_policy: actor
update=1000 loss=0.234899 pred_error_mean=1.129753 successes=0/64 lateral_mean=0.031516 orientation_mean=0.112044 depth_mean=0.013553
final_checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-34-12_step6_sc_bc_actor_e88ad49/model_1000.pt
elapsed_seconds: 189.65
STEP6_BC_ACTOR_TRAIN_EXIT:0
```

Actor-rollout BC evaluation command:

```bash
tmux new-session -d -s isaac-step6-bc-actor-eval-e88ad49 \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_ACTOR_EVAL_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --num_eval_episodes 256 --max_episode_steps 300 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-34-12_step6_sc_bc_actor_e88ad49/model_1000.pt --headless --enable_cameras\"; echo STEP6_BC_ACTOR_EVAL_EXIT:\$?; sleep 120"
```

Actor-rollout BC evaluation result:

```text
progress: 64/256 successes=0
progress: 128/256 successes=0
progress: 192/256 successes=0
progress: 256/256 successes=0

episodes: 256
successes: 0
success_rate: 0.000000
mean_episode_length: 300.000
mean_lateral_error_at_termination: 0.307786
mean_orientation_error_at_termination: 1.494016
mean_insertion_depth_at_termination: -0.383904
failure_breakdown:
  timeout: 256
  lateral_miss: 256
  orientation_miss: 253
  depth_shortfall: 254
per_target:
  sc_port: episodes=134 successes=0 success_rate=0.000000
  sc_port_2: episodes=122 successes=0 success_rate=0.000000
STEP6_BC_ACTOR_EVAL_EXIT:0
```

Conclusion:

- Actor-rollout BC at learning rate `3e-4` is not an improvement over the
  expert-rollout BC checkpoint. It pushes the policy far outside the insertion
  corridor and should not be preferred.
- The current best checkpoint remains:
  `/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-03-31_step6_sc_bc_8a19d2a/model_1000.pt`.
- Before another training run, add evaluator diagnostics for signed lateral
  components, optional terminal actor-vs-scripted action error, and failure
  samples. Use those diagnostics on the useful expert-rollout BC checkpoint to
  decide whether the remaining misses are systematic calibration bias,
  target-specific geometry bias, or recovery failure.

## Diagnostic And Strict-Alignment BC Results

Commit `2e5987b` added evaluator diagnostics:

```text
--action_error_diagnostics
--failure_sample_count <N>
```

The evaluator now reports signed lateral components in the active SC port frame,
optional terminal actor-vs-scripted action error, sample failure rows, and
per-target terminal means.

Diagnostic repeat of the useful expert-rollout BC checkpoint:

```bash
tmux new-session -d -s isaac-step6-bc-best-diag-2e5987b \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_BEST_DIAG_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --num_envs 64 --num_eval_episodes 256 --max_episode_steps 300 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-03-31_step6_sc_bc_8a19d2a/model_1000.pt --action_error_diagnostics --failure_sample_count 12 --headless --enable_cameras\"; echo STEP6_BC_BEST_DIAG_EXIT:\$?; sleep 120"
```

Result:

```text
successes: 189/256
success_rate: 0.738281
mean_lateral_error_at_termination: 0.009456
mean_signed_lateral_x_at_termination: -0.003685
mean_signed_lateral_z_at_termination: -0.004418
mean_terminal_action_error_vs_scripted: 0.470200
failure_breakdown:
  timeout: 67
  lateral_miss: 67
  orientation_miss: 4
  depth_shortfall: 11
per_target:
  sc_port: episodes=134 successes=90 success_rate=0.671642
  sc_port_2: episodes=122 successes=99 success_rate=0.811475
STEP6_BC_BEST_DIAG_EXIT:0
```

Interpretation: failures were still mostly lateral misses with a consistent
negative signed lateral bias in both port-frame axes.

Blended DAgger-style BC with a gentler actor mix:

```bash
tmux new-session -d -s isaac-step6-bc-blend-train-2e5987b \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_BLEND_TRAIN_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_15-03-31_step6_sc_bc_8a19d2a --checkpoint model_1000.pt --num_envs 64 --max_updates 1000 --report_every 50 --save_every 250 --learning_rate 1e-4 --rollout_policy blend --rollout_actor_weight 0.1 --run_name step6_sc_bc_blend_2e5987b --headless --enable_cameras\"; echo STEP6_BC_BLEND_TRAIN_EXIT:\$?; sleep 120"
```

Evaluation result:

```text
checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_15-54-03_step6_sc_bc_blend_2e5987b/model_1000.pt
successes: 195/256
success_rate: 0.761719
mean_lateral_error_at_termination: 0.007967
mean_signed_lateral_x_at_termination: -0.002007
mean_signed_lateral_z_at_termination: -0.004785
failure_breakdown:
  timeout: 61
  lateral_miss: 61
  orientation_miss: 3
  depth_shortfall: 6
STEP6_BC_BLEND_EVAL_EXIT:0
```

This was only a small improvement over the original expert-rollout checkpoint.

Strict scripted alignment BC:

```bash
tmux new-session -d -s isaac-step6-bc-strict-train-2e5987b \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_STRICT_TRAIN_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_15-03-31_step6_sc_bc_8a19d2a --checkpoint model_1000.pt --num_envs 64 --max_updates 1000 --report_every 50 --save_every 250 --learning_rate 1e-4 --rollout_policy blend --rollout_actor_weight 0.1 --align_lateral_threshold 0.01 --align_orientation_threshold 0.20 --run_name step6_sc_bc_strict_2e5987b --headless --enable_cameras\"; echo STEP6_BC_STRICT_TRAIN_EXIT:\$?; sleep 120"
```

Evaluation result:

```text
checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_16-04-17_step6_sc_bc_strict_2e5987b/model_1000.pt
successes: 233/256
success_rate: 0.910156
mean_episode_length: 35.016
mean_episode_length_on_success: 8.858
mean_lateral_error_at_termination: 0.004744
mean_signed_lateral_x_at_termination: 0.001731
mean_signed_lateral_z_at_termination: -0.002253
mean_orientation_error_at_termination: 0.013835
mean_insertion_depth_at_termination: 0.017545
mean_terminal_action_error_vs_scripted: 0.359335
failure_breakdown:
  timeout: 23
  lateral_miss: 23
  orientation_miss: 0
  depth_shortfall: 5
per_target:
  sc_port: episodes=134 successes=124 success_rate=0.925373
  sc_port_2: episodes=122 successes=109 success_rate=0.893443
STEP6_BC_STRICT_EVAL_EXIT:0
```

This is the current best SC checkpoint, but Step 7 remains gated because `23/256`
episodes still timed out and missed laterally under the fixed 300-step gate.

Failed refinement:

```bash
tmux new-session -d -s isaac-step6-bc-refine-train-2e5987b \
  "pgrep -af \"rsl_rl/train.py|rsl_rl/evaluate.py|pretrain_sc_bc.py|isaaclab.sh\"; echo STEP6_BC_REFINE_TRAIN_STALE_BEFORE_EXIT:\$?; nvidia-smi; docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/pretrain_sc_bc.py --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point --resume --load_run 2026-05-10_16-04-17_step6_sc_bc_strict_2e5987b --checkpoint model_1000.pt --num_envs 64 --max_updates 500 --report_every 50 --save_every 250 --learning_rate 5e-5 --rollout_policy blend --rollout_actor_weight 0.1 --align_lateral_threshold 0.005 --align_orientation_threshold 0.20 --target_depth 0.025 --run_name step6_sc_bc_strict_refine_2e5987b --headless --enable_cameras\"; echo STEP6_BC_REFINE_TRAIN_EXIT:\$?; sleep 120"
```

Evaluation result:

```text
checkpoint: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_16-13-10_step6_sc_bc_strict_refine_2e5987b/model_500.pt
successes: 0/256
success_rate: 0.000000
mean_lateral_error_at_termination: 0.027971
mean_signed_lateral_x_at_termination: 0.007720
mean_signed_lateral_z_at_termination: -0.026474
mean_insertion_depth_at_termination: 0.028446
failure_breakdown:
  timeout: 256
  lateral_miss: 256
  orientation_miss: 119
  depth_shortfall: 0
STEP6_BC_REFINE_EVAL_EXIT:0
```

Do not prefer the refinement checkpoint. It inserts deeply but creates a large
negative lateral-z bias.
