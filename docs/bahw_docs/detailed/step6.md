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
