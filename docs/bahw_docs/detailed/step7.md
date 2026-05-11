# Step 7: Extend The MDP To SFP

Date: 2026-05-11

Commit under test:

- `ffdc704 Add SFP insertion task interface`

Decision carried in from Step 6:

- The current best SC gate is the strict-label BC neural actor at `233/256`
  deterministic successes (`0.910156`).
- This is accepted as enough to unblock SFP work because the user accepted
  `>90%` plus a saved video artifact.
- PPO remains the preferred final/generalizable specialist policy path. The BC
  result is a practical SC gate and diagnostic, not the final training choice.

## Local Changes

Implemented SFP task support using the same MDP shape as SC:

- Added SFP geometry helpers in `mdp/geometry.py`.
- Added SFP privileged observation terms in `mdp/observations.py`.
- Added SFP reward terms in `mdp/rewards.py`.
- Added `sfp_insertion_success` in `mdp/terminations.py`.
- Added `AICTaskSfpEnvCfg` plus SFP events, rewards, observations, and
  terminations in `aic_task_env_cfg.py`.
- Registered `AIC-SFP-Task-v0` in `aic_task/__init__.py`.
- Added SFP RSL-RL config in `agents/rsl_rl_ppo_sfp_cfg.py`.
- Extended `inspect_aic_geometry.py` and `check_aic_rewards.py` so they can
  exercise the SFP task.

Geometry choices:

- SFP plug tip uses runtime body `robot.sfp_tip_link`.
- SFP port entrances are not runtime rigid bodies, so the helper derives fixed
  offsets from `nic_card`.
- Port offsets came from `aic_assets/models/NIC Card/model.sdf`:
  - port 0 local pose: `(0.01295, -0.031572, 0.00501)`,
    rpy `(4.69895, 0, 0)`
  - port 1 local pose: `(-0.01025, -0.031572, 0.00501)`,
    rpy `(4.69895, 0, 0)`
  - entrance offset: `(0, 0, -0.0458)`
  - insertion axis local: `(0, 0, 1)`
- The SFP plug axis helper currently uses local `(0, 0, -1)` so that the
  helper axis aligns with the derived port insertion axis.

Important caveat:

- Step 0 confirmed `sfp_tip_link` exists as a runtime body.
- Step 6 showed that body-name presence alone is not enough for SC, because
  `sc_tip_link` did not behave as the controlled gripped insertion point.
- Before long SFP training, repeat the Step 6 drift/scripted-control diagnostic
  for SFP. If `sfp_tip_link` is not controllable, switch SFP to a virtual TCP
  helper offset before training.

## Local Checks

Run from local repo root:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

## Push And Host Pull

Run locally:

```bash
git add \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
git commit -m "Add SFP insertion task interface"
git push
```

Result:

```text
[dev/stage0 ffdc704] Add SFP insertion task interface
 10 files changed, 932 insertions(+), 1 deletion(-)
 create mode 100644 aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
```

Run on the host in tmux session `isaac-step7-sfp-pull-ffdc704`:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py
```

Result:

```text
From github.com:BHAW-Intrinsic/aic
   ccb6b7f..ffdc704  dev/stage0 -> origin/dev/stage0
Updating ccb6b7f..ffdc704
Fast-forward
STEP7_SFP_PULL_COPY_EXIT:0
```

## SFP Task Registration Smoke

Run on the host in tmux session `isaac-step7-list-envs2-ffdc704`:

```bash
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py"
```

Result:

```text
STEP7_LIST_ENVS_EXIT:0
```

The captured pane mostly contained Isaac startup warnings, but the process
exited successfully after loading the task registry.

## SFP Geometry Inspect

Run on the host in tmux session `isaac-step7-sfp-inspect-ffdc704`:

```bash
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py --task AIC-SFP-Task-v0 --num_envs 4 --headless --enable_cameras"
```

Result:

```text
STEP7_SFP_INSPECT_EXIT:0
Log path: /workspace/isaaclab/aic/logs/aic_geometry/20260511_042247_AIC-SFP-Task-v0.log
```

Important output:

```text
sfp_tip_link:
  runtime body: robot.sfp_tip_link
  USD prims: /World/envs/env_*/Robot/cable/sfp_module/sfp_tip_link

sfp_port_0_link_entrance:
  runtime matches: none
  USD prims: /World/envs/env_*/nic_card/sfp_port_0_link/sfp_port_0_link_entrance

sfp_port_1_link_entrance:
  runtime matches: none
  USD prims: /World/envs/env_*/nic_card/sfp_port_1_link/sfp_port_1_link_entrance

active_sfp_target_ids: [0, 1, 0, 0]
active_sfp_target_names: ['sfp_port_0', 'sfp_port_1', 'sfp_port_0', 'sfp_port_0']

sfp_plug_tip_pos_w shape: (4, 3)
sfp_plug_tip_quat_w shape: (4, 4)
sfp_port_entry_pos_w shape: (4, 3)
sfp_port_entry_quat_w shape: (4, 4)
sfp_plug_axis_w shape: (4, 3)
sfp_port_insertion_axis_w shape: (4, 3)
sfp_plug_to_port shape: (4, 3)
sfp_lateral_error shape: (4,)
sfp_insertion_depth shape: (4,)
sfp_orientation_error shape: (4,)
```

Environment 0 helper values:

```text
sfp_plug_tip_pos_w: [2.231548, -2.208287, 0.241388]
sfp_port_entry_pos_w: [2.238400, -1.713274, 0.151672]
sfp_plug_axis_w: [0.005682, -0.302441, -0.953151]
sfp_port_insertion_axis_w: [0.000000, 0.012642, -0.999920]
sfp_lateral_error: 0.493887
sfp_insertion_depth: -0.095966
sfp_orientation_error: 0.319948

sfp_port_0 entry: [2.238400, -1.713274, 0.151672]
sfp_port_1 entry: [2.261600, -1.713274, 0.151672]
both insertion axes: [0.000000, 0.012642, -0.999920]
```

## SFP Reward Check

Run on the host in tmux session `isaac-step7-sfp-reward-ffdc704`:

```bash
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-SFP-Task-v0 --num_envs 8 --num_steps 3 --headless --enable_cameras"
```

Result:

```text
STEP7_SFP_REWARD_EXIT:0
Log path: /workspace/isaaclab/aic/logs/aic_rewards/20260511_042323_AIC-SFP-Task-v0.log
```

Important output:

```text
Event terms:
  reset_robot_joints
  randomize_light
  randomize_board_and_parts
  sample_active_sfp_target
  reset_sfp_progress_buffers

Observation groups:
  policy: (3149,)
  privileged: (20,)

Termination terms:
  time_out
  sfp_insertion_success

Reward terms:
  sfp_approach
  sfp_distance_progress
  sfp_lateral_progress
  sfp_orientation_progress
  sfp_depth_progress
  sfp_coarse_lateral_alignment
  sfp_coarse_orientation_alignment
  sfp_lateral_alignment
  sfp_orientation_alignment
  sfp_insertion_depth
  sfp_insertion_success
  action_rate
  joint_vel
  joint_acc
  joint_torques
  joint_pos_limits

overall_finite: True
```

Reset reward examples:

```text
sfp_approach mean=0.518217
sfp_coarse_lateral_alignment mean=0.067343
sfp_coarse_orientation_alignment mean=0.838764
sfp_orientation_alignment mean=0.271968
sfp_insertion_depth mean=0.000000
sfp_insertion_success mean=0.000000
```

Random-policy rewards were finite for steps 0 through 2.

## SFP PPO Smoke Run

Run on the host in tmux session `isaac-step7-sfp-smoke-ffdc704`:

```bash
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 16 --max_iterations 1 --run_name step7_sfp_smoke_ffdc704 --headless --enable_cameras"
```

Result:

```text
STEP7_SFP_SMOKE_EXIT:0
Log dir: /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_04-24-06_step7_sfp_smoke_ffdc704
```

Important output:

```text
Resolved observation sets:
     actor :  ['policy']
     critic :  ['policy', 'privileged']

Actor Model:
  Linear(in_features=3149, out_features=512)
  ...
  Linear(in_features=128, out_features=6)

Critic Model:
  Linear(in_features=3169, out_features=512)
  ...
  Linear(in_features=128, out_features=1)

Learning iteration 0/1
Run name: step7_sfp_smoke_ffdc704
Total steps: 384
Steps per second: 124
Mean reward: 0.60
Mean episode length: 5.00
Episode_Termination/sfp_insertion_success: 0.0000
```

Interpretation:

- The SFP task creates successfully.
- The RSL-RL config uses eval-compatible actor observations and privileged
  critic observations.
- The PPO loop starts and completes one iteration.
- This is a smoke test only. It does not show SFP learning yet.

## Log Copy

Run on the host in tmux session `isaac-step7-sfp-logcopy-ffdc704`:

```bash
cd ~/IsaacLab/aic
mkdir -p logs/aic_geometry logs/aic_rewards logs/rsl_rl/aic_sfp_insert
docker cp \
  isaac-lab-base:/workspace/isaaclab/aic/logs/aic_geometry/20260511_042247_AIC-SFP-Task-v0.log \
  logs/aic_geometry/
docker cp \
  isaac-lab-base:/workspace/isaaclab/aic/logs/aic_rewards/20260511_042323_AIC-SFP-Task-v0.log \
  logs/aic_rewards/
docker cp \
  isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_04-24-06_step7_sfp_smoke_ffdc704 \
  logs/rsl_rl/aic_sfp_insert/
```

Result:

```text
Successfully copied 25.6kB (transferred 27.1kB) to /var/home/bahw/IsaacLab/aic/logs/aic_geometry/
Successfully copied 6.42kB (transferred 8.19kB) to /var/home/bahw/IsaacLab/aic/logs/aic_rewards/
Successfully copied 43MB to /var/home/bahw/IsaacLab/aic/logs/rsl_rl/aic_sfp_insert/
STEP7_LOGCOPY_EXIT:0
```

Postcheck run on the host in tmux session
`isaac-step7-sfp-postcheck-ffdc704`:

```bash
pgrep -af "rsl_rl/train.py|check_aic_rewards.py|inspect_aic_geometry.py" || true
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

Result:

```text
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
491 MiB, 24564 MiB, 0 %
STEP7_POSTCHECK_EXIT:0
```

Only the postcheck command matched `pgrep`; no stale Isaac training/check
process remained.

## Impact On Future Steps

Step 8 should start with an SFP drift/scripted-control diagnostic before any
long PPO training. The current Step 7 task interface is ready, but SFP physical
tip controllability still needs the same kind of validation that exposed the SC
tip problem in Step 6.

No change to the high-level PPO direction:

- prefer PPO specialist checkpoints for SC and SFP
- keep BC/DAgger as diagnostics, warm starts, or provisional gates only
- do not start Step 9 distillation until both specialists solve their Isaac
  tasks or an explicit routing/export shortcut is chosen
