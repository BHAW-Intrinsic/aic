# Step 8: Specialist Checkpoints

Date: 2026-05-11

Starting point:

- Step 7 added `AIC-SFP-Task-v0` and smoke-tested one PPO iteration.
- The main training direction remains PPO for generalizable specialist
  checkpoints.
- The current SC `>90%` BC checkpoint is only an accepted gate for moving to
  SFP; it is not the preferred final path.

Immediate Step 8 task:

- Before long SFP PPO training, validate whether the SFP tip geometry is
  actually controllable by the IK action body.
- This repeats the Step 6 lesson from SC: an Isaac runtime body can exist
  without being the gripped insertion point that the policy controls.

## SFP Scripted Diagnostic Support

Local change:

- Generalized `scripts/check_aic_scripted_insert.py` from SC-only to
  connector-aware.
- Added `--connector auto|sc|sfp`; `auto` selects SFP when the task name
  contains `sfp`, otherwise SC.
- The scripted controller now routes through the appropriate geometry helpers:
  - SC: `sc_plug_tip_pose`, `sc_port_entry_pose`, `sc_insertion_success_mask`
  - SFP: `sfp_plug_tip_pose`, `sfp_port_entry_pose`,
    `sfp_insertion_success_mask`
- Diagnostic offset logging now reports transforms to the active connector tip
  rather than hard-coding `sc_tip_link`.
- Default success thresholds remain SC-compatible for SC and use the Step 7 SFP
  thresholds for SFP:
  - SC: lateral `<0.005`, orientation `<0.20`, depth `>0.012`
  - SFP: lateral `<0.004`, orientation `<0.20`, depth `>0.015`

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-lateralguard-pull-3526909 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py; echo STEP8_LATERALGUARD_PULL_EXIT:\$?; sleep 120'"
```

Reward smoke key output:

```text
Reward Manager contains 17 active terms.
  sfp_lateral_progress weight: 5.0
  sfp_lateral_alignment weight: 4.0
  sfp_insertion_action weight: 15.0
overall_finite: True
STEP8_SFP_LATERALGUARD_REWARD_3526909_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-lateralguard-3526909 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_lateralguard_3526909 --headless --enable_cameras\"; echo STEP8_SFP_PPO_LATERALGUARD_3526909_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-47-49_step8_sfp_ppo_lateralguard_3526909
```

Key output:

```text
iteration 60:
  Mean reward: 2.91
  Mean episode length: 6.47
  Episode_Reward/sfp_lateral_progress: -0.0341
  Episode_Reward/sfp_insertion_depth: 0.0046
  Episode_Reward/sfp_insertion_action: 0.0203
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000

iteration 115:
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000
```

Interpretation:

- The lateral guard reduced the inward-action reward substantially, but the
  fixed-NIC warm-start policy still immediately left the corridor.
- This run was stopped. Lateral state weighting alone was not enough.

## Lateral-Correction Action Reward

Code change:

```text
cee91e6 Reward SFP lateral correction actions
```

Change:

- Added `mdp.sfp_lateral_correction_action_reward`.
- The reward projects the raw relative-IK translation command into world frame,
  computes the SFP plug-tip lateral vector away from the active port axis, and
  rewards commands that move opposite that lateral vector.
- Added `SfpRewardsCfg.sfp_lateral_correction_action` with weight `20.0`.
- The reward is active only inside the temporary corridor:
  lateral between `0.002` and `0.060`, orientation `<0.80`, and depth between
  `-0.080` and `0.060`.
- This is still PPO reward shaping, not BC or a scripted policy.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-lateralcorrection-pull-6956029 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py; echo STEP8_LATERALCORRECTION_PULL_EXIT:\$?; sleep 120'"
```

Reward smoke key output:

```text
Reward Manager contains 18 active terms.
  sfp_lateral_correction_action weight: 20.0
overall_finite: True
STEP8_SFP_LATERALCORRECTION_REWARD_6956029_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-lateralcorrection-6956029 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_lateralcorrection_6956029 --headless --enable_cameras\"; echo STEP8_SFP_PPO_LATERALCORRECTION_6956029_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-59-21_step8_sfp_ppo_lateralcorrection_6956029
```

Key output:

```text
iteration 64:
  Episode_Reward/sfp_lateral_correction_action: 0.0485
  Episode_Termination/sfp_insertion_success: 0.0104
  Episode_Termination/sfp_corridor_violation: 0.9896

iteration 129:
  Episode_Reward/sfp_lateral_correction_action: 0.0455
  Episode_Termination/sfp_insertion_success: 0.0111
  Episode_Termination/sfp_corridor_violation: 0.9889

iteration 222:
  Episode_Reward/sfp_lateral_correction_action: 0.0569
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000
```

Evaluation of `model_200.pt`:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-lateralcorrection200-6956029 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 50 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-59-21_step8_sfp_ppo_lateralcorrection_6956029/model_200.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_LATERALCORRECTION200_6956029_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 0
success_rate: 0.000000
mean_lateral_error_at_termination: 0.142818
mean_signed_lateral_x_at_termination: 0.112403
mean_signed_lateral_z_at_termination: -0.056010
mean_orientation_error_at_termination: 1.119678
mean_insertion_depth_at_termination: -0.170864
STEP8_SFP_EVAL_LATERALCORRECTION200_6956029_EXIT:0
```

Interpretation:

- The lateral-correction reward recovered intermittent coarse training success
  but did not produce a usable checkpoint.
- Evaluation shows the policy backs away from the port under rollout
  (`mean_depth=-0.170864` after 50 steps), so the next PPO shaping should reward
  pre-insertion actions that move toward the port entry.

## Port-Approach Action Reward

Code change:

```text
16159eb Reward SFP port approach actions
```

Change:

- Added `mdp.sfp_port_approach_action_reward`.
- The reward projects the raw relative-IK translation command onto the vector
  from SFP plug tip to active port entry and rewards motion toward the entry.
- The term is active before insertion: depth between `-0.080` and `0.005`,
  distance between `0.001` and `0.120`, and orientation `<1.20`.
- Added `SfpRewardsCfg.sfp_port_approach_action` with weight `20.0`.

Reason:

- `model_200.pt` from the lateral-correction run still backed away from the
  port. This term gives PPO an immediate action-level reward for moving toward
  the entry before the insertion-depth reward should dominate.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Next remote command after commit/push/host pull:

```bash
cd ~/IsaacLab/aic
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py

docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 500 --report_every 50 --headless --enable_cameras"
```

What to look for:

- If scripted SFP insertion succeeds and offsets to `sfp_tip_link` stay stable,
  `sfp_tip_link` is likely usable for SFP training.
- If it fails with large or drifting `gripper_tcp_to_sfp_tip_pos_drift`, add a
  virtual SFP helper from `gripper_tcp` before long PPO training.

## Remote Pull

Commit:

```text
64dd864 Add SFP scripted insertion diagnostic
```

Host tmux session:

```bash
tmux new-session -d -s isaac-step8-sfp-diag-pull-64dd864 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py; echo STEP8_SFP_DIAG_PULL_EXIT:\$?; sleep 120'"
```

Result:

```text
Fast-forward to 64dd864
Successfully copied check_aic_scripted_insert.py into isaac-lab-base
STEP8_SFP_DIAG_PULL_EXIT:0
```

## SFP Scripted Diagnostics

Loose/default gate:

```bash
tmux new-session -d -s isaac-step8-sfp-scripted-diag-64dd864 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 500 --report_every 50 --headless --enable_cameras\"; echo STEP8_SFP_SCRIPTED_DIAG_EXIT:\$?; sleep 120'"
```

Key output:

```text
successes: 0/16
per target: sfp_port_0 0/11, sfp_port_1 0/5
final lateral mean: 0.012049
final depth mean: -0.001367
final orientation mean: 0.042459
gripper_tcp_to_sfp_tip_pos_drift mean=0.000000
sfp_module_link_to_sfp_tip_pos_drift mean=0.000000
sfp_tip_link_to_sfp_tip_pos_drift mean=0.000000
```

Interpretation:

- The SFP tip is stable relative to the controlled gripper/TCP path.
- The loose default alignment gate starts insertion too early and jams with a
  lateral miss.

Tight alignment gate:

```bash
tmux new-session -d -s isaac-step8-sfp-scripted-tight-64dd864 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 700 --report_every 50 --align_lateral_threshold 0.004 --align_orientation_threshold 0.10 --headless --enable_cameras\"; echo STEP8_SFP_SCRIPTED_TIGHT_EXIT:\$?; sleep 120'"
```

Key output:

```text
successes: 0/16
per target: sfp_port_0 0/7, sfp_port_1 0/9
final lateral mean: 0.005954
final depth mean: -0.034301
final orientation mean: 0.017710
gripper_tcp_to_sfp_tip_pos_drift mean=0.000000
sfp_module_link_to_sfp_tip_pos_drift mean=0.000000
sfp_tip_link_to_sfp_tip_pos_drift mean=0.000000
```

Interpretation:

- Tight alignment improves lateral error but keeps the tip outside the port.
- There is no scripted SFP insertion corridor yet.

## Signed Port-Frame Delta Diagnostic

Commit:

```text
e89a184 Report port-frame deltas in scripted diagnostic
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-delta-pull-e89a184 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py; echo STEP8_DELTA_PULL_EXIT:\$?; sleep 120'"
```

Result:

```text
Fast-forward to e89a184
Successfully copied check_aic_scripted_insert.py into isaac-lab-base
STEP8_DELTA_PULL_EXIT:0
```

Tight gate with signed deltas:

```bash
tmux new-session -d -s isaac-step8-sfp-delta-tight-e89a184 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 300 --report_every 100 --align_lateral_threshold 0.004 --align_orientation_threshold 0.10 --headless --enable_cameras\"; echo STEP8_SFP_DELTA_TIGHT_EXIT:\$?; sleep 120'"
```

Key output:

```text
successes: 0/16
per target: sfp_port_0 0/8, sfp_port_1 0/8
final lateral mean: 0.006258
final depth mean: -0.032340
final orientation mean: 0.019000
final_port_frame_delta x mean=-0.004948 min=-0.006890 max=-0.002182
final_port_frame_delta y mean=-0.002980 min=-0.004735 max=0.002825
final_port_frame_delta z mean=-0.032340 min=-0.038837 max=-0.001486
```

Legacy wrist-frame control:

```bash
tmux new-session -d -s isaac-step8-sfp-delta-legacy-e89a184 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 300 --report_every 100 --align_lateral_threshold 0.004 --align_orientation_threshold 0.10 --control_frame wrist_legacy --headless --enable_cameras\"; echo STEP8_SFP_DELTA_LEGACY_EXIT:\$?; sleep 120'"
```

Key output:

```text
successes: 0/16
per target: sfp_port_0 0/7, sfp_port_1 0/9
final lateral mean: 0.005366
final depth mean: -0.028610
final orientation mean: 0.022202
final_port_frame_delta x mean=-0.004180 min=-0.005914 max=-0.001956
final_port_frame_delta y mean=-0.002033 min=-0.004413 max=0.002801
final_port_frame_delta z mean=-0.028610 min=-0.038782 max=-0.001204
```

Middle alignment gate:

```bash
tmux new-session -d -s isaac-step8-sfp-delta-mid-e89a184 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 500 --report_every 100 --align_lateral_threshold 0.008 --align_orientation_threshold 0.10 --headless --enable_cameras\"; echo STEP8_SFP_DELTA_MID_EXIT:\$?; sleep 120'"
```

Key output:

```text
successes: 0/16
per target: sfp_port_0 0/10, sfp_port_1 0/6
final lateral mean: 0.008056
final depth mean: -0.018507
final orientation mean: 0.028459
final_port_frame_delta x mean=-0.006953 min=-0.008628 max=-0.004863
final_port_frame_delta y mean=-0.001310 min=-0.006439 max=0.004837
final_port_frame_delta z mean=-0.018507 min=-0.038422 max=-0.001070
```

Interpretation:

- `sfp_tip_link` is not loose in the way SC `sc_tip_link` was loose.
- The scripted controller consistently misses the SFP port in signed port-frame
  `x` by roughly 4-7 mm.
- The scripted controller is not a usable SFP expert yet.
- PPO remains the preferred training path; this diagnostic only says not to use
  the current scripted controller as a BC teacher.

## SFP Helper Vs USD Frame Check

Commit:

```text
65efb8d Compare SFP helpers against USD frames
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-inspect-pull-65efb8d \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py; echo STEP8_INSPECT_PULL_EXIT:\$?; sleep 120'"
```

Result:

```text
Fast-forward to 65efb8d
Successfully copied inspect_aic_geometry.py into isaac-lab-base
STEP8_INSPECT_PULL_EXIT:0
```

Inspection run:

```bash
tmux new-session -d -s isaac-step8-sfp-inspect-65efb8d \
  "bash -lc 'docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py --task AIC-SFP-Task-v0 --num_envs 2 --headless --enable_cameras\"; echo STEP8_SFP_INSPECT_EXIT:\$?; sleep 120'"
```

Log path:

```text
/workspace/isaaclab/aic/logs/aic_geometry/20260511_045441_AIC-SFP-Task-v0.log
```

Key output:

```text
== SFP Helper Vs USD Semantic Frames Env0 ==
sfp plug tip: /World/envs/env_0/Robot/cable/sfp_module/sfp_tip_link
  helper_minus_usd_m: [1.243098, 0.416048, 0.221814]
  pos_delta_norm_m:   1.329508
  quat_angle_error_rad: 2.862581
sfp_port_0 entry: /World/envs/env_0/nic_card/sfp_port_0_link/sfp_port_0_link_entrance
  helper_minus_usd_m: [0.000000, 0.000000, 0.000000]
  pos_delta_norm_m:   0.000000
  quat_angle_error_rad: 0.000000
sfp_port_1 entry: /World/envs/env_0/nic_card/sfp_port_1_link/sfp_port_1_link_entrance
  helper_minus_usd_m: [0.000000, 0.000000, 0.000000]
  pos_delta_norm_m:   0.000000
  quat_angle_error_rad: 0.000000
STEP8_SFP_INSPECT_EXIT:0
```

Interpretation:

- The fixed SFP port entrance helpers exactly match the USD semantic entrance
  prims for both SFP ports.
- The SFP tip helper uses the Isaac runtime articulation body. The USD-stage
  SFP tip pose is not a reliable dynamic comparison under the default headless
  fabric run, but the scripted diagnostic already confirmed zero runtime drift
  between the controlled gripper path and `sfp_tip_link`.
- No SFP virtual tip helper is needed at this point.

Log copy to host:

```bash
tmux new-session -d -s isaac-step8-logcopy-65efb8d \
  "bash -lc 'mkdir -p ~/IsaacLab/aic/logs/aic_scripted_insert ~/IsaacLab/aic/logs/aic_geometry && docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_scripted_insert/. ~/IsaacLab/aic/logs/aic_scripted_insert/ && docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_geometry/20260511_045441_AIC-SFP-Task-v0.log ~/IsaacLab/aic/logs/aic_geometry/; echo STEP8_LOGCOPY_EXIT:\$?; sleep 120'"
```

Result:

```text
Successfully copied 124kB to ~/IsaacLab/aic/logs/aic_scripted_insert/
Successfully copied 26.8kB to ~/IsaacLab/aic/logs/aic_geometry/
STEP8_LOGCOPY_EXIT:0
```

## Step 8 Decision

Proceed with SFP PPO training.

Reasoning:

- The SFP runtime tip is stable.
- The SFP static port-entry helper geometry matches the USD semantic entrance
  frames exactly.
- The current scripted controller is not a solved expert and should not be used
  as the primary SFP teacher.
- The project direction remains PPO for generalizable specialist policies; BC or
  DAgger should stay diagnostic/warm-start only unless explicitly accepted later.

## Near-Port Reset Curriculum

The first SFP PPO run from normal resets was expected to be wasteful because the
policy rarely samples the final insertion region. Step 8 therefore added a
temporary near-port reset curriculum for SFP, similar in spirit to the SC
near-port curriculum.

Final joint preset support:

```text
a8fc591 Report final scripted joint presets
```

Remote preset extraction:

```bash
tmux new-session -d -s isaac-step8-sfp-preset-a8fc591 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 32 --max_steps 300 --report_every 100 --align_lateral_threshold 0.004 --align_orientation_threshold 0.10 --headless --enable_cameras\"; echo STEP8_SFP_PRESET_EXIT:\$?; sleep 120'"
```

Result:

```text
successes: 0/32
final lateral mean: 0.006185
final depth mean: -0.031279
final orientation mean: 0.018250
```

The random-card joint means were not good reset anchors after NIC
randomization was removed, so a fixed-card preset pass was run instead.

Curriculum implementation:

```text
3ad791a Add SFP near-port PPO reset curriculum
04c70e1 Freeze NIC pose for initial SFP curriculum
977ac05 Update fixed-card SFP reset presets
```

The fixed-card reset presets from the final diagnostic were:

```text
sfp_port_0:
[0.8269333839416504, -1.6315652132034302, -1.792166829109192,
 -1.1168278455734253, 1.8379584550857544, 2.102725028991699]

sfp_port_1:
[0.7965589165687561, -1.671587347984314, -1.7475603818893433,
 -1.1217869520187378, 1.837986946105957, 2.108513116836548]
```

Reset validation:

```bash
tmux new-session -d -s isaac-step8-sfp-reset-977ac05 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_scripted_insert.py --task AIC-SFP-Task-v0 --connector sfp --num_envs 16 --max_steps 1 --report_every 1 --headless --enable_cameras\"; echo STEP8_SFP_RESET_977AC05_EXIT:\$?; sleep 120'"
```

Key output:

```text
step=0 successes=0/16
lateral mean: 0.003303
orientation mean: 0.012314
depth mean: -0.001980
final_port_frame_delta x mean=-0.001654
final_port_frame_delta y mean=0.002603
final_port_frame_delta z mean=-0.001980
```

Interpretation:

- The reset curriculum now starts close to the SFP entrance.
- NIC `y` randomization is intentionally frozen only for this first PPO
  curriculum stage and must be reintroduced after fixed-card insertion works.

## Fixed-NIC PPO Attempt

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-977ac05 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_fixednic_977ac05 --headless --enable_cameras\"; echo STEP8_SFP_PPO_977AC05_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05
```

The run was stopped after the early metrics showed a flat insertion signal:

```text
iteration 50:
  Mean reward: 31.29
  Mean episode length: 650.57
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 60:
  Mean reward: 36.85
  Mean episode length: 982.23
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

Interpretation:

- PPO learned to stay aligned or near the port but did not move into insertion.
- `model_50.pt` was kept as a useful alignment warm start.

## SFP Depth Reward Remediation

Code change:

```text
491fc43 Shape SFP insertion depth reward
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Reward/config change:

- `sfp_insertion_depth_reward` now accepts `min_depth`.
- The configured SFP depth reward uses `min_depth=-0.006`, `depth_scale=0.018`,
  `max_depth=0.045`, lateral gate `<0.010`, and orientation gate `<0.35`.
- `sfp_depth_progress` weight changed from `1.0` to `2.0`.
- `sfp_insertion_depth` weight changed from `4.0` to `8.0`.

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-depth-pull-491fc43 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py; echo STEP8_DEPTH_PULL_EXIT:\$?; sleep 120'"
```

Reward smoke:

```bash
tmux new-session -d -s isaac-step8-sfp-reward-491fc43 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-SFP-Task-v0 --num_envs 16 --headless --enable_cameras\"; echo STEP8_SFP_REWARD_491FC43_EXIT:\$?; sleep 120'"
```

Key output:

```text
Reward Manager weights:
  sfp_depth_progress: 2.0
  sfp_insertion_depth: 8.0
overall_finite: True
STEP8_SFP_REWARD_491FC43_EXIT:0
```

Note: `check_aic_rewards.py` directly calls the SFP reward functions with its
own hard-coded diagnostic parameters, so its direct `sfp_insertion_depth` line
does not prove whether the configured `min_depth=-0.006` is active. The reward
manager table and the training config are the source of truth for PPO.

## Depth-Reward PPO Retry

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-depthreward-491fc43 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_depthreward_491fc43 --headless --enable_cameras\"; echo STEP8_SFP_PPO_DEPTHREWARD_491FC43_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-21-25_step8_sfp_ppo_depthreward_491fc43
```

The first monitor pass found the older `step8_sfp_ppo_fixednic_977ac05` process
still running in the container, even though its tmux session was gone. It was
stopped so the depth-reward run had the GPU to itself:

```bash
tmux new-session -d -s isaac-step8-kill-stale-977ac05 \
  "bash -lc 'docker exec isaac-lab-base bash -lc \"kill -INT 26509\"; echo STEP8_KILL_STALE_977AC05_EXIT:\$?; sleep 60'"
```

Early depth-reward PPO monitor output after the stale process was stopped:

```text
iteration 62:
  Mean reward: 42.77
  Mean episode length: 183.00
  Episode_Reward/sfp_insertion_depth: 0.0005
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 65:
  Mean reward: 27.44
  Mean episode length: 250.00
  Episode_Reward/sfp_insertion_depth: 0.0008
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

Current interpretation:

- The depth signal is no longer exactly flat, but it is still very small.
- Continue monitoring before accepting or rejecting this PPO remediation.

Later monitor output before stopping:

```text
iteration 155:
  Mean reward: 63.44
  Mean episode length: 1431.00
  Episode_Reward/sfp_depth_progress: 0.0018
  Episode_Reward/sfp_insertion_depth: 0.0004
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

The run saved `model_150.pt` but still had zero insertion success, so it was
stopped and treated as an insufficient remediation.

## Inward-Action PPO Reward

Code change:

```text
3b9e781 Reward SFP inward insertion actions
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Change:

- Added `sfp_insertion_action_reward`.
- The term reads the raw relative-IK translation command, converts it from
  robot-root frame to world frame, projects it onto the active SFP port
  insertion axis, and rewards positive inward command when lateral/orientation
  alignment is already within the SFP depth gate.
- This is PPO reward shaping, not BC. It uses privileged geometry in the reward
  only; the actor observation group remains eval-compatible.

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-action-pull-3b9e781 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py; echo STEP8_ACTION_PULL_EXIT:\$?; sleep 120'"
```

Reward smoke:

```bash
tmux new-session -d -s isaac-step8-sfp-action-reward-3b9e781 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py --task AIC-SFP-Task-v0 --num_envs 16 --headless --enable_cameras\"; echo STEP8_SFP_ACTION_REWARD_3B9E781_EXIT:\$?; sleep 120'"
```

Key output:

```text
Reward Manager contains 17 active terms.
sfp_insertion_action weight: 3.0
overall_finite: True
STEP8_SFP_ACTION_REWARD_3B9E781_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-action-3b9e781 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_action_3b9e781 --headless --enable_cameras\"; echo STEP8_SFP_PPO_ACTION_3B9E781_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-33-23_step8_sfp_ppo_action_3b9e781
```

Early status:

```text
iteration 50:
  Run loaded model_50.pt from the fixed-NIC alignment warm start.
  Reward Manager includes sfp_insertion_action.
```

The weak inward-action run still had zero success:

```text
iteration 120:
  Mean reward: 44.12
  Mean episode length: 1055.20
  Episode_Reward/sfp_insertion_depth: 0.0002
  Episode_Reward/sfp_insertion_action: 0.0005
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

It was stopped and treated as insufficient.

## Stronger Inward-Action Curriculum

Code change:

```text
27ae683 Strengthen SFP final insertion curriculum
```

Changes:

- Increased `sfp_insertion_action` weight from `3.0` to `30.0`.
- Increased the action reward command scale to `0.010`.
- Loosened the action reward alignment gate to lateral `<0.020`,
  orientation `<0.50`.
- Shortened `AICTaskSfpEnvCfg` episodes to `20.0` seconds so failed attempts
  reset back into the near-port curriculum more often.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-strongaction-27ae683 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_strongaction_27ae683 --headless --enable_cameras\"; echo STEP8_SFP_PPO_STRONGACTION_27AE683_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-40-34_step8_sfp_ppo_strongaction_27ae683
```

Key monitor output:

```text
iteration 50:
  Episode_Reward/sfp_insertion_depth: 0.0042
  Episode_Reward/sfp_insertion_action: 0.1083
  Episode_Reward/sfp_insertion_success: 0.0000

iteration 100:
  Episode_Reward/sfp_insertion_depth: 0.0060
  Episode_Reward/sfp_insertion_action: 0.1823
  Episode_Reward/sfp_insertion_success: 0.0000
```

Interpretation:

- The stronger action reward produced a much clearer inward-command signal.
- Strict success was still zero, so a temporary coarse success gate was added.

## Temporary Coarse Success Gate

Code change:

```text
b47cc33 Add coarse SFP success curriculum gate
```

Change:

- Temporarily changed the first SFP curriculum success gate to lateral
  `<0.020`, orientation `<0.50`, depth `>0.005`.
- Increased SFP success bonus weight to `25.0`.
- This gate must be tightened back to the strict insertion gate after PPO
  reliably reaches the coarse final-insertion band.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-coarsegate-b47cc33 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-40-34_step8_sfp_ppo_strongaction_27ae683 --checkpoint model_100.pt --run_name step8_sfp_ppo_coarsegate_b47cc33 --headless --enable_cameras\"; echo STEP8_SFP_PPO_COARSEGATE_B47CC33_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-47-25_step8_sfp_ppo_coarsegate_b47cc33
```

Key monitor output:

```text
iteration 180:
  Mean episode length: 600.00
  Episode_Reward/sfp_insertion_depth: 0.0057
  Episode_Reward/sfp_insertion_action: 0.1376
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 210:
  Mean episode length: 600.00
  Episode_Reward/sfp_insertion_depth: 0.0013
  Episode_Reward/sfp_insertion_action: 0.1500
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

The run saved `model_200.pt` and was stopped for evaluation because it still had
zero coarse success during training.

## SFP Evaluation Diagnostics

Code change:

```text
93e9b1a Support SFP policy evaluation diagnostics
```

Change:

- Extended `scripts/rsl_rl/evaluate.py` to select SFP target names, SFP target
  ids, SFP lateral/orientation/depth metrics, and signed SFP lateral components
  when the task name contains `SFP`.
- Evaluation still disables environment terminations and measures success
  inside the evaluator using the CLI thresholds.

Evaluation command:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-coarsegate-fullpath-200-93e9b1a \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 128 --max_episode_steps 600 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_05-47-25_step8_sfp_ppo_coarsegate_b47cc33/model_200.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_COARSEGATE_FULLPATH_200_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 128
successes: 0
success_rate: 0.000000
mean_episode_length: 600.000
mean_lateral_error_at_termination: 0.957564
mean_signed_lateral_x_at_termination: 0.672674
mean_signed_lateral_z_at_termination: -0.634553
mean_orientation_error_at_termination: 1.172962
mean_insertion_depth_at_termination: -0.254260
failure_breakdown:
  timeout: 128
  lateral_miss: 128
  orientation_miss: 116
  depth_shortfall: 112
per_target:
  sfp_port_0: episodes=68 successes=0 success_rate=0.000000
  sfp_port_1: episodes=60 successes=0 success_rate=0.000000
STEP8_SFP_EVAL_COARSEGATE_FULLPATH_200_EXIT:0
```

Interpretation:

- The `model_200.pt` coarse-gate policy is not narrowly missing insertion; by
  timeout it has moved far away from the port.
- Continuing the same reward shaping is unlikely to help unless failed attempts
  are reset as soon as they leave the useful near-port corridor.

## SFP Corridor Reset Remediation

Code change:

```text
8c987b8 Reset failed SFP attempts outside corridor
```

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Change:

- Added `mdp.sfp_insertion_corridor_violation`.
- Added SFP termination `sfp_corridor_violation` with initial limits:
  lateral `>0.060`, orientation `>0.80`, depth `<-0.080`, or depth `>0.060`.
- This is a temporary curriculum guard. It should keep PPO rollouts in the
  near-port insertion band instead of spending full episodes far from the port.

Evaluator follow-up:

```text
f64c274 Ignore SFP corridor termination during evaluation
```

The evaluator disables `sfp_corridor_violation` just like success and timeout
terminations, so evaluation still measures success/failure with its own
thresholds instead of letting the env auto-reset on corridor exits.

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-corridor-pull-f64c274 \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py; echo STEP8_CORRIDOR_PULL_F64C274_EXIT:\$?; sleep 120'"
```

Smoke command:

```bash
tmux new-session -d -s isaac-step8-sfp-corridor-smoke-f64c274 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 16 --max_iterations 1 --run_name step8_sfp_corridor_smoke_f64c274 --headless --enable_cameras\"; echo STEP8_SFP_CORRIDOR_SMOKE_F64C274_EXIT:\$?; sleep 120'"
```

Key output:

```text
Termination Manager contains 3 active terms:
  time_out
  sfp_insertion_success
  sfp_corridor_violation

Learning iteration 0/1:
  Mean episode length: 18.62
  Episode_Termination/sfp_corridor_violation: 0.2656
  STEP8_SFP_CORRIDOR_SMOKE_F64C274_EXIT:0
```

## Corridor Run From Strong-Action Warm Start

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-corridor-f64c274 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-40-34_step8_sfp_ppo_strongaction_27ae683 --checkpoint model_100.pt --run_name step8_sfp_ppo_corridor_f64c274 --headless --enable_cameras\"; echo STEP8_SFP_PPO_CORRIDOR_F64C274_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-16-36_step8_sfp_ppo_corridor_f64c274
```

Key output:

```text
iteration 100:
  Mean episode length: 4.45
  Episode_Termination/sfp_corridor_violation: 0.7949
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 160:
  Mean episode length: 5.78
  Episode_Termination/sfp_corridor_violation: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

The run was stopped. The strong-action warm start had learned to leave the
near-port corridor too aggressively.

Short-horizon evaluation of this run's `model_150.pt` with terminations disabled:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-corridor150-short-f64c274 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 10 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-16-36_step8_sfp_ppo_corridor_f64c274/model_150.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_CORRIDOR150_SHORT_F64C274_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 0
mean_lateral_error_at_termination: 0.205032
mean_orientation_error_at_termination: 0.965474
mean_insertion_depth_at_termination: -0.077171
```

Interpretation:

- The policy is outside the lateral and orientation corridor almost immediately.
- Do not resume SFP PPO from the strong-action/coarse-gate checkpoints for the
  corridor curriculum.

## Reduced SFP Action Scale

Code change:

```text
21fe0ef Reduce SFP insertion action scale
```

Change:

- Set `AICTaskSfpEnvCfg.actions.arm_action.scale = 0.01`.
- Set `sfp_insertion_action_reward`'s physical `action_scale` to `0.01` and
  `command_scale` to `0.004`.

Reason:

- SFP is millimeter-tolerance insertion. The shared `0.05` relative-IK scale
  moves too far per policy step for the near-port curriculum.

Short-horizon evaluation of the earlier fixed-NIC `model_50.pt` before this
change:

```text
mean_lateral_error_at_termination: 0.105937
mean_orientation_error_at_termination: 0.758474
mean_insertion_depth_at_termination: -0.031109
```

Same fixed-NIC `model_50.pt` after reducing SFP action scale:

```text
mean_lateral_error_at_termination: 0.125998
mean_orientation_error_at_termination: 0.588953
mean_insertion_depth_at_termination: 0.036265
```

The reduced scale did not solve lateral alignment by itself, but it changed the
failure mode from backing away to reaching positive insertion depth while
laterally offset. That is a better PPO starting point for the corridor
curriculum than the strong-action checkpoint.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-smallscale-corridor-21fe0ef \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_smallscale_corridor_21fe0ef --headless --enable_cameras\"; echo STEP8_SFP_PPO_SMALLSCALE_CORRIDOR_21FE0EF_EXIT:\$?; sleep 120'"
```

Run directory:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_06-27-54_step8_sfp_ppo_smallscale_corridor_21fe0ef
```

Early key output:

```text
iteration 52:
  Episode_Reward/sfp_insertion_success: 0.0002
  Episode_Termination/sfp_insertion_success: 0.0078
  Episode_Termination/sfp_corridor_violation: 0.9922

iteration 55:
  Episode_Reward/sfp_insertion_success: 0.0003
  Episode_Termination/sfp_insertion_success: 0.0065
  Episode_Termination/sfp_corridor_violation: 0.9935

iteration 125:
  Episode_Reward/sfp_insertion_success: 0.0004
  Episode_Termination/sfp_insertion_success: 0.0124
  Episode_Termination/sfp_corridor_violation: 0.9876

iteration 127:
  Episode_Reward/sfp_insertion_success: 0.0012
  Episode_Termination/sfp_insertion_success: 0.0208
  Episode_Termination/sfp_corridor_violation: 0.9792

iteration 130:
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000

iteration 185:
  Episode_Reward/sfp_insertion_success: 0.0002
  Episode_Termination/sfp_insertion_success: 0.0072
  Episode_Termination/sfp_corridor_violation: 0.9928

iteration 209:
  Episode_Reward/sfp_insertion_success: 0.0002
  Episode_Termination/sfp_insertion_success: 0.0085
  Episode_Termination/sfp_corridor_violation: 0.9915

iteration 215:
  Episode_Reward/sfp_insertion_success: 0.0002
  Episode_Termination/sfp_insertion_success: 0.0059
  Episode_Termination/sfp_corridor_violation: 0.9941
```

Interpretation:

- This is the first PPO SFP run in Step 8 to produce nonzero coarse success.
- The signal is still weak and unstable: most episodes terminate through the
  corridor guard, but there are now intermittent coarse successes up to roughly
  `2%` of terminations.
- By iteration `215`, the signal had not climbed materially and corridor exits
  were still roughly `99-100%`.
- The run was stopped after saving `model_200.pt`.
- The next PPO remediation should target lateral drift directly rather than
  switching to BC.

## Lateral-Guard PPO Remediation

Code change:

```text
8db3fd3 Add SFP lateral guard reward shaping
```

Change:

- Added optional `lateral_std` to `mdp.sfp_insertion_action_reward`.
- When `lateral_std > 0`, the inward-action reward is multiplied by
  `1 - tanh(lateral_error / lateral_std)`, so inward motion is only strongly
  rewarded when the tip is close to the port axis.
- Reduced `sfp_insertion_action` weight from `30.0` to `15.0`.
- Tightened `sfp_insertion_action` gates from lateral `<0.020` and orientation
  `<0.50` to lateral `<0.010` and orientation `<0.35`.
- Set `sfp_insertion_action.params["lateral_std"] = 0.008`.
- Increased `sfp_lateral_progress` weight from `1.0` to `5.0`.
- Increased fine `sfp_lateral_alignment` weight from `1.0` to `4.0`.

Reason:

- The reduced-scale run learned positive insertion depth and inward action, but
  it still drifted laterally and immediately tripped the corridor guard.
- This stays on PPO: no BC labels or scripted expert are introduced. The reward
  uses privileged geometry only as a training signal.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```
