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

Code changes:

```text
cee91e6 Reward SFP lateral correction actions
6956029 Document SFP lateral correction remediation
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

Code changes:

```text
16159eb Reward SFP port approach actions
87b0517 Document SFP port approach remediation
```

Change:

- Added `mdp.sfp_port_approach_action_reward`.
- The reward projects the raw relative-IK translation command onto the vector
  from SFP plug tip to active port entry and rewards motion toward the entry.
- The term is active before insertion: depth between `-0.080` and `0.005`,
  distance between `0.001` and `0.120`, and orientation `<1.20`.
- Added `SfpRewardsCfg.sfp_port_approach_action` with weight `20.0`.

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

Remote smoke key output:

```text
Reward Manager contains 19 active terms.
  sfp_lateral_correction_action weight: 20.0
  sfp_port_approach_action weight: 20.0
overall_finite: True
STEP8_SFP_PORTAPPROACH_REWARD_87B0517_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-portapproach-87b0517 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_05-09-54_step8_sfp_ppo_fixednic_977ac05 --checkpoint model_50.pt --run_name step8_sfp_ppo_portapproach_87b0517 --headless --enable_cameras\"; echo STEP8_SFP_PPO_PORTAPPROACH_87B0517_EXIT:\$?; sleep 120'"
```

Key output:

```text
iteration 66:
  Episode_Reward/sfp_lateral_correction_action: 0.0088
  Episode_Reward/sfp_port_approach_action: 0.0378
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000

iteration 130:
  Episode_Reward/sfp_lateral_correction_action: 0.0089
  Episode_Reward/sfp_port_approach_action: 0.0372
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000
```

Interpretation:

- The port-approach term made the fixed-NIC warm-start run worse. It suppressed
  the intermittent coarse successes seen in the lateral-correction run.
- The run was stopped at iteration `130`.
- The active config should not include this term for the next attempt.

Follow-up code change:

```text
5100b1c Disable SFP port approach reward term
```

The function remains in `rewards.py` for provenance, but
`SfpRewardsCfg.sfp_port_approach_action` was removed so the active reward set
returns to the lateral-correction configuration.

## Fresh-Start Lateral-Correction PPO

The next run restarted PPO from scratch after disabling the port-approach term,
instead of resuming the old fixed-NIC optimizer state.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-fresh-lateralcorrection-5100b1c \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_fresh_lateralcorrection_5100b1c --headless --enable_cameras\"; echo STEP8_SFP_PPO_FRESH_LATERALCORRECTION_5100B1C_EXIT:\$?; sleep 120'"
```

Key output:

```text
iteration 112:
  Mean action std: 0.20
  Mean episode length: 5.60
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000
```

Interpretation:

- Fresh PPO with default exploration noise still left the near-port corridor
  almost immediately and produced no SFP successes.
- The next remediation is to reduce PPO exploration noise instead of adding
  another behavior-cloning path.

## Low-Std Fresh PPO

Code change:

```text
9ff4939 Reduce SFP PPO exploration noise
```

Change:

- Reduced SFP PPO actor `init_std` from `0.2` to `0.05`.
- Reduced SFP PPO entropy coefficient from `0.001` to `0.0002`.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-lowstd-9ff4939 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_lowstd_9ff4939 --headless --enable_cameras\"; echo STEP8_SFP_PPO_LOWSTD_9FF4939_EXIT:\$?; sleep 120'"
```

Key output:

```text
Mean action std: 0.05
Episode_Termination/sfp_insertion_success: 0.0000
Episode_Termination/sfp_corridor_violation: 1.0000
```

Interpretation:

- Lower PPO exploration noise was active, but the policy still exited the
  corridor in roughly 5-6 steps.
- The next attempt reduced the relative-IK action scale further.

## Reduced Scale 0.005 PPO

Code change:

```text
2f06127 Reduce SFP relative IK scale further
```

Change:

- Reduced `AICTaskSfpEnvCfg.actions.arm_action.scale` from `0.01` to `0.005`.
- Reduced `sfp_lateral_correction_action` and `sfp_insertion_action` physical
  action scales from `0.01` to `0.005`.
- Reduced their command normalization scales from `0.004` to `0.002`.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-scale005-2f06127 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_scale005_2f06127 --headless --enable_cameras\"; echo STEP8_SFP_PPO_SCALE005_2F06127_EXIT:\$?; sleep 120'"
```

Early key output:

```text
iteration 16:
  Mean action std: 0.05
  Mean episode length: 7.80
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000

iteration 56:
  Mean action std: 0.05
  Mean episode length: 8.05
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000

iteration 152:
  Mean action std: 0.06
  Mean episode length: 7.87
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_violation: 1.0000
```

Interpretation:

- The smaller action scale improved early episode length from roughly 5-6 steps
  to roughly 8 steps, but has not yet produced a success signal.
- The run stayed flat through iteration `152`, so it was stopped.
- The next PPO remediation was to split the single `sfp_corridor_violation`
  termination into reason-specific diagnostics so the logs report whether
  lateral, orientation, min-depth, or max-depth ends each rollout.

## Corridor-Reason Diagnostics

Code change:

```text
2dedf4c Split SFP corridor termination diagnostics
```

Change:

- Replaced the active aggregate `sfp_corridor_violation` termination with four
  explicit termination terms:
  `sfp_corridor_lateral_violation`,
  `sfp_corridor_orientation_violation`,
  `sfp_corridor_min_depth_violation`, and
  `sfp_corridor_max_depth_violation`.
- Kept the old aggregate helper function available for provenance.
- Updated `scripts/rsl_rl/evaluate.py` to disable all four split corridor terms
  during offline evaluation.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Remote smoke key output:

```text
Termination Manager contains 6 active terms:
  time_out
  sfp_insertion_success
  sfp_corridor_lateral_violation
  sfp_corridor_orientation_violation
  sfp_corridor_min_depth_violation
  sfp_corridor_max_depth_violation

STEP8_SFP_CORRIDOR_REASONS_2DEDF4C_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-corridor-reasons-2dedf4c \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_corridor_reasons_2dedf4c --headless --enable_cameras\"; echo STEP8_SFP_PPO_CORRIDOR_REASONS_2DEDF4C_EXIT:\$?; sleep 120'"
```

Key output:

```text
iteration 1:
  Episode_Termination/sfp_corridor_lateral_violation: 0.8861
  Episode_Termination/sfp_corridor_orientation_violation: 0.0000
  Episode_Termination/sfp_corridor_min_depth_violation: 0.0000
  Episode_Termination/sfp_corridor_max_depth_violation: 0.0000

iteration 16:
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 1.0000
  Episode_Termination/sfp_corridor_orientation_violation: 0.0000
  Episode_Termination/sfp_corridor_min_depth_violation: 0.0000
  Episode_Termination/sfp_corridor_max_depth_violation: 0.0000
```

Interpretation:

- The dominant SFP failure is lateral drift out of the near-port corridor.
- Orientation and depth are not the primary early termination causes in this
  curriculum stage.

## Lateral-Drift Penalty

Code change:

```text
5848f4b Penalize SFP lateral corridor drift
```

Change:

- Added `mdp.sfp_lateral_error_penalty`.
- Added active reward term `sfp_lateral_error` with weight `-6.0`, scale
  `0.060`, and clip `1.0`.
- This remains PPO reward shaping from privileged training geometry; it is not BC
  and does not add privileged actor observations.

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

Remote smoke key output:

```text
Reward Manager contains 19 active terms.
  sfp_lateral_error weight: -6.0
STEP8_SFP_LATERALPENALTY_SMOKE_5848F4B_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-lateralpenalty-5848f4b \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_lateralpenalty_5848f4b --headless --enable_cameras\"; echo STEP8_SFP_PPO_LATERALPENALTY_5848F4B_EXIT:\$?; sleep 120'"
```

Early key output:

```text
iteration 4:
  Episode_Reward/sfp_lateral_error: -0.0378
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 1.0000
```

Current interpretation:

- The penalty is active but early magnitude is still small relative to the rest
  of the reward mix.
- Continue only briefly; if lateral exits remain near `1.0000`, the next PPO
  adjustment should make lateral control much more dominant.

## Strong Lateral-Control Rewards

Code change:

```text
9117c07 Strengthen SFP lateral control rewards
```

Change:

- Increased `sfp_lateral_progress` weight from `5.0` to `20.0`.
- Increased `sfp_lateral_error` penalty from weight `-6.0`, scale `0.060` to
  weight `-40.0`, scale `0.020`.
- Increased `sfp_lateral_correction_action` weight from `20.0` to `80.0`.
- Reduced `sfp_lateral_correction_action` command scale from `0.002` to `0.001`.
- Reduced `sfp_lateral_correction_action` lateral gain scale from `0.012` to
  `0.006`.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Remote smoke key output:

```text
Reward Manager contains 19 active terms.
  sfp_lateral_progress weight: 20.0
  sfp_lateral_error weight: -40.0
  sfp_lateral_correction_action weight: 80.0

iteration 0:
  Episode_Reward/sfp_lateral_error: -0.1002
  Episode_Termination/sfp_corridor_lateral_violation: 0.0000
STEP8_SFP_STRONGLATERAL_SMOKE_9117C07_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-stronglateral-9117c07 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_stronglateral_9117c07 --headless --enable_cameras\"; echo STEP8_SFP_PPO_STRONGLATERAL_9117C07_EXIT:\$?; sleep 120'"
```

Early key output:

```text
iteration 1:
  Mean reward: -4.64
  Episode_Reward/sfp_lateral_error: -0.7784
  Episode_Reward/sfp_lateral_correction_action: 0.0854
  Episode_Termination/sfp_corridor_lateral_violation: 0.6595

iteration 4:
  Mean reward: -4.68
  Episode_Reward/sfp_lateral_error: -0.3232
  Episode_Reward/sfp_lateral_correction_action: 0.0081
  Episode_Termination/sfp_corridor_lateral_violation: 1.0000
```

Current interpretation:

- The stronger terms are active and materially change reward scale.
- Early rollouts still mostly leave through lateral violation, so monitor briefly
  before deciding whether a reset/action-scale curriculum change is needed.

## Reduced Scale 0.001 PPO

Code change:

```text
e4df956 Reduce SFP action scale for lateral stability
```

Change:

- Reduced SFP PPO actor `init_std` from `0.05` to `0.02`.
- Reduced SFP PPO entropy coefficient from `0.0002` to `0.0001`.
- Reduced `AICTaskSfpEnvCfg.actions.arm_action.scale` from `0.005` to `0.001`.
- Reduced the SFP lateral-correction reward physical `action_scale` from
  `0.005` to `0.001` and `command_scale` from `0.001` to `0.0002`.
- Reduced the SFP insertion-action reward physical `action_scale` from `0.005`
  to `0.001` and `command_scale` from `0.002` to `0.0004`.

Reason:

- Stronger lateral reward terms alone did not reduce lateral exits by iteration
  `44`.
- The SFP tolerance is millimeter-scale; the next curriculum change is to reduce
  the actual action delta so random PPO actions do not immediately throw the tip
  out of the lateral corridor.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Remote smoke key output:

```text
iteration 0:
  Mean action std: 0.02
  Mean episode length: 20.00
  Episode_Reward/sfp_lateral_error: -0.0705
  Episode_Termination/sfp_corridor_lateral_violation: 0.0000
STEP8_SFP_SCALE001_SMOKE_E4DF956_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-scale001-e4df956 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_scale001_e4df956 --headless --enable_cameras\"; echo STEP8_SFP_PPO_SCALE001_E4DF956_EXIT:\$?; sleep 120'"
```

Early key output:

```text
iteration 1:
  Mean action std: 0.02
  Mean episode length: 39.49
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 0.1133

iteration 4:
  Mean episode length: 28.55
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 1.0000
```

Current interpretation:

- The first update showed the best lateral-stability signal so far, but the run
  initially regressed back to lateral exits.
- Continue monitoring before deciding whether to stop or keep training.

Later key output before stopping:

```text
iteration 52:
  Mean action std: 0.03
  Mean episode length: 15.93
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 1.0000
```

Interpretation:

- The actual action scale helped initially, but the learned action std grew from
  `0.02` to roughly `0.03`, and lateral exits returned to `1.0000`.
- RSL-RL's current Gaussian distribution config exposes `init_std` and
  `std_type`, but not an explicit maximum std clamp. The next attempt therefore
  lowers initial std further, removes entropy pressure, and reduces learning
  rate.

## Stabilized Low-Noise PPO

Code change:

```text
3f5ffd1 Stabilize SFP low-noise PPO
```

Change:

- Reduced SFP PPO actor `init_std` from `0.02` to `0.005`.
- Set SFP PPO entropy coefficient from `0.0001` to `0.0`.
- Reduced SFP PPO learning rate from `1.0e-3` to `3.0e-4`.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Remote smoke key output:

```text
iteration 0:
  Mean action std: 0.00
  Mean episode length: 20.00
  Episode_Termination/sfp_corridor_lateral_violation: 0.0000
STEP8_SFP_STABLELOWSTD_SMOKE_3F5FFD1_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-stablelowstd-3f5ffd1 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_stablelowstd_3f5ffd1 --headless --enable_cameras\"; echo STEP8_SFP_PPO_STABLELOWSTD_3F5FFD1_EXIT:\$?; sleep 120'"
```

Early key output:

```text
iteration 2:
  Mean episode length: 42.80
  Episode_Termination/sfp_corridor_lateral_violation: 0.0059

iteration 4:
  Mean episode length: 83.57
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Termination/sfp_corridor_lateral_violation: 0.4395
```

Current interpretation:

- This is the first SFP PPO run to sustain substantially longer episodes after
  the first few updates.
- Lateral stability is still not solved and insertion success remains zero, but
  this run is worth monitoring longer than the immediately flat variants.

## Signed Depth Action Shaping

Code changes:

```text
7448d8b Use signed SFP depth action shaping
```

Change:

- Changed `mdp.sfp_port_frame_depth_action_reward` from one-sided reward-only
  shaping to signed shaping:
  - raw `z-` is rewarded because the Step 8 action-frame diagnostic showed it is
    the clearest positive SFP depth direction.
  - raw `z+` is penalized so PPO receives an immediate signal when it chooses the
    wrong depth direction.
- Increased `sfp_port_frame_depth_action` weight from `80.0` to `120.0`.
- Reduced its command normalization scale from `0.02` to `0.005`.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-signeddepth-7448d8b \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_signeddepth_7448d8b --headless --enable_cameras\"; echo STEP8_SFP_PPO_SIGNEDDEPTH_7448D8B_EXIT:\$?; sleep 120'"
```

Key output:

```text
iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: -0.9328
  Episode_Reward/sfp_depth_backout: 0.0000
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 32:
  Episode_Reward/sfp_port_frame_depth_action: -101.6597
  Episode_Reward/sfp_depth_backout: -117.8170
  Episode_Termination/sfp_insertion_success: 0.0000
```

Interpretation:

- The signed term was active, but PPO still chose the wrong depth direction.
- The run was stopped at iteration `32`.
- The next remediation was to initialize the SFP PPO actor with a small inward
  raw-action bias while keeping the policy fully trainable.

## SFP Actor Output Bias

Code changes:

```text
139c40d Bias SFP PPO initial depth action
ab0c252 Fix SFP actor bias hook
6b32679 Zero SFP actor output head initially
```

Change:

- Added an AIC-local training hook in `scripts/rsl_rl/train.py` that can set an
  initial actor output bias after the RSL-RL runner is constructed.
- The hook strips the custom config keys before handing the config dictionary to
  installed RSL-RL.
- The installed RSL-RL PPO stores the actor at `runner.alg.actor`, so the hook
  was fixed to target that attribute.
- Set SFP initial actor output bias to raw action
  `[0.0, 0.0, -0.05, 0.0, 0.0, 0.0]`.
- Added `aic_actor_output_zero_weights=True` so the final actor output layer
  starts from a known constant mean instead of random features overpowering the
  bias.
- This is PPO initialization only. The output head remains trainable, and no
  scripted controller is deployed.

Smoke output:

```text
[INFO] Applied AIC actor output bias to algorithm.actor:
  [0.0, 0.0, -0.05000000074505806, 0.0, 0.0, 0.0]
  (zero_output_weights=True)

iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 0.5314
  Episode_Reward/sfp_depth_progress: 0.0004
  Episode_Reward/sfp_insertion_action: 0.0433
  STEP8_ZEROHEAD_SMOKE_EXIT:0
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-zerohead-6b32679 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_zerohead_6b32679 --headless --enable_cameras\"; echo STEP8_SFP_PPO_ZEROHEAD_EXIT:\$?; sleep 120'"
```

Key output:

```text
iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 0.8808
  Episode_Reward/sfp_depth_progress: 0.0007
  Episode_Reward/sfp_insertion_action: 0.0625
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 56:
  Episode_Reward/sfp_depth_backout: 0.0000
  Episode_Reward/sfp_port_frame_depth_action: -9.1438
  Episode_Reward/sfp_depth_progress: -0.0000
  Episode_Termination/time_out: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

Interpretation:

- Actor output-head zeroing fixed the early depth direction and prevented the
  previous backing-out failure.
- The run reached a timeout-only local optimum: no backout, no corridor exits,
  but still no final insertion success.
- The run was stopped after `model_50.pt` was saved.

Evaluation of `model_50.pt`:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-zerohead50-6b32679 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 150 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_09-54-55_step8_sfp_ppo_zerohead_6b32679/model_50.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_ZEROHEAD50_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 0
success_rate: 0.000000
mean_lateral_error_at_termination: 0.004905
mean_orientation_error_at_termination: 0.023362
mean_insertion_depth_at_termination: -0.001481
failure_breakdown:
  timeout: 64
  lateral_miss: 0
  orientation_miss: 0
  depth_shortfall: 64
```

Interpretation:

- The policy is aligned laterally and angularly under the coarse SFP gate.
- The remaining blocker is final insertion depth. Mean depth is about `6.5 mm`
  short of the coarse success threshold (`-0.001481` vs `>0.005`).

## Final-Depth Curriculum

Code change:

```text
c3504ec Emphasize SFP final insertion depth
```

Change:

- Reduced `sfp_port_frame_lateral_action` weight from `100.0` to `20.0` because
  the previous policy collected large lateral-action reward while staying just
  outside insertion success.
- Increased `sfp_port_frame_depth_action` weight from `120.0` to `200.0`.
- Increased its target depth from `0.005` to `0.012`.
- Increased `sfp_depth_progress` weight from `2.0` to `10.0`.
- Increased `sfp_insertion_depth` weight from `8.0` to `60.0`.
- Increased `sfp_insertion_action` weight from `15.0` to `80.0` and reduced its
  command scale from `0.0004` to `0.0002`.
- Increased SFP actor output bias from raw `z=-0.05` to raw `z=-0.10`.

Local checks:

```bash
python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
```

Result:

```text
py_compile passed
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-finaldepth-c3504ec \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_finaldepth_c3504ec --headless --enable_cameras\"; echo STEP8_SFP_PPO_FINALDEPTH_EXIT:\$?; sleep 120'"
```

Smoke result:

```text
[INFO] Applied AIC actor output bias to algorithm.actor:
  [0.0, 0.0, -0.10000000149011612, 0.0, 0.0, 0.0]
  (zero_output_weights=True)

iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 1.5506
  Episode_Reward/sfp_depth_progress: 0.0022
  Episode_Reward/sfp_insertion_depth: 0.4043
  Episode_Reward/sfp_insertion_action: 0.9347
  STEP8_FINALDEPTH_SMOKE_EXIT:0
```

Training monitor:

```text
iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 2.4567
  Episode_Reward/sfp_depth_progress: 0.0045
  Episode_Reward/sfp_insertion_depth: 0.4874
  Episode_Reward/sfp_insertion_action: 1.3323
  Episode_Termination/sfp_insertion_success: 0.0000

iteration 68:
  Episode_Termination/time_out: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Reward/sfp_port_frame_depth_action: about 29-30
  Episode_Reward/sfp_insertion_depth: about 9-11
```

The run was stopped after `model_50.pt` because training still had zero coarse
success and timeout-only terminations.

Evaluation command:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-finaldepth50-c3504ec \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 150 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_11-20-10_step8_sfp_ppo_finaldepth_c3504ec/model_50.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_FINALDEPTH50_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 0
success_rate: 0.000000
mean_lateral_error_at_termination: 0.009654
mean_orientation_error_at_termination: 0.023373
mean_insertion_depth_at_termination: -0.001222
failure_breakdown:
  timeout: 64
  lateral_miss: 0
  orientation_miss: 0
  depth_shortfall: 64
per_target:
  sfp_port_0: episodes=34 successes=0 mean_lateral=0.009905 mean_depth=-0.001218
  sfp_port_1: episodes=30 successes=0 mean_lateral=0.009371 mean_depth=-0.001226
STEP8_SFP_EVAL_FINALDEPTH50_EXIT:0
```

Interpretation:

- The final-depth curriculum did not solve coarse insertion.
- Mean insertion depth improved only slightly from the zero-head evaluation
  (`-0.001481` to `-0.001222`).
- Lateral alignment worsened but stayed inside the temporary coarse gate.
- The main issue is not depth sign; it is that reward was available for raw
  inward intent and slightly negative depth without actual final insertion.

## Progress-Gated Final-Depth Curriculum

Forced-action diagnostic:

```bash
tmux new-session -d -s isaac-step8-sfp-actionframe-c3504ec \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_sfp_action_frame.py --task AIC-SFP-Task-v0 --num_envs 16 --raw_action 0.1 --num_steps 150 --headless --enable_cameras\"; echo STEP8_SFP_ACTIONFRAME_C3504EC_EXIT:\$?; sleep 120'"
```

Critical rows:

```text
action=tz+
  d_depth_mean=+0.000222
  after_depth_mean=-0.002236

action=tz-
  d_depth_mean=+0.001539
  after_depth_mean=-0.001076
  sfp_port_0 d_depth_mean=+0.002615
  sfp_port_1 d_depth_mean=+0.000893
```

The diagnostic was stopped after the `tz-` row because the remaining rotation
probes were not needed. Result: raw `z-` is still the best positive-depth
direction, but the measured motion is small enough that rewarding raw intent
alone can plateau outside the port.

Code change:

```text
a79737c Gate SFP final-depth rewards on progress
```

Change:

- Gate SFP approach and distance-progress rewards off after
  `insertion_depth >= -0.002` so entrance-distance rewards do not pull the
  policy back toward the entrance once it should be inserting.
- Add a separate SFP previous-depth buffer for action shaping.
- Change `sfp_port_frame_depth_action_reward` so raw `z-` reward is multiplied
  by realized positive signed-depth progress from the previous step.
- Reduce the raw depth-action weight from `200.0` to `80.0` and increase its
  command scale from `0.005` to `0.20`.
- Make insertion-depth reward zero until positive depth by setting
  `min_depth=0.0`, and increase the useful ramp with `depth_scale=0.006`,
  weight `120.0`.
- Increase SFP relative-IK scale from `0.001` to `0.002`.
- Increase the training-only actor output bias from raw `z=-0.10` to
  raw `z=-0.20`.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-progressgate-pull-a79737c \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py; echo STEP8_PROGRESSGATE_PULL_EXIT:\$?; sleep 120'"
```

Result:

```text
Fast-forward to a79737c
STEP8_PROGRESSGATE_PULL_EXIT:0
```

Smoke command:

```bash
tmux new-session -d -s isaac-step8-progressgate-smoke-a79737c \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 16 --max_iterations 1 --run_name step8_sfp_progressgate_smoke_a79737c --headless --enable_cameras\"; echo STEP8_PROGRESSGATE_SMOKE_EXIT:\$?; sleep 120'"
```

Key output:

```text
Reward Manager:
  sfp_port_frame_depth_action: 80.0
  sfp_insertion_depth: 120.0

[INFO] Applied AIC actor output bias to algorithm.actor:
  [0.0, 0.0, -0.20000000298023224, 0.0, 0.0, 0.0]
  (zero_output_weights=True)

iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 0.5218
  Episode_Reward/sfp_depth_progress: 0.0039
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_action: 1.7301
  Episode_Termination/sfp_insertion_success: 0.0000
STEP8_PROGRESSGATE_SMOKE_EXIT:0
```

Interpretation:

- The new config loads.
- The actor starts with the intended raw `z=-0.20` bias.
- The progress-gated depth-action reward is positive when measured depth
  increases.
- The insertion-depth state reward no longer pays at negative depth.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-progressgate-a79737c \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_progressgate_a79737c --headless --enable_cameras\"; echo STEP8_SFP_PPO_PROGRESSGATE_EXIT:\$?; sleep 120'"
```

Status:

```text
Run started in tmux session isaac-step8-sfp-ppo-progressgate-a79737c.
```

Training monitor:

```text
iteration 12:
  Episode_Reward/sfp_depth_progress: 0.0201
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_action: 18.9017
  Episode_Reward/sfp_insertion_success: 0.0278
  Episode_Termination/sfp_insertion_success: 0.0078

iteration 64:
  Episode_Reward/sfp_depth_progress: 0.0050
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_action: 23.2279
  Episode_Reward/sfp_insertion_success: 0.0000
  Episode_Termination/time_out: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
```

The run was stopped after `model_50.pt` because it collapsed back to timeout
only after a brief early success signal.

Evaluation command:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-progressgate50-a79737c \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 150 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_11-41-28_step8_sfp_ppo_progressgate_a79737c/model_50.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_PROGRESSGATE50_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 1
success_rate: 0.015625
mean_lateral_error_at_termination: 0.009848
mean_orientation_error_at_termination: 0.059943
mean_insertion_depth_at_termination: -0.001483
mean_success_lateral_error: 0.014801
mean_success_insertion_depth: 0.005114
failure_breakdown:
  timeout: 63
  lateral_miss: 0
  orientation_miss: 0
  depth_shortfall: 63
per_target:
  sfp_port_0: episodes=34 successes=1 mean_depth=-0.001388
  sfp_port_1: episodes=30 successes=0 mean_depth=-0.001591
STEP8_SFP_EVAL_PROGRESSGATE50_EXIT:0
```

Interpretation:

- The patch produced the first nonzero detached SFP evaluation success, but only
  `1/64`.
- Mean depth is still essentially the old timeout plateau.
- `sfp_insertion_action` remained large while measured depth stayed negative,
  so the next patch should gate insertion-action reward by realized positive
  signed-depth progress too, and increase the sparse success bonus now that PPO
  occasionally samples a valid coarse success.

## Insertion-Action Progress Gate

Code change:

```text
f5d3eed Gate SFP insertion actions on progress
```

Change:

- Add a separate previous-depth buffer for `sfp_insertion_action`.
- Multiply `sfp_insertion_action_reward` by realized positive signed-depth
  progress, just like the raw depth-action term.
- Increase actual-depth signals:
  - `sfp_depth_progress`: `10.0 -> 40.0`
  - `sfp_insertion_depth`: `120.0 -> 160.0`
  - `sfp_insertion_success`: `25.0 -> 100.0`
- Reduce insertion-action weight from `80.0` to `60.0` because it is now only a
  measured-progress-gated helper.
- Increase SFP relative-IK scale from `0.002` to `0.003`.
- Increase the initial actor raw depth bias from `z=-0.20` to `z=-0.25`.

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py
git diff --check
```

Result:

```text
py_compile passed
git diff --check passed
```

Host pull/copy:

```bash
tmux new-session -d -s isaac-step8-actionprogress-pull-f5d3eed \
  "bash -lc 'cd ~/IsaacLab/aic && git pull --ff-only && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py && docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py; echo STEP8_ACTIONPROGRESS_PULL_EXIT:\$?; sleep 120'"
```

Result:

```text
Fast-forward to f5d3eed
STEP8_ACTIONPROGRESS_PULL_EXIT:0
```

Smoke command:

```bash
tmux new-session -d -s isaac-step8-actionprogress-smoke-f5d3eed \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 16 --max_iterations 1 --run_name step8_sfp_actionprogress_smoke_f5d3eed --headless --enable_cameras\"; echo STEP8_ACTIONPROGRESS_SMOKE_EXIT:\$?; sleep 120'"
```

Key output:

```text
Reward Manager:
  sfp_depth_progress: 40.0
  sfp_insertion_depth: 160.0
  sfp_insertion_action: 60.0
  sfp_insertion_success: 100.0

[INFO] Applied AIC actor output bias to algorithm.actor:
  [0.0, 0.0, -0.25, 0.0, 0.0, 0.0]
  (zero_output_weights=True)

iteration 0:
  Episode_Reward/sfp_port_frame_depth_action: 0.4345
  Episode_Reward/sfp_depth_progress: 0.0204
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_action: 0.9750
  Episode_Reward/sfp_insertion_success: 0.0000
STEP8_ACTIONPROGRESS_SMOKE_EXIT:0
```

Interpretation:

- The new config loads.
- `sfp_insertion_action` is no longer a large reward for command intent alone.
- The run to monitor is `step8_sfp_ppo_actionprogress_f5d3eed`.

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-actionprogress-f5d3eed \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_actionprogress_f5d3eed --headless --enable_cameras\"; echo STEP8_SFP_PPO_ACTIONPROGRESS_EXIT:\$?; sleep 120'"
```

Training monitor:

```text
iteration 55:
  Episode_Reward/sfp_insertion_success: 0.0694
  Episode_Termination/sfp_insertion_success: 0.0189

iteration 56:
  Episode_Termination/sfp_insertion_success: 0.0169

iteration 100:
  Episode_Termination/time_out: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Reward/sfp_insertion_depth: 0.0000
```

Interpretation:

- The measured-progress gate reduced the action-intent exploit, and PPO briefly
  found coarse successes.
- The run still collapsed to timeout-only behavior by iteration `100`.
- The next diagnostic tested whether a stronger deterministic push can cross the
  SFP depth threshold from the current near-port reset.

## Strong SFP Forced-Action Diagnostic

Command:

```bash
tmux new-session -d -s isaac-step8-sfp-actionframe-strong-f5d3eed \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_sfp_action_frame.py --task AIC-SFP-Task-v0 --num_envs 16 --raw_action 1.0 --num_steps 150 --headless --enable_cameras\"; echo STEP8_SFP_ACTIONFRAME_STRONG_EXIT:\$?; sleep 120'"
```

Critical rows:

```text
action=tx+
  d_lateral_x=+0.163086
  d_lateral_y=-0.007386
  d_depth=+0.014318
  after_lateral=0.161685
  after_depth=0.012513
  orientation=0.078266

action=ty+
  d_lateral_x=+0.012965
  d_lateral_y=-0.182977
  d_depth=+0.014021
  after_lateral=0.180083
  after_depth=0.011559
  orientation=0.027020

action=tz+
  d_lateral_x=+0.011740
  d_lateral_y=-0.008985
  d_depth=-0.156781
  after_lateral=0.014415
  after_depth=-0.158948
  orientation=0.097445

action=tz-
  d_lateral_x=-0.021373
  d_lateral_y=+0.018711
  d_depth=+0.036686
  after_lateral=0.031734
  after_depth=0.033635
  orientation=0.303476
```

Interpretation:

- A strong raw `tz-` push can cross the coarse insertion-depth threshold.
- The same push also creates lateral drift beyond the coarse lateral gate
  (`0.031734` vs `0.020`).
- Approximate linear compensation from the diagnostic suggested trying an
  initial raw bias around `(x=0.13, y=0.10, z=-1.0)`.

## Coupled Initial Insertion Push

Code change:

```text
6c3fbf2 Initialize SFP PPO with coupled insertion push
```

Change:

- Updated the SFP training-only actor-output bias from pure raw `z=-0.25` to
  `(0.13, 0.10, -1.0, 0.0, 0.0, 0.0)`.
- This is still a PPO initialization. It is not a hardcoded runtime policy;
  PPO can immediately update the actor parameters.

Smoke command:

```bash
tmux new-session -d -s isaac-step8-coupledbias-smoke-6c3fbf2 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 16 --max_iterations 1 --run_name step8_sfp_coupledbias_smoke_6c3fbf2 --headless --enable_cameras\"; echo STEP8_COUPLEDBIAS_SMOKE_EXIT:\$?; sleep 120'"
```

Key output:

```text
[INFO] Applied AIC actor output bias to algorithm.actor:
  [0.13, 0.10, -1.0, 0.0, 0.0, 0.0]
  (zero_output_weights=True)

iteration 0:
  Mean episode length: 13.00
  Episode_Reward/sfp_port_frame_lateral_action: 1.0487
  Episode_Reward/sfp_port_frame_depth_action: 0.2607
  Episode_Reward/sfp_depth_progress: 0.0353
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_insertion_action: 0.5619
  Episode_Termination/sfp_insertion_success: 0.0000
```

Training command:

```bash
tmux new-session -d -s isaac-step8-sfp-ppo-coupledbias-6c3fbf2 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step8_sfp_ppo_coupledbias_6c3fbf2 --headless --enable_cameras\"; echo STEP8_SFP_PPO_COUPLEDBIAS_EXIT:\$?; sleep 120'"
```

Training monitor:

```text
iteration 9:
  Episode_Reward/sfp_insertion_success: 0.0278
  Episode_Termination/sfp_insertion_success: 0.0111

iteration 72:
  Episode_Termination/time_out: 1.0000
  Episode_Termination/sfp_insertion_success: 0.0000
  Episode_Reward/sfp_insertion_depth: 0.0000
  Episode_Reward/sfp_depth_progress: -0.0325
```

The run was stopped after `model_50.pt` because it had collapsed to timeout-only
behavior.

Evaluation command:

```bash
tmux new-session -d -s isaac-step8-sfp-eval-coupledbias50-6c3fbf2 \
  "bash -lc 'cd ~/IsaacLab && docker exec isaac-lab-base bash -lc \"cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 64 --max_episode_steps 150 --checkpoint /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_12-12-34_step8_sfp_ppo_coupledbias_6c3fbf2/model_50.pt --lateral_threshold 0.020 --orientation_threshold 0.50 --depth_threshold 0.005 --failure_sample_count 10 --headless --enable_cameras\"; echo STEP8_SFP_EVAL_COUPLEDBIAS50_EXIT:\$?; sleep 120'"
```

Key output:

```text
episodes: 64
successes: 0
success_rate: 0.000000
mean_episode_length: 150.000
mean_lateral_error_at_termination: 0.013022
mean_signed_lateral_x_at_termination: -0.009134
mean_signed_lateral_z_at_termination: 0.009250
mean_orientation_error_at_termination: 0.342141
mean_insertion_depth_at_termination: -0.003365
failure_breakdown:
  timeout: 64
  lateral_miss: 0
  orientation_miss: 0
  depth_shortfall: 64
per_target:
  sfp_port_0: episodes=34 successes=0 mean_depth=-0.003395
  sfp_port_1: episodes=30 successes=0 mean_depth=-0.003330
STEP8_SFP_EVAL_COUPLEDBIAS50_EXIT:0
```

Interpretation:

- The coupled initial bias did not improve the detached checkpoint; it regressed
  mean depth relative to the progress-gated checkpoint.
- The SFP blocker is now clearly a final-insertion control problem, not missing
  semantic geometry.
- The next step should be a custom combined-action diagnostic in
  `check_sfp_action_frame.py`, then a reset/curriculum or reward update based
  on measured combined-action behavior.
