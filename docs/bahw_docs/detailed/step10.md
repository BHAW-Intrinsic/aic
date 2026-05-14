# Step 10: Gazebo Transfer Audit

## Goal

Bring the official Gazebo deployment path up to qualification standard without
using hidden runtime state. The current best legal run gets proximity credit but
does not trigger insertion:

- SFP trial 1: final distance about `0.05m`
- SFP trial 2: final distance about `0.04m`
- SC trial 3: final distance about `0.29m`

The first pass is instrumentation, not behavior tuning. We need to see exactly
what the exported actor receives from the Gazebo adapter and what commands it
sends to the controller.

## Legality Boundary

Runtime traces use only:

- official `Task` metadata received by `Policy.insert_cable()`
- official `Observation` messages returned by `get_observation()`
- actor outputs
- emitted `MotionUpdate` command fields

The policy trace does not subscribe to `/scoring`, `/gazebo`, hidden TF, or
simulator internals. Scoring bags remain offline diagnostics after the run.

## Implementation Plan

- Add `AIC_RSLRL_TRACE_DIR` to `RslRlCheckpointPolicy`.
- Add per-step JSONL summaries for:
  - task metadata and inferred target selection
  - joint observations
  - TCP pose/reference/error
  - wrist wrench and body-force observation block
  - ResNet18 feature norms for all three cameras
  - actor action vector
  - emitted `MotionUpdate` frame, pose, stiffness, damping, and mode
- Add optional `AIC_RSLRL_TRACE_FULL_OBS=true` to save compressed full actor
  observations/actions for deeper offline comparison.
- Add `AIC_RSLRL_ZERO_BODY_FORCES=true` as a legal diagnostic ablation because
  the Isaac actor expects a 42D body-force block, while Gazebo supplies only a
  wrist wrench.
- Add `--record-policy-trace` to `run_gazebo_checkpoint_eval.py` so traces land
  under the same result directory as `scoring.yaml`.
- Add `summarize_policy_trace.py` for a quick post-run trace summary.

## Commands To Verify Locally

```bash
python3 -m py_compile \
  aic_model/aic_model/RslRlCheckpointPolicy.py \
  aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  aic_utils/aic_training_utils/scripts/summarize_policy_trace.py
```

```bash
git diff --check
```

## Planned Remote Eval Command

Run from the host repo copy:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-transfer-audit \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --record-policy-trace \
  --policy-trace-every-n 1 \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true
```

Expected trace location:

```text
~/ws_aic/src/aic/logs/gazebo_eval/<timestamp>/policy_trace/
```

Summarize after the run:

```bash
python3 aic_utils/aic_training_utils/scripts/summarize_policy_trace.py \
  logs/gazebo_eval/<timestamp>/policy_trace
```

## First Traced Official Gazebo Run

Implementation commit:

```text
b71a942 Add Gazebo policy transfer tracing
```

The remote host could not resolve `github.com` during the first sync, so the
commit was applied to `~/ws_aic/src/aic` with a git bundle. The host repo was
then at `b71a942`.

Run command from the host repo copy:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-transfer-audit \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --record-policy-trace \
  --policy-trace-every-n 1 \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --replace
```

Artifacts:

```text
logs/gazebo_eval/20260514_193445/scoring.yaml
logs/gazebo_eval/20260514_193445/policy_trace/
logs/gazebo_eval/20260514_193445/bag_trial_1_20260514_193509_825/
logs/gazebo_eval/20260514_193445/bag_trial_2_20260514_193526_148/
logs/gazebo_eval/20260514_193445/bag_trial_3_20260514_193540_425/
logs/gazebo_eval/20260514_193445/camera_bags/wrist_cameras/
```

Scoring:

```text
total: 95.939107820001894
trial_1: tier_1=1, tier_2=22.148262217361868, tier_3=24.666992313196427
  No insertion detected. Final plug port distance: 0.05m.
trial_2: tier_1=1, tier_2=22.566228426391909, tier_3=23.557624863051686
  No insertion detected. Final plug port distance: 0.05m.
trial_3: tier_1=1, tier_2=0, tier_3=0
  No insertion detected. Final plug port distance: 0.33m.
```

Policy trace summary:

```text
trial_1 SFP target_module=nic_card_mount_0 port=sfp_port_0
  action_norm mean=0.9995
  first_action=[-0.0063, -0.0033, -0.9994, 0.0001, 0.0014, -0.0065]
  tcp_delta=[0.0072, 0.0267, -0.0958], tcp_error_norm mean=0.0016
  ResNet18 feature norms were nonzero for all cameras.

trial_2 SFP target_module=nic_card_mount_1 port=sfp_port_0
  action_norm mean=0.9996
  first_action=[-0.0053, -0.0052, -0.9982, -0.0006, 0.0018, -0.0036]
  tcp_delta=[-0.0457, 0.0102, -0.1127], tcp_error_norm mean=0.0015
  ResNet18 feature norms were nonzero for all cameras.

trial_3 SC target_module=sc_port_1 port=sc_port_base
  action_norm mean=0.3732
  tcp_delta=[0.0912, -0.0518, -0.0393], tcp_error_norm mean=0.0120
  ResNet18 feature norms were nonzero for all cameras.
```

Interpretation:

- The Gazebo Cartesian controller follows emitted `MotionUpdate` targets closely.
  SFP TCP tracking error was about `1-2 mm`, so the immediate SFP failure is not
  dominated by ignored controller commands.
- The SFP actor saturates almost entirely along its learned insertion/depth
  action and reaches depth, but it drifts laterally by centimeters.
- Camera features are available and nonzero. The issue is not a missing ResNet18
  encoder in this run.

Offline scoring-TF bag analysis:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc \
  "cd /var/home/bahw/ws_aic/src/aic && source /ws_aic/install/setup.bash && \
   python3 aic_utils/aic_training_utils/scripts/analyze_gazebo_eval_bag.py \
   logs/gazebo_eval/20260514_193445/bag_trial_1_20260514_193509_825 \
   --include-scoring-tf"
```

Important SFP observations from the offline TF data:

- Trial 1 target was `nic_card_mount_0/sfp_port_0`.
  - SFP tip moved by roughly `[-0.0053, -0.0289, -0.1021]` in scoring world.
  - Final distance to `sfp_port_0_link_entrance` was `0.03864 m`.
  - The final tip was past the entrance in depth, but laterally offset.
- Trial 2 target was `nic_card_mount_1/sfp_port_0`.
  - SFP tip moved by roughly `[0.0482, -0.0103, -0.1167]` in scoring world.
  - Final distance to target `sfp_port_0_link_entrance` was `0.05257 m`.
  - It ended closer to the neighboring `sfp_port_1_link` (`0.03678 m`) than to
    the requested `sfp_port_0` on the requested mount.
- Official SFP task metadata uses `port_name=sfp_port_0` for both SFP trials and
  changes `target_module_name` between `nic_card_mount_0` and
  `nic_card_mount_1`.
- Existing Isaac SFP training sampled `sfp_port_0` versus `sfp_port_1` within a
  single NIC, and the accepted randomized stage only moved the NIC by
  `[-0.002, 0.002] m` in y. That does not match the official Gazebo target
  structure, where the target is port 0 on two different mounts separated by
  roughly `0.04 m`.

Decision:

- Do not continue blind training on the old SFP setup.
- Add a separate Gazebo-transfer SFP Isaac task variant:
  - active target fixed to `sfp_port_0`
  - NIC/card y randomized over mount-scale offsets
  - same eval-compatible actor observation shape, so exported policies remain
    loadable by the current Gazebo wrapper
  - no hidden runtime geometry in the deployed policy

## Gazebo-Transfer SFP Task Variant

Code changes:

- `mdp.sample_active_sfp_target(...)` now accepts optional `target_id`.
  - Existing behavior is unchanged when `target_id` is omitted.
  - The Gazebo-transfer variant passes `target_id=0`, which maps to
    `sfp_port_0` in `SFP_TARGET_NAMES`.
- Added `SfpGazeboTransferEventCfg`.
  - Inherits the normal SFP reset events.
  - Overrides NIC/card y randomization from the previous `[-0.002, 0.002] m`
    stage to `[-0.045, 0.005] m`.
  - Keeps `snap_step.y = 0.0` for continuous robustness across and around the
    mount-scale shift.
- Added `AICTaskSfpGazeboTransferEnvCfg`.
  - Inherits `AICTaskSfpEnvCfg`.
  - Uses the Gazebo-transfer event config.
- Registered new Gym task id:

```text
AIC-SFP-Gazebo-Transfer-Task-v0
```

- Added a separate PPO runner config:

```text
aic_sfp_gazebo_transfer
```

The actor observation shape is intentionally unchanged at `3149`:

- 2D SFP task metadata remains `[sfp_port_0, sfp_port_1]`
- Gazebo-transfer training always emits `[1, 0]`
- the actor must infer the mount-scale NIC shift from camera/proprioception,
  matching what the official runtime policy can legally observe

Step 11 later revised this after the first scratch run plateaued: the
Gazebo-transfer SFP task can now opt into 4D official task metadata
`[sfp_port_0, sfp_port_1, nic_card_mount_0, nic_card_mount_1]`. This makes the
new SFP actor observation shape `3151` while preserving the old `3149` path for
existing artifacts.

Local verification:

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

Next remote checks:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
```

Expected: `AIC-SFP-Gazebo-Transfer-Task-v0` appears in the task list.

Then smoke-test and train:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-SFP-Gazebo-Transfer-Task-v0 \
  --agent rsl_rl_sfp_cfg_entry_point \
  --num_envs 64 \
  --max_iterations 1500 \
  --headless \
  --enable_cameras
```
