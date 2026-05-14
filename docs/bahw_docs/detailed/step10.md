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
