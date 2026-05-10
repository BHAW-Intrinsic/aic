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

Local checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py
git diff --check
```

Status: passed locally.
