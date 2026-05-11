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
