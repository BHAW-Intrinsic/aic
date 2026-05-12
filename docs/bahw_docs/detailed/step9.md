# Step 9: Distillation And Routing

Status: blocked pending randomized SFP reliability pass.

Step 9 should not start until the specialist policies are reliable enough to be
worth distilling or exporting. The actor observation groups are already
eval-compatible for both SC and SFP, so direct export may be sufficient later
without a separate distillation pass.

Current specialist status:

- SC has an accepted neural checkpoint at `233/256` deterministic Isaac
  successes plus a saved video artifact.
- SFP has an accepted PPO checkpoint for fixed-NIC final-stage insertion with
  small reset noise:
  `/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b/model_19.pt`
- SFP fixed-NIC detached eval:
  `118/128` successes under `lateral <0.015`, `orientation <0.25`,
  `depth >0.015`.
- SFP with `position_noise=0.002` detached eval:
  `121/128` successes, with both SFP ports above `94%`.

Current blocker:

- SFP has not yet been validated with NIC/card y randomization.
- The current SFP near-port reset uses fixed per-target joint presets:
  `reset_robot_near_sfp_port` selects from `SFP_NEAR_PORT_JOINT_PRESETS` and
  adds optional joint noise.
- Because the reset does not adapt to randomized NIC pose, enabling NIC y
  randomization directly would move the target away from the fixed reset
  curriculum.

Pre-Step-9 randomized SFP plan:

1. Change the SFP reset/randomization curriculum so the target port location can
   vary independently of the robot joint state. The policy should not be able to
   solve the randomized task by memorizing fixed near-port joint presets.
2. Keep the actor eval-compatible. It may use the existing wrist-camera ResNet18
   image features, proprioception, forces, last action, and official task
   metadata. It must not receive privileged plug-to-port geometry.
3. Run two PPO tracks under the same randomized SFP setup:
   - Track A: warm-start from the best fixed-NIC checkpoint as a weight
     initialization only:
     `/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b/model_19.pt`
   - Track B: PPO from scratch as a control, using the same randomized reset,
     rewards, observations, and evaluation gates.
4. Evaluate both tracks with deterministic playback and per-target metrics. If
   the scratch control learns better randomized insertion, prefer the scratch
   checkpoint over the warm-started checkpoint.
5. Only after randomized SFP is reliable, decide between direct export and
   distillation.
6. Add final Gazebo wrapper routing using official `Task.msg` metadata:
   `plug_type` / `port_type` select SC vs SFP checkpoint.

Reason for the two-track plan:

- The current fixed-NIC checkpoint contains useful insertion behavior, but may
  also encode a fixed-location shortcut.
- The scratch run tests whether the randomized setup is learnable without that
  shortcut.
- The selected SFP candidate should be the policy that performs best under the
  randomized evaluation, not necessarily the one initialized from the older
  checkpoint.

Open decisions before launching remote runs:

- Initial NIC/card y-randomization range for the first curriculum stage:
  accepted at `[-0.002, 0.002]` meters.
- Perception path: keep the existing ResNet18 camera features first. Add a
  separate port-entrance detector only if randomized PPO stalls.
- Acceptance: `>90%` deterministic Isaac success over randomized port
  positions, with both SFP targets above `90%`, using the current intermediate
  gate (`lateral <0.015`, `orientation <0.25`, `depth >0.015`).
- Training schedule: run the warm-start and scratch PPO tracks in parallel if
  the remote 4090 has enough available capacity.

Implementation start:

- Changed `SfpEventCfg.randomize_board_and_parts` so `nic_card` samples
  continuous `y` offsets from `[-0.002, 0.002]` meters.
- Set `snap_step.y` to `0.0` for this SFP curriculum. Keeping the previous
  `0.04` meter snap grid would make all samples in the `[-0.002, 0.002]` range
  snap back to `0.0`, silently disabling the intended randomization.
- Actor observations are unchanged and remain eval-compatible.
- Updated `scripts/rsl_rl/evaluate.py` to print
  `active_port_entry_y_range_env` plus per-target port-entry `y` min/max, so
  success logs show that the evaluated episodes used randomized target
  positions.

## Randomized SFP Stage 1 Runs

Commit:

```text
f2cd192 Add randomized SFP port curriculum
```

Remote sync:

```bash
cd ~/IsaacLab/aic
git fetch origin
git switch aloy
git pull --ff-only
git status --short
```

Result:

```text
Fast-forward to f2cd192
?? logs/
```

Smoke session:

```bash
tmux new-session -d -s isaac-step9-smoke-f2cd192
tmux send-keys -t isaac-step9-smoke-f2cd192 "cd ~/IsaacLab" C-m
tmux send-keys -t isaac-step9-smoke-f2cd192 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./_isaac_sim/python.sh -m py_compile aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 8 --max_iterations 1 --run_name step9_sfp_randy002_smoke_f2cd192 --headless --enable_cameras'; echo STEP9_SMOKE_EXIT:\$?" C-m
```

Smoke result:

```text
STEP9_SMOKE_EXIT:0
log_dir: /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-38-11_step9_sfp_randy002_smoke_f2cd192
```

Warm-start PPO session:

```bash
tmux new-session -d -s isaac-step9-warm-randy002-f2cd192
tmux send-keys -t isaac-step9-warm-randy002-f2cd192 "cd ~/IsaacLab" C-m
tmux send-keys -t isaac-step9-warm-randy002-f2cd192 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --resume --load_run 2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b --checkpoint model_19.pt --run_name step9_sfp_randy002_warm_f2cd192 --headless --enable_cameras'; echo STEP9_WARM_RANDY002_EXIT:\$?" C-m
```

Initial warm-start output:

```text
log_dir: /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-39-53_step9_sfp_randy002_warm_f2cd192
Learning iteration 19/1519
Episode_Termination/sfp_insertion_success: 0.2909
Episode_Termination/time_out: 0.3468
```

Scratch PPO session:

```bash
tmux new-session -d -s isaac-step9-scratch-randy002-f2cd192
tmux send-keys -t isaac-step9-scratch-randy002-f2cd192 "cd ~/IsaacLab" C-m
tmux send-keys -t isaac-step9-scratch-randy002-f2cd192 \
  "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 64 --max_iterations 1500 --run_name step9_sfp_randy002_scratch_f2cd192 --headless --enable_cameras'; echo STEP9_SCRATCH_RANDY002_EXIT:\$?" C-m
```

Initial scratch output:

```text
log_dir: /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-40-05_step9_sfp_randy002_scratch_f2cd192
Learning iteration 0/1500
Episode_Termination/sfp_insertion_success: 0.3098
Episode_Termination/time_out: 0.3425
```

Resource check after both long runs started:

```text
GPU: NVIDIA GeForce RTX 4090
memory_used: 22233 MiB / 24564 MiB
processes: two Isaac Python training processes, about 10.8 GiB each
```
