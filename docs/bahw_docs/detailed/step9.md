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

Progress check after both runs had trained for roughly two hours:

```text
warm latest checkpoint: model_140.pt
warm latest observed rollout success termination: 0.8826 to 0.9322
scratch latest checkpoint: model_120.pt
scratch latest observed rollout success termination: 0.8669 to 0.9120
GPU memory while both train: about 22249 MiB / 24564 MiB
```

Progress check after roughly five hours:

```text
watcher state: still waiting for both Step 9 Python training processes
warm latest checkpoint: model_290.pt
warm latest observed rollout success termination: 0.9091 to 0.9271
scratch latest checkpoint: model_270.pt
scratch latest observed rollout success termination: 0.9145 to 0.9534
GPU memory while both train: about 22249 MiB / 24564 MiB
```

These are training-time rollout termination rates. They are useful health
signals, but they are not the Step 9 acceptance result. Acceptance still depends
on the queued deterministic randomized SFP evaluation reporting overall and
per-target success above `90%`.

Because the two training runs nearly fill the RTX 4090, do not start evaluation
concurrently with both training processes unless one exits or memory headroom
changes. A third Isaac process is likely to starve or OOM.

Post-training evaluation watcher:

```bash
tmux new-session -d -s isaac-step9-eval-after-python-f2cd192
tmux send-keys -t isaac-step9-eval-after-python-f2cd192 "cd ~/IsaacLab" C-m
tmux send-keys -t isaac-step9-eval-after-python-f2cd192 \
  "while docker exec isaac-lab-base bash -lc 'pgrep -f \"[p]ython3 .*step9_sfp_randy002_warm_f2cd192|[p]ython3 .*step9_sfp_randy002_scratch_f2cd192\" >/dev/null'; do date; echo waiting_for_step9_python_training; sleep 300; done; docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && warm=\$(ls -v logs/rsl_rl/aic_sfp_insert/2026-05-12_01-39-53_step9_sfp_randy002_warm_f2cd192/model_*.pt | tail -n 1) && scratch=\$(ls -v logs/rsl_rl/aic_sfp_insert/2026-05-12_01-40-05_step9_sfp_randy002_scratch_f2cd192/model_*.pt | tail -n 1) && echo WARM_CKPT:\$warm && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 256 --max_episode_steps 150 --checkpoint \$warm --lateral_threshold 0.015 --orientation_threshold 0.25 --depth_threshold 0.015 --failure_sample_count 10 --headless --enable_cameras && echo SCRATCH_CKPT:\$scratch && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py --task AIC-SFP-Task-v0 --agent rsl_rl_sfp_cfg_entry_point --num_envs 32 --num_eval_episodes 256 --max_episode_steps 150 --checkpoint \$scratch --lateral_threshold 0.015 --orientation_threshold 0.25 --depth_threshold 0.015 --failure_sample_count 10 --headless --enable_cameras'; echo STEP9_POSTTRAIN_EVAL_EXIT:\$?" C-m
```

The first watcher session, `isaac-step9-eval-after-train-f2cd192`, was
interrupted with `Ctrl-C` and replaced by the stricter Python-process watcher
above so the polling command does not accidentally match its own shell.

Post-training deterministic randomized SFP evaluation result:

```text
warm checkpoint:
  /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-39-53_step9_sfp_randy002_warm_f2cd192/model_1518.pt
  log: /workspace/isaaclab/aic/logs/aic_eval/20260513_061332_AIC-SFP-Task-v0.log
  overall: 235/256 = 0.917969
  sfp_port_0: 114/129 = 0.883721
  sfp_port_1: 121/127 = 0.952756
  decision: reject for Step 9 acceptance because sfp_port_0 is below 90%.

scratch checkpoint:
  /workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-40-05_step9_sfp_randy002_scratch_f2cd192/model_1499.pt
  log: /workspace/isaaclab/aic/logs/aic_eval/20260513_061729_AIC-SFP-Task-v0.log
  overall: 238/256 = 0.929688
  sfp_port_0: 123/132 = 0.931818
  sfp_port_1: 115/124 = 0.927419
  decision: select scratch model_1499.pt as the randomized SFP candidate.
```

## Official Gazebo Eval Wrapper Scaffold

While the randomized SFP training runs continue, an official Gazebo evaluation
wrapper scaffold was added for the eventual selected checkpoint:

- `aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py`
  starts the official `aic_eval` Gazebo/AIC engine path, the `aic_model`
  participant policy process, and optional wrist-camera topic recording in
  separate tmux sessions.
- `aic_model/aic_model/RslRlCheckpointPolicy.py` is the ROS policy scaffold.
  It receives raw checkpoint paths or exported policy artifact paths through
  environment variables and logs official `Task` / `Observation` metadata.
- `docs/bahw_docs/eval_wrapper/README.md` documents usage, expected outputs,
  and the remaining adapter work.

This is intentionally not marked as a functional checkpoint replay path yet.
The official Gazebo eval stack runs `aic_model.Policy.insert_cable()` against
ROS `Observation` messages, while the Isaac RSL-RL `.pt` checkpoints expect the
Isaac actor observation vector:

```text
task_metadata
joint_pos_rel
joint_vel_rel
eef_pose
body_forces
center_rgb_resnet18
left_rgb_resnet18
right_rgb_resnet18
last_action
```

The remaining deployment work is to export/load the actor artifact, reconstruct
that observation vector from official Gazebo observations and task metadata,
and translate the six-dimensional relative-IK actor action into safe
`MotionUpdate` or `JointMotionUpdate` commands. Until then,
`RslRlCheckpointPolicy` returns failure after logging the received task and
observation snapshot.

Static checks run locally:

```bash
python3 -m py_compile \
  aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  aic_model/aic_model/RslRlCheckpointPolicy.py
```

Dry-run command:

```bash
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --checkpoint /tmp/model.pt \
  --session-prefix dryrun-rslrl \
  --record-camera-bag \
  --camera-bag-duration-sec 10 \
  --dry-run
```

Dry-run result:

```text
eval_session: dryrun-rslrl-eval
model_session: dryrun-rslrl-model
camera_session: dryrun-rslrl-camera-bag
Expected scoring output:
/Users/aloy/projects/aic/logs/gazebo_eval/<timestamp>/scoring.yaml
```

Host scaffold run against selected SFP checkpoint:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --checkpoint /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_model_1499.pt \
  --task-kind sfp \
  --session-prefix gazebo-rslrl-sfp-model1499 \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace
```

Run notes:

```text
local wrapper fixes pushed on aloy:
  8f1b390 Fix Gazebo eval wrapper policy import path
  1d75df4 Run Gazebo model policy inside pixi shell

selected checkpoint copied from Isaac container to host:
  /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_model_1499.pt

wrapper result directory:
  /var/home/bahw/ws_aic/src/aic/logs/gazebo_eval/20260513_193949

official scoring:
  total: 3
  trial_1 tier_1 score: 1, tier_2 score: 0, tier_3 score: 0
  trial_2 tier_1 score: 1, tier_2 score: 0, tier_3 score: 0
  trial_3 tier_1 score: 1, tier_2 score: 0, tier_3 score: 0
```

The scaffold run is useful because it proves the official Gazebo stack can
launch, load the branch-local `RslRlCheckpointPolicy`, receive official task
metadata and camera observations, and write `scoring.yaml` plus trial bags. The
zero tier-2/tier-3 scores are expected: the scaffold returns `False` until the
Gazebo observation/action adapter is implemented.

## First SFP Actor Artifact Export and Official Eval Run

The selected randomized SFP checkpoint was exported to a lightweight
TorchScript MLP actor without launching Isaac Sim, because the Isaac
`play.py` export path timed out before writing `exported/policy.pt`.

Host export command:

```bash
cd ~/ws_aic/src/aic
pixi run python3 aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py \
  --checkpoint logs/checkpoints/step9_sfp_randy002_scratch_model_1499.pt \
  --output logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --obs-dim 3149 \
  --action-dim 6
```

Exported artifact:

```text
/var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt
```

The SFP ROS adapter was then implemented in
`aic_model/aic_model/RslRlCheckpointPolicy.py`:

- loads the exported TorchScript artifact
- routes SFP using official `Task` metadata
- uses only official `Observation` fields
- reconstructs the 3149D Isaac actor input
- fills six Gazebo UR joints into the 46D Isaac joint slots
- applies the known Gazebo-to-Isaac shoulder-pan sign conversion
- extracts ResNet18 ImageNet-V1 logits from `center_image`, `left_image`, and
  `right_image`
- converts the 6D actor output to small Cartesian `MotionUpdate` deltas

First actor-backed official eval command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-sfp-adapter-04cbb82 \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace
```

Run directory:

```text
/var/home/bahw/ws_aic/src/aic/logs/gazebo_eval/20260513_203832
```

Official scoring result:

```text
total: 3
trial_1:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: No insertion detected. Final plug port distance: 0.30m.
trial_2:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: No insertion detected. Final plug port distance: 0.20m.
trial_3:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: Task not completed.
```

Interpretation:

- Tier 1 passed on all official trials, so the model package was accepted by
  the official Gazebo path.
- The two SFP trials did not insert. The current action conversion is therefore
  not a functional Gazebo deployment mapping yet.
- The third official trial was SC. The current adapter rejects SC because only
  SFP is implemented.
- The first camera recorder used direct camera topics from the host pixi
  environment and exited without writing a bag because that environment does
  not include the `ros2 bag` verb. The wrapper was updated to record
  `/observations` from the sourced `aic_eval` container by default, because
  that is the official policy observation stream and includes all three wrist
  images.

Updated video-evidence path:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-sfp-adapter-obsvideo \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace
```

After the run, convert one camera field from the observation bag:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc '
  cd /var/home/bahw/ws_aic/src/aic &&
  source /ws_aic/install/setup.bash &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/<timestamp>/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field center_image \
    --output logs/gazebo_eval/<timestamp>/videos/center_image.mp4 \
    --fps 20
'
```

Repeat with `--image-field left_image` or `--image-field right_image` for
additional review videos.

## Final Video-Recorded Official Eval Run

After fixing the recorder to use the sourced `aic_eval` container, the
video-recorded official run was repeated from a clean process state.

Command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-sfp-obsvideo-final-7d9f825 \
  --record-camera-bag \
  --camera-bag-duration-sec 240 \
  --replace
```

Run directory:

```text
/var/home/bahw/ws_aic/src/aic/logs/gazebo_eval/20260513_205601
```

Observation bag:

```text
logs/gazebo_eval/20260513_205601/camera_bags/wrist_cameras/
  metadata.yaml
  wrist_cameras_0.mcap
```

Bag metadata:

```text
topic: /observations
type: aic_model_interfaces/msg/Observation
message_count: 2021
duration: 137.578657180s
bag size: about 20G
```

Official scoring:

```text
total: 3
trial_1:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: No insertion detected. Final plug port distance: 0.27m.
trial_2:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: No insertion detected. Final plug port distance: 0.17m.
trial_3:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: Task not completed.
```

MP4 export command:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc '
  cd /var/home/bahw/ws_aic/src/aic &&
  source /ws_aic/install/setup.bash &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260513_205601/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field center_image \
    --output logs/gazebo_eval/20260513_205601/videos/center_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260513_205601/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field left_image \
    --output logs/gazebo_eval/20260513_205601/videos/left_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260513_205601/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field right_image \
    --output logs/gazebo_eval/20260513_205601/videos/right_image.mp4 \
    --fps 20
'
```

Video outputs:

```text
logs/gazebo_eval/20260513_205601/videos/center_image.mp4  22M
logs/gazebo_eval/20260513_205601/videos/left_image.mp4    20M
logs/gazebo_eval/20260513_205601/videos/right_image.mp4   28M
```

Each MP4 contains `2021` frames at `20 FPS`.

Current blockers after this run:

- The SFP actor runs legally from official task/observation inputs, but the
  Gazebo action conversion does not reproduce Isaac insertion behavior.
- The SC adapter is not implemented, so any official SC trial returns failure.
- Next deployment work should focus on validating the action-frame/control
  mapping before more training.

## SFP Gazebo Adapter Diagnostics And Rerun

The failed `20260513_205601` run was analyzed with a new host-side rosbag
diagnostic:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc '
  cd /var/home/bahw/ws_aic/src/aic &&
  source /ws_aic/install/setup.bash &&
  python3 aic_utils/aic_training_utils/scripts/analyze_gazebo_eval_bag.py \
    logs/gazebo_eval/20260513_205601/bag_1_20260513_205601 \
    --include-scoring-tf
'
```

Implemented diagnostic/fix commits:

```text
2aee2a6 Add Gazebo eval bag analyzer
42be090 Summarize scoring TF in eval bag analyzer
2d1b138 Compose scoring TF in eval bag analyzer
e6ca495 Fix SFP Gazebo action replay horizon
cbc4131 Disable harmful SFP Gazebo prepose
49a88e8 Tune SFP Gazebo actor horizon
```

Findings:

- The old legal SFP prepose command moved the plug away from the official
  Gazebo task start. The official task already starts the SFP trial close to
  the target port, so replaying the Isaac near-port joint preset was harmful.
- Long SFP actor replay horizons moved the TCP past the useful approach region
  and degraded the second SFP trial.
- ResNet18 model loading was happening inside the first control loop and
  consumed part of the time budget.
- Replaying only translational deltas ignored the actor's rotational output.

SFP adapter changes in `aic_model/aic_model/RslRlCheckpointPolicy.py`:

- `AIC_RSLRL_ENABLE_SFP_PREPOSE` now defaults to `false`.
- `AIC_RSLRL_SFP_MAX_CONTROL_SEC` now defaults to `9.0`.
- ResNet18 is loaded before the SFP control timer starts.
- Actor rotation output is replayed as a small axis-angle delta composed with
  the current Gazebo TCP orientation.
- The SFP near-port presets were updated from the final randomized SFP training
  event values, but they remain disabled by default for official Gazebo eval.

Wrapper change:

- `run_gazebo_checkpoint_eval.py` accepts repeatable
  `--model-env KEY=VALUE` overrides for controlled evaluation experiments.

Best post-fix official Gazebo command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-sfp-horizon9-49a88e8 \
  --record-camera-bag \
  --camera-bag-duration-sec 150 \
  --replace
```

Run directory:

```text
/var/home/bahw/ws_aic/src/aic/logs/gazebo_eval/20260514_005106
```

Official scoring:

```text
total: 92.514068059037598
trial_1:
  tier_1: 1
  tier_2: 21.910012164914278
  tier_3: 21.970096731026825
  message: No insertion detected. Final plug port distance: 0.05m.
trial_2:
  tier_1: 1
  tier_2: 22.577460031209593
  tier_3: 23.056499131886891
  message: No insertion detected. Final plug port distance: 0.05m.
trial_3:
  tier_1: 1
  tier_2: 0
  tier_3: 0
  message: Task not completed.
```

The `9s` horizon is the best committed default so far. A `15s` control-horizon
experiment was run with:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-sfp-horizon15 \
  --record-camera-bag \
  --camera-bag-duration-sec 180 \
  --model-env AIC_RSLRL_SFP_MAX_CONTROL_SEC=15 \
  --replace
```

The `15s` run regressed to total `73.452717931903607`, so `9s` remains the
checked-in default.

Video export for the best post-fix run:

```bash
cd ~/ws_aic/src/aic
mkdir -p logs/gazebo_eval/20260514_005106/videos
distrobox enter -r aic_eval -- bash -lc '
  source /ws_aic/install/setup.bash &&
  cd /var/home/bahw/ws_aic/src/aic &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_005106/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field center_image \
    --output logs/gazebo_eval/20260514_005106/videos/center_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_005106/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field left_image \
    --output logs/gazebo_eval/20260514_005106/videos/left_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_005106/camera_bags/wrist_cameras \
    --topic /observations \
    --image-field right_image \
    --output logs/gazebo_eval/20260514_005106/videos/right_image.mp4 \
    --fps 20
'
```

Video outputs on the host:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_005106/videos/center_image.mp4  7.6M
~/ws_aic/src/aic/logs/gazebo_eval/20260514_005106/videos/left_image.mp4    9.7M
~/ws_aic/src/aic/logs/gazebo_eval/20260514_005106/videos/right_image.mp4   11M
```

Current blockers after the rerun:

- The SFP deployment path is much closer, but still does not trigger official
  insertion. The next SFP work should inspect the final `5cm` approach and
  decide whether the remaining issue is action-frame mismatch, controller
  convergence, policy observation mismatch, or the need for a separate final
  insertion approach stage.
- In that rerun, SC routing still failed because the SC Gazebo adapter was not
  implemented yet.

## Adapter Review And Local Fixes

Recent-commit review identified these concrete risks:

- `RslRlCheckpointPolicy` still rejected all non-SFP tasks, so official SC trial
  3 could never run a learned actor.
- The hand-built Gazebo actor observation may have the correct shape but still
  needs semantic validation against Isaac observations, especially image
  features, TCP pose frame, and wrench/body-force padding.
- The SFP action replay is only an approximation of Isaac relative IK. It sends
  small Cartesian targets through the official controller, but official Gazebo
  still stops about `5cm` from insertion.
- The lightweight exporter did not accept the documented `--obs-dim` /
  `--action-dim` flags and only handled unprefixed `mlp.*` state-dict keys.
- The wrapper records `/observations` by default, while the video converter
  defaulted to `/center_camera/image`.

Local fixes:

- Added first SC route/observation support in
  `aic_model/aic_model/RslRlCheckpointPolicy.py`.
  - SC now maps official `sc_port_0` / `sc_port_1`-style module names to the
    Isaac one-hot targets `[sc_port, sc_port_2]`.
  - SC and SFP share the same 3149D actor observation builder.
  - SC action scale defaults to the Isaac SC relative-IK scale `0.05` and can be
    overridden with `AIC_RSLRL_SC_POSITION_SCALE` /
    `AIC_RSLRL_SC_ROTATION_SCALE`.
- Generalized `MotionUpdate` generation so actor replay can use either
  `base_link` absolute targets or `gripper/tcp` relative targets via
  `AIC_RSLRL_SC_COMMAND_FRAME` / `AIC_RSLRL_SFP_COMMAND_FRAME`.
- Added optional SFP final-settle experiment controls:
  `AIC_RSLRL_SFP_FINAL_SETTLE_SEC` and `AIC_RSLRL_SFP_FINAL_SETTLE_STEP`. This is
  legal because it uses only the current official controller observation and a
  TCP-frame relative command; it does not use scoring TF or hidden Gazebo state.
- Added optional `AIC_RSLRL_REQUIRE_RESNET18=true` so future validation can fail
  fast if the image encoder is unavailable instead of silently zeroing 3000
  observation dimensions.
- Updated `export_rslrl_mlp_actor.py` to accept `--obs-dim` / `--action-dim` and
  strip common actor prefixes such as `actor.mlp.*`.
- Changed `rosbag_images_to_video.py` to default to `/observations`.

Local checks:

```bash
python3 -m py_compile \
  aic_model/aic_model/RslRlCheckpointPolicy.py \
  aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py \
  aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py
```

Next host validation:

1. Export the accepted SC actor to TorchScript:

   ```bash
   cd ~/ws_aic/src/aic
   pixi run python3 aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py \
     --checkpoint <accepted-sc-checkpoint.pt> \
     --output logs/checkpoints/step6_sc_policy.pt \
     --obs-dim 3149 \
     --action-dim 6
   ```

2. Run official Gazebo with both actor artifacts:

   ```bash
   cd ~/ws_aic/src/aic
   python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
     --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
     --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
     --session-prefix gazebo-final-adapter \
     --record-camera-bag \
     --camera-bag-duration-sec 180 \
     --replace
   ```

3. If SFP remains at the `5cm` miss, test only one controlled variable per run,
   starting with:

   ```bash
   --model-env AIC_RSLRL_SFP_FINAL_SETTLE_SEC=2 \
   --model-env AIC_RSLRL_SFP_FINAL_SETTLE_STEP=-0.002
   ```

## SC/SFP Combined Gazebo Baseline

After commit `0e12ccd`, the accepted SC checkpoint was exported on the host
without touching the dirty Isaac training checkout:

```bash
cd ~/ws_aic/src/aic
mkdir -p logs/checkpoints
docker cp \
  isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/aic_sc_insert/2026-05-10_16-04-17_step6_sc_bc_strict_2e5987b/model_1000.pt \
  logs/checkpoints/step6_sc_model_1000.pt
pixi run python3 aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py \
  --checkpoint logs/checkpoints/step6_sc_model_1000.pt \
  --output logs/checkpoints/step6_sc_policy.pt \
  --obs-dim 3149 \
  --action-dim 6
```

Export result:

```text
input_dim: 3149
output_dim: 6
logs/checkpoints/step6_sc_policy.pt  6.8M
```

Qualification-like combined run:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-base \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true
```

Result path:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_093740/scoring.yaml
~/ws_aic/src/aic/logs/gazebo_eval/20260514_093740/videos/center_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260514_093740/videos/left_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260514_093740/videos/right_image.mp4
```

Score:

```text
total: 91.318105837048677
trial_1 SFP: tier_1=1, tier_2=21.6578, tier_3=19.7977, no insertion, final distance 0.06m
trial_2 SFP: tier_1=1, tier_2=22.5798, tier_3=24.2828, no insertion, final distance 0.05m
trial_3 SC:  tier_1=1, tier_2=0,       tier_3=0,       no insertion, final distance 0.32m
```

Key diagnostics:

- ResNet18 loaded successfully with ImageNet V1 weights, so the run did not use
  zeroed image features.
- SFP still stops around `5cm` from the target. Offline scoring-TF diagnostics
  are for analysis only; the runtime policy still uses only official task and
  observation messages.
- SC now routes and executes the SC actor, but the official SC trial starts
  after both SFP trials with the TCP near the NIC/SFP area. The accepted SC
  checkpoint was trained from near-SC reset states, so the direct actor replay
  starts out of distribution. Trial 3 moved the TCP by roughly
  `[+0.0846, -0.0913, +0.0563]m`, but the physical `sc_tip_link` ended about
  `0.32m` from `task_board/sc_port_1/sc_port_base_link_entrance`.

Next legal adapter change:

- Add an optional SC joint-space prepose using the Step 6 first-success joint
  seeds, selected only from official SC task metadata. This is legal for eval
  because it uses no scoring TF, hidden Gazebo state, or ground-truth geometry.
  It is intended to bring the official sequential trial back into the same
  near-port distribution as the accepted SC actor.

## SC Prepose And SFP Final-Settle Test

Commit `1c18264` added the legal SC joint-space prepose. A clean
qualification-like rerun also enabled the optional SFP final-settle experiment:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-prepose-settle-clean \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_SFP_FINAL_SETTLE_SEC=2 \
  --model-env AIC_RSLRL_SFP_FINAL_SETTLE_STEP=-0.002
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_095112/scoring.yaml
total: 58.255106729418905
trial_1 SFP: no insertion, final distance 0.08m
trial_2 SFP: no insertion, final distance 0.09m
trial_3 SC:  no insertion, final distance 0.28m
```

Decision:

- Reject `AIC_RSLRL_SFP_FINAL_SETTLE_SEC=2` / `STEP=-0.002`; it worsened SFP
  from the prior `0.05-0.06m` miss to `0.08-0.09m`.
- The SC prepose with shoulder mirroring only improved the SC final distance
  from `0.32m` to about `0.28m`. Offline diagnostics showed the SC joint preset
  command used Gazebo shoulder-pan `-0.7603` for `sc_port_1`.
- A follow-up no-mirror isolation run under
  `logs/gazebo_eval/20260514_095648/` regressed SC to `0.60m`, so keep the
  default `AIC_RSLRL_SC_PREPOSE_MIRROR_SHOULDER=true`.
- The next legal check is mirrored SC prepose with SFP final-settle disabled, so
  the SFP result is not contaminated by the rejected final-settle experiment.

## Final mirrored SC prepose, no SFP final settle

Host command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-sc-mirror \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/scoring.yaml
total: 92.631565804455263
trial_1 SFP: tier1=1, tier2=18.320959564659177, tier3=23.714641175284299,
             no insertion, final distance 0.05m
trial_2 SFP: tier1=1, tier2=22.595965064511788, tier3=25,
             no insertion, final distance 0.04m
trial_3 SC:  tier1=1, tier2=0, tier3=0,
             no insertion, final distance 0.29m
```

Video export:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc '
  cd /var/home/bahw/ws_aic/src/aic &&
  source /ws_aic/install/setup.bash &&
  ros2 bag reindex logs/gazebo_eval/20260514_100007/camera_bags/wrist_cameras &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_100007/camera_bags/wrist_cameras \
    --image-field center_image \
    --output logs/gazebo_eval/20260514_100007/videos/center_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_100007/camera_bags/wrist_cameras \
    --image-field left_image \
    --output logs/gazebo_eval/20260514_100007/videos/left_image.mp4 \
    --fps 20 &&
  python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
    logs/gazebo_eval/20260514_100007/camera_bags/wrist_cameras \
    --image-field right_image \
    --output logs/gazebo_eval/20260514_100007/videos/right_image.mp4 \
    --fps 20
'
```

Exported files:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/videos/center_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/videos/left_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/videos/right_image.mp4
```

Decision:

- This is the best legal official run so far by total score.
- Keep `AIC_RSLRL_SC_PREPOSE_MIRROR_SHOULDER=true`.
- Keep SFP final-settle disabled; the controlled final-settle run worsened SFP.
- The run still does not solve insertion. SFP remains a close controller/action
  mapping miss, and SC remains out of distribution even with the legal prepose.

## Best-run scoring bag diagnostics

Offline-only diagnostic command:

```bash
cd ~/ws_aic/src/aic
distrobox enter -r aic_eval -- bash -lc '
  cd /var/home/bahw/ws_aic/src/aic &&
  source /ws_aic/install/setup.bash &&
  python3 aic_utils/aic_training_utils/scripts/analyze_gazebo_eval_bag.py \
    logs/gazebo_eval/20260514_100007/bag_trial_1_20260514_100050_977 \
    --sample-limit 8 --include-scoring-tf &&
  python3 aic_utils/aic_training_utils/scripts/analyze_gazebo_eval_bag.py \
    logs/gazebo_eval/20260514_100007/bag_trial_2_20260514_100125_194 \
    --sample-limit 8 --include-scoring-tf &&
  python3 aic_utils/aic_training_utils/scripts/analyze_gazebo_eval_bag.py \
    logs/gazebo_eval/20260514_100007/bag_trial_3_20260514_100137_430 \
    --sample-limit 8 --include-scoring-tf
'
```

Key observations:

- SFP trial 1 moved the TCP by `[-0.0255, 0.0027, -0.0846]` and the SFP tip by
  `[0.0254, -0.0026, -0.0846]`. The final SFP tip was `0.01120m` from one
  entrance frame but `0.04921m` from the corresponding deeper SFP port link.
- SFP trial 2 moved the TCP by `[-0.0038, 0.0142, -0.1026]` and the SFP tip by
  `[0.0037, -0.0141, -0.1026]`. The final SFP tip was `0.02820m` from one
  entrance frame and `0.04308m` from the corresponding deeper SFP port link.
- The actor replay mostly commands small base-frame negative-z targets, with
  mean pose-command delta from latest TCP around `[0.00029, -0.00005, -0.00184]`
  in trial 1 and `[0.00001, -0.00019, -0.00170]` in trial 2.
- SC trial 3 used the mirrored legal joint prepose command for `sc_port_1`, then
  the SC actor moved the SC tip to about `0.29065m` from the target port. This
  motivates an actor-disabled diagnostic to measure whether the actor handoff is
  worse than the legal prepose alone.

Local adapter changes for the next controlled eval:

- Added disabled-by-default `AIC_RSLRL_SFP_BASE_INSERT_SEC` and
  `AIC_RSLRL_SFP_BASE_INSERT_STEP`. This runs a base-frame negative-z insertion
  push after the SFP actor loop, using only the official controller observation.
- Added `AIC_RSLRL_SC_ACTOR_ENABLED=false` as a diagnostic toggle. This lets the
  official eval measure the legal SC prepose without actor handoff.

## Controlled timing and base-insert evals

30 Hz replay command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-hz30 \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_CONTROL_HZ=30
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_101451/scoring.yaml
total: 90.001609774891335
trial_1 SFP: no insertion, final distance 0.05m
trial_2 SFP: no insertion, final distance 0.05m
trial_3 SC:  no insertion, final distance 0.29m
```

Decision:

- Reject `AIC_RSLRL_CONTROL_HZ=30` as a default. It reduced total score.
- The logs still revealed a real replay issue: the actor loop planned 90 SFP
  steps at 10 Hz or 270 steps at 30 Hz, but stopped early because it mixed
  wall-clock elapsed checks with sim-time sleeps. Patch the wrapper to replay
  the fixed planned step count and sleep in sim time each iteration.

Base-frame SFP insert command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-baseinsert \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_SFP_BASE_INSERT_SEC=2 \
  --model-env AIC_RSLRL_SFP_BASE_INSERT_STEP=-0.002
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_101713/scoring.yaml
total: 91.55788002615509
trial_1 SFP: no insertion, final distance 0.04m
trial_2 SFP: no insertion, final distance 0.05m
trial_3 SC:  no insertion, final distance 0.29m
```

Decision:

- Reject the tested base-frame insert setting. It improved trial 1 distance
  slightly but worsened total score and did not trigger insertion.
- Keep `AIC_RSLRL_SFP_BASE_INSERT_SEC=0` by default.
- Next eval should test the fixed-step replay patch without extra base insert.

Fixed-step replay command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-fixedstep \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_102325/scoring.yaml
total: 79.399977111014948
trial_1 SFP: no insertion, final distance 0.06m
trial_2 SFP: no insertion, final distance 0.06m
trial_3 SC:  no insertion, final distance 0.29m
```

Decision:

- Reject fixed-step replay as the default. It ran the intended 90 actor steps,
  but official SFP scoring worsened, so full replay likely overshoots the useful
  approach region.
- Keep the previous wall-budgeted replay default and expose full fixed-step
  replay only behind `AIC_RSLRL_FIXED_STEP_REPLAY=true`.

SC prepose-only diagnostic command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-sc-prepose-only \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_SC_ACTOR_ENABLED=false
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_102722/scoring.yaml
total: 78.759224629052937
trial_1 SFP: no insertion, final distance 0.06m
trial_2 SFP: no insertion, final distance 0.08m
trial_3 SC:  no insertion, final distance 0.28m
```

Decision:

- The SC actor is not the primary source of the SC failure. Disabling it after
  the legal prepose only changes SC from about `0.29m` to `0.28m`.
- The SC legal prepose itself is not close enough for final qualification.

Latest-code default confirmation command:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-latest-default \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_102952/scoring.yaml
total: 90.51278931029411
trial_1 SFP: no insertion, final distance 0.05m
trial_2 SFP: no insertion, final distance 0.05m
trial_3 SC:  no insertion, final distance 0.32m
```

Decision:

- Branch defaults are still legal and match the intended safe settings:
  `AIC_RSLRL_FIXED_STEP_REPLAY=false`, `AIC_RSLRL_SFP_BASE_INSERT_SEC=0`, and
  `AIC_RSLRL_SC_ACTOR_ENABLED=true`.
- The best saved official evidence remains
  `logs/gazebo_eval/20260514_100007/`, total `92.631565804455263`.

SFP command-frame diagnostic:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-sfp-tcpframe \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_SFP_COMMAND_FRAME=gripper/tcp
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_103409/scoring.yaml
total: 25.33397048270108
trial_1 SFP: no insertion, final distance 0.12m
trial_2 SFP: no insertion, final distance 0.10m
trial_3 SC:  no insertion, final distance 0.29m
```

Decision:

- Reject `AIC_RSLRL_SFP_COMMAND_FRAME=gripper/tcp`. It is much worse than the
  base-link replay default.
- Added disabled-by-default `AIC_RSLRL_ZERO_JOINT_OBS=true` as the next
  observation-mismatch diagnostic. Gazebo joint values appear to be in a
  different convention from the Isaac actor's joint observations, so this tests
  whether removing misleading joint inputs helps.

Zero-joint-observation diagnostic:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact /var/home/bahw/ws_aic/src/aic/logs/checkpoints/step9_sfp_randy002_scratch_policy.pt \
  --session-prefix gazebo-final-zerojoint \
  --record-camera-bag \
  --camera-bag-duration-sec 900 \
  --replace \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_ZERO_JOINT_OBS=true
```

Result:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260514_103958/scoring.yaml
total: 82.898135541085352
trial_1 SFP: no insertion, final distance 0.05m
trial_2 SFP: no insertion, final distance 0.08m
trial_3 SC:  no insertion, final distance 0.27m
```

Decision:

- Reject `AIC_RSLRL_ZERO_JOINT_OBS=true`. It does not improve insertion and
  worsens total score.
- Keep Gazebo joint observations enabled by default.
- Remaining blocker is not a single wrapper toggle: SFP needs a better final
  approach/controller strategy from the `0.04-0.05m` miss, and SC needs a better
  legal official-start approach before actor handoff.
