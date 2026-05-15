# Step 11: Gazebo-Compatible SFP Retraining

## Goal

Train a deployment-compatible SFP actor on the target structure used by the
official Gazebo qualification eval.

Step 10 showed that the old SFP Isaac task and the official Gazebo SFP task do
not ask quite the same problem:

- old Isaac SFP: choose between `sfp_port_0` and `sfp_port_1` on one NIC
- official Gazebo SFP: insert into `sfp_port_0` on different
  `nic_card_mount_*` modules

The task must keep the actor observation legal for eval, so it does not add
privileged port geometry to the actor. The first implementation kept the
existing 3149D observation shape and forced the policy to infer the mount-scale
shift from camera/proprioception; the later revision adds only official
`Task.target_module_name` metadata and expands the Gazebo-transfer SFP actor to
3151D.

## Initial Implementation

Added a sibling Isaac Lab task:

```text
AIC-SFP-Gazebo-Transfer-Task-v0
```

Files changed:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_gazebo_transfer_cfg.py`

Behavior:

- existing `AIC-SFP-Task-v0` remains unchanged
- active SFP target is fixed to `sfp_port_0`
- NIC/card y randomization is widened to `[-0.045, 0.005] m`
- PPO logs go under experiment `aic_sfp_gazebo_transfer`

Local checks:

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

Commit:

```text
d9ff95e Add Gazebo-transfer SFP task
```

## Remote Plan

After commit/push and host sync:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

If the container copy needs updating:

```bash
docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/
```

Task discovery check:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
```

Training command:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-SFP-Gazebo-Transfer-Task-v0 \
  --agent rsl_rl_sfp_cfg_entry_point \
  --num_envs 64 \
  --max_iterations 1500 \
  --run_name step11_sfp_gazebo_transfer \
  --headless \
  --enable_cameras
```

## Remote Sync And Smoke Test

Pushed branch:

```bash
git push origin aloy
```

Remote host sync:

```bash
cd ~/IsaacLab/aic
git fetch origin
git switch aloy
git pull --ff-only
```

Result:

```text
0e12ccd..d9ff95e  aloy -> origin/aloy
Fast-forward
ISAAC_STEP11_SYNC_EXIT:0
```

Container sync:

```bash
cd ~/IsaacLab
docker cp ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/
docker cp ~/IsaacLab/aic/docs/bahw_docs \
  isaac-lab-base:/workspace/isaaclab/aic/docs/
```

Result:

```text
ISAAC_STEP11_CONTAINER_SYNC_EXIT:0
```

Smoke command:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 1 \
   --max_iterations 0 \
   --headless \
   --enable_cameras"
```

Important output:

```text
Parsing configuration from:
  aic_task.tasks.manager_based.aic_task.aic_task_env_cfg:AICTaskSfpGazeboTransferEnvCfg
Parsing configuration from:
  aic_task.tasks.manager_based.aic_task.agents.rsl_rl_ppo_sfp_gazebo_transfer_cfg:PPORunnerCfg
Logging experiment in directory:
  /workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer
Active Observation Terms in Group: 'policy' (shape: (3149,))
Actor Model: Linear(in_features=3149, out_features=512, ...)
Critic Model: Linear(in_features=3169, out_features=512, ...)
ISAAC_STEP11_SMOKE_EXIT:0
```

## Live Training Run

Started in tmux:

```text
isaac-step11-sfp-gazebo-train-d9ff95e
```

Command:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 64 \
   --max_iterations 1500 \
   --run_name step11_sfp_gazebo_transfer_d9ff95e \
   --headless \
   --enable_cameras"
```

Log path:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_11-50-22_step11_sfp_gazebo_transfer_d9ff95e
```

First iteration:

```text
Learning iteration 0/1500
Total steps: 9600
Steps per second: 296
Mean reward: 209.13
Mean episode length: 67.79
Episode_Reward/sfp_insertion_depth: 53.3603
Episode_Reward/sfp_insertion_success: 0.0911
Episode_Termination/time_out: 0.4899
Episode_Termination/sfp_insertion_success: 0.0692
ETA: 13:28:26
```

Interpretation:

- The widened port-0 mount-shift task is harder than the previous narrow
  randomization task, but not fully sparse.
- Training should be allowed to continue before deciding whether to warm-start
  from the previous SFP checkpoint or adjust the curriculum.

## First Scratch Run Plateau

The first `d9ff95e` scratch run was monitored through iteration 52 and then
stopped cleanly with Ctrl-C after `model_50.pt` was saved. The observed SFP
insertion success stayed roughly in the `0.14-0.26` band and did not show a
clear upward trend:

```text
iteration 0:  sfp_insertion_success=0.0692
iteration 1:  sfp_insertion_success=0.2655
iteration 50: sfp_insertion_success=0.1714
iteration 52: sfp_insertion_success=0.2568
```

Conclusion:

- The task was legal, but too much of the 4 cm Gazebo mount switch had to be
  inferred indirectly from vision/proprioception.
- PPO exploration was also too small for the 0.003 m SFP action scale, so the
  initial policy mostly pushed along the insertion axis and did not explore
  enough lateral correction.

## Legal Target-Module Metadata Revision

The official Gazebo `Task` includes both `port_name` and `target_module_name`.
Using those fields is legal because they are passed directly to the submitted
policy. No scoring TF, Gazebo model state, or hidden geometry is used.

Implementation changes:

- Added Gazebo-style SFP mount IDs in `mdp/geometry.py`:
  `nic_card_mount_0` and `nic_card_mount_1`.
- Added `mdp.active_sfp_gazebo_task_one_hot`, returning:

```text
[sfp_port_0, sfp_port_1, nic_card_mount_0, nic_card_mount_1]
```

- Updated `AIC-SFP-Gazebo-Transfer-Task-v0` to use the 4D metadata term.
  Actor observation shape becomes `3151`; critic shape becomes `3171`.
- Changed NIC randomization from one continuous y range to sampled mount IDs:
  `0.0 m` for mount 0 and `-0.04 m` for mount 1, with `+/-0.005 m` jitter.
- Kept the original `AIC-SFP-Task-v0` and older 3149D Gazebo wrapper path
  unchanged.
- Added `AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA=true` to opt the Gazebo wrapper
  into the 3151D SFP actor input. The default remains 3149D for old artifacts.
- Increased the Gazebo-transfer PPO actor initial std to `0.25` and entropy
  coefficient to `0.002` so lateral correction is actually explored.

Local syntax check:

```bash
python3 -m py_compile \
  aic_model/aic_model/RslRlCheckpointPolicy.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_gazebo_transfer_cfg.py
```

Result:

```text
no output; exit 0
```

Verification command on the host:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 1 \
   --max_iterations 0 \
   --headless \
   --enable_cameras"
```

Run session:

```text
isaac-step11-meta-smoke-f00eb1e
```

Important output:

```text
Active Observation Terms in Group: 'policy' (shape: (3151,))
task_metadata shape: (4,)
Actor Model: Linear(in_features=3151, out_features=512, ...)
Critic Model: Linear(in_features=3171, out_features=512, ...)
ISAAC_META_SMOKE_EXIT:0
```

## Live Metadata Training Run

Started in tmux:

```text
isaac-step11-sfp-gazebo-meta-train-3bd2119
```

Command:

```bash
cd ~/IsaacLab
docker exec isaac-lab-base bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p \
   aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
   --task AIC-SFP-Gazebo-Transfer-Task-v0 \
   --agent rsl_rl_sfp_cfg_entry_point \
   --num_envs 64 \
   --max_iterations 1500 \
   --run_name step11_sfp_gazebo_meta_3bd2119 \
   --headless \
   --enable_cameras"
```

Log path:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_12-42-06_step11_sfp_gazebo_meta_3bd2119
```

Early metrics:

```text
iteration 0:  sfp_insertion_success=0.1584
iteration 5:  sfp_insertion_success=0.4409
iteration 8:  sfp_insertion_success=0.3876
iteration 32: sfp_insertion_success=0.4208
iteration 36: sfp_insertion_success=0.4110
```

Interpretation:

- This is better than the first Gazebo-transfer scratch run, which stayed near
  `0.14-0.26` through iteration 52.
- It is not yet qualification-ready; let it continue to at least the
  iteration-100 range before deciding whether to export/evaluate or change the
  curriculum again.

## Metadata Run Failure Mode

The metadata run regressed by the iteration-100 range:

```text
iteration 87:  sfp_insertion_success=0.1236
iteration 90:  sfp_insertion_success=0.0712
iteration 101: sfp_insertion_success=0.0684
iteration 104: sfp_insertion_success=0.0757
```

At the same time, mean reward kept increasing to about `700+`, and
`sfp_insertion_depth` reward stayed high. This means PPO found a reward shortcut:
push deep and collect depth/action/alignment shaping while missing the final
strict lateral success threshold.

The run was stopped after `model_100.pt` was saved:

```text
/workspace/isaaclab/logs/rsl_rl/aic_sfp_gazebo_transfer/2026-05-14_12-42-06_step11_sfp_gazebo_meta_3bd2119/model_100.pt
```

## Reward Tightening Revision

Changed the Gazebo-transfer SFP curriculum so success dominates depth:

- `sfp_depth_progress_reward` now optionally gates depth progress by lateral and
  orientation thresholds.
- Gazebo-transfer config gates depth progress at lateral `<0.018 m` and
  orientation `<0.25 rad`.
- `sfp_port_frame_depth_action` weight reduced from `80` to `30` and gated at
  lateral `<0.018 m`.
- `sfp_insertion_depth` gate tightened from lateral `<0.030 m` to
  `<0.015 m`.
- sparse `sfp_insertion_success` bonus increased from `100` to `1000`.
- lateral pressure increased:
  `sfp_lateral_progress=60`, `sfp_lateral_error=-120`, and
  `sfp_lateral_corridor=-160` with a `0.012-0.030 m` corridor.
- Added Gazebo-transfer max-depth termination at `0.055 m` so overshooting while
  misaligned cannot remain a long high-reward timeout trajectory.

## Final Packaging Pass

The current Docker submission candidate was built on the host from commit
`ebb57a2`:

```bash
cd ~/ws_aic/src/aic
docker build --network=host \
  --add-host pixi.sh:104.21.63.195 \
  --add-host prefix.dev:34.90.252.205 \
  --add-host github.com:20.205.243.166 \
  --add-host release-assets.githubusercontent.com:185.199.109.133 \
  --add-host raw.githubusercontent.com:185.199.108.133 \
  --add-host download.pytorch.org:65.8.76.66 \
  --add-host repo.ros2.org:54.176.191.34 \
  -t my-solution:v1 \
  -f docker/aic_model/Dockerfile .
```

Build log:

```text
/tmp/aic_model_build_ebb57a2_clean.log
```

Important build output:

```text
✔ The default environment has been installed.
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
#18 naming to docker.io/library/my-solution:v1
#18 DONE 1943.3s
```

Final image:

```text
my-solution:v1 406a86a04849 40.6GB
```

Smoke test:

```bash
docker run --rm --network none --entrypoint bash my-solution:v1 -lc \
  'set -e
   ls -lh /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
   ls -lh /ws_aic/src/aic/aic_model/artifacts
   cd /ws_aic/src/aic
   pixi run --as-is python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT); from aic_model.RslRlCheckpointPolicy import RslRlCheckpointPolicy; print(\"IMPORT_OK\", RslRlCheckpointPolicy.__name__)"'
```

Smoke log:

```text
/tmp/aic_model_smoke_ebb57a2.log
```

Important smoke output:

```text
/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
step11_sfp_gazebo_tight_a23f1da_model_100_policy.pt
step6_sc_policy.pt
IMPORT_OK RslRlCheckpointPolicy
```

## Official Docker Compose Verification

The package was verified with the official compose path and no rebuild:

```bash
cd ~/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml down
docker compose -f docker/docker-compose.yaml up --no-build \
  --abort-on-container-exit --exit-code-from eval
```

Compose log:

```text
/tmp/aic_docker_compose_eval_ebb57a2.log
```

Result:

```text
total: 137.19207522758077
trial_1: tier_1=1, tier_2=20.139221787410996, tier_3=24.354275175905077
trial_2: tier_1=1, tier_2=21.858664475509293, tier_3=25
trial_3: tier_1=1, tier_2=17.839913788755418, tier_3=25
```

All three trials completed and passed model validation. No insertion was
detected in the compose run; final plug-port distances were `0.05 m`, `0.03 m`,
and `0.01 m`.

Compose artifacts were copied from the stopped eval container:

```bash
docker cp aic-eval-1:/root/aic_results /tmp/aic_compose_results_ebb57a2
mkdir -p ~/ws_aic/src/aic/logs/docker_compose_eval/20260514_214210_ebb57a2
cp -a /tmp/aic_compose_results_ebb57a2/. \
  ~/ws_aic/src/aic/logs/docker_compose_eval/20260514_214210_ebb57a2/
```

Artifact path:

```text
~/ws_aic/src/aic/logs/docker_compose_eval/20260514_214210_ebb57a2/
```

The compose scoring bags do not contain `/observations`; they contain scoring
and controller topics only. Camera review videos therefore use the wrapper run
below, which records the same legal observation stream consumed by the policy.

## Fresh Legal Video Verification Run

A separate qualification-like wrapper run was started with `ground_truth:=false`
and `/observations` recording enabled. The wrapper used the same policy class,
SC/SFP policy artifacts, and runtime options as the Docker image defaults:

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --policy aic_model.RslRlCheckpointPolicy \
  --sc-policy-artifact logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact logs/checkpoints/step11_sfp_gazebo_tight_a23f1da_model_100_policy.pt \
  --task-kind auto \
  --record-camera-bag \
  --camera-bag-duration-sec 240 \
  --results-dir logs/gazebo_eval/20260515_video_ebb57a2 \
  --session-prefix gazebo-video-ebb57a2 \
  --replace \
  --model-env AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA=true \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_ENABLE_SFP_TERMINAL_TARGET=true \
  --model-env AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC=1.30 \
  --model-env AIC_RSLRL_ENABLE_SFP_TERMINAL_ORIENTATION=false \
  --model-env AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH=false \
  --model-env AIC_RSLRL_ENABLE_SC_TERMINAL_TARGET=true \
  --model-env AIC_RSLRL_SC_TERMINAL_TARGET_DWELL_SEC=1.30
```

Wrapper score:

```text
total: 154.38066039880212
trial_1: final plug-port distance 0.05 m, no insertion
trial_2: final plug-port distance 0.03 m, no insertion
trial_3: partial insertion detected with distance 0.01 m
```

Wrapper scoring and bag path:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260515_video_ebb57a2/
```

The `/observations` camera bag was converted in the ROS-sourced `aic_eval`
distrobox:

```bash
cd ~/ws_aic/src/aic
source /opt/ros/kilted/setup.bash
source .pixi/envs/default/setup.bash
python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
  logs/gazebo_eval/20260515_video_ebb57a2/camera_bags/wrist_cameras \
  --topic /observations --image-field center_image \
  --output logs/gazebo_eval/20260515_video_ebb57a2/videos/center_image.mp4 \
  --fps 10
```

Equivalent commands were run for `left_image` and `right_image`.

Generated videos:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260515_video_ebb57a2/videos/left_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260515_video_ebb57a2/videos/center_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260515_video_ebb57a2/videos/right_image.mp4
```

Conversion output:

```text
wrote 1152 frames to .../left_image.mp4
wrote 1152 frames to .../center_image.mp4
wrote 1152 frames to .../right_image.mp4
```

## Current Step 11 Conclusion

The Docker image is legal and package-valid, but it is not a solved
qualification submission:

- it uses only official `Task` metadata, official `Observation` images/joints,
  previous legal actions, and internal command state;
- it does not subscribe to scoring internals, hidden Gazebo transforms, or
  ground-truth topics;
- compose verification completes all three trials with tier-1 validation;
- full insertion remains unreliable, with SFP near misses and only partial SC
  insertion in the fresh video run.

The next useful technical work is not more packaging. It is improving the final
Gazebo approach/control mapping or retraining a policy whose terminal behavior
matches the official Gazebo mount distribution closely enough to trigger the
contact/insertion detectors.

## Final Host Package Candidate

After the first Docker package pass, the source default was restored to the
better-performing terminal dwell:

```text
commit: 0e6e100 Restore package SFP dwell default
Dockerfile default: AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC=1.30
```

The host image was retagged by committing the corrected environment into
`my-solution:v1`. The first retag accidentally inherited a temporary
`/bin/true` entrypoint from a smoke container, so a second `docker commit`
restored the intended runtime:

```text
Entrypoint=["/entrypoint.sh"]
Cmd=["--ros-args","-p","policy:=aic_model.RslRlCheckpointPolicy","-p","use_sim_time:=true"]
AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC=1.30
```

Final image verification on the host:

```text
my-solution:fc7fb3a-0e6e100 fc7fb3a897d1 40.6GB
my-solution:v1                 fc7fb3a897d1 40.6GB
ID=sha256:fc7fb3a897d19d072553ae089f72e973160fa2dc4b6d1c4a370a3c975ec66a1f
CHECK_FINAL_IMAGE_EXIT:0
```

This image is the current local submission candidate. It is package-valid and
legal: the runtime policy uses official `Task` metadata, official
`Observation` images/joints/wrench/controller fields, previous legal actions,
and internal command state. It does not use scoring TF, hidden Gazebo state, or
ground-truth topics.

## Final Docker Compose Verification

The final image was verified from the official Compose file with no rebuild:

```bash
cd ~/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml down
docker compose -f docker/docker-compose.yaml up --no-build \
  --abort-on-container-exit --exit-code-from eval
```

Log:

```text
~/ws_aic/src/aic/logs/docker_build/docker_compose_eval_fc7fb_final.log
```

Result:

```text
COMPOSE_EXIT:0
total: 154.62729379313998
trial_1: tier_1=1, tier_2=21.349914814607473, tier_3=24.664026275345606
trial_2: tier_1=1, tier_2=21.822713523265058, tier_3=24.936866484798827
trial_3: tier_1=1, tier_2=18.26616459376131, tier_3=40.587608101361681
```

Trial notes:

```text
trial_1: no insertion, final distance 0.05 m
trial_2: no insertion, final distance 0.05 m
trial_3: partial insertion, final distance 0.01 m
```

Final Compose artifacts were copied to:

```text
~/ws_aic/src/aic/logs/docker_compose_eval/20260515_fc7fb_final/compose.log
~/ws_aic/src/aic/logs/docker_compose_eval/20260515_fc7fb_final/results/scoring.yaml
~/ws_aic/src/aic/logs/docker_compose_eval/20260515_fc7fb_final/results/bag_trial_1_20260515_010628_258/
~/ws_aic/src/aic/logs/docker_compose_eval/20260515_fc7fb_final/results/bag_trial_2_20260515_010646_972/
~/ws_aic/src/aic/logs/docker_compose_eval/20260515_fc7fb_final/results/bag_trial_3_20260515_010701_870/
```

Compose shutdown still produced known ROS/container shutdown noise in some
runs, including `ExternalShutdownException` and a killed component container
after scoring finished. The final run exited with `COMPOSE_EXIT:0` and wrote a
complete `scoring.yaml`, so the shutdown noise is not treated as model failure.

## Final Review Videos

The official Compose scoring bags do not contain `/observations`, so review
videos come from a separate legal wrapper run that records the same
`/observations` camera stream consumed by the policy:

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --policy aic_model.RslRlCheckpointPolicy \
  --sc-policy-artifact logs/checkpoints/step6_sc_policy.pt \
  --sfp-policy-artifact logs/checkpoints/step11_sfp_gazebo_tight_a23f1da_model_100_policy.pt \
  --task-kind auto \
  --record-camera-bag \
  --camera-bag-duration-sec 240 \
  --results-dir logs/gazebo_eval/20260515_final_fc7fb_video \
  --session-prefix gazebo-video-final-fc7fb \
  --replace \
  --model-env AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA=true \
  --model-env AIC_RSLRL_REQUIRE_RESNET18=true \
  --model-env AIC_RSLRL_ENABLE_SFP_TERMINAL_TARGET=true \
  --model-env AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC=1.30 \
  --model-env AIC_RSLRL_ENABLE_SFP_TERMINAL_ORIENTATION=false \
  --model-env AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH=false \
  --model-env AIC_RSLRL_ENABLE_SC_TERMINAL_TARGET=true \
  --model-env AIC_RSLRL_SC_TERMINAL_TARGET_DWELL_SEC=1.30
```

Video-run score:

```text
total: 138.8247978715018
trial_1: no insertion, final distance 0.05 m
trial_2: no insertion, final distance 0.03 m
trial_3: no insertion, final distance 0.01 m
```

The score is lower than the final Compose run because the evaluation is
stochastic; use the videos for qualitative review and the Compose run above for
package score evidence.

Generated videos:

```text
~/ws_aic/src/aic/logs/gazebo_eval/20260515_final_fc7fb_video/videos/left_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260515_final_fc7fb_video/videos/center_image.mp4
~/ws_aic/src/aic/logs/gazebo_eval/20260515_final_fc7fb_video/videos/right_image.mp4
```

Conversion output:

```text
wrote 1130 frames to .../left_image.mp4
wrote 1130 frames to .../center_image.mp4
wrote 1130 frames to .../right_image.mp4
```

## Final Step 11 Status

The final package is legal, reproducible on the host, and ready to tag/push to
the team ECR repository once the team slug or repository URI is provided. It is
the best package-valid candidate currently available, but it is still a partial
insertion solution rather than a solved policy:

- SFP gets close enough for tier-2/tier-3 partial credit but does not reliably
  trigger insertion.
- SC can reach partial insertion in some official trials.
- Further technical improvement should focus on terminal SFP insertion under
  official mount metadata and the official-start SC approach.
