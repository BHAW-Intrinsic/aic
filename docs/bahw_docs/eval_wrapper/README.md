# Official Gazebo Eval Wrapper

This directory documents the wrapper for running the official Gazebo
qualification evaluation path against our checkpoint policy package.

## Current State

The wrapper now handles official evaluation orchestration and a first SFP
TorchScript actor adapter:

- starts the `aic_eval` Gazebo/AIC engine container in a tmux session
- starts `aic_model` in a separate tmux session
- passes checkpoint/artifact paths into the ROS policy via environment variables
- writes official scoring output under a unique `AIC_RESULTS_DIR`
- can optionally record wrist camera image topics to a separate rosbag
- converts recorded camera rosbags to MP4 with
  `aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py`
- can export simple RSL-RL MLP actor checkpoints with
  `aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py`

Raw Isaac Lab RSL-RL `.pt` checkpoints are still not directly deployable in
Gazebo. Export the actor first with Isaac Lab `play.py` and pass the exported
TorchScript artifact with `--sfp-policy-artifact` or `--policy-artifact`.
If `play.py` fails before export, the selected SFP checkpoint can be exported
with the lightweight MLP exporter:

```bash
python3 aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py \
  --checkpoint logs/checkpoints/step9_sfp_randy002_scratch_model_1499.pt \
  --output logs/checkpoints/step9_sfp_randy002_scratch_policy.pt
```

`aic_model.RslRlCheckpointPolicy` currently implements the SFP adapter only.
The SC adapter and final SC/SFP routing still need to be completed before this
is a full qualification policy.

## Files

- `aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py`
  starts the official eval/model/camera-recording tmux sessions.
- `aic_model/aic_model/RslRlCheckpointPolicy.py`
  is the ROS policy for exported actor-backed Gazebo eval.
- `aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py`
  converts recorded image-topic bags into MP4 review videos.
- `aic_utils/aic_training_utils/scripts/export_rslrl_mlp_actor.py`
  exports simple RSL-RL MLP actor checkpoints without launching Isaac Sim.
- `docs/bahw_docs/eval_wrapper/README.md`
  is this usage and implementation guide.

## Basic Usage

Run this from the host repo copy, not inside the Isaac Lab container:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sfp-policy-artifact /path/to/exported/policy.pt \
  --task-kind sfp \
  --session-prefix gazebo-rslrl-sfp-candidate \
  --record-camera-bag \
  --camera-bag-duration-sec 900
```

The wrapper prints the tmux session names and the expected scoring path. Inspect
sessions with:

```bash
tmux capture-pane -t gazebo-rslrl-sfp-candidate-eval -p -S -200
tmux capture-pane -t gazebo-rslrl-sfp-candidate-model -p -S -200
tmux capture-pane -t gazebo-rslrl-sfp-candidate-camera-bag -p -S -200
```

Default output is:

```text
logs/gazebo_eval/<timestamp>/scoring.yaml
logs/gazebo_eval/<timestamp>/bag_<trial>_<timestamp>/
logs/gazebo_eval/<timestamp>/camera_bags/wrist_cameras/
```

The model session launches `ros2 run ...` inside `pixi run bash -lc ...` and
prepends `<repo>/aic_model` to `PYTHONPATH` inside that pixi shell. This makes
branch-local policy modules, such as `aic_model.RslRlCheckpointPolicy`,
importable even when the pixi environment was built before the latest branch
checkout.

## Final SC/SFP Routing Shape

For final submission-style routing, pass separate exported actor artifacts:

```bash
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-policy-artifact /path/to/sc_exported/policy.pt \
  --sfp-policy-artifact /path/to/sfp_exported/policy.pt \
  --session-prefix gazebo-rslrl-final \
  --record-camera-bag \
  --camera-bag-duration-sec 900
```

The policy will later route with official `Task` metadata such as `plug_type`,
`port_type`, `port_name`, and `target_module_name`. It must not use hidden
Gazebo state or ground-truth transforms.

## Qualification-Like Settings

For qualification-like checks, keep the defaults:

```text
ground_truth:=false
gazebo_gui:=false
launch_rviz:=false
start_aic_engine:=true
shutdown_on_aic_engine_exit:=true
```

Use `--ground-truth` only for debugging. A run with ground truth enabled is not
a qualification-like result.

## Verification Video

The official Gazebo eval stack does not provide a built-in MP4/video artifact.
It writes `scoring.yaml` and scoring rosbags. The wrapper can also record wrist
camera image topics:

```bash
--record-camera-bag
```

By default this records:

```text
/left_camera/image
/center_camera/image
/right_camera/image
```

Use `--camera-topics` to record a subset, for example:

```bash
--record-camera-bag --camera-topics /center_camera/image
```

For visual inspection as an actual video, convert the camera bag:

```bash
python3 aic_utils/aic_training_utils/scripts/rosbag_images_to_video.py \
  logs/gazebo_eval/<timestamp>/camera_bags/wrist_cameras \
  --topic /center_camera/image \
  --output logs/gazebo_eval/<timestamp>/videos/center_camera.mp4 \
  --fps 20
```

Repeat with `/left_camera/image` and `/right_camera/image` if needed. The MP4s
remain host-side review artifacts under `~/ws_aic/src/aic/logs/gazebo_eval/...`.

The scoring rosbags are not camera videos; they are for official scoring topics.

## Implementation Details

The Isaac actor observation order that the Gazebo adapter must reproduce is:

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

The official Gazebo `Observation` message provides:

```text
left_image, center_image, right_image
left_camera_info, center_camera_info, right_camera_info
wrist_wrench
joint_states
controller_state
```

The SFP adapter currently defines:

- target one-hot from official `Task.port_name` / `target_module_name`
- 46D Isaac joint position and velocity observations, with the six official UR
  arm joints filled from `joint_states` and unobserved cable joints held at
  default-relative zero
- a Gazebo-to-Isaac shoulder-pan sign conversion, matching the known home-joint
  convention difference
- TCP pose from `controller_state.tcp_pose`
- wrist wrench padded into the final six slots of the 42D force observation
- three torchvision ResNet18 ImageNet-V1 1000D camera logits using Isaac Lab's
  preprocessing convention
- last-action bookkeeping in the ROS policy loop
- optional legal SFP warm-start joint preset selected only by official
  `Task.port_name`
- actor action conversion to small Cartesian `MotionUpdate` position targets
  in `base_link`

Still required:

- validate the Cartesian action-frame mapping in Gazebo
- implement the SC adapter
- decide whether the warm-start should remain in the final policy or be
  replaced by a learned approach stage
- produce qualification-like `ground_truth:=false` scoring and MP4 evidence

## Remote Resource Note

Do not run this while the Isaac 4090 is saturated by training unless you intend
to stop one training job first. During the Step 9 randomized SFP runs, the host
had roughly `22249 MiB / 24564 MiB` VRAM in use, leaving too little headroom for
another Gazebo/GUI/eval workload.
