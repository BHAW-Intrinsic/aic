# Official Gazebo Eval Wrapper

This directory documents the wrapper for running the official Gazebo
qualification evaluation path against our checkpoint policy package.

## Current State

The wrapper now handles official evaluation orchestration:

- starts the `aic_eval` Gazebo/AIC engine container in a tmux session
- starts `aic_model` in a separate tmux session
- passes checkpoint/artifact paths into the ROS policy via environment variables
- writes official scoring output under a unique `AIC_RESULTS_DIR`
- can optionally record wrist camera image topics to a separate rosbag

The raw Isaac Lab RSL-RL `.pt` checkpoint is not directly deployable in Gazebo
yet. The missing implementation is the runtime adapter that converts official
Gazebo `Observation` + `Task` messages into the exact Isaac actor observation
vector, runs the exported actor, and converts the actor action into
`MotionUpdate` or `JointMotionUpdate`.

Until that adapter is implemented, `aic_model.RslRlCheckpointPolicy` is a
scaffold. It verifies lifecycle, task metadata, checkpoint-path plumbing, and
scoring-output collection, then returns `False` for the task.

## Files

- `aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py`
  starts the official eval/model/camera-recording tmux sessions.
- `aic_model/aic_model/RslRlCheckpointPolicy.py`
  is the ROS policy scaffold for checkpoint-backed Gazebo eval.
- `docs/bahw_docs/eval_wrapper/README.md`
  is this usage and implementation guide.

## Basic Usage

Run this from the host repo copy, not inside the Isaac Lab container:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --checkpoint /path/to/model.pt \
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

The model session prepends `<repo>/aic_model` to `PYTHONPATH` before launching
`pixi run ros2 run aic_model ...`. This makes branch-local policy modules, such
as `aic_model.RslRlCheckpointPolicy`, importable even when the pixi environment
was built before the latest branch checkout.

## Final SC/SFP Routing Shape

For final submission-style routing, pass separate checkpoint or exported actor
artifacts:

```bash
python3 aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py \
  --sc-checkpoint /path/to/sc_model.pt \
  --sfp-checkpoint /path/to/sfp_model.pt \
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

For visual inspection as an actual video, use one of these paths:

- Run with `--gazebo-gui --launch-rviz` and manually screen-record the remote
  desktop.
- Record camera rosbags with the wrapper and convert or replay them later.

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

The adapter still needs to define and validate:

- Isaac-compatible joint position and velocity normalization from Gazebo
  `joint_states`.
- The `gripper_tcp` pose convention from Gazebo `controller_state.tcp_pose`.
- The body-force observation replacement. Gazebo exposes wrist wrench, while
  Isaac used incoming wrench terms over multiple robot bodies.
- ResNet18 preprocessing parity for the three camera images.
- Last-action bookkeeping in the ROS policy loop.
- Conversion from the six-dimensional relative IK actor action to safe Gazebo
  `MotionUpdate` or `JointMotionUpdate` commands.

## Remote Resource Note

Do not run this while the Isaac 4090 is saturated by training unless you intend
to stop one training job first. During the Step 9 randomized SFP runs, the host
had roughly `22249 MiB / 24564 MiB` VRAM in use, leaving too little headroom for
another Gazebo/GUI/eval workload.
