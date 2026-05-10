# Step 2: Add Eval-Compatible And Privileged Observations

This note records the observation split added after Step 1 geometry helpers.

## Goal

The actor should only receive information that can exist during Gazebo
evaluation. The critic can receive simulator-only plug-to-port geometry during
training.

Step 1 produced shared SC geometry helpers for:

- active SC target selection
- SC plug tip pose
- active SC port entry pose
- plug-to-port vector
- lateral error
- insertion depth
- orientation error

Step 2 wires those helpers into observation groups.

## Implemented Locally

Changed `mdp/observations.py`:

- Added `active_sc_target_one_hot`, an eval-compatible one-hot task metadata
  vector:
  - `[1, 0]` means active target `sc_port`
  - `[0, 1]` means active target `sc_port_2`
- Added privileged observation wrappers:
  - `sc_plug_to_port_vec`
  - `sc_lateral_error_obs`
  - `sc_orientation_error_obs`
  - `sc_insertion_depth_obs`
  - `sc_active_port_pose`
  - `sc_plug_tip_pose_obs`

Changed `aic_task_env_cfg.py`:

- Removed `pose_command` from `PolicyCfg`.
- Added `task_metadata` to `PolicyCfg`.
- Added `PrivilegedCfg` with plug-to-port geometry terms.
- Kept both observation groups concatenated with stable term dimensions.

Changed `rsl_rl_ppo_cfg.py`:

```python
obs_groups = {
    "actor": ["policy"],
    "critic": ["policy", "privileged"],
}
```

## Observation Boundary

Policy group:

- task metadata one-hot
- joint positions
- joint velocities
- end-effector pose
- robot body wrench signal
- center/left/right camera features
- last action

Privileged group:

- plug-to-port vector
- lateral error
- orientation error
- insertion depth
- active port entrance pose
- plug tip pose

`pose_command` is intentionally no longer visible to the actor. The existing
command generator remains configured for now because the Step 3 reward cleanup
has not happened yet.

## Local Verification

Local syntax checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py
```

Status: passed locally.

## Remote Verification

Verification commit:

```text
88e5b53 Add privileged observation group
```

Host pull:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

Result:

```text
HOST_PULL_EXIT:0
88e5b53
```

The running container could not pull directly in earlier steps because its git
remote requires `ssh`, so the changed files were copied from the updated host
checkout into the running container:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py

docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py

docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py
```

Inspector command:

```bash
docker exec isaac-lab-base bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py --task AIC-Task-v0 --num_envs 4 --headless --enable_cameras'
```

Result:

```text
STEP2_EXIT:0
```

Final log copied to the host:

```text
~/IsaacLab/aic/logs/aic_geometry/20260510_101621_AIC-Task-v0.log
```

Observation Manager output:

```text
[INFO] Observation Manager: <ObservationManager> contains 2 groups.
+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (3149,)) |
+------------+----------------------------+----------------+
|   Index    | Name                       |     Shape      |
+------------+----------------------------+----------------+
|     0      | task_metadata              |      (2,)      |
|     1      | joint_pos                  |     (46,)      |
|     2      | joint_vel                  |     (46,)      |
|     3      | eef_pose                   |      (7,)      |
|     4      | body_forces                |     (42,)      |
|     5      | center_rgb                 |    (1000,)     |
|     6      | left_rgb                   |    (1000,)     |
|     7      | right_rgb                  |    (1000,)     |
|     8      | actions                    |      (6,)      |
+------------+----------------------------+----------------+
+-------------------------------------------------------------+
| Active Observation Terms in Group: 'privileged' (shape: (20,)) |
+------------+-----------------------------------+------------+
|   Index    | Name                              |   Shape    |
+------------+-----------------------------------+------------+
|     0      | plug_to_port_vec                  |    (3,)    |
|     1      | lateral_error                     |    (1,)    |
|     2      | orientation_error                 |    (1,)    |
|     3      | insertion_depth                   |    (1,)    |
|     4      | active_port_pose                  |    (7,)    |
|     5      | plug_tip_pose                     |    (7,)    |
+------------+-----------------------------------+------------+
```

Gym spaces:

```text
gym observation space: Dict('policy': Box(-inf, inf, (4, 3149), float32), 'privileged': Box(-inf, inf, (4, 20), float32))
gym action space: Box(-inf, inf, (4, 6), float32)
```

RSL-RL config check inside the container:

```text
21:    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
RSL_CHECK_EXIT:0
```

Interpretation:

- `policy` contains no direct plug-to-port geometry and no `pose_command`.
- `task_metadata` is present as the eval-compatible active SC target one-hot.
- `privileged` contains plug-to-port geometry only for the critic.
- Observation dimensions are stable after reset for `num_envs=4`.
- RSL-RL is configured for asymmetric actor-critic inputs.
