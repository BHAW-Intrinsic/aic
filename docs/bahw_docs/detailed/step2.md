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

## Remote Verification Still Needed

Run the inspector in Isaac after pushing and pulling/copying the changed files
into the container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  --task AIC-Task-v0 --num_envs 4 --headless --enable_cameras
```

Expected checks:

- Observation Manager prints both `policy` and `privileged` groups.
- `policy` no longer contains `pose_command`.
- `policy` contains `task_metadata` with shape `(2,)`.
- `privileged` contains plug-to-port geometry terms.
- RSL-RL config maps actor to `policy` and critic to `policy + privileged`.
