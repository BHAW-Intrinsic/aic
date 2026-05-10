# Step 1: Add Geometry Helpers

This note records the implementation details for the first SC plug-to-port
geometry helper pass.

## Step 0 Inputs

Step 0 confirmed:

- SC plug tip is available directly as runtime body `robot.sc_tip_link`.
- SC port entrance `sc_port_base_link_entrance` is not available as a runtime
  body or USD prim in Isaac.
- The nearest SC port runtime frames are `sc_port` and `sc_port_2`, each with
  body `sc_port_visual` and a root pose.
- SFP plug tip is available directly as runtime body `robot.sfp_tip_link`.
- SFP port entrances exist as USD prims under `nic_card`, but not as runtime
  rigid bodies.

## SC Port Entrance Offset

The missing SC entrance helper pose is derived from
`aic_assets/models/SC Port/model.sdf`.

Relevant SDF links:

```xml
<link name="sc_port_base_link">
  <pose>0 -0.002 0 1.5708 3.14159 0</pose>
</link>

<link name="sc_port_base_link_entrance">
  <pose relative_to="sc_port_base_link">0 0 -0.01564 0 0 0</pose>
</link>
```

Using the SDF roll-pitch-yaw convention, the entrance position in the
`sc_port_link` frame is approximately:

```text
SC_PORT_ENTRY_POS_LOCAL = (0.0, 0.01364, 0.0)
```

The entrance lies on the positive-Y face of the SC port model, so the insertion
axis from the entrance into the port is:

```text
SC_PORT_INSERTION_AXIS_LOCAL = (0.0, -1.0, 0.0)
```

The helper orientation for `sc_port_base_link_entrance` relative to the SC port
root is:

```text
SC_PORT_ENTRY_QUAT_LOCAL = (0.0, 0.0, 0.7071067811865476, -0.7071067811865476)
```

## Implemented Locally

Added:

- `mdp/geometry.py`
- `sample_active_sc_target` reset event in `aic_task_env_cfg.py`
- geometry exports in `mdp/__init__.py`
- geometry-helper output in `scripts/inspect_aic_geometry.py`

The first helper set includes:

- `active_sc_target_ids`
- `active_sc_target_names`
- `sample_active_sc_target`
- `sc_plug_tip_pose`
- `sc_plug_axis`
- `active_sc_port_root_pose`
- `sc_port_root_pose_for_target`
- `sc_port_entry_pose`
- `sc_port_entry_pose_for_target`
- `sc_port_insertion_axis`
- `sc_port_insertion_axis_for_target`
- `sc_plug_to_port_vector`
- `sc_lateral_error`
- `sc_insertion_depth`
- `sc_orientation_error`

## Conventions

- `sc_plug_to_port_vector = port_entry_pos_w - plug_tip_pos_w`
- `sc_insertion_depth = dot(plug_tip_pos_w - port_entry_pos_w, port_axis_w)`
- Insertion depth is intended to be zero at the entrance and increase as the
  plug tip moves from the entrance into the port.
- `sc_lateral_error` removes the axial depth component and measures distance
  from the plug tip to the active port insertion axis.
- `sc_orientation_error` is the angle between the plug axis and port insertion
  axis.

## Remote Verification

Final verification commit:

```text
996f525 Add per-target SC geometry helpers
```

Host-side commands run:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

The running container copy could not `git pull` because its origin was configured
for SSH and the container did not have `ssh` available:

```text
error: cannot run ssh: No such file or directory
fatal: unable to fork
```

Because the container was already running and only Step 1 files changed, the
updated host files were copied into the container:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py

docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py
```

Inspector command run from the host:

```bash
docker exec isaac-lab-base bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py --task AIC-Task-v0 --num_envs 4 --headless --enable_cameras'
```

Log copy command:

```bash
mkdir -p ~/IsaacLab/aic/logs
docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_geometry \
  ~/IsaacLab/aic/logs/
```

Final log:

```text
~/IsaacLab/aic/logs/aic_geometry/20260510_100156_AIC-Task-v0.log
```

Relevant output:

```text
== AIC Geometry Helper Values ==
active_sc_target_ids all envs: [1, 0, 1, 1]
active_sc_target_names: ['sc_port_2', 'sc_port', 'sc_port_2', 'sc_port_2']
sc_port_entry_pos_local: (0.0, 0.01364, 0.0)
sc_port_insertion_axis_local: (0.0, -1.0, 0.0)
sc_plug_axis_local: (0.0, 0.0, 1.0)
helper tensor shapes:
  plug_tip_pos_w:        (4, 3)
  plug_tip_quat_w:       (4, 4)
  port_entry_pos_w:      (4, 3)
  port_entry_quat_w:     (4, 4)
  plug_axis_w:           (4, 3)
  port_insertion_axis_w: (4, 3)
  plug_to_port_vec_w:    (4, 3)
  lateral_error:         (4,)
  insertion_depth:       (4,)
  orientation_error:     (4,)
plug_tip_pos_w env0:       [2.053993, -1.962219, 1.190298]
plug_tip_quat_w env0:      [-0.977973, 0.174040, 0.106342, -0.044379]
port_entry_pos_w env0:     [2.286846, -1.853438, 0.018640]
port_entry_quat_w env0:    [0.000000, 0.681992, 0.731360, -0.000000]
plug_axis_w env0:          [-0.223446, 0.330975, 0.916803]
port_insertion_axis_w env0:[-0.000000, -0.000000, -1.000000]
plug_to_port_vec_w env0:   [0.232853, 0.108781, -1.171658]
lateral_error env0:        0.257010
insertion_depth env0:      -1.171658
orientation_error env0:    2.730796
per-target SC helper poses env0:
  sc_port:
    port_entry_pos_w:      [2.297784, -1.811338, 0.018640]
    port_entry_quat_w:     [0.000000, 0.681992, 0.731360, -0.000000]
    port_insertion_axis_w: [-0.000000, -0.000000, -1.000000]
  sc_port_2:
    port_entry_pos_w:      [2.286846, -1.853438, 0.018640]
    port_entry_quat_w:     [0.000000, 0.681992, 0.731360, -0.000000]
    port_insertion_axis_w: [-0.000000, -0.000000, -1.000000]
```

Interpretation:

- Helper tensor shapes are correct for `num_envs=4`.
- The reset event sampled both active target IDs in one run: `0` for `sc_port`
  and `1` for `sc_port_2`.
- The active env0 target was `sc_port_2`, and the active `port_entry_pos_w`
  matches the per-target `sc_port_2` helper pose.
- Both SC port helper poses share the same orientation and world insertion axis
  in this scene instance.
- With `port_insertion_axis_w = [0, 0, -1]`, the plug starts above the entrance,
  so depth is negative. Moving the plug tip from the entrance into the port
  means moving along negative world Z, which increases
  `dot(plug_tip_pos_w - port_entry_pos_w, port_axis_w)`.
- Orientation error is large in the reset pose because the plug is not aligned
  with the port yet; the value is still meaningful for rewards/observations.

## Remaining Step 1 Note

Active SC target IDs are now sampled and stored on the env object, but the
eval-compatible actor metadata observation is not added yet. That belongs with
the Step 2 observation group changes.
