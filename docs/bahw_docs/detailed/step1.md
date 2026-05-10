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
- `sc_port_entry_pose`
- `sc_port_insertion_axis`
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

## Verification Still Needed

The next remote Isaac run should verify:

- helper tensor shapes are `(num_envs, ...)`
- active target selection switches between `sc_port` and `sc_port_2`
- derived `port_entry_pos_w` lands at the expected SC port opening
- `sc_insertion_depth` sign is correct
- orientation error is meaningful for the imported `sc_tip_link` frame

Run the updated inspector inside the Isaac Lab container after pushing this
branch:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  --task AIC-Task-v0 --num_envs 1 --headless --enable_cameras
```
