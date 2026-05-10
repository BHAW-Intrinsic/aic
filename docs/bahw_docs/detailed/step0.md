# Step 0: Confirm Asset Frames In Isaac

This note records the exact remote commands and key output from the Step 0
geometry inspection. Commands are written relative to the remote host; no `ssh`
prefix is included.

For future runs, start container setup and inspection commands inside named
`tmux` sessions such as `isaac-stage0-container` and `isaac-stage0-inspect`.

## Goal

Gazebo SDF assets have semantic frames for the real insertion geometry:

- SC plug tip: `sc_tip_link`
- SC port entrance: `sc_port_base_link_entrance`
- SFP plug/module tip: `sfp_tip_link`
- SFP port entrances: `sfp_port_0_link_entrance`, `sfp_port_1_link_entrance`

Step 0 checks whether the Isaac USD assets expose equivalent runtime bodies or
USD prims. If they are missing, Step 1 must derive helper poses from the nearest
available Isaac frame.

## Branch Sync

Run on the remote host:

```bash
cd ~/IsaacLab/aic
git switch dev/stage0
git pull --ff-only
```

Observed output:

```text
Already up to date.
## dev/stage0...origin/dev/stage0
```

## Headless Container Setup

For headless SSH/container use, X11 forwarding was disabled in the Isaac Lab
Docker helper config:

```bash
cd ~/IsaacLab
cp docker/.container.cfg docker/.container.cfg.bak-stage0-x11
sed -i "s/x11_forwarding_enabled = 1/x11_forwarding_enabled = 0/" docker/.container.cfg
```

Observed config:

```text
[X11]
x11_forwarding_enabled = 0
__isaaclab_tmp_xauth = /tmp/tmp.X6PB5f7aEy/tmp.HhlvyrPpli.xauth
```

Start the Isaac Lab base container:

```bash
cd ~/IsaacLab
./docker/container.py start base
```

Observed container state after the start/build finished:

```text
CONTAINER ID   IMAGE            COMMAND   STATUS          NAMES
9b87b7fc7b39   isaac-lab-base   "bash"    Up 49 seconds   isaac-lab-base
```

Note: `isaaclab` was not on `PATH` in the rebuilt `isaac-lab-base` container, so
the successful run used `./isaaclab.sh -p ...`.

## Inspection Command

Run inside the container from the remote host:

```bash
docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && \
  ./isaaclab.sh -p -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task && \
  ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
    --task AIC-Task-v0 --num_envs 1 --headless --enable_cameras'
```

The command completed with:

```text
STEP0_EXIT:0
```

Headless Isaac emitted display warnings such as `GLFW initialization failed` and
`failed to open the default display`, but the process continued and completed
successfully.

## Key Output

```text
== AIC Geometry Inspection ==
task: AIC-Task-v0
num_envs: 1
log_path: /workspace/isaaclab/aic/logs/aic_geometry/20260510_090800_AIC-Task-v0.log

gym observation space: Dict('policy': Box(-inf, inf, (1, 3154), float32))
gym action space: Box(-inf, inf, (1, 6), float32)

== Scene Collections ==
articulations: ['robot']
rigid_objects: ['nic_card', 'sc_port', 'sc_port_2', 'task_board']
sensors: ['center_camera', 'left_camera', 'right_camera']
extras: ['aic_scene', 'ground', 'light']

== All Scene Asset Names ==
- aic_scene
- center_camera
- ground
- left_camera
- light
- nic_card
- right_camera
- robot
- sc_port
- sc_port_2
- task_board
- terrain

robot body_names include:
  'sfp_tip_link', 'sc_plug_link', 'sc_tip_link'

sc_port:
  body_names (1): ['sc_port_visual']
  root_pos_w env0:  [0.289148, 0.190034, 0.005000]
  root_quat_w env0: [0.517150, 0.517150, -0.482241, -0.482241]

sc_port_2:
  body_names (1): ['sc_port_visual']
  root_pos_w env0:  [0.293923, 0.147934, 0.005000]
  root_quat_w env0: [0.517150, 0.517150, -0.482241, -0.482241]

nic_card:
  body_names (1): ['nic_card']
  root_pos_w env0:  [0.247762, 0.329524, 0.074300]
  root_quat_w env0: [-0.000000, 0.000000, -0.706825, 0.707389]

robot.sc_plug_link body_id=54
  body_pos_w env0:  [0.191221, 0.101897, 1.253416]
  body_quat_w env0: [0.505221, 0.459952, -0.317968, 0.657338]

robot.sc_tip_link body_id=55
  body_pos_w env0:  [0.190448, 0.106227, 1.264203]
  body_quat_w env0: [-0.970239, 0.192321, 0.005065, -0.147050]

== Gazebo Semantic Name Search ==
sc_tip_link:
  runtime matches: ['body:robot.sc_tip_link']
  USD prim matches: ['/World/envs/env_0/Robot/cable/sc_plug/sc_tip_link']
sc_port_base_link_entrance:
  runtime matches: none
  USD prim matches: none
sfp_tip_link:
  runtime matches: ['body:robot.sfp_tip_link']
  USD prim matches: ['/World/envs/env_0/Robot/cable/sfp_module/sfp_tip_link', ...]
sfp_port_0_link_entrance:
  runtime matches: none
  USD prim matches: ['/World/envs/env_0/nic_card/sfp_port_0_link/sfp_port_0_link_entrance']
sfp_port_1_link_entrance:
  runtime matches: none
  USD prim matches: ['/World/envs/env_0/nic_card/sfp_port_1_link/sfp_port_1_link_entrance']

== Done ==
STEP0_EXIT:0
```

## Copy Log To Host Checkout

The script wrote the log inside the container. Copy it to the remote host
checkout:

```bash
mkdir -p ~/IsaacLab/aic/logs
docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_geometry ~/IsaacLab/aic/logs/
cd ~/IsaacLab/aic
find logs/aic_geometry -maxdepth 1 -type f -print -exec wc -l {} \;
```

Observed output:

```text
Successfully copied 15.4kB to /var/home/bahw/IsaacLab/aic/logs/
logs/aic_geometry/20260510_090800_AIC-Task-v0.log
187 logs/aic_geometry/20260510_090800_AIC-Task-v0.log
```

## Interpretation

- Isaac exposes the SC plug tip:
  - runtime body: `robot.sc_tip_link`
  - USD prim: `/World/envs/env_0/Robot/cable/sc_plug/sc_tip_link`
- Isaac does not expose the SC port entrance:
  - `sc_port_base_link_entrance` has no runtime match
  - `sc_port_base_link_entrance` has no USD prim match
- The nearest SC port runtime frames are:
  - asset `sc_port`, body `sc_port_visual`
  - asset `sc_port_2`, body `sc_port_visual`
- Isaac exposes the SFP plug tip as runtime body `robot.sfp_tip_link`.
- SFP port entrances exist as USD prims under `nic_card`, but not as runtime
  rigid bodies.

Step 1 should use `robot.sc_tip_link` directly for the SC plug tip and derive
the SC port entrance helper pose from `sc_port` or `sc_port_2`.
