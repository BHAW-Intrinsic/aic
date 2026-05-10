# Workspace Setup

## Machines

| Machine | Role | Access |
|---|---|---|
| Local (aloy) | Development, code editing | — |
| `bahw@100.103.111.75` | Training (RTX 4090, 24GB) | `ssh bahw@100.103.111.75` (no password) |

---

## Gazebo Evaluation Environment (bahw machine)

Workspace at `~/ws_aic/src/aic`. Uses distrobox + pixi.

**Run simulation + engine (terminal 1):**
```bash
distrobox enter -r aic_eval

# inside distrobox:
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

**Run policy (terminal 2, at almost exactly the same time):**
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

Swap `CheatCode` for any policy class (e.g. `aic_example_policies.ros.RunACT`, or your own).  
Drop `ground_truth:=true` when not using CheatCode.

---

## Isaac Lab (bahw machine)

Isaac Lab 2.3.2 at `~/IsaacLab`. AIC repo cloned inside at `~/IsaacLab/aic`.  
Intrinsic assets in place at both `~/IsaacLab/Intrinsic_assets` and symlinked into `aic_task`.  
Runs inside a **Docker container** (not on the host directly).

**Enter the container:**
```bash
cd ~/IsaacLab
./docker/container.py start base
./docker/container.py enter base
```

**Install aic_task (first time only, inside container):**
```bash
python -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
```

**Run RL training (inside container):**
```bash
cd ~/IsaacLab
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 64 --headless --enable_cameras
```

**Run teleoperation (inside container):**
```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras
```

---

## Vulkan Fix (one-time, already applied)

**Why it was needed:** The Isaac Sim container requires Vulkan for offscreen camera rendering (`--enable_cameras`). On this Fedora host, `nvidia-container-toolkit` v1.18+ defaults to CDI mode, which has a [known bug](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1124) where `libGLX_nvidia.so.0` — the library the Vulkan ICD loads — is not injected into the container. The result is `ERROR_INCOMPATIBLE_DRIVER` at startup and a hang/crash when any camera sensor initialises.

**Fix:** `~/IsaacLab/docker/docker-compose.yaml` was edited to bind-mount the library from the host into the container on every start:

```yaml
# in x-default-isaac-lab-volumes
- type: bind
  source: /usr/lib64/libGLX_nvidia.so.0
  target: /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
  read_only: true
```

**If you ever need to reapply this** (e.g. after a clean OS reinstall or if the container starts throwing `ERROR_INCOMPATIBLE_DRIVER` again):
1. Check the mount is still in `docker-compose.yaml` — it should be, since the file is committed.
2. If the host driver was updated, verify the library still exists at `/usr/lib64/libGLX_nvidia.so.0`.
3. If the container is already running without it, a temporary workaround is:
   ```bash
   docker cp /usr/lib64/libGLX_nvidia.so.0 isaac-lab-base:/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
   docker exec isaac-lab-base ldconfig
   ```
   Then restart training — no container rebuild needed.
