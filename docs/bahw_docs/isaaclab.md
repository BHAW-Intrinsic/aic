# Isaac Lab Setup & Commands

All commands run on `bahw@100.103.111.75` unless noted.

---

## First-Time Setup

Add user to docker group (one-time, then log out and back in):
```bash
sudo usermod -aG docker $USER
newgrp docker  # only needed until next login
```

Install `aic_task` inside the container (one-time):
```bash
cd ~/IsaacLab
./docker/container.py enter base

# inside container:
python -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
```

---

## Every Session

**Enter the container** (from the remote desktop terminal, not SSH — needs `$DISPLAY`):
```bash
cd ~/IsaacLab
./docker/container.py start base
./docker/container.py enter base
```

All `isaaclab` commands below run **inside the container** from `/workspace/isaaclab`.

---

## Verify Environment

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
# should show AIC-Task-v0
```

---

## Run RL Training

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --num_envs 64 --headless --enable_cameras
```

Logs saved to `/workspace/isaaclab/logs/rsl_rl/aic_task/`.

---

## Notes

- Vulkan errors on startup are harmless — headless training uses CUDA, not Vulkan.
- `.glb` mesh warnings are cosmetic — visuals missing but physics unaffected.
- `--enable_cameras` is required whenever the env config includes camera sensors.
- `aic_task` pip install must be re-run if the container is rebuilt from scratch.
