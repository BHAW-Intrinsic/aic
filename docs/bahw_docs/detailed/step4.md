# Step 4: Add True Success Termination

This note records the first SC insertion success termination pass.

## Step 3 Input

Step 3 added SC insertion rewards based on the shared plug-to-port geometry:

- lateral error
- orientation error
- insertion depth

Step 4 reuses the same geometry thresholds for episode termination so the reward
bonus and the done condition do not drift apart.

## Remaining Plan Check

No direction change is needed for later steps. Step 4 strengthens Step 5 and
Step 6 because PPO can now distinguish successful insertions from timeouts.

Visual confirmation that success fires only on true physical insertion still
belongs with Step 6 videos; this step verifies the runtime termination wiring and
threshold logic in headless Isaac Lab.

## Implemented Locally

Changed `mdp/geometry.py`:

- Added `sc_insertion_success_from_errors`.
- Added `sc_insertion_success_mask`.

Changed `mdp/rewards.py`:

- Updated `sc_insertion_success_bonus` to reuse
  `geometry.sc_insertion_success_mask`.

Added `mdp/terminations.py`:

- Added `sc_insertion_success`, returning a boolean tensor of shape
  `(num_envs,)`.

Changed `mdp/__init__.py`:

- Exported `sc_insertion_success`.

Changed `aic_task_env_cfg.py`:

- Added `sc_insertion_success` to `TerminationsCfg`.
- Kept `time_out` as the only timeout term.

Added `scripts/check_aic_terminations.py`:

- Builds `AIC-Task-v0`.
- Prints the termination manager terms.
- Checks analytic success threshold cases:
  - inserted state succeeds
  - hovering fails
  - lateral miss fails
  - orientation miss fails
  - depth shortfall fails
  - exact-threshold equality fails because the termination uses strict
    inequalities
- Checks direct success tensors after reset for both `sc_port` and `sc_port_2`.
- Steps random actions and verifies termination tensor shape/type.
- Writes logs under `logs/aic_terminations/`.

## Local Verification

Syntax checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
```

Status: passed locally.

Whitespace check:

```bash
git diff --check
```

Status: passed locally.

## Remote Verification

Implementation commit:

- `43e88c0 Add SC insertion success termination`

Pulled on the remote host:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

Copied the changed source files into the running Isaac Lab container:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_terminations.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_terminations.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py
```

Ran inside the Isaac Lab container through host `tmux` session
`isaac-step4-terminations-43e88c0`:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_terminations.py \
  --task AIC-Task-v0 --num_envs 16 --num_steps 8 --headless --enable_cameras
```

Key output:

```text
Termination Manager contains 2 active terms:
time_out             Time Out=True
sc_insertion_success Time Out=False
```

Analytic threshold checks:

```text
inserted: success=True expected=True lateral=0.000000 orientation=0.000000 depth=0.020000
hovering: success=False expected=False lateral=0.000000 orientation=0.000000 depth=0.000000
lateral_miss: success=False expected=False lateral=0.020000 orientation=0.000000 depth=0.020000
orientation_miss: success=False expected=False lateral=0.000000 orientation=0.500000 depth=0.020000
depth_shortfall: success=False expected=False lateral=0.000000 orientation=0.000000 depth=0.006000
at_thresholds: success=False expected=False lateral=0.005000 orientation=0.200000 depth=0.012000
analytic_success_checks_ok: True
```

Reset and random-policy checks:

```text
sc_port reset success: shape=(16,) dtype=torch.bool true_count=0/16 shape_ok=True dtype_ok=True
sc_port_2 reset success: shape=(16,) dtype=torch.bool true_count=0/16 shape_ok=True dtype_ok=True
step 00 gym terminated: shape=(16,) dtype=torch.bool true_count=0/16
...
step 07 gym terminated: shape=(16,) dtype=torch.bool true_count=0/16
overall_termination_check_ok: True
STEP4_TERMINATION_EXIT:0
```

Copied the termination log back to the host:

```bash
mkdir -p ~/IsaacLab/aic/logs
docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/aic_terminations \
  ~/IsaacLab/aic/logs/
```

Result:

```text
COPY_EXIT:0
~/IsaacLab/aic/logs/aic_terminations/20260510_104604_AIC-Task-v0.log
```
