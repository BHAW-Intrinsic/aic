# Step 5: Add SC PPO Teacher Config

This note records the SC-specific RSL-RL PPO teacher config.

## Step 4 Input

Step 4 added true SC insertion success termination. The teacher config can now
train against insertion rewards and an insertion success done term instead of
pure timeout episodes.

## Remaining Plan Check

No direction change is needed for later steps. Step 5 prepares Step 6 training.
Steps 7 to 9 remain gated on teacher results:

- Step 7 should reuse the same MDP interface for SFP after the SC teacher path
  is smoke-tested.
- Step 8 needs actual SC/SFP checkpoint metrics before it can be completed.
- Step 9 distillation should not start until the PPO teachers are useful.

## Implemented Locally

Added `agents/rsl_rl_ppo_sc_cfg.py`:

- Uses `experiment_name = "aic_sc_insert"`.
- Keeps the current PPO hyperparameters and MLP sizes.
- Preserves asymmetric observation groups:
  - actor: `["policy"]`
  - critic: `["policy", "privileged"]`

Changed `aic_task/__init__.py`:

- Registered `rsl_rl_sc_cfg_entry_point`.
- Kept the existing default `rsl_rl_cfg_entry_point` unchanged.

Training can now select the SC teacher config without creating a second Gym task
id:

```bash
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 16 --headless --enable_cameras --max_iterations 10
```

## Local Verification

Syntax checks:

```bash
python3 -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sc_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py
```

Status: passed locally.

Whitespace check:

```bash
git diff --check
```

Status: passed locally.

## Remote Verification Needed

Run a small smoke train inside the Isaac Lab container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 4 --headless --enable_cameras --max_iterations 1 \
  --run_name step5_smoke
```

Expected checks:

- `AIC-Task-v0` loads with `rsl_rl_sc_cfg_entry_point`.
- Logs are written under `logs/rsl_rl/aic_sc_insert/`.
- Actor observations use the policy group.
- Critic observations use policy plus privileged groups.
- The smoke run completes at least one PPO iteration.
