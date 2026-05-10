# Step 5: Add SC PPO Teacher Config

This note records the SC-specific RSL-RL PPO teacher config.

## Step 4 Input

Step 4 added true SC insertion success termination. The teacher config can now
train against insertion rewards and an insertion success done term instead of
pure timeout episodes.

## Remaining Plan Check

Step 6 commands should now pass `--agent rsl_rl_sc_cfg_entry_point` so training
and play use the SC teacher config and write under `logs/rsl_rl/aic_sc_insert`.

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

Changed `scripts/rsl_rl/cli_args.py`, `train.py`, and `play.py` after the first
remote smoke attempt:

- Added `runner_cfg_to_dict`.
- Removed model config keys that the installed RSL-RL `MLPModel` does not
  accept (`stochastic`, `init_noise_std`, `noise_std_type`,
  `state_dependent_std`) before constructing `OnPolicyRunner` or
  `DistillationRunner`.
- This keeps the config source compatible with Isaac Lab's config class while
  matching the installed RSL-RL runner API.
- Wrapped `simulation_app.close()` in `finally` in both scripts so failed smoke
  runs do not leave orphaned Isaac Python processes.

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
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/cli_args.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sc_cfg.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py
```

Status: passed locally.

Whitespace check:

```bash
git diff --check
```

Status: passed locally.

## Remote Verification

Implementation commits:

- `ab13e0a Add SC PPO teacher config`
- `9ff42f5 Sanitize RSL-RL model config for runner`
- `cdec30a Close Isaac app on RSL-RL script failure`

Pulled on the remote host:

```bash
cd ~/IsaacLab/aic
git pull --ff-only
```

Copied the changed source files into the running Isaac Lab container:

```bash
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sc_cfg.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sc_cfg.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/cli_args.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/cli_args.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py
docker cp \
  ~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  isaac-lab-base:/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py
```

First smoke attempt loaded the new config and observation groups but failed
before training:

```text
Parsing configuration from: ...agents.rsl_rl_ppo_sc_cfg:PPORunnerCfg
Resolved observation sets:
     actor :  ['policy']
     critic :  ['policy', 'privileged']
TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'
```

Inspected the installed RSL-RL API in the container:

```text
RslRlMLPModelCfg annotations include:
stochastic, init_noise_std, noise_std_type, state_dependent_std

rsl_rl.models.mlp_model.MLPModel.__init__ accepts:
obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
obs_normalization, distribution_cfg
```

After adding the config sanitizer and cleaning the stale failed process, ran a
fresh smoke train inside the Isaac Lab container through host `tmux` session
`isaac-step5-sc-smoke-cdec30a`:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 4 --headless --enable_cameras --max_iterations 1 \
  --run_name step5_smoke
```

Key output:

```text
Parsing configuration from: ...agents.rsl_rl_ppo_sc_cfg:PPORunnerCfg
Logging experiment in directory: /workspace/isaaclab/logs/rsl_rl/aic_sc_insert
Resolved observation sets:
     actor :  ['policy']
     critic :  ['policy', 'privileged']
Actor Model first layer: Linear(in_features=3149, out_features=512)
Critic Model first layer: Linear(in_features=3169, out_features=512)
Learning iteration 0/1
Total steps: 96
Mean reward: -0.07
Episode_Termination/time_out: 0.1667
Episode_Termination/sc_insertion_success: 0.0000
Training time: 5.32 seconds
STEP5_SMOKE_EXIT:0
```

Copied the smoke logs back to the host:

```bash
mkdir -p ~/IsaacLab/aic/logs/rsl_rl
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/aic_sc_insert \
  ~/IsaacLab/aic/logs/rsl_rl/
```

Result:

```text
COPY_EXIT:0
~/IsaacLab/aic/logs/rsl_rl/aic_sc_insert/2026-05-10_11-01-12_step5_smoke
```
