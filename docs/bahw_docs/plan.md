# Concrete Isaac Lab Implementation Plan

This document is the implementation checklist for the strategy in
`docs/bahw_docs/overview.md`.

Scope:

- Isaac Lab training path only
- SC first, then SFP
- PPO with asymmetric actor-critic
- eval-compatible actor from the start
- privileged critic during training
- direct plug-to-port rewards, not `ee_pose` command rewards
- high-level distillation only after PPO teachers work

## Current Decisions

- Start with SC insertion because the current Isaac scene already includes
  `sc_port` and `sc_port_2`.
- Use direct plug-to-port geometry for reward and success. Do not keep the
  current `ee_pose` command rewards as the learning objective.
- Train the actor with eval-compatible observations from day one. The critic gets
  privileged geometry.
- Train separate SC and SFP teachers first. A single submitted `aic_model` can
  later route to the right checkpoint using eval-provided `Task` metadata.
- Treat distillation as a later phase. Do not implement it before the privileged
  PPO teacher can solve insertion.

## Relevant Files

Primary Isaac Lab task files:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py`

Recommended new files:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py`

Reference Gazebo asset files:

- `aic_assets/models/SC Plug/model.sdf`
- `aic_assets/models/SC Port/model.sdf`
- `aic_assets/models/SFP Module/model.sdf`
- `aic_assets/models/NIC Card/model.sdf`
- `aic_assets/models/NIC Card Mount/model.sdf`

Remote Isaac assets:

- machine: `bahw@100.103.111.75`
- path: `~/IsaacLab/Intrinsic_assets`
- Isaac repo path: `~/IsaacLab/aic`
- training runs inside the Isaac Lab Docker container at `/workspace/isaaclab`
  - this path is verified from the remote Isaac Lab Docker config, not from the
    local AIC repo

## Step 0: Confirm Asset Frames In Isaac

Why:

Gazebo SDF has semantic frames for the real insertion geometry. We need to verify
whether the Isaac USD assets expose equivalent frames. If they do not, we must
add fixed helper offsets in Isaac Lab.

Known Gazebo SDF frames:

- SC plug tip: `sc_tip_link`
- SC port entrance: `sc_port_base_link_entrance`
- SFP plug/module tip: `sfp_tip_link`
- SFP port entrances: `sfp_port_0_link_entrance`, `sfp_port_1_link_entrance`

Work:

- [x] Write `scripts/inspect_aic_geometry.py`.
  - Added local script at
    `aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py`.
    It inspects both SC and SFP semantic frame names and writes timestamped
    logs under `logs/aic_geometry/`, which remain untracked.
- [x] Create `AIC-Task-v0` with `num_envs=1`.
- [x] Print all robot body names.
- [x] Print all scene asset names.
- [x] Print all rigid body names for `sc_port`, `sc_port_2`, `nic_card`, and the
  robot articulation.
- [x] Search for tip/entrance names above.
- [x] Print root poses for `sc_port`, `sc_port_2`, and any available plug bodies.

Result from remote log
`logs/aic_geometry/20260510_090800_AIC-Task-v0.log`:

- Isaac exposes `sc_tip_link` as runtime body `robot.sc_tip_link` and USD prim
  `/World/envs/env_0/Robot/cable/sc_plug/sc_tip_link`.
- Isaac does not expose `sc_port_base_link_entrance` as a runtime body or USD
  prim.
- The nearest SC port frames available in the runtime are `sc_port` and
  `sc_port_2`, each with body `sc_port_visual` plus root pose.
- Isaac exposes `sfp_tip_link` as runtime body `robot.sfp_tip_link`.
- SFP port entrances are present as USD prims under `nic_card`, but not as
  runtime rigid bodies.

Expected command, inside the Isaac Lab container:

```bash
cd /workspace/isaaclab
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py \
  --task AIC-Task-v0 --num_envs 1 --headless --enable_cameras
```

Actual remote run used `./isaaclab.sh -p ...` because `isaaclab` was not on
`PATH` in the rebuilt `isaac-lab-base` container.

Done when:

- [x] We know whether the USD assets expose SC tip and SC port entrance frames.
- [x] If exposed, the plan uses those bodies directly.
  - Use `robot.sc_tip_link` directly for the SC plug tip.
- [x] If not exposed, document the nearest available body/root frame and defer
  fixed transform derivation to Step 1.
  - Derive the SC port entrance helper pose from `sc_port` or `sc_port_2`
    root/body `sc_port_visual` in Step 1.
- [x] The script output is saved under `logs/aic_geometry/`.

## Step 1: Add Geometry Helpers

Why:

Rewards, observations, and terminations must use exactly the same plug-to-port
geometry. Do not duplicate transform math across files.

Work:

- [ ] Add `mdp/geometry.py`.
- [ ] Implement helper functions for SC first:
  - [ ] active target selection
  - [ ] plug tip pose
  - [ ] port entry pose
  - [ ] port insertion axis
  - [ ] plug-to-port vector
  - [ ] lateral error
  - [ ] insertion depth
  - [ ] orientation error
- [ ] Prefer named USD bodies if Step 0 confirms them.
- [ ] If named bodies are missing, compute helper poses from known fixed offsets
  derived from the Gazebo SDF.

Initial SC target selection:

- [ ] Sample the active SC target per environment on reset: `sc_port` or `sc_port_2`.
- [ ] Store the active target index on the env object as a tensor.
- [ ] Expose active target metadata to the actor as eval-compatible task information.

Important:

- Task metadata is not privileged. During Gazebo evaluation, `Task.msg` gives
  `plug_type`, `port_type`, `plug_name`, `port_name`, and
  `target_module_name`.
- Exact plug-to-port geometry is privileged. It must not be used by the deployed
  actor.

Done when:

- [ ] Geometry helpers return tensors shaped `(num_envs, ...)`.
- [ ] Geometry helpers work for both `sc_port` and `sc_port_2`.
- [ ] `inspect_aic_geometry.py` prints sane values for plug tip pose, port entry
  pose, lateral error, orientation error, and insertion depth.
- [ ] Insertion depth sign is verified visually or numerically: moving the plug into
  the port increases the chosen depth metric.

## Step 2: Add Eval-Compatible And Privileged Observations

Why:

The actor should learn from data that can exist at evaluation time. The critic
can use extra simulator geometry to make PPO training easier.

Work in `mdp/observations.py`:

- [ ] Add policy observation terms:
  - [ ] task metadata one-hot or small numeric vector
  - [ ] keep joint position and velocity
  - [ ] keep end-effector pose
  - [ ] keep force/wrench-like robot signal
  - [ ] keep camera features
  - [ ] keep last action
- [ ] Add privileged observation terms:
  - [ ] `plug_to_port_vec`
  - [ ] `lateral_error`
  - [ ] `orientation_error`
  - [ ] `insertion_depth`
  - [ ] active port pose if useful
  - [ ] plug tip pose if useful

Work in `aic_task_env_cfg.py`:

- [ ] Remove `pose_command` from the policy observation group.
- [ ] Add a new `PrivilegedCfg` observation group.
- [ ] Keep `PolicyCfg` eval-compatible.
- [ ] Make sure term concatenation is stable and dimensions do not change by episode.

Work in `rsl_rl_ppo_cfg.py`:

- [ ] Update the existing `obs_groups` mapping so the critic receives the new
  privileged observation group.

```python
obs_groups = {
    "actor": ["policy"],
    "critic": ["policy", "privileged"],
}
```

Done when:

- [ ] `policy` observation contains no privileged geometry.
- [ ] `privileged` observation contains plug-to-port geometry.
- [ ] Actor and critic observation dimensions are stable after reset.
- [ ] RSL-RL config maps actor to `policy` and critic to `policy + privileged`.

Verification:

```bash
rg -n "pose_command|PrivilegedCfg|obs_groups|plug_to_port|insertion_depth" \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task
```

## Step 3: Replace Command-Pose Rewards With Insertion Rewards

Why:

The current rewards optimize reaching a sampled `ee_pose`. That must stop being
the training objective.

Work in `aic_task_env_cfg.py`:

- [ ] Remove or set weight to zero for:
  - [ ] `position_command_error`
  - [ ] `position_command_error_tanh`
  - [ ] `position_command_error_exp`
  - [ ] `orientation_command_error`
  - [ ] `orientation_command_error_tanh`
  - [ ] `ee_reaching_bonus`
- [ ] Keep smoothness and safety penalties if still useful:
  - [ ] action rate
  - [ ] joint velocity
  - [ ] joint acceleration
  - [ ] joint torques
  - [ ] joint position limits

Work in `mdp/rewards.py`:

- [ ] Add insertion-specific terms:
  - [ ] lateral alignment reward
  - [ ] orientation alignment reward
  - [ ] approach reward
  - [ ] insertion depth reward
  - [ ] success bonus
  - [ ] optional force/torque penalty near contact

Initial reward shape:

- lateral reward: high when plug tip is centered on the port entrance plane
- orientation reward: high when plug axis aligns with port axis
- approach reward: helps move from start pose to near-port pose
- depth reward: increases only when aligned and moving into the port
- success bonus: large sparse bonus when all success thresholds pass

Guard against bad shaping:

- Do not give large depth reward if lateral error is too high.
- Do not count hovering near the entrance as success.
- Do not reward pushing past the port with bad orientation.

Done when:

- [ ] No nonzero reward term depends on `ee_pose`.
- [ ] Random policy reward logs show each reward term is finite.
- [ ] Manually moving the plug closer to the port improves lateral/approach rewards.
- [ ] Manually increasing insertion depth improves depth reward only when alignment
  is reasonable.

Verification command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/check_aic_rewards.py \
  --task AIC-Task-v0 --num_envs 16 --headless --enable_cameras
```

## Step 4: Add True Success Termination

Why:

Timeout-only episodes do not tell training or evaluation whether insertion was
actually completed.

Work:

- [ ] Add `mdp/terminations.py`.
- [ ] Implement `sc_insertion_success`.
- [ ] Wire it into `TerminationsCfg` in `aic_task_env_cfg.py`.

Initial success definition:

- lateral error below threshold
- orientation error below threshold
- insertion depth above threshold
- optional low relative velocity or short dwell period

Initial thresholds should be conservative and then tuned after geometry
inspection:

- lateral error: start around `0.003` to `0.005` m
- orientation error: start around `0.10` to `0.20` rad
- insertion depth: derive from SC port entrance geometry; CheatCode uses descent
  until about `-0.015` m relative offset

Done when:

- [ ] Success termination fires only when the plug is visibly inserted.
- [ ] Success does not fire when the plug is hovering near the entrance.
- [ ] Success works for both `sc_port` and `sc_port_2`.
- [ ] Timeout still works for failed episodes.

Verification:

```bash
rg -n "sc_insertion_success|time_out|TerminationsCfg" \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task
```

## Step 5: Add SC PPO Teacher Config

Why:

Keep the current generic PPO config intact enough to compare, but create a clear
SC insertion teacher config.

Work:

- [ ] Add a new RSL-RL config or clearly rename the existing one:
  - [ ] `agents/rsl_rl_ppo_sc_cfg.py`, or
  - [ ] keep `rsl_rl_ppo_cfg.py` but set `experiment_name = "aic_sc_insert"`.
- [ ] Prefer a separate config if we will soon add SFP.
- [ ] Register the config entry point if adding a new config.
- [ ] Preserve the existing `obs_groups` setting but update its critic entry
  from `["policy"]` to `["policy", "privileged"]`.

Teacher setup:

- actor obs: `policy`
- critic obs: `policy + privileged`
- actor hidden dims: start with current MLP dimensions
- critic hidden dims: start with current MLP dimensions
- keep PPO initially close to current values to reduce variables

Done when:

- [ ] `AIC-Task-v0` loads with the SC PPO teacher config.
- [ ] RSL-RL sees different actor and critic observation dimensions.
- [ ] A 1 to 10 iteration smoke run starts and writes logs.

Smoke command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --num_envs 16 --headless --enable_cameras \
  --max_iterations 10
```

## Step 6: Train And Evaluate SC Teacher

Why:

This is the first proof that the Isaac MDP is correct.

Work:

- [ ] Train with enough parallel envs for throughput.
- [ ] Start with 64 envs if camera memory allows.
- [ ] Increase after smoke runs are stable.
- [ ] Save videos periodically for qualitative checks.

Training command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --num_envs 64 --headless --enable_cameras
```

Evaluation command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  --task AIC-Task-v0 --num_envs 16 --headless --enable_cameras \
  --checkpoint <checkpoint_path>
```

Done when:

- [ ] SC teacher inserts reliably in simulation.
- [ ] Success rate is measured over randomized `sc_port` and `sc_port_2` targets.
- [ ] Videos show true insertion, not hovering or target cheating.
- [ ] Reward curves are dominated by insertion progress and success, not smoothness
  penalties.

Minimum metric to record:

- [ ] success rate
- [ ] mean episode length on success
- [ ] mean lateral error at termination
- [ ] mean insertion depth at termination
- [ ] failure breakdown: timeout, lateral miss, orientation miss, depth shortfall

## Step 7: Extend The Same MDP To SFP

Why:

Qualification trials 1 and 2 are SFP, so SFP must be trained directly.

Work:

- [ ] Add SFP assets to Isaac scene if not already present.
- [ ] Add or expose SFP plug tip geometry.
- [ ] Add or expose SFP port entrance geometry.
- [ ] Add active SFP target metadata for `sfp_port_0` and `sfp_port_1`.
- [ ] Reuse the same geometry helper interface from SC.
- [ ] Add SFP reward thresholds and insertion depth thresholds.

Reference Gazebo SDF frames:

- `sfp_tip_link`
- `sfp_port_0_link_entrance`
- `sfp_port_1_link_entrance`

Done when:

- [ ] The same observation/reward/termination API works for SFP.
- [ ] SFP target can switch between port 0 and port 1.
- [ ] SFP PPO teacher smoke run starts.
- [ ] SFP teacher learns insertion in simulation.

## Step 8: Keep Separate Specialist Checkpoints

Why:

SC and SFP are different enough that specialist policies are the safest first
qualification path.

Output checkpoints:

- `sc_teacher.pt`
- `sfp_teacher.pt`

Later distilled outputs:

- `sc_student.pt`
- `sfp_student.pt`

Done when:

- [ ] SC checkpoint solves SC validation.
- [ ] SFP checkpoint solves SFP validation.
- [ ] Each checkpoint has recorded config, commit hash, and success metrics.

## Step 9: Distillation, High Level Only

Do not start this until the PPO teacher is useful.

High-level work later:

- [ ] Create RSL-RL distillation config for SC.
- [ ] Create RSL-RL distillation config for SFP.
- [ ] Load trained PPO teacher checkpoints.
- [ ] Train students using eval-compatible observations only.
- [ ] Export student policies.
- [ ] In the final Gazebo wrapper, route using official `Task` metadata:
  - [ ] `plug_type == "sc"` or `port_type == "sc"` uses SC checkpoint
  - [ ] `plug_type == "sfp"` or `port_type == "sfp"` uses SFP checkpoint

Done when:

- [ ] Distilled student success is close to teacher success in Isaac.
- [ ] Student receives no privileged geometry.
- [ ] Wrapper routing uses only official `Task.msg` fields.

## Implementation Order

- [ ] 1. Add `inspect_aic_geometry.py`.
- [ ] 2. Confirm or reconstruct SC plug-tip and port-entry poses.
- [ ] 3. Add `mdp/geometry.py`.
- [ ] 4. Add SC active target selection.
- [ ] 5. Add policy and privileged observation terms.
- [ ] 6. Update PPO obs groups for asymmetric actor-critic.
- [ ] 7. Replace command-pose rewards with insertion rewards.
- [ ] 8. Add success termination.
- [ ] 9. Add `check_aic_rewards.py`.
- [ ] 10. Run SC smoke training.
- [ ] 11. Train SC teacher.
- [ ] 12. Extend the same geometry/reward interface to SFP.
- [ ] 13. Train SFP teacher.
- [ ] 14. Revisit distillation only after both teachers work.

## Global Done Criteria

The SC implementation is done when:

- [ ] `AIC-Task-v0` no longer trains on random `ee_pose` reaching.
- [ ] Actor observations are eval-compatible.
- [ ] Critic observations include privileged insertion geometry.
- [ ] SC plug-to-port geometry is correct for both SC ports.
- [ ] SC PPO teacher inserts in randomized simulation.

The SFP implementation is done when:

- [ ] The same MDP structure supports SFP.
- [ ] SFP plug and port entrance geometry are correct.
- [ ] SFP PPO teacher inserts in randomized simulation.

The training path is ready for distillation when:

- [ ] Both specialist teachers solve their respective Isaac tasks.
- [ ] Metrics and videos confirm actual insertion.
- [ ] There is no dependence on privileged geometry in actor observations.
