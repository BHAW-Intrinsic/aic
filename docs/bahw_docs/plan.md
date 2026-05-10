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

Detailed remote reproduction notes and output are in
`docs/bahw_docs/detailed/step0.md`.

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

- [x] Add `mdp/geometry.py`.
- [x] Implement helper functions for SC first:
  - [x] active target selection
  - [x] plug tip pose
  - [x] port entry pose
  - [x] port insertion axis
  - [x] plug-to-port vector
  - [x] lateral error
  - [x] insertion depth
  - [x] orientation error
- [x] Use Step 0's confirmed SC plug body directly:
  - [x] `robot.sc_tip_link` for the SC plug tip pose.
- [x] Derive SC port entry poses from Step 0's nearest available Isaac frames:
  - [x] `sc_port` root/body `sc_port_visual`
  - [x] `sc_port_2` root/body `sc_port_visual`
- [x] Compute the missing `sc_port_base_link_entrance` helper pose from fixed
  offsets derived from the Gazebo SDF. Do not block on finding a named SC port
  entrance body in Isaac; Step 0 confirmed it is absent.
- [x] Keep the geometry helper interface generic enough that SFP can later use
  either runtime bodies or USD-derived helper poses.

Initial SC target selection:

- [x] Sample the active SC target per environment on reset: `sc_port` or `sc_port_2`.
- [x] Store the active target index on the env object as a tensor.
- [x] Expose active target metadata to the actor as eval-compatible task information.
  - Completed in Step 2 as `task_metadata` one-hot.

Important:

- Task metadata is not privileged. During Gazebo evaluation, `Task.msg` gives
  `plug_type`, `port_type`, `plug_name`, `port_name`, and
  `target_module_name`.
- Exact plug-to-port geometry is privileged. It must not be used by the deployed
  actor.

Done when:

- [x] Geometry helpers return tensors shaped `(num_envs, ...)`.
- [x] Geometry helpers work for both `sc_port` and `sc_port_2`.
- [x] `inspect_aic_geometry.py` prints sane values for plug tip pose, port entry
  pose, lateral error, orientation error, and insertion depth.
- [x] Insertion depth sign is verified visually or numerically: moving the plug into
  the port increases the chosen depth metric.

Result from remote log
`logs/aic_geometry/20260510_100156_AIC-Task-v0.log`:

- Helper tensors printed with shapes `(4, 3)`, `(4, 4)`, or `(4,)` for
  `num_envs=4`.
- Reset sampled active target IDs `[1, 0, 1, 1]`, covering both `sc_port` and
  `sc_port_2`.
- The active `sc_port_2` entry pose matched the per-target `sc_port_2` helper
  pose.
- The SC port insertion axis in world frame was `[0, 0, -1]`; the reset plug tip
  was above the entrance, giving negative depth, and moving along the insertion
  axis increases the depth metric.
- Detailed remote reproduction notes and output are in
  `docs/bahw_docs/detailed/step1.md`.

## Step 2: Add Eval-Compatible And Privileged Observations

Why:

The actor should learn from data that can exist at evaluation time. The critic
can use extra simulator geometry to make PPO training easier.

Work in `mdp/observations.py`:

- [x] Add policy observation terms:
  - [x] task metadata one-hot or small numeric vector
  - [x] keep joint position and velocity
  - [x] keep end-effector pose
  - [x] keep force/wrench-like robot signal
  - [x] keep camera features
  - [x] keep last action
- [x] Add privileged observation terms:
  - [x] `plug_to_port_vec`
  - [x] `lateral_error`
  - [x] `orientation_error`
  - [x] `insertion_depth`
  - [x] active port pose if useful
  - [x] plug tip pose if useful

Work in `aic_task_env_cfg.py`:

- [x] Remove `pose_command` from the policy observation group.
- [x] Add a new `PrivilegedCfg` observation group.
- [x] Keep `PolicyCfg` eval-compatible.
- [x] Make sure term concatenation is stable and dimensions do not change by episode.

Work in `rsl_rl_ppo_cfg.py`:

- [x] Update the existing `obs_groups` mapping so the critic receives the new
  privileged observation group.

```python
obs_groups = {
    "actor": ["policy"],
    "critic": ["policy", "privileged"],
}
```

Done when:

- [x] `policy` observation contains no privileged geometry.
- [x] `privileged` observation contains plug-to-port geometry.
- [x] Actor and critic observation dimensions are stable after reset.
- [x] RSL-RL config maps actor to `policy` and critic to `policy + privileged`.

Result from remote run `20260510_101621_AIC-Task-v0.log`:

- `policy` group shape is `(3149,)` and contains `task_metadata`, joint state,
  end-effector pose, body forces, camera features, and last action.
- `policy` no longer contains `pose_command`.
- `privileged` group shape is `(20,)` and contains plug-to-port vector, lateral
  error, orientation error, insertion depth, active port pose, and plug tip pose.
- Gym observation space is
  `Dict('policy': Box(..., (4, 3149)), 'privileged': Box(..., (4, 20)))`.
- RSL-RL config maps `actor` to `["policy"]` and `critic` to
  `["policy", "privileged"]`.
- Detailed remote reproduction notes and output are in
  `docs/bahw_docs/detailed/step2.md`.

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

- [x] Remove or set weight to zero for:
  - [x] `position_command_error`
  - [x] `position_command_error_tanh`
  - [x] `position_command_error_exp`
  - [x] `orientation_command_error`
  - [x] `orientation_command_error_tanh`
  - [x] `ee_reaching_bonus`
- [x] Keep smoothness and safety penalties if still useful:
  - [x] action rate
  - [x] joint velocity
  - [x] joint acceleration
  - [x] joint torques
  - [x] joint position limits

Work in `mdp/rewards.py`:

- [x] Add insertion-specific terms:
  - [x] lateral alignment reward
  - [x] orientation alignment reward
  - [x] approach reward
  - [x] insertion depth reward
  - [x] success bonus
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

- [x] No nonzero reward term depends on `ee_pose`.
- [x] Random policy reward logs show each reward term is finite.
- [x] Manually moving the plug closer to the port improves lateral/approach rewards.
- [x] Manually increasing insertion depth improves depth reward only when alignment
  is reasonable.

Result:

- Commit `ca3456f` verified `AIC-Task-v0` remotely with
  `scripts/check_aic_rewards.py`.
- Active reward terms are SC insertion shaping plus smoothness/safety penalties;
  the command manager still creates `ee_pose`, but no reward term uses it.
- Analytic shape checks verified approach, lateral, orientation, and depth reward
  monotonicity, and random-policy reward tensors were finite.
- Reward log copied on the host to
  `~/IsaacLab/aic/logs/aic_rewards/20260510_103443_AIC-Task-v0.log`.

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

- [x] Add `mdp/terminations.py`.
- [x] Implement `sc_insertion_success`.
- [x] Wire it into `TerminationsCfg` in `aic_task_env_cfg.py`.

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

- [x] Success termination fires only when the plug is visibly inserted.
- [x] Success does not fire when the plug is hovering near the entrance.
- [x] Success works for both `sc_port` and `sc_port_2`.
- [x] Timeout still works for failed episodes.

Result:

- Commit `43e88c0` added `mdp/terminations.py` and wired
  `sc_insertion_success` into `TerminationsCfg`.
- The same shared success mask now drives the sparse reward bonus and the
  termination.
- Remote headless verification showed `time_out` and `sc_insertion_success` as
  active termination terms, with `time_out` remaining the only timeout term.
- Analytic checks confirmed inserted geometry succeeds while hovering, lateral
  miss, orientation miss, depth shortfall, and exact-threshold equality fail.
- Reset-state success was false for both `sc_port` and `sc_port_2`.
- Visual confirmation remains part of Step 6 video review, but the Step 4
  threshold logic and runtime wiring are complete.
- Termination log copied on the host to
  `~/IsaacLab/aic/logs/aic_terminations/20260510_104604_AIC-Task-v0.log`.

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

- [x] Add a new RSL-RL config or clearly rename the existing one:
  - [x] `agents/rsl_rl_ppo_sc_cfg.py`, or
  - [ ] keep `rsl_rl_ppo_cfg.py` but set `experiment_name = "aic_sc_insert"`.
- [x] Prefer a separate config if we will soon add SFP.
- [x] Register the config entry point if adding a new config.
- [x] Preserve the existing `obs_groups` setting but update its critic entry
  from `["policy"]` to `["policy", "privileged"]`.

Teacher setup:

- actor obs: `policy`
- critic obs: `policy + privileged`
- actor hidden dims: start with current MLP dimensions
- critic hidden dims: start with current MLP dimensions
- keep PPO initially close to current values to reduce variables

Done when:

- [x] `AIC-Task-v0` loads with the SC PPO teacher config.
- [x] RSL-RL sees different actor and critic observation dimensions.
- [x] A 1 to 10 iteration smoke run starts and writes logs.

Result:

- Commit `ab13e0a` added `agents/rsl_rl_ppo_sc_cfg.py` and registered
  `rsl_rl_sc_cfg_entry_point`.
- Commits `9ff42f5` and `cdec30a` made local RSL-RL train/play scripts
  compatible with the installed RSL-RL model constructor and ensured failed
  scripts close the Isaac app.
- Remote smoke run used `--agent rsl_rl_sc_cfg_entry_point`, loaded
  `experiment_name = "aic_sc_insert"`, resolved actor observations to
  `["policy"]`, and resolved critic observations to `["policy", "privileged"]`.
- Actor first layer used `3149` inputs; critic first layer used `3169` inputs.
- One PPO iteration completed with `STEP5_SMOKE_EXIT:0`.
- Smoke logs copied on the host to
  `~/IsaacLab/aic/logs/rsl_rl/aic_sc_insert/2026-05-10_11-01-12_step5_smoke`.

Smoke command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 16 --headless --enable_cameras \
  --max_iterations 10
```

## Step 6: Train And Evaluate SC Teacher

Why:

This is the first proof that the Isaac MDP is correct.

Work:

- [x] Train with enough parallel envs for throughput.
- [x] Start with 64 envs if camera memory allows.
  - Baseline run `step6_sc_teacher` ran to iteration `523` before being stopped.
    Infrastructure was stable, but learning failed: insertion-depth reward,
    insertion-success reward, and success termination stayed at `0.0`.
- [ ] Increase after smoke runs are stable.
- [ ] Save videos periodically for qualitative checks.
- [x] Add a first Step 6 remediation before SFP:
  - added reset-initialized SC progress buffers
  - added distance, lateral, orientation, and signed-depth progress rewards
  - increased coarse approach reward width/weight
- [x] Verify the remediated reward terms in Isaac.
  - First remote reward check saw `14` active terms and exited with
    `STEP6_REWARD_CHECK_EXIT:0`.
  - Second remote reward check saw `16` active terms after adding coarse
    alignment and exited with `STEP6_COARSE_REWARD_EXIT:0`.
- [ ] Retrain the SC teacher with remediated reward shaping.
  - Stopped `step6_sc_progress` at iteration `106`; it still had zero fine
    lateral alignment, insertion depth, and success.
  - Stopped `step6_sc_coarse` at iteration `45`; task rewards were active but
    smoothness penalties dominated and insertion terms stayed zero.
  - Stopped `step6_sc_rebalanced` at iteration `110`; reward balance improved,
    but fine lateral alignment, insertion depth, and success stayed at zero.
  - Stopped `step6_sc_virtual_tip_a7144d2` at iteration `65`; virtual helper
    made the geometry controllable for scripted IK, but PPO still did not sample
    the fine insertion corridor from the normal reset distribution.
- [x] Add a headless scripted SC insertion check before changing the trainer
  again.
  - Added `scripts/check_aic_scripted_insert.py` to test whether privileged
    plug-to-port geometry can drive the existing relative IK action into the SC
    success condition.
- [x] Run the scripted SC insertion check remotely.
  - Result: `0/8` successes over `1500` steps. The scripted controller could
    move the plug to positive depth, but lateral/orientation errors remained too
    large for success.
  - A second run with `--control_frame tip` also produced `0/8` successes. It
    logged `wrist_to_sc_tip_pos_drift` mean `1.185182` m, so `sc_tip_link` is
    not behaving like a fixed helper frame rigidly attached to the
    `wrist_3_link` action target.
- [ ] If scripted IK succeeds, use it to derive a near-port reset/curriculum or
  demonstration seed before more PPO.
- [ ] If scripted IK fails, fix the geometry/action convention before more PPO.
  - Current next work: inspect and repair the plug control path. Determine the
    actual gripper/TCP frame, whether the SC plug is rigidly attached to it in
    Isaac, and then set the IK action target/body offset or asset attachment so
    `sc_tip_link` motion is controllable.
  - Added the next diagnostic to the scripted checker: set the IK action body to
    `gripper_tcp` and log drift from `wrist_3_link`, `gripper_tcp`,
    `ati_tool_link`, `tool0`, and `sc_plug_link` to `sc_tip_link`.
  - The `gripper_tcp` diagnostic confirmed the SC tip is rigid to
    `sc_plug_link` but not to the gripper/TCP: `gripper_tcp_to_sc_tip_pos_drift`
    mean `0.953258` m, while `sc_plug_link_to_sc_tip_pos_drift` mean `0.0`.
  - Added a temporary virtual gripped SC tip helper from `gripper_tcp` and
    changed the IK target/eef observation to `gripper_tcp` so the SC training
    geometry is controllable while the USD attachment issue remains unresolved.
  - The first identity TCP helper stalled outside the port; a `0.05` m helper
    offset reached positive but insufficient depth.
  - A `0.07` m helper offset produced `3/4` scripted successes over `500`
    steps. The remaining miss was a strict lateral-threshold miss on one
    `sc_port_2` episode, so the next gate is an expanded scripted validation
    across more envs/seeds before PPO retraining.
  - Reward smoke check after the virtual helper saw `16` active reward terms,
    active success termination, `analytic_shape_checks_ok: True`, and
    `overall_finite: True`.
- [ ] Run expanded virtual-tip scripted validation.
  - Use more envs/seeds than the first `4`-env check and inspect per-target
    `sc_port` vs `sc_port_2` success before deciding whether to launch PPO,
    tune the helper offset, or adjust per-target port-entry geometry.
  - Expanded `16`-env check reached `14/16` successes: `7/8` on `sc_port` and
    `7/8` on `sc_port_2`. The two misses were near-threshold lateral/depth
    cases, so PPO retraining is justified; if PPO fails next, prefer curriculum
    or lateral convergence work over another blind reward rebalance.
- [ ] Add a near-port reset curriculum for Step 6.
  - Extracted first-success arm joint seeds from scripted insertion:
    `sc_port` seed `[0.8142, -1.8485, -1.8316, -1.0275, 1.5704, 2.1715]`,
    `sc_port_2` seed `[0.7603, -1.8014, -1.8958, -1.0112, 1.5705, 2.1117]`.
  - Added `reset_robot_near_sc_port` as a temporary curriculum reset with
    `blend=0.95`, `position_noise=0.01`, and `probability=1.0`.
  - First smoke with `blend=0.85` was wired correctly but still too far from
    the fine corridor: lateral mean `0.058786`, depth mean `-0.025474`.
  - Tightened to `blend=0.95`, `position_noise=0.01`. Reset starts outside
    success but near the port: lateral mean `0.023949`, depth mean `0.001344`.
  - Scripted insertion from the tightened reset reached `16/16` successes by
    step `6`.
  - PPO retry from the tightened reset had nonzero insertion samples at
    iteration `0`, but stayed flat through iteration `25` with action std `1.0`.
- [ ] Reduce SC PPO exploration scale for the near-port curriculum.
  - Changed actor Gaussian `init_std` from `1.0` to `0.2` and entropy coefficient
    from `0.006` to `0.001`.
  - Reduced-std PPO rose from `0.0495` to `0.1250` success termination but
    plateaued through iteration `70`.
- [x] Freeze board/SC port randomization for the first final-insertion
  curriculum stage.
  - Temporarily set board `x/y` randomization and SC port `x` randomization to
    zero while keeping active target sampling between `sc_port` and `sc_port_2`.
  - Fixed-port reset smoke remained near the final corridor: lateral mean
    `0.024799`, orientation mean `0.010348`, depth mean `0.000837`.
  - Fixed-port PPO reached early nonzero success but plateaued at
    `0.2656` success termination by iteration `110`, with insertion-depth
    reward returning to `0.0`.
  - Result: fixed-port reset alone is not enough for reliable final insertion.
- [x] Add a privileged scripted-action-prior reward for the Step 6 teacher
  curriculum.
  - Compare the raw `arm_action` against the successful scripted relative-IK
    action computed from privileged SC geometry.
  - Cache the desired scripted action on reset so the reward compares the action
    just taken against the prior for the state it was taken from.
  - Keep this as reward-only teacher shaping; do not add privileged geometry to
    actor observations.
  - Reward smoke passed with `17` active terms and `overall_finite: True`.
  - PPO retry with the action-prior reward plateaued at `0.1406` success
    termination by iteration `30`, worse than the previous fixed-port plateau.
  - Result: scalar action-prior reward is not enough.
- [ ] Investigate direct scripted actor bootstrap before more PPO.
  - Prefer a supervised/behavior-cloning pretrain from eval-compatible
    `policy` observations to scripted SC actions, then resume PPO.
  - If no small RSL-RL hook exists, add a focused pretrain script rather than
    continuing blind reward tuning.
  - Added `scripts/rsl_rl/pretrain_sc_bc.py` to train the normal RSL-RL actor
    with supervised MSE on scripted raw `arm_action` labels, then save a normal
    RSL-RL checkpoint for PPO resume.
  - Exposed `mdp.sc_scripted_raw_action` as the shared label generator.
  - Next: smoke-test the BC script remotely, then run a short pretrain and
    evaluate/resume PPO from the saved checkpoint.

Training command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 64 --headless --enable_cameras
```

Evaluation command:

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  --task AIC-Task-v0 --agent rsl_rl_sc_cfg_entry_point \
  --num_envs 16 --headless --enable_cameras \
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

Current gate:

- Do not start Step 7 until Step 6 produces a useful SC teacher. Baseline PPO
  and three reward-only remediation attempts did not reach insertion depth or
  success. Step 6 later found that the physical Isaac SC tip was not rigidly
  attached to the controlled TCP path, and a temporary virtual `gripper_tcp`
  tip helper reached scripted insertion. Near-port and fixed-port curricula have
  produced nonzero PPO success samples but still plateau below reliable
  insertion; scalar scripted-action-prior reward shaping also failed to improve.
  This is progress, but it is not a trained SC teacher.

Work:

- [ ] Add SFP assets to Isaac scene if not already present.
- [ ] Use Step 0's confirmed runtime body `robot.sfp_tip_link` for the SFP plug
  tip geometry.
  - Before relying on it, repeat the Step 6 drift/scripted-control diagnostic
    pattern for SFP. The SC body name existed but was not the controlled gripped
    insertion tip, so SFP must not assume body-name presence equals controllable
    insertion geometry.
- [ ] Add or expose SFP port entrance helper poses.
  - Step 0 found `sfp_port_0_link_entrance` and `sfp_port_1_link_entrance` as
    USD prims under `nic_card`, but not as runtime rigid bodies.
  - Prefer reading those USD prim transforms if stable; otherwise derive fixed
    offsets from `nic_card`.
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

- [x] 1. Add `inspect_aic_geometry.py`.
- [x] 2. Confirm or reconstruct SC plug-tip and port-entry poses.
- [x] 3. Add `mdp/geometry.py`.
- [x] 4. Add SC active target selection.
- [x] 5. Add policy and privileged observation terms.
- [x] 6. Update PPO obs groups for asymmetric actor-critic.
- [x] 7. Replace command-pose rewards with insertion rewards.
- [x] 8. Add success termination.
- [x] 9. Add `check_aic_rewards.py`.
- [x] 10. Run SC smoke training.
- [ ] 11. Train SC teacher.
  - [x] Baseline training infrastructure worked.
  - [x] Added first progress-reward remediation after baseline learning failed.
  - [x] Verified remediated rewards load in Isaac.
  - [x] Added second coarse-alignment remediation after progress-only shaping
    failed to reach the port axis by iteration `106`.
  - [x] Rebalanced task reward scale after coarse alignment was still dominated
    by smoothness penalties.
  - [x] Confirmed reward-only shaping still failed after rebalanced training to
    iteration `110`.
  - [x] Added a headless scripted SC insertion check.
  - [x] Ran scripted SC insertion remotely.
  - [x] Ran the tip-frame scripted variant and confirmed `sc_tip_link` is not a
    fixed helper frame relative to `wrist_3_link`.
  - [x] Added a temporary virtual gripped SC tip from `gripper_tcp` to make the
    SC geometry controllable while the USD attachment issue remains unresolved.
  - [x] Validated the virtual helper across more envs/seeds.
  - [x] Added first near-port reset curriculum after PPO from normal reset still
    produced zero insertion-depth samples.
  - [x] Smoke-tested and trained with the near-port curriculum.
  - [x] Froze board/SC port randomization for the first final-insertion
    curriculum stage; fixed-port PPO still plateaued.
  - [x] Added and validated privileged scripted-action-prior reward shaping;
    PPO with this scalar prior still failed to improve.
  - [ ] Add and validate direct scripted actor bootstrap before more PPO.
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
