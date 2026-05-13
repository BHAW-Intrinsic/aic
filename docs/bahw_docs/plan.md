# Concrete Isaac Lab Implementation Plan

This document is the implementation checklist for the strategy in
`docs/bahw_docs/overview.md`.

Scope:

- Isaac Lab training path only
- SC first, then SFP
- PPO with asymmetric actor-critic remains the preferred path for
  generalization.
- Supervised BC/DAgger bootstraps are allowed as diagnostics or warm starts, but
  they are not the preferred final training strategy unless explicitly accepted.
- eval-compatible actor from the start
- privileged critic during training
- direct plug-to-port rewards, not `ee_pose` command rewards
- high-level distillation only after reliable specialist teachers/policies work

## Current Decisions

- Start with SC insertion because the current Isaac scene already includes
  `sc_port` and `sc_port_2`.
- Use direct plug-to-port geometry for reward and success. Do not keep the
  current `ee_pose` command rewards as the learning objective.
- Train the actor with eval-compatible observations from day one. The critic gets
  privileged geometry.
- Train separate SC and SFP teachers or specialist policies first. A single
  submitted `aic_model` can later route to the right checkpoint using
  eval-provided `Task` metadata.
- Treat distillation as a later phase. Do not implement it before a reliable
  teacher/policy can solve insertion.
- Step 6 gate decision: SC at or above `90%` deterministic Isaac success is
  acceptable for unblocking SFP work, provided a video artifact is saved. The
  current BC checkpoint is a neural actor, not a runtime hardcoded CheatCode,
  but PPO remains the preferred final/generalizable training path.
- Step 9 precondition decision: before distillation/export, run a randomized SFP
  reliability pass with two PPO tracks: one warm-started from the successful
  fixed-NIC SFP checkpoint and one control run from scratch. If the scratch run
  learns better randomized insertion, prefer it over the warm-started run.
- Step 9 result: the scratch randomized SFP PPO run is the selected candidate.
  Deterministic Isaac eval was `238/256` overall, with `123/132` on
  `sfp_port_0` and `115/124` on `sfp_port_1`. The warm-start run passed
  overall but failed the per-port gate on `sfp_port_0`.
- Official Gazebo eval wrapper status: scaffold orchestration works and writes
  `scoring.yaml`/trial bags, but final functional Gazebo eval remains blocked
  on the observation/action adapter.

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
- [x] Save videos periodically for qualitative checks.
  - Recorded the current best SC checkpoint with `play.py --video` after fixing
    playback export for bare-actor checkpoints. Local copy:
    `/Users/aloy/Downloads/step6_sc_strict_20260511_d731182c.mp4`.
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
  - Expert-rollout BC smoke passed and `model_1000.pt` reached `193/256`
    deterministic evaluation successes with a 300-step cap (`0.753906`).
  - PPO resume from the BC checkpoint regressed: the first PPO checkpoint
    evaluated at `0/128` successes, so do not prefer PPO-resumed checkpoint.
  - Added `--rollout_policy actor|blend` support for DAgger-style BC on states
    reached by the actor's own actions.
  - Actor-rollout BC from the useful checkpoint completed, but regressed hard:
    `0/256` deterministic evaluation successes, all timeouts, all lateral
    misses, `253/256` orientation misses, and `254/256` depth shortfalls.
    Do not prefer this checkpoint.
- [x] Add failure diagnostics before the next Step 6 training attempt.
  - Extend `evaluate.py` to report signed lateral components in the active port
    frame, optional terminal actor-vs-scripted action error, and sample failure
    rows. Use this to decide whether the useful `193/256` BC checkpoint has a
    systematic calibration bias, a target-specific issue, or a recovery-policy
    issue.
  - Diagnostic repeat of the useful expert-rollout BC checkpoint reached
    `189/256`; failures had mostly negative signed lateral components.
  - Blended rollout BC with `--rollout_actor_weight 0.1` and `lr=1e-4` reached
    `195/256`, only a small improvement.
  - Strict scripted alignment labels with `--align_lateral_threshold 0.01` and
    `--align_orientation_threshold 0.20` reached the current best result:
    `233/256` (`0.910156`) over the fixed 300-step evaluation gate.
  - A tighter/deeper refinement
    (`--align_lateral_threshold 0.005 --target_depth 0.025`) regressed to
    `0/256`; do not prefer it.
  - User decision on 2026-05-11: accept `>90%` SC as sufficient to proceed to
    Step 7, but continue treating PPO as the preferred final path over BC.

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

- [x] SC teacher/policy reaches the accepted provisional reliability gate in
  simulation.
- [x] Success rate is measured over randomized `sc_port` and `sc_port_2` targets.
- [x] Video artifact is saved for qualitative review.
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

- Step 7 is now unblocked by the accepted SC gate from Step 6.
  Baseline PPO
  and three reward-only remediation attempts did not reach insertion depth or
  success. Step 6 later found that the physical Isaac SC tip was not rigidly
  attached to the controlled TCP path, and a temporary virtual `gripper_tcp`
  tip helper reached scripted insertion. Near-port and fixed-port curricula have
  produced nonzero PPO success samples but still plateau below reliable
  insertion; scalar scripted-action-prior reward shaping also failed to improve.
  Expert-rollout BC reached `193/256` successes. The current best strict-label
  BC checkpoint reached `233/256` successes, but PPO resume, actor-rollout BC,
  and overly tight/deep refinement all regressed. The `233/256` result exceeds
  the accepted `90%` SC gate and has a saved video artifact, so SFP work can
  start while PPO remains the preferred final/generalizable training path.

Work:

- [x] Add SFP assets to Isaac scene if not already present.
  - `AIC-SFP-Task-v0` loads the existing `nic_card` and unified robot/cable
    scene; no new scene asset was needed for the Step 7 smoke test.
- [x] Use Step 0's confirmed runtime body `robot.sfp_tip_link` for the SFP plug
  tip geometry.
  - Before relying on it, repeat the Step 6 drift/scripted-control diagnostic
    pattern for SFP. The SC body name existed but was not the controlled gripped
    insertion tip, so SFP must not assume body-name presence equals controllable
    insertion geometry.
  - Step 7 uses `robot.sfp_tip_link` for the initial helper. The diagnostic is
    still required before long SFP training.
  - Step 8 confirmed zero runtime drift from the controlled gripper path to
    `sfp_tip_link`; no SFP virtual tip helper is needed at this point.
- [x] Add or expose SFP port entrance helper poses.
  - Step 0 found `sfp_port_0_link_entrance` and `sfp_port_1_link_entrance` as
    USD prims under `nic_card`, but not as runtime rigid bodies.
  - Prefer reading those USD prim transforms if stable; otherwise derive fixed
    offsets from `nic_card`.
  - Step 7 derives fixed offsets from `nic_card` using the NIC Card SDF port and
    entrance poses.
  - Step 8 confirmed the derived fixed offsets exactly match the USD semantic
    entrance prims for both SFP ports in env 0.
- [x] Add active SFP target metadata for `sfp_port_0` and `sfp_port_1`.
- [x] Reuse the same geometry helper interface from SC.
- [x] Add SFP reward thresholds and insertion depth thresholds.

Reference Gazebo SDF frames:

- `sfp_tip_link`
- `sfp_port_0_link_entrance`
- `sfp_port_1_link_entrance`

Done when:

- [x] The same observation/reward/termination API works for SFP.
- [x] SFP target can switch between port 0 and port 1.
- [x] SFP teacher/policy smoke run starts.
  - Carry over the Step 6 scripted-control plus BC/DAgger bootstrap path if
    reward-only PPO remains unreliable.
- [ ] SFP teacher/policy learns insertion in simulation.

## Step 8: Keep Separate Specialist Checkpoints

Why:

SC and SFP are different enough that specialist policies are the safest first
qualification path.

Output checkpoints:

- `sc_teacher.pt`
- `sfp_teacher.pt`

Prefer PPO checkpoints for final/generalizable specialist policies. BC/DAgger
checkpoints may be kept as diagnostics, warm starts, or provisional gates, and
their provenance must be recorded explicitly.

Immediate Step 8 note from Step 7:

- Completed before long SFP PPO training:
  - `sfp_tip_link` is stable relative to the controlled gripper path.
  - SFP port entrance helpers match the USD semantic entrance frames exactly.
  - The current scripted controller remains `0/16` on SFP and has a systematic
    signed port-frame `x` miss of roughly 4-7 mm, so it is not a suitable SFP BC
    expert. PPO remains the main training path.
- Current SFP PPO curriculum status:
  - Added a temporary fixed-NIC near-port reset curriculum for the first SFP PPO
    stage. Reintroduce NIC `y` randomization after fixed-card insertion works.
  - Fixed-card reset validation starts near the entrance: lateral mean
    `0.003303`, orientation mean `0.012314`, depth mean `-0.001980`.
  - First fixed-NIC PPO attempt from the near-port reset plateaued with
    `sfp_insertion_depth: 0.0000` and no success by iteration `60`.
  - Added shaped SFP depth reward from slightly outside the port
    (`min_depth=-0.006`, `depth_scale=0.018`) and resumed PPO from the
    alignment warm-start `model_50.pt` in run
    `2026-05-11_05-21-25_step8_sfp_ppo_depthreward_491fc43`.
  - Depth-reward PPO still had `0.0000` success by iteration `155`; added a
    privileged inward-action PPO reward and started run
    `2026-05-11_05-33-23_step8_sfp_ppo_action_3b9e781`.
  - Weak inward-action PPO still had zero success by iteration `120`; stronger
    action shaping reached nonzero depth/action rewards but zero strict success
    by iteration `100`.
  - Added a temporary coarse success gate (`lateral <0.020`,
    `orientation <0.50`, `depth >0.005`) and trained to `model_200.pt`; training
    still had zero coarse success.
  - SFP-aware evaluation of
    `2026-05-11_05-47-25_step8_sfp_ppo_coarsegate_b47cc33/model_200.pt` produced
    `0/128` successes under the coarse gate. Terminal failures were far from the
    port (`mean_lateral=0.957564`, `mean_orientation=1.172962`,
    `mean_depth=-0.254260`), so the next remediation is a temporary SFP
    corridor-violation termination to reset attempts once they leave the
    near-port insertion band.
  - Corridor termination smoke passed, but resuming from the strong-action
    checkpoint immediately violated the corridor and remained at zero success.
    Short-horizon evaluation showed it was far off within 10 steps
    (`mean_lateral=0.205032`, `mean_orientation=0.965474`).
  - Reduced the SFP relative-IK action scale from `0.05` to `0.01`, restarted
    from fixed-NIC `model_50.pt`, and observed the first nonzero coarse SFP PPO
    success signal (`Episode_Termination/sfp_insertion_success` around
    `0.0065-0.0078` by iterations `52-55`, intermittently reaching roughly
    `0.0208` by iteration `127`). Most episodes still terminate through the
    corridor guard.
  - The reduced-scale run still had only intermittent low coarse success by
    iteration `215` (`0.0059` at that snapshot) and roughly `99-100%` corridor
    exits, so it was stopped after `model_200.pt`.
  - Added a PPO lateral-guard remediation: reduce/gate inward-action reward,
    multiply it by a smooth lateral-centering factor, and increase lateral
    progress/alignment weights. This stays on PPO rather than BC.
  - The lateral-guard run still had zero success and `1.0000` corridor exits by
    iteration `115`, so it was stopped and replaced with a more direct PPO
    reward for lateral corrective actions.
  - Lateral-correction action reward smoke passed and the run from fixed-NIC
    `model_50.pt` recovered intermittent coarse success around `0.010-0.011` by
    iterations `64`, `127`, and `129`, but the signal did not improve by
    iteration `222`.
  - Evaluation of lateral-correction `model_200.pt` over 64 episodes with
    `max_episode_steps=50` produced `0/64` successes and backed away from the
    port (`mean_lateral=0.142818`, `mean_orientation=1.119678`,
    `mean_depth=-0.170864`), so add a pre-insertion port-approach action reward.
  - Added PPO port-approach action reward to reward motion from SFP plug tip
    toward the active port entry before the insertion-depth reward takes over.
	  - The port-approach warm-start run was worse: it stayed at `0.0000` success
	    and `1.0000` corridor exits through iteration `130`, so the term was
	    disabled again. Next comparison should use the lateral-correction reward set
	    from a fresh PPO start instead of resuming the old fixed-NIC optimizer state.
	  - Fresh-start lateral-correction PPO, lower exploration noise
	    (`init_std=0.05`, entropy coefficient `0.0002`), and a smaller relative-IK
	    scale (`0.005`) all remained at `0.0000` success with near-total corridor
	    exits.
	  - Split the SFP corridor termination into separate lateral, orientation,
	    min-depth, and max-depth terms. The diagnostic run showed lateral drift is
	    the dominant failure: by iteration `16`,
	    `sfp_corridor_lateral_violation` was `1.0000` while the other corridor
	    reasons were `0.0000`.
	  - Added a direct SFP lateral-error PPO penalty (`sfp_lateral_error`,
	    weight `-6.0`) and started a fresh lateral-penalty run. Early iterations
	    still show lateral exits near `1.0000`, so monitor briefly before deciding
	    whether to strengthen lateral control further.
	  - The initial lateral-error penalty was too weak; strengthened the lateral
	    curriculum (`sfp_lateral_progress=20.0`, `sfp_lateral_error=-40.0`,
	    `sfp_lateral_correction_action=80.0`) and started another PPO run.
	  - Strong lateral-control rewards still stayed at `1.0000` lateral exits
	    through iteration `44`, so the next change reduced actual SFP action scale
	    to `0.001` and PPO `init_std` to `0.02`.
	  - The 0.001-scale run produced the best early lateral-stability signal so far
	    at iteration `1` (`mean episode length=39.49`,
	    `sfp_corridor_lateral_violation=0.1133`), but regressed to `1.0000`
	    lateral exits by iteration `52` as learned action std grew.
	  - RSL-RL's current Gaussian distribution config exposes `init_std` and
	    `std_type`, but not a max-std clamp. Lowered SFP PPO `init_std` to `0.005`,
	    set entropy coefficient to `0.0`, and reduced learning rate to `3.0e-4`.
	    This stabilized run is the first to sustain longer early episodes
	    (`mean_episode_length=83.57`, lateral exits `0.4395` at iteration `4`);
	    keep monitoring.
	  - Low-noise PPO then became a timeout/backout local optimum. Added signed
	    port-frame depth-action shaping, but the policy still selected the wrong
	    raw `z` direction by iteration `32`.
	  - Added a training-only SFP PPO actor-output initialization hook. The hook
	    sets the initial actor output bias and can zero the actor output-head
	    weights before PPO starts. This is not a scripted controller; PPO still
	    updates the policy normally.
	  - Zero-head SFP PPO with raw `z=-0.05` fixed the backing-out failure but
	    plateaued as a timeout-only policy. Evaluation of `model_50.pt` over 64
	    episodes produced `0/64` successes with good lateral/orientation alignment
	    (`mean_lateral=0.004905`, `mean_orientation=0.023362`) but a depth
	    shortfall (`mean_depth=-0.001481` vs coarse success `>0.005`).
	  - Added final-depth curriculum `c3504ec`: reduce lateral-action reward,
	    strengthen depth/action rewards, target a deeper depth, and initialize raw
	    `z=-0.10`.
	  - Final-depth PPO still produced `0/64` successes when evaluated at
	    `model_50.pt`; depth improved only slightly to `mean_depth=-0.001222`,
	    with no lateral or orientation misses under the coarse gate.
	  - Forced raw-action diagnostic confirmed raw `tz-` still increases SFP
	    depth, but only by about `+0.001539 m` over `150` steps from the
	    near-port reset, ending at about `-0.001076 m`.
	  - Added progress-gated final-depth curriculum `a79737c`: gate off
	    entrance-distance rewards near the entrance, stop rewarding negative
	    insertion depth, multiply raw depth-action reward by measured positive
	    depth progress, raise SFP IK scale to `0.002`, and initialize raw
	    `z=-0.20`.
	  - Progress-gated PPO briefly recovered a nonzero training success signal,
	    but collapsed to timeout-only behavior by iteration `64`. Evaluation of
	    `model_50.pt` produced `1/64` coarse successes; failures still had no
	    lateral/orientation misses but remained depth shortfalls
	    (`mean_depth=-0.001483`).
	  - Next remediation: gate `sfp_insertion_action` by measured positive depth
	    progress as well, because it still pays large inward-intent reward while
	    actual insertion depth remains negative.
	  - Added insertion-action progress gate `f5d3eed`: `sfp_insertion_action`
	    now also requires measured positive signed-depth progress, true-depth
	    rewards were strengthened, sparse coarse-success bonus was increased to
	    `100.0`, SFP IK scale moved to `0.003`, and the initial raw depth bias is
	    `z=-0.25`.
	  - Insertion-action-progress PPO had intermittent early training successes
	    around iterations `51-56`, then collapsed to timeout-only behavior by
	    iteration `100`.
	  - Strong forced-action diagnostic with raw action magnitude `1.0` confirmed
	    raw `tz-` can cross the depth threshold (`d_depth_mean=+0.036686`), but it
	    also drifts laterally outside the coarse corridor (`after_lateral=0.031734`
	    vs gate `0.020`).
	  - Added coupled initial actor bias `6c3fbf2` with raw action
	    `(x=0.13, y=0.10, z=-1.0)` to compensate the measured lateral drift while
	    pushing inward.
	  - Coupled-bias PPO briefly showed early training success, then collapsed to
	    timeout-only behavior by iteration `72`. Evaluation of `model_50.pt`
	    produced `0/64` successes, `mean_lateral=0.013022`,
	    `mean_orientation=0.342141`, and `mean_depth=-0.003365`.
	  - Added custom combined-action and sequence diagnostics for
	    `check_sfp_action_frame.py`. The direct combined compensation
	    `(0.13, 0.10, -1.0)` did not work (`0/64`), but a two-phase sequence did:
	    `(0.5, 0.5, 0.0)@30; (0.0, 0.0, -1.0)@120` reached `46/64` final
	    successes and `46/64` ever-successes.
	  - Used the successful lateral pre-correction diagnostic to derive
	    per-target SFP reset joint presets. From those pre-corrected resets, pure
	    raw `z-` succeeds in `64/64` deterministic action-probe episodes with
	    mean first success step `8.14`.
	  - Current SFP status: the deterministic final-insertion curriculum is now
	    controllable. PPO run `step8_sfp_ppo_precorr_54b5879` reached `1.0000`
	    `Episode_Termination/sfp_insertion_success` by iteration `20`.
	  - Detached evaluation of
	    `2026-05-11_15-49-50_step8_sfp_ppo_precorr_54b5879/model_50.pt` reached
	    `64/64` successes under the temporary coarse SFP gate
	    (`lateral <0.020`, `orientation <0.50`, `depth >0.005`).
	  - Next blocker: this solves only the fixed/pre-corrected final-stage reset
	    under a coarse lateral gate. The evaluated mean lateral error is
	    `0.017177`, close to the `0.020` threshold and far outside the older
	    strict SFP scripted gate of `0.004`. Improve lateral centering, then
	    reintroduce reset/NIC randomization gradually before starting Step 9.
	  - A strict 16-env action-sequence grid found no successes under
	    `lateral <0.004`, `orientation <0.20`, `depth >0.015`. Negative lateral
	    corrections can reach mean lateral `0.003805`, but depth remains negative;
	    positive corrections preserve depth but leave lateral around `0.0145` to
	    `0.0164`. The next PPO stage is therefore an intermediate gate:
	    `lateral <0.015`, `orientation <0.25`, `depth >0.015`.
	  - The first intermediate-gate resume briefly recovered low success
	    (`0.0156-0.0312`) but was stopped because
	    `sfp_port_frame_depth_action` still rewarded inward action outside the new
	    lateral/orientation gate and targeted only `0.012 m` depth. Tighten that
	    shaping term to the intermediate gate and raise its target depth before
	    retrying.
	  - The gated-depth retry was evaluated at `model_100.pt` and reached only
	    `2/64` successes under the intermediate gate. It ended close on depth
	    (`mean_depth=0.019861`) but still missed lateral/orientation
	    (`mean_lateral=0.016383`, `mean_orientation=0.301070`). The reward logs
	    showed `sfp_port_frame_lateral_action` saturating while measured lateral
	    error stayed clipped, so the next patch makes the lateral penalty
	    informative across the corridor, requires realized lateral improvement
	    for the lateral-action reward, and loosens depth-shaping gates to
	    lateral `<0.030` while keeping terminal success at `<0.015`.
	  - The realized-lateral-action retry also evaluated at `2/64` under the
	    intermediate gate (`mean_lateral=0.016442`,
	    `mean_orientation=0.296399`, `mean_depth=0.021180`).
	  - Action-sequence search showed the intermediate gate is reachable:
	    pure `z-` reached only `2/32`, while
	    `0.5,0.75,0@10;0,0,-1@130` reached `23/32` final successes and `28/32`
	    ever-successes. The next curriculum reset uses the joint state after the
	    `0.5,0.75,0@10` pre-correction so PPO does not have to discover that
	    first phase from scratch.
	  - A later target-specific reset pass replaced both SFP reset presets with
	    joint means after `-0.25,-0.25,0@4` from the current reset. From this
	    final deterministic reset, pure `z-` reached `122/128` first-hit
	    successes under the intermediate gate.
	  - SFP PPO was changed to full 150-step rollouts because the useful
	    insertion signal usually appears around step `83`; the previous 24-step
	    rollout updated PPO before it saw success.
	  - Full-rollout PPO run
	    `2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b/model_19.pt`
	    evaluated at `118/128` successes (`92.1875%`) under the intermediate
	    gate. Per-target rates were `67/74` for `sfp_port_0` and `51/54` for
	    `sfp_port_1`.
	  - First reset-noise reintroduction is stable at
	    `reset_robot_near_sfp_port.position_noise=0.002`: the same `model_19.pt`
	    evaluated at `121/128` successes, with both SFP ports above `94%`.
	  - `position_noise=0.005` is not accepted as the default yet. PPO resume
	    reached `118/128` overall, but `sfp_port_0` stayed below `90%`
	    (`61/69`). The checked-in curriculum was backed off to `0.002`.

Later distilled outputs:

- `sc_student.pt`
- `sfp_student.pt`

Done when:

- [ ] SC checkpoint solves SC validation.
- [x] SFP checkpoint solves SFP validation.
  - Current accepted deterministic SFP checkpoint:
    `/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b/model_19.pt`
  - Detached eval: `118/128` successes (`92.1875%`) at
    `lateral <0.015`, `orientation <0.25`, `depth >0.015`.
  - This is still fixed-NIC/final-stage curriculum validation; randomized SFP
    insertion remains future work before final qualification confidence.
- [ ] Each checkpoint has recorded config, commit hash, and success metrics.

## Step 9: Distillation, High Level Only

Do not start this until reliable specialist teachers/policies exist.

High-level work later:

- [ ] Create RSL-RL distillation config for SC.
- [ ] Create RSL-RL distillation config for SFP.
- [ ] Load trained teacher checkpoints, whether PPO or BC/DAgger.
- [ ] Train students using eval-compatible observations only.
- [ ] Export student policies.
  - If the reliable specialist policy already uses only eval-compatible actor
    observations, direct export may be valid and distillation may be unnecessary.
- [x] Add official Gazebo eval wrapper scaffold for checkpoint-backed policies.
  - Added `aic_utils/aic_training_utils/scripts/run_gazebo_checkpoint_eval.py`
    to start official `aic_eval`, `aic_model`, and optional camera-topic rosbag
    recording in separate tmux sessions.
  - Added `aic_model/aic_model/RslRlCheckpointPolicy.py` as the ROS policy
    scaffold that receives checkpoint/artifact paths and logs official task and
    observation metadata.
  - Documented usage and limitations in `docs/bahw_docs/eval_wrapper/README.md`.
  - Fixed the wrapper to launch branch-local policy modules inside the pixi
    shell, avoiding stale installed `aic_model` packages.
  - Ran the scaffold wrapper with `ground_truth:=false` against the selected
    randomized SFP checkpoint. Tier-1 model validation passed for all three
    official trials; tier-2/tier-3 scores remained zero as expected because the
    policy scaffold still returns failure.
- [ ] Implement the Gazebo `Observation` + `Task` to Isaac actor-observation
  adapter for exported SC/SFP actor artifacts.
- [ ] Convert actor actions into safe Gazebo `MotionUpdate` or
  `JointMotionUpdate` commands.
- [ ] Run official Gazebo eval with `ground_truth:=false` and preserve
  `scoring.yaml`, scoring bags, and optional camera rosbags.
  - Scaffold smoke run completed under
    `logs/gazebo_eval/20260513_193949/`; keep this unchecked until the adapter
    drives the robot and produces functional scores.
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
  - [x] Added direct scripted actor bootstrap before more PPO; expert-rollout
    BC reached `193/256`, while PPO resume and actor-rollout BC regressed.
  - [x] Added diagnostics for the current BC failure modes before another
    training variant.
  - [x] Improved BC with strict scripted alignment labels; current best SC
    checkpoint reached `233/256`.
  - [x] User accepted `>90%` SC plus saved video as sufficient to unblock SFP.
    Current best is a BC-trained neural actor checkpoint at `233/256`; PPO
    remains the preferred final path.
- [x] 12. Extend the same geometry/reward interface to SFP.
- [ ] 13. Train SFP teacher/policy.
  - [x] Run SFP scripted-control diagnostic before long training.
  - [x] Verify SFP port helper geometry against USD semantic entrance frames.
  - [x] Add first SFP near-port reset curriculum.
  - [x] Run first fixed-NIC SFP PPO attempt.
  - [x] Add depth reward shaping after fixed-NIC PPO learned alignment without
    insertion.
  - [x] Add inward-action PPO reward after depth-only shaping still had zero
    success by iteration `155`.
  - [x] Add temporary coarse SFP success gate after stronger inward-action PPO
    still had zero strict success.
  - [x] Add SFP-aware evaluation diagnostics and confirm coarse-gate
    `model_200.pt` is `0/128`.
  - [x] Add temporary SFP corridor-violation termination after evaluation showed
    timeout failures far outside the near-port corridor.
  - [x] Reduce SFP relative-IK action scale after short-horizon diagnostics
    showed the policy left the near-port corridor within 10 steps.
  - [x] Continue reduced-scale SFP PPO from fixed-NIC `model_50.pt` until it
    stalled with low intermittent coarse success and near-total corridor exits.
  - [x] Add PPO lateral-guard reward/config remediation.
  - [x] Run lateral-guard SFP PPO from fixed-NIC `model_50.pt` until it stalled
    at zero success with immediate corridor exits.
  - [x] Add PPO lateral-correction action reward.
  - [x] Run lateral-correction SFP PPO from fixed-NIC `model_50.pt`.
  - [x] Evaluate lateral-correction `model_200.pt` and confirm it still backs
    away from the port.
  - [x] Add PPO port-approach action reward.
	  - [x] Run port-approach SFP PPO from fixed-NIC `model_50.pt`; stopped because
	    it stayed at zero success with immediate corridor exits.
	  - [x] Disable port-approach term from active SFP reward config.
	  - [x] Run fresh-start lateral-correction SFP PPO.
	  - [x] Reduce SFP PPO exploration noise after fresh-start PPO still left the
	    corridor immediately.
	  - [x] Reduce SFP relative-IK action scale to `0.005`.
	  - [x] Split SFP corridor terminations by failure reason and confirm lateral
	    violation is the dominant early failure.
	  - [x] Add SFP lateral-error penalty.
	  - [x] Monitor SFP PPO with the initial lateral-error penalty; stopped because
	    lateral exits remained near `1.0000`.
	  - [x] Strengthen SFP lateral-control reward weights.
	  - [x] Monitor SFP PPO with strong lateral-control rewards; stopped because
	    lateral exits remained `1.0000` through iteration `44`.
	  - [x] Reduce SFP action scale to `0.001` and PPO `init_std` to `0.02`.
	  - [x] Monitor SFP PPO with 0.001 action scale; stopped because learned std
	    grew and lateral exits returned to `1.0000`.
	  - [x] Stabilize SFP low-noise PPO with `init_std=0.005`, zero entropy
	    coefficient, and lower learning rate.
	  - [x] Monitor stabilized low-noise SFP PPO; stopped after it became a
	    timeout/backout local optimum.
	  - [x] Add signed SFP depth-action shaping.
	  - [x] Add training-only SFP PPO actor output bias and zero-head
	    initialization.
	  - [x] Run zero-head SFP PPO; stopped after `model_50.pt` plateaued at
	    timeout-only behavior.
	  - [x] Evaluate zero-head `model_50.pt`; confirmed the remaining blocker is
	    insertion depth, not lateral or orientation alignment.
	  - [x] Add final-depth reward curriculum after zero-head evaluation showed a
	    `6-7 mm` depth shortfall.
	  - [x] Run final-depth SFP PPO and evaluate `model_50.pt`; still `0/64`
	    coarse successes.
	  - [x] Run forced SFP raw-action diagnostic for final-depth motion.
	  - [x] Add progress-gated final-depth reward curriculum.
	  - [x] Run progress-gated SFP PPO and evaluate `model_50.pt`; result was
	    `1/64` coarse successes.
	  - [x] Gate insertion-action reward by measured positive depth progress.
	  - [x] Run insertion-action-progress SFP PPO.
	  - [x] Run strong forced-action diagnostic for final SFP insertion motion.
	  - [x] Add and evaluate coupled initial insertion push bias.
	  - [x] Add custom combined-action SFP diagnostic.
	  - [x] Add action-sequence SFP diagnostic and derive pre-corrected reset
	    presets.
	  - [x] Validate pre-corrected reset with pure raw `z-` action probe
	    (`64/64` deterministic successes).
	  - [x] Train and evaluate SFP PPO specialist checkpoint from the
	    pre-corrected reset (`64/64` coarse deterministic successes).
	  - [x] Tighten/improve SFP lateral centering beyond the temporary coarse
	    `0.020` lateral gate.
	  - [x] Train/evaluate first intermediate-gate SFP PPO
	    (`2/64`; not sufficient).
	  - [x] Retry intermediate-gate SFP PPO with realized-lateral-action shaping
	    and widened lateral-error scale (`2/64`; not sufficient).
	  - [x] Run positive-precorrection action grid and derive new deterministic
	    SFP reset presets.
	  - [x] Validate pure `z-` insertion from the positive/final pre-corrected
	    reset (`122/128` first-hit successes under the intermediate gate).
	  - [x] Retry intermediate-gate PPO from the pre-corrected reset
	    (`118/128` detached eval successes with full 150-step PPO rollouts).
	  - [x] Reintroduce small SFP reset noise while preserving success
	    (`0.002` joint noise: `121/128`, both ports above `94%`).
	  - [ ] Reintroduce SFP reset/NIC randomization after the deterministic
	    final-stage checkpoint is reliable under the tighter gate.
	    - Current decision: train randomized SFP with two PPO tracks. Track A
	      warm-starts from the best fixed-NIC SFP checkpoint as a weight
	      initialization only; Track B starts PPO from scratch under the same
	      randomized setup as a control.
	    - Randomize NIC/port pose independently enough that joint state alone
	      cannot identify the target correction, so the actor must use camera
	      features plus proprioception.
	    - First accepted curriculum stage uses continuous NIC/card `y`
	      randomization in `[-0.002, 0.002]` meters and keeps the existing
	      ResNet18 camera features. `snap_step.y` must stay `0.0` for this
	      stage, otherwise the old `0.04` meter grid silently disables the small
	      randomization range.
	    - Do not simply enable NIC y randomization against the old fixed reset
	      without changing the reset curriculum; the current SFP reset uses
	      fixed joint presets and does not adapt to randomized NIC pose.
	    - Prefer the better randomized policy. If the scratch control does
	      better than the warm-started checkpoint, use the scratch run as the
	      SFP candidate.
	    - Status on 2026-05-12 14:50 +08: both randomized SFP PPO tracks were
	      still training in remote tmux. Warm-start had reached `model_290.pt`
	      with recent rollout success around `0.91-0.93`; scratch had reached
	      `model_270.pt` with recent rollout success around `0.91-0.95`. These
	      are training health signals only; the item stays open until
	      deterministic randomized SFP eval passes overall and per-target gates.
- [ ] 14. Revisit distillation only after both teachers work.

## Global Done Criteria

The SC implementation is done when:

- [ ] `AIC-Task-v0` no longer trains on random `ee_pose` reaching.
- [ ] Actor observations are eval-compatible.
- [ ] Critic observations include privileged insertion geometry.
- [ ] SC plug-to-port geometry is correct for both SC ports.
- [ ] SC teacher/policy inserts in randomized simulation.

The SFP implementation is done when:

- [x] The same MDP structure supports SFP.
- [x] SFP plug and port entrance geometry are correct.
- [ ] SFP teacher/policy inserts in randomized simulation.
  - Current best accepted checkpoint solves fixed-NIC final-stage validation and
    small reset noise (`0.002`), but not NIC/card randomization.

The training path is ready for distillation when:

- [ ] Both specialist teachers solve their respective Isaac tasks.
- [ ] Metrics and videos confirm actual insertion.
- [ ] There is no dependence on privileged geometry in actor observations.
