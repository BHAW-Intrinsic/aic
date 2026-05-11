
# BAHW Isaac Lab Work Notes

This directory tracks the Isaac Lab training path for the AIC cable insertion
work.

- Current plan and checklist: `plan.md`
- High-level strategy: `overview.md`
- Setup notes: `setup.md`
- Reproducible step logs: `detailed/stepX.md`

## Current Status

The main path is PPO with an asymmetric actor-critic in Isaac Lab. Actor
observations are kept eval-compatible; critic observations can use privileged
plug-to-port geometry.

Completed work so far:

- Step 0 verified the relevant Isaac/Gazebo asset frames. Isaac exposes
  `robot.sfp_tip_link`; SFP port entrances exist as USD semantic prims and are
  reproduced by fixed helper poses. SC needed a virtual gripped-tip helper
  because the named `sc_tip_link` is not the controlled gripped insertion tip.
- Steps 1-5 added shared insertion geometry helpers, active target selection,
  policy observations, privileged observations, insertion rewards, and success
  terminations.
- Step 6 trained and evaluated the SC path. The accepted SC gate is the best
  saved neural checkpoint at `233/256` successes, plus a saved video artifact.
  This was accepted as enough to unblock SFP work.
- Step 7 extended the same MDP structure to SFP with `AIC-SFP-Task-v0`, active
  SFP port metadata, SFP helper geometry, SFP observations/rewards/terminations,
  and SFP PPO config.
- Step 8 is in progress. The SFP scripted controller was tested and rejected as
  a BC expert because it systematically misses the port by a few millimeters.
  Multiple PPO curriculum and reward variants have been tested. The current
  SFP direction is a pre-corrected final-insertion reset derived from action
  sequence diagnostics, followed by gradual reintroduction of randomization.

Current Step 8 best result:

- Best detached SFP PPO eval before reset pre-correction is
  `step8_sfp_ppo_purez_noneg_c2a6b11/model_100.pt`: `5/64` coarse successes.
- The action-sequence diagnostic showed the old SFP reset was laterally biased:
  `(0.5, 0.5, 0.0)@30; (0.0, 0.0, -1.0)@120` reached `46/64` deterministic
  successes.
- The current pre-corrected SFP reset plus pure raw `z-` action probe reaches
  `64/64` deterministic successes.
- PPO checkpoint
  `step8_sfp_ppo_precorr_54b5879/model_50.pt` also reaches `64/64` detached
  successes under the temporary coarse SFP gate.
- The first intermediate-gate retry,
  `step8_sfp_ppo_gateddepth_777d095/model_100.pt`, reached only `2/64`
  detached successes at `lateral <0.015`, `orientation <0.25`,
  `depth >0.015`.

Current blocker:

- The deterministic final-stage SFP reset is now controllable under the coarse
  SFP gate, but lateral centering is not good enough yet: the detached eval mean
  lateral error is `0.017177`, close to the temporary `0.020` threshold.
- A strict action-sequence grid found no `0.004` lateral-threshold successes, so
  the next training stage uses an intermediate gate of `lateral <0.015`,
  `orientation <0.25`, `depth >0.015`.
- The first intermediate-gate retry exposed a reward-shaping conflict: the raw
  lateral-action reward could saturate while measured lateral error stayed poor.
  The next retry uses a wider lateral-error scale, realized lateral improvement
  gating, and looser depth-shaping gates before the terminal success gate.
- After lateral centering improves, reset noise and NIC randomization must be
  reintroduced without losing success.
- Step 9 remains blocked until both SC and SFP have reliable specialist
  checkpoints.

Next recommended work:

- Improve the SFP final-stage lateral centering from the pre-corrected reset.
- Train and evaluate the revised intermediate-gate SFP PPO stage.
- Then gradually reintroduce SFP reset/NIC randomization.
- After SFP has a reliable specialist checkpoint, revisit Step 9 distillation
  and final Gazebo routing.

## Key Code References

- Isaac task config:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- Geometry helpers:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/geometry.py`
- Observations:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py`
- Rewards:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- Terminations:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/terminations.py`
- SC PPO config:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py`
- SFP PPO config:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_sfp_cfg.py`
- Evaluation script:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/evaluate.py`
- SFP action-frame diagnostic:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/check_sfp_action_frame.py`
- Geometry inspection:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_aic_geometry.py`

## Detailed Notes

- Step 0 geometry inspection: `detailed/step0.md`
- Step 1 geometry helper work: `detailed/step1.md`
- Step 2 active target selection: `detailed/step2.md`
- Step 3 observations/asymmetric critic: `detailed/step3.md`
- Step 4 insertion rewards: `detailed/step4.md`
- Step 5 success termination: `detailed/step5.md`
- Step 6 SC training and accepted checkpoint: `detailed/step6.md`
- Step 7 SFP task extension: `detailed/step7.md`
- Step 8 SFP specialist PPO work: `detailed/step8.md`

## Per-Step Workflow

Use this workflow for each implementation step unless the step-specific notes say
otherwise.

1. Check local state.

   ```bash
   git status --short
   sed -n '<step-range>p' docs/bahw_docs/plan.md
   ```

2. Read the relevant code and the previous detailed step note before editing.
   Prefer `rg` / `rg --files` for search.

3. Implement the smallest code change that satisfies the current step. Keep
   docs updated while working:
   - update the current `docs/bahw_docs/detailed/stepX.md`
   - update `docs/bahw_docs/plan.md` checkboxes when the result is verified
   - keep `AGENTS.md` and `CLAUDE.md` identical if project guidance changes

4. Run local checks that do not require Isaac Sim.

   ```bash
   python3 -m py_compile <changed-python-files>
   git diff --check
   ```

5. Commit and push the local implementation.

   ```bash
   git status --short
   git add <changed-files>
   git commit -m "<short step message>"
   git push
   ```

6. Pull the pushed commit on the remote host repo copy.

   Run on the host:

   ```bash
   cd ~/IsaacLab/aic
   git pull --ff-only
   ```

7. Sync changed source files into the already-running Isaac Lab container when
   needed. The container often cannot pull directly because its git remote uses
   SSH, so copy files from the host repo into `/workspace/isaaclab/aic`.

   Run on the host:

   ```bash
   docker cp ~/IsaacLab/aic/<relative-path> \
     isaac-lab-base:/workspace/isaaclab/aic/<relative-path>
   ```

8. Run remote checks inside named `tmux` sessions. Use descriptive names such as
   `isaac-step3-rewards`, `isaac-step4-termination`, or
   `isaac-step5-smoke-train`.

   Run on the host:

   ```bash
   tmux new-session -d -s <session-name> \
     "docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && <isaac-command>'; echo EXIT:\$?"
   tmux capture-pane -t <session-name> -p -S -200
   ```

9. Copy generated logs from the container back to the host repo logs directory.
   Logs remain untracked.

   Run on the host:

   ```bash
   mkdir -p ~/IsaacLab/aic/logs
   docker cp isaac-lab-base:/workspace/isaaclab/aic/logs/<log-dir> \
     ~/IsaacLab/aic/logs/
   ```

10. Paste the important replication details into
    `docs/bahw_docs/detailed/stepX.md`:
    - commit hash
    - host pull result
    - container copy commands
    - Isaac command
    - relevant output excerpts
    - log path

11. Before starting the next step, check whether the completed result changes
    any later steps in `docs/bahw_docs/plan.md`. Document the decision in the
    completed step note and update the future checklist if needed.

12. Use subagents when needed for bounded research or codebase exploration that
    will clarify the next implementation decision. Keep delegated tasks
    read-only unless assigning a clearly separate code-change scope.

13. Commit and push the documentation updates, then pull them on the remote host
    so local and host copies stay aligned.
