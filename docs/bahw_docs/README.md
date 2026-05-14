
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
- Step 8 produced the fixed-NIC SFP PPO checkpoint that unblocked randomized SFP
  work.
- Step 9 trained two randomized SFP PPO tracks. The scratch run beat the warm
  start and is the selected randomized SFP candidate:
  `/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-12_01-40-05_step9_sfp_randy002_scratch_f2cd192/model_1499.pt`.
- Deterministic randomized Isaac eval for that checkpoint reached `238/256`
  overall (`92.97%`), with `123/132` on `sfp_port_0` (`93.18%`) and `115/124`
  on `sfp_port_1` (`92.74%`).
- The official Gazebo eval wrapper scaffold now launches `aic_eval`, loads the
  branch-local `RslRlCheckpointPolicy`, receives official task/camera
  observations, and writes `scoring.yaml` plus trial bags.
- The SFP Gazebo adapter reconstructs the 3149D Isaac actor observation from
  official `Task`/`Observation` fields and reaches partial tier-2/tier-3 scores
  in official Gazebo, but it still misses insertion.
- The current best legal official Gazebo run is
  `~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/scoring.yaml`, total
  `92.631565804455263`, with videos under
  `~/ws_aic/src/aic/logs/gazebo_eval/20260514_100007/videos/`.
- Later controlled runs rejected `AIC_RSLRL_CONTROL_HZ=30`, fixed-step replay,
  TCP/base-frame final-settle pushes, and SC prepose-only handoff as defaults.
  SFP `gripper/tcp` command-frame replay and zeroed joint observations were also
  rejected. None produced insertion or beat the `20260514_100007` total score.

Current blocker:

- The SFP deployment mapping is close but not complete. The best official run so
  far ended the two SFP trials at `0.05m` and `0.04m` plug-port distance without
  triggering insertion.
- SC routing and observation reconstruction now run with the exported SC actor
  in official Gazebo, but the legal near-port joint prepose only improves the SC
  final distance to about `0.29m`.
- The highest-risk deployment assumptions are observation equivalence
  (ResNet18 image features, TCP pose frame, wrench/body-force padding) and the
  relative-IK action replay frame.
- Step 10 is a transfer-audit pass. It adds opt-in policy traces so the next
  change is based on legal Gazebo observation/action evidence rather than
  another blind training run.

Next recommended work:

- For SFP, continue the controller/action-frame mapping work from the current
  `0.04-0.05m` miss. The tested TCP-frame final-settle hook worsened official
  scoring, the tested base-frame insert worsened total score, and fixed-step
  replay overshot the useful approach region; keep these disabled by default.
- For SC, improve the official-start approach path before actor handoff. The
  current legal prepose is not close enough even without actor handoff.
- Compare one Isaac actor observation and one reconstructed Gazebo actor
  observation around a near-port pose to find semantic mismatches.

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
- Step 9 distillation/export blocker: `detailed/step9.md`
- Step 10 Gazebo transfer audit: `detailed/step10.md`

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
