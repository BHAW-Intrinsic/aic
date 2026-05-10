
# BAHW Isaac Lab Work Notes

Overall plan: `plan.md`

Details for reproducing each step: `detailed/stepX.md` for each step X.

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

12. Use subagents for bounded research or codebase exploration when it will
    clarify the next implementation decision. Keep delegated tasks read-only
    unless assigning a clearly separate code-change scope.

13. Commit and push the documentation updates, then pull them on the remote host
    so local and host copies stay aligned.
