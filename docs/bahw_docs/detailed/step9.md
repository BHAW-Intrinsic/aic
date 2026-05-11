# Step 9: Distillation And Routing

Status: blocked.

Step 9 should not start until the specialist policies are reliable enough to be
worth distilling or exporting. The actor observation groups are already
eval-compatible for both SC and SFP, so direct export may be sufficient later
without a separate distillation pass.

Current specialist status:

- SC has an accepted neural checkpoint at `233/256` deterministic Isaac
  successes plus a saved video artifact.
- SFP has an accepted PPO checkpoint for fixed-NIC final-stage insertion with
  small reset noise:
  `/workspace/isaaclab/logs/rsl_rl/aic_sfp_insert/2026-05-11_18-14-11_step8_sfp_ppo_fullrollout_303652b/model_19.pt`
- SFP fixed-NIC detached eval:
  `118/128` successes under `lateral <0.015`, `orientation <0.25`,
  `depth >0.015`.
- SFP with `position_noise=0.002` detached eval:
  `121/128` successes, with both SFP ports above `94%`.

Current blocker:

- SFP has not yet been validated with NIC/card y randomization.
- The current SFP near-port reset uses fixed per-target joint presets:
  `reset_robot_near_sfp_port` selects from `SFP_NEAR_PORT_JOINT_PRESETS` and
  adds optional joint noise.
- Because the reset does not adapt to randomized NIC pose, enabling NIC y
  randomization directly would move the target away from the fixed reset
  curriculum.

Next required work before Step 9:

1. Derive per-NIC-offset SFP reset presets, or implement an adaptive reset helper
   that can place the robot near the active randomized SFP port.
2. Reintroduce NIC/card y randomization gradually.
3. Resume/evaluate SFP PPO until the randomized SFP specialist is reliable.
4. Then decide between direct export and distillation.
5. Add final Gazebo wrapper routing using official `Task.msg` metadata:
   `plug_type` / `port_type` select SC vs SFP checkpoint.
