# Qualification Plan

## Task

3 trials evaluated in Gazebo — no ground truth transforms available at eval:
- **Trials 1 & 2:** Insert `SFP_MODULE` into `SFP_PORT` on a randomised NIC card
- **Trial 3:** Insert `SC_PLUG` into `SC_PORT` (tests generalisation across plug types)

Robot starts a few cm from the target with the plug already in hand. Eval inputs: 3 wrist cameras + joint states + F/T wrench only.

---

## Approach: Oracle-Seeded SAC + Student Distillation

### Step 1 — Oracle Policy (Isaac Lab)

A CheatCode-equivalent implemented natively in Isaac Lab using direct scene state access (`env.scene["sc_port"].data.root_pos_w` etc.) — no ROS, no TF tree. Runs vectorised across 64 parallel envs simultaneously on the GPU, each with independent domain randomisation, generating perfect insertion demos automatically.

Used to pre-fill the SAC replay buffer before training starts.

**Domain randomisation (already in Isaac env):**
- Board pose, NIC card rail/position, SC port position
- Light intensity and colour

---

### Step 2 — SAC Training (Teacher)

Pre-fill SAC replay buffer with oracle transitions → start SAC training. Because the buffer already contains successful insertions, SAC learns from good examples immediately rather than exploring randomly.

**Observations (privileged — Isaac only):**
- Joint positions + velocities
- EE pose
- F/T wrench (wrist force-torque sensor — 6 values: Fx, Fy, Fz, Tx, Ty, Tz)
- `plug_to_port_vec` — ground truth 3D vector from plug tip to port, read directly from Isaac scene state

**Reward:**
- **Dense:** exponential kernel `exp(-d²/σ²)` — gives reward proportional to proximity at all distances, with sharpest gradient near the port where precision matters most
- **Sparse:** large bonus when plug is within ~5mm of port — reinforces actual insertion rather than just hovering close
- **Penalties:** action rate, joint torques, joint acceleration — encourage smooth, low-force motion (matters for Tier 2 scoring)

Both plug types (SFP + SC) trained jointly from the start, conditioned on a binary task ID flag in the observation.

---

### Step 3 — Student Distillation

The teacher policy relies on `plug_to_port_vec` which is unavailable at eval. The student replaces it with camera features, trained to match teacher actions using RSL-RL's `DistillationRunner` (already in the repo).

**Observations (eval-compatible — matches Gazebo exactly):**
- Joint positions + velocities
- EE pose
- F/T wrench
- Frozen ResNet18 features from left, centre, and right wrist cameras

The student implicitly learns to extract plug-to-port geometry from pixels — no separate visual estimator needed. F/T wrench is kept because it is the primary signal during the final contact phase of insertion, where camera pixels change very little.

---

### Step 4 — Gazebo Deployment

Wrap the student in a `Policy` subclass. `insert_cable()`:
1. Reads `get_observation()` → maps to student obs tensor
2. Runs student forward pass
3. Maps output delta pose → `MotionUpdate` → `move_robot()`

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Isaac has no cable dynamics; Gazebo does | Add random wrench noise to F/T obs during training to simulate cable drag |
| Sparse insertion reward slows early RL | Oracle replay seeding ensures SAC sees successful insertions from step 1 |
| Trial 3 SC generalisation | Both plug types trained jointly with task ID flag from the start |
| Timeline (eval May 18) | Fallback always ready in parallel |

---

## Fallback

If SAC/distillation does not converge in time: collect CheatCode demos in Gazebo using `lerobot-record` → train ACT policy via LeRobot → submit. Tooling already exists in the repo. Likely sufficient to qualify.
