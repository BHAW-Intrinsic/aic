# Isaac Lab Training Plan

## Summary

We want to train policies in Isaac Lab that can insert both connector
types used in qualification:

- SFP module into SFP port
- SC plug into SC port

The important first step is not choosing PPO versus SAC. The important first
step is making sure the training environment rewards the robot for the real
task: inserting the held connector into the actual target port.

Right now, the Isaac Lab environment named `AIC-Task-v0` contains useful scene
assets, but the learning objective is still too generic. It rewards the robot for
moving its end effector toward a randomly generated pose, not for inserting a
plug into the actual port in the scene.

If we train on that as-is, the robot can learn to reach a floating target while
ignoring the port. 

## What `AIC-Task-v0` Is

`AIC-Task-v0` is the Isaac Lab training environment. It defines the simulated
world used for RL training:

- robot
- cameras
- task board
- ports and connector assets
- actions the policy can output
- observations the policy receives
- rewards used for learning
- reset randomization
- success and timeout conditions

The RL library, such as RSL-RL or SKRL, is only the trainer. The environment
defines the problem. If the environment rewards the wrong behavior, any trainer
will learn the wrong behavior.

## Current Problem

The current environment has a command called `ee_pose`. This command samples a
random target pose for the end effector.

The current rewards then tell the robot:

> Move the wrist/end effector toward this sampled target pose.

That is a reaching task.

The task we actually need is:

> Move the held connector so its tip aligns with the target port, enters the
> port, and reaches the required insertion depth.

Those are not the same task.

The port position is randomized on reset, but the current random `ee_pose` target
is not derived from that port. This means the port can be present in the scene
without being what the policy is rewarded to interact with.

## What We Need To Fix

We need to change the environment so the reward and success condition are based
on actual connector geometry.

For every connector type, the environment should know:

- where the plug tip is
- where the port entrance is
- how far the plug tip is from the port entrance
- whether the plug is laterally aligned with the port opening
- whether the plug orientation matches the port orientation
- how deep the plug has entered along the port axis
- whether insertion should count as successful

In other words, training should be driven by plug-to-port geometry, not by an
unrelated random target pose.

## Main Training Path

Use PPO with asymmetric actor-critic as the main training path.

In plain terms:

- The actor is the policy that will eventually run at evaluation time.
- The critic is a training-only helper that estimates how good the actor's
  actions are.
- "Asymmetric" means the critic is allowed to see privileged simulator state
  during training, while the actor only sees information that should be available
  at evaluation time.

Recommended observation split:

- actor sees eval-compatible observations: cameras, robot state, wrench/force
  information, end-effector state, and last action
- critic sees actor observations plus privileged geometry: plug-to-port vector,
  port pose, insertion depth, and orientation error

After a privileged PPO teacher can solve insertion in simulation, distill it into
an eval-compatible student policy.

In plain terms, distillation means:

> Train a deployable student policy to copy the trained teacher's actions, but
> only using observations that will be available during evaluation.

## Why Not SAC 

The local repo is already wired for PPO with the RSL-RL library, and RSL-RL also supports
distillation. SAC would require adding another training backend, such as SKRL or
RL-Games.

More importantly, SAC would still fail if the environment rewards the wrong task.
So the priority is to fix the environment first, then measure a PPO teacher
baseline. SAC can be revisited later as an experiment.

## Connector Strategy

We should train for both connector types, but stage the work.

### Stage 1: Fix The SC Task First

Start with SC because the current Isaac Lab scene already contains `sc_port` and
`sc_port_2`.

This is the fastest validation target. The goal is to prove that the corrected
environment can train a real insertion behavior at all.

Success criteria:

- the policy inserts the SC plug into the actual SC port
- success is measured by plug-tip and port-entry geometry
- the policy does not merely hover near the port
- training works across randomized board and port positions

### Stage 2: Extend The Same Design To SFP

Qualification trials 1 and 2 use SFP.

The SFP task should reuse the same structure:

- SFP plug/module tip frame
- SFP port entrance frame
- lateral alignment reward
- orientation alignment reward
- insertion depth reward
- insertion success condition
- privileged critic observations

SC is only the first validation step. We should not assume that an SC-only policy
will transfer to SFP.

### Stage 3: Train PPO Teachers

Train privileged-state PPO teachers after the task geometry is fixed.

Main recommendation: train two specialist teachers first.

- one SC teacher
- one SFP teacher

SC and SFP have different geometry, visual appearance, contact behavior, and
insertion depth. Separate teachers are easier to debug and let us optimize SFP
without accidentally hurting SC performance, or vice versa.

### Stage 4: Distill To Eval-Compatible Policy

Only start distillation after the PPO teacher can solve insertion with privileged
training information.

The student should not receive privileged geometry such as exact
`plug_to_port_vec` at deployment. It should rely on eval-compatible inputs such
as cameras, robot state, and force/wrench information.

The final submitted `aic_model` still has to handle all qualification trials, but
it can contain multiple internal checkpoints. During Gazebo evaluation, the
official `Task` message is passed into `Policy.insert_cable()`. That message
includes connector and target metadata such as `plug_type`, `port_type`,
`plug_name`, `port_name`, and `target_module_name`.

Therefore, the runtime wrapper can legally choose the SC policy or SFP policy
based on the provided `Task` metadata. This is different from using privileged
geometry: task metadata tells us what connector and target port the task is
about, but it does not reveal the exact 3D port pose or plug-to-port vector.

## Immediate Next Milestone

Make the SC version of `AIC-Task-v0` train the actual insertion task.

Concrete work:

- identify or create the SC plug-tip frame
- identify or create the SC port-entrance frame
- replace random command-pose rewards with plug-to-port rewards
- add insertion-depth reward
- add true insertion success termination
- add a privileged observation group for training
- configure PPO so the actor sees eval-compatible observations and the critic
  also sees privileged geometry

After this, train an SC PPO teacher and check whether it inserts reliably in
simulation.

## Non-Goals For This Document

This document only covers the Isaac Lab training path.

It does not cover:

- Gazebo deployment wrapper
- ROS policy interface
- LeRobot/ACT fallback
- final challenge submission packaging
