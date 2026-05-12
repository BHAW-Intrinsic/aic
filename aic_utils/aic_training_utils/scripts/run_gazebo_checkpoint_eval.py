#!/usr/bin/env python3
"""Launch official Gazebo eval for an Isaac checkpoint-backed policy.

This is an orchestration wrapper. It starts the official Gazebo/AIC engine path
and an ``aic_model`` policy process in separate tmux sessions. The current
checkpoint policy is a scaffold until the Gazebo Observation -> Isaac actor
observation adapter is implemented.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "bahw_docs" / "overview.md").exists():
            return parent
    return Path.cwd()


def _quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def _run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(_quote(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _tmux_new(session: str, command: str, dry_run: bool) -> None:
    _run(["tmux", "new-session", "-d", "-s", session, command], dry_run=dry_run)


def _tmux_kill(session: str, dry_run: bool) -> None:
    command = (
        f"tmux has-session -t {_quote(session)} 2>/dev/null "
        f"&& tmux kill-session -t {_quote(session)} || true"
    )
    _run(["bash", "-lc", command], dry_run=dry_run)


def _bool_launch(value: bool) -> str:
    return "true" if value else "false"


def _ros_env_exports() -> list[str]:
    return [
        "export RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        "export ZENOH_ROUTER_CHECK_ATTEMPTS=-1",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start official Gazebo qualification eval sessions for a checkpoint "
            "policy wrapper."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Isaac/RSL-RL checkpoint path to pass to the ROS policy wrapper.",
    )
    parser.add_argument(
        "--sc-checkpoint",
        default="",
        help="SC-specific Isaac/RSL-RL checkpoint path for final routing.",
    )
    parser.add_argument(
        "--sfp-checkpoint",
        default="",
        help="SFP-specific Isaac/RSL-RL checkpoint path for final routing.",
    )
    parser.add_argument(
        "--policy-artifact",
        default="",
        help=(
            "Optional exported actor artifact path, for example exported/policy.pt. "
            "The current ROS policy scaffold can only best-effort load TorchScript."
        ),
    )
    parser.add_argument(
        "--sc-policy-artifact",
        default="",
        help="SC-specific exported actor artifact path for final routing.",
    )
    parser.add_argument(
        "--sfp-policy-artifact",
        default="",
        help="SFP-specific exported actor artifact path for final routing.",
    )
    parser.add_argument(
        "--policy",
        default="aic_model.RslRlCheckpointPolicy",
        help="aic_model policy module/class to run.",
    )
    parser.add_argument(
        "--task-kind",
        choices=("auto", "sc", "sfp"),
        default="auto",
        help="Checkpoint task kind passed through to the policy wrapper.",
    )
    parser.add_argument(
        "--repo",
        default=str(_repo_root()),
        help="Host repo path containing the pixi workspace.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help=(
            "AIC_RESULTS_DIR for scoring.yaml and scoring bags. Defaults to "
            "logs/gazebo_eval/<timestamp> under the repo."
        ),
    )
    parser.add_argument(
        "--session-prefix",
        default="gazebo-rslrl-eval",
        help="tmux session prefix. Sessions use -eval, -model, and optional -camera-bag.",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Run eval with ground_truth:=true. Do not use for qualification-like checks.",
    )
    parser.add_argument(
        "--gazebo-gui",
        action="store_true",
        help="Enable Gazebo GUI. Default is headless to reduce GPU load.",
    )
    parser.add_argument(
        "--launch-rviz",
        action="store_true",
        help="Launch RViz. Default is false to match automation and reduce GPU load.",
    )
    parser.add_argument(
        "--record-camera-bag",
        action="store_true",
        help=(
            "Start a third tmux session recording wrist camera image topics. "
            "This produces rosbag evidence, not MP4."
        ),
    )
    parser.add_argument(
        "--camera-topics",
        nargs="+",
        default=["/left_camera/image", "/center_camera/image", "/right_camera/image"],
        help="ROS image topics to record when --record-camera-bag is set.",
    )
    parser.add_argument(
        "--camera-bag-duration-sec",
        type=int,
        default=0,
        help=(
            "Optional duration for camera rosbag recording. The default 0 records "
            "until the camera-bag tmux session is stopped."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Kill existing tmux sessions with the selected names before starting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tmux commands without running them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any(
        (
            args.checkpoint,
            args.sc_checkpoint,
            args.sfp_checkpoint,
            args.policy_artifact,
            args.sc_policy_artifact,
            args.sfp_policy_artifact,
        )
    ):
        raise SystemExit(
            "Provide at least one checkpoint or policy artifact path. "
            "Use --checkpoint for a single candidate or --sc-checkpoint and "
            "--sfp-checkpoint for final routing."
        )

    repo = Path(args.repo).expanduser().resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = (
        Path(args.results_dir).expanduser()
        if args.results_dir
        else repo / "logs" / "gazebo_eval" / stamp
    )

    eval_session = f"{args.session_prefix}-eval"
    model_session = f"{args.session_prefix}-model"
    camera_session = f"{args.session_prefix}-camera-bag"

    if args.replace:
        for session in (eval_session, model_session, camera_session):
            _tmux_kill(session, dry_run=args.dry_run)

    eval_launch_args = [
        f"ground_truth:={_bool_launch(args.ground_truth)}",
        "start_aic_engine:=true",
        "shutdown_on_aic_engine_exit:=true",
        "model_discovery_timeout_seconds:=600",
        f"gazebo_gui:={_bool_launch(args.gazebo_gui)}",
        f"launch_rviz:={_bool_launch(args.launch_rviz)}",
    ]

    eval_cmd_parts = [
        "set -e",
        f"cd {_quote(repo)}",
        f"mkdir -p {_quote(results_dir)}",
        "export DBX_CONTAINER_MANAGER=docker",
        f"export AIC_RESULTS_DIR={_quote(results_dir)}",
        "distrobox enter -r aic_eval -- /entrypoint.sh "
        + " ".join(_quote(arg) for arg in eval_launch_args),
    ]
    eval_cmd = " && ".join(eval_cmd_parts)

    model_env = [
        f"export AIC_RSLRL_CHECKPOINT={_quote(args.checkpoint)}",
        f"export AIC_RSLRL_SC_CHECKPOINT={_quote(args.sc_checkpoint)}",
        f"export AIC_RSLRL_SFP_CHECKPOINT={_quote(args.sfp_checkpoint)}",
        f"export AIC_RSLRL_POLICY_ARTIFACT={_quote(args.policy_artifact)}",
        f"export AIC_RSLRL_SC_POLICY_ARTIFACT={_quote(args.sc_policy_artifact)}",
        f"export AIC_RSLRL_SFP_POLICY_ARTIFACT={_quote(args.sfp_policy_artifact)}",
        f"export AIC_RSLRL_TASK_KIND={_quote(args.task_kind)}",
    ]
    model_cmd_parts = [
        "set -e",
        f"cd {_quote(repo)}",
        "sleep 10",
        *_ros_env_exports(),
        *model_env,
        "pixi run ros2 run aic_model aic_model --ros-args "
        "-p use_sim_time:=true "
        f"-p policy:={_quote(args.policy)}",
    ]
    model_cmd = " && ".join(model_cmd_parts)

    print(f"results_dir: {results_dir}")
    print(f"eval_session: {eval_session}")
    print(f"model_session: {model_session}")
    if args.record_camera_bag:
        print(f"camera_session: {camera_session}")

    _tmux_new(eval_session, eval_cmd, dry_run=args.dry_run)
    _tmux_new(model_session, model_cmd, dry_run=args.dry_run)

    if args.record_camera_bag:
        bag_dir = results_dir / "camera_bags" / "wrist_cameras"
        record_cmd_parts = [
            "set -e",
            f"cd {_quote(repo)}",
            "sleep 20",
            *_ros_env_exports(),
            f"mkdir -p {_quote(bag_dir.parent)}",
        ]
        rosbag_command = (
            "pixi run ros2 bag record "
            f"-o {_quote(bag_dir)} "
            + " ".join(_quote(topic) for topic in args.camera_topics)
        )
        if args.camera_bag_duration_sec > 0:
            rosbag_command = (
                f"timeout {_quote(str(args.camera_bag_duration_sec))} {rosbag_command}"
            )
        record_cmd_parts.append(rosbag_command)
        _tmux_new(camera_session, " && ".join(record_cmd_parts), dry_run=args.dry_run)

    print()
    print("Inspect sessions with:")
    print(f"  tmux capture-pane -t {eval_session} -p -S -200")
    print(f"  tmux capture-pane -t {model_session} -p -S -200")
    if args.record_camera_bag:
        print(f"  tmux capture-pane -t {camera_session} -p -S -200")
    print()
    print(f"Expected scoring output: {results_dir / 'scoring.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
