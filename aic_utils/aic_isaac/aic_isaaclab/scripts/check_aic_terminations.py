# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check AIC insertion termination terms in Isaac Lab."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check AIC termination tensors.")
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs.")
parser.add_argument("--num_steps", type=int, default=8, help="Random action steps.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.25,
    help="Uniform random action range is [-scale, scale].",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for the timestamped termination log. Defaults to <repo>/logs/aic_terminations.",
)
parser.add_argument(
    "--no_log_file",
    action="store_true",
    default=False,
    help="Print only to stdout instead of also writing a log file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.aic_task import mdp


class Reporter:
    """Small stdout/file tee for termination-check output."""

    def __init__(self, path: Path | None):
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        if self._file is not None:
            self._file.write(f"{text}\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _repo_root() -> Path:
    """Find the AIC repo root from this script path."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "bahw_docs" / "overview.md").exists():
            return parent
    return Path.cwd()


def _default_log_path(task_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = task_name.replace("/", "_")
    return _repo_root() / "logs" / "aic_terminations" / f"{stamp}_{safe_task}.log"


def _bool_summary(value: Any) -> tuple[tuple[int, ...], str, int, int]:
    tensor = torch.as_tensor(value).detach()
    true_count = int(tensor.bool().sum().cpu().item())
    total = int(tensor.numel())
    return tuple(tensor.shape), str(tensor.dtype), true_count, total


def _print_termination_manager_terms(report: Reporter, env: Any) -> bool:
    termination_manager = getattr(env, "termination_manager", None)
    report.line("== Termination Manager Introspection ==")
    if termination_manager is None:
        report.line("termination_manager: unavailable")
        return False

    term_names = None
    for attr_name in ("_term_names", "active_terms"):
        value = getattr(termination_manager, attr_name, None)
        if value is not None:
            report.line(f"{attr_name}: {value}")
            if attr_name == "_term_names":
                term_names = list(value)

    term_cfgs = getattr(termination_manager, "_term_cfgs", None)
    if term_cfgs is not None:
        report.line("_term_cfgs:")
        for cfg in term_cfgs:
            name = getattr(cfg, "name", "<unnamed>")
            time_out = getattr(cfg, "time_out", "<unknown>")
            report.line(f"  {name}: time_out={time_out}")

    if term_names is None:
        return True
    return "time_out" in term_names and "sc_insertion_success" in term_names


def _success_mask(env: Any) -> torch.Tensor:
    return mdp.sc_insertion_success(
        env,
        lateral_threshold=0.005,
        orientation_threshold=0.20,
        depth_threshold=0.012,
    )


def _print_success_check(report: Reporter, env: Any, label: str) -> tuple[bool, bool]:
    value = _success_mask(env)
    shape, dtype, true_count, total = _bool_summary(value)
    shape_ok = shape == (env.num_envs,)
    dtype_ok = value.dtype == torch.bool
    report.line(
        f"{label}: shape={shape} dtype={dtype} true_count={true_count}/{total} "
        f"shape_ok={shape_ok} dtype_ok={dtype_ok}"
    )
    return shape_ok and dtype_ok, true_count == 0


def _print_target_reset_checks(report: Reporter, env: Any) -> bool:
    report.line("== Direct Success Term Checks: after reset ==")
    all_ok = True
    active_ids = mdp.active_sc_target_ids(env)
    original_ids = active_ids.clone()
    try:
        for target_id, target_name in enumerate(mdp.SC_TARGET_NAMES):
            active_ids[:] = target_id
            tensor_ok, all_false = _print_success_check(
                report, env, f"{target_name} reset success"
            )
            all_ok = all_ok and tensor_ok and all_false
    finally:
        active_ids[:] = original_ids
    return all_ok


def _print_analytic_threshold_checks(report: Reporter) -> bool:
    report.line("== Analytic Success Threshold Checks ==")
    cases = (
        ("inserted", 0.0, 0.0, 0.020, True),
        ("hovering", 0.0, 0.0, 0.000, False),
        ("lateral_miss", 0.020, 0.0, 0.020, False),
        ("orientation_miss", 0.0, 0.50, 0.020, False),
        ("depth_shortfall", 0.0, 0.0, 0.006, False),
        ("at_thresholds", 0.005, 0.20, 0.012, False),
    )
    lateral = torch.tensor([case[1] for case in cases])
    orientation = torch.tensor([case[2] for case in cases])
    depth = torch.tensor([case[3] for case in cases])
    expected = torch.tensor([case[4] for case in cases])
    actual = mdp.sc_insertion_success_from_errors(
        lateral,
        orientation,
        depth,
        lateral_threshold=0.005,
        orientation_threshold=0.20,
        depth_threshold=0.012,
    )
    all_ok = bool(torch.equal(actual, expected))
    for case, actual_value, expected_value in zip(cases, actual.tolist(), expected.tolist()):
        report.line(
            f"{case[0]}: success={actual_value} expected={expected_value} "
            f"lateral={case[1]:.6f} orientation={case[2]:.6f} depth={case[3]:.6f}"
        )
    report.line(f"analytic_success_checks_ok: {all_ok}")
    return all_ok


def main() -> int:
    log_path = None
    if not args_cli.no_log_file:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = args_cli.task.replace("/", "_")
        log_path = (
            Path(args_cli.output_dir) / f"{stamp}_{safe_task}.log"
            if args_cli.output_dir
            else _default_log_path(args_cli.task)
        )

    report = Reporter(log_path)
    env = None
    failed = False
    try:
        report.line("== AIC Termination Check ==")
        report.line(f"task: {args_cli.task}")
        report.line(f"num_envs: {args_cli.num_envs}")
        report.line(f"num_steps: {args_cli.num_steps}")
        if log_path is not None:
            report.line(f"log_path: {log_path}")

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped

        report.line()
        report.line(f"gym observation space: {env.observation_space}")
        report.line(f"gym action space: {env.action_space}")

        failed = failed or not _print_termination_manager_terms(report, base_env)
        failed = failed or not _print_analytic_threshold_checks(report)

        with torch.inference_mode():
            env.reset()

        failed = failed or not _print_target_reset_checks(report, base_env)

        report.line("== Random Policy Termination Checks ==")
        for step in range(args_cli.num_steps):
            actions = torch.empty(
                env.action_space.shape, device=base_env.device, dtype=torch.float32
            ).uniform_(-args_cli.action_scale, args_cli.action_scale)
            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(actions)
            term_shape, term_dtype, term_true, term_total = _bool_summary(terminated)
            trunc_shape, trunc_dtype, trunc_true, trunc_total = _bool_summary(truncated)
            success_ok, _ = _print_success_check(
                report, base_env, f"step {step:02d} direct success"
            )
            failed = failed or not success_ok
            report.line(
                f"step {step:02d} gym terminated: shape={term_shape} "
                f"dtype={term_dtype} true_count={term_true}/{term_total}"
            )
            report.line(
                f"step {step:02d} gym truncated: shape={trunc_shape} "
                f"dtype={trunc_dtype} true_count={trunc_true}/{trunc_total}"
            )

        report.line()
        report.line("== Done ==")
        report.line(f"overall_termination_check_ok: {not failed}")
    finally:
        if env is not None:
            env.close()
        report.close()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
