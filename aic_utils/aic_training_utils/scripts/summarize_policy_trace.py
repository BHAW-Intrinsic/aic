#!/usr/bin/env python3
"""Summarize JSONL traces emitted by RslRlCheckpointPolicy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def _norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _fmt(values: list[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_file(path: Path) -> None:
    events = _load_events(path)
    start = next((event for event in events if event.get("event") == "start"), {})
    finish = next((event for event in events if event.get("event") == "finish"), {})
    steps = [event for event in events if event.get("event") == "actor_step"]
    task = start.get("task", {})

    print(f"trace: {path}")
    print(
        "  task: "
        f"{start.get('task_kind', 'unknown')} "
        f"target_module={task.get('target_module_name', '')} "
        f"port={task.get('port_name', '')}"
    )
    print(f"  status: {finish.get('status', 'unknown')} steps_logged={len(steps)}")
    if finish.get("full_obs_npz"):
        print(f"  full_obs_npz: {finish['full_obs_npz']}")
    if not steps:
        print()
        return

    actions = [step["action"] for step in steps]
    action_norms = [_norm(action) for action in actions]
    frames = Counter(step["command"]["frame_id"] for step in steps)
    tcp_first = steps[0]["observation"]["tcp_pose"]["xyz"]
    tcp_last = steps[-1]["observation"]["tcp_pose"]["xyz"]
    tcp_delta = [last - first for first, last in zip(tcp_first, tcp_last)]
    tcp_error_norms = [
        _norm(step["observation"].get("tcp_error", [])[:3]) for step in steps
    ]
    body_force_norms = [
        _norm(step["actor_obs"]["body_forces_last6"]) for step in steps
    ]

    print(f"  command_frames: {dict(frames)}")
    print(
        "  action_norm: "
        f"first={action_norms[0]:.4f} mean={mean(action_norms):.4f} "
        f"last={action_norms[-1]:.4f}"
    )
    print(f"  first_action: {_fmt(actions[0])}")
    print(f"  last_action:  {_fmt(actions[-1])}")
    print(f"  tcp_first: {_fmt(tcp_first)}")
    print(f"  tcp_last:  {_fmt(tcp_last)}")
    print(f"  tcp_delta: {_fmt(tcp_delta)} norm={_norm(tcp_delta):.4f}")
    print(
        "  tcp_error_norm: "
        f"first={tcp_error_norms[0]:.4f} mean={mean(tcp_error_norms):.4f} "
        f"last={tcp_error_norms[-1]:.4f}"
    )
    print(
        "  body_force_last6_norm: "
        f"first={body_force_norms[0]:.4f} mean={mean(body_force_norms):.4f} "
        f"last={body_force_norms[-1]:.4f}"
    )
    for key in ("center_rgb_resnet18", "left_rgb_resnet18", "right_rgb_resnet18"):
        norms = [step["actor_obs"][key]["norm"] for step in steps]
        print(
            f"  {key}_norm: "
            f"first={norms[0]:.4f} mean={mean(norms):.4f} last={norms[-1]:.4f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", help="Trace JSONL file or policy_trace directory")
    args = parser.parse_args()

    path = Path(args.trace_path).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]
    if not files:
        raise SystemExit(f"No JSONL trace files found under {path}")
    for trace_file in files:
        summarize_file(trace_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
