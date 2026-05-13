#!/usr/bin/env python3
"""Summarize controller commands and TCP motion from a Gazebo eval rosbag."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COMMAND_TOPICS = {
    "/aic_controller/joint_commands",
    "/aic_controller/pose_commands",
    "/aic_controller/controller_state",
    "/joint_states",
}


@dataclass
class TopicInfo:
    msg_type: str
    count: int = 0


def _reader(bag_uri: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def _topic_types(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
        if topic.name in COMMAND_TOPICS
    }


def _pose_to_xyz(pose) -> np.ndarray:
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)


def _quat_to_xyzw(pose) -> np.ndarray:
    return np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )


def _format_vec(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def analyze_bag(bag_uri: str, sample_limit: int, include_scoring_tf: bool) -> None:
    reader = _reader(bag_uri)
    topic_types = _topic_types(reader)
    if include_scoring_tf:
        topic_types.update(
            {
                topic.name: topic.type
                for topic in reader.get_all_topics_and_types()
                if topic.name == "/scoring/tf"
            }
        )
    msg_classes = {topic: get_message(msg_type) for topic, msg_type in topic_types.items()}
    infos = {topic: TopicInfo(msg_type=msg_type) for topic, msg_type in topic_types.items()}

    first_tcp = None
    last_tcp = None
    tcp_bounds_min = None
    tcp_bounds_max = None
    first_joint = None
    last_joint = None
    pose_samples = []
    joint_command_samples = []
    pose_delta_norms = []
    last_controller_tcp = None
    pose_command_delta_from_tcp = []
    tf_first = {}
    tf_last = {}

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in msg_classes:
            continue
        msg = deserialize_message(data, msg_classes[topic])
        infos[topic].count += 1
        t = timestamp_ns * 1.0e-9

        if topic == "/aic_controller/controller_state":
            tcp = _pose_to_xyz(msg.tcp_pose)
            if first_tcp is None:
                first_tcp = tcp.copy()
            last_tcp = tcp.copy()
            last_controller_tcp = tcp.copy()
            if tcp_bounds_min is None:
                tcp_bounds_min = tcp.copy()
                tcp_bounds_max = tcp.copy()
            else:
                tcp_bounds_min = np.minimum(tcp_bounds_min, tcp)
                tcp_bounds_max = np.maximum(tcp_bounds_max, tcp)
        elif topic == "/joint_states":
            if first_joint is None:
                first_joint = np.array(msg.position[:6], dtype=np.float64)
            last_joint = np.array(msg.position[:6], dtype=np.float64)
        elif topic == "/aic_controller/pose_commands":
            target = _pose_to_xyz(msg.pose)
            pose_delta = None
            if last_controller_tcp is not None:
                pose_delta = target - last_controller_tcp
                pose_delta_norms.append(_norm(pose_delta))
                pose_command_delta_from_tcp.append(pose_delta)
            if len(pose_samples) < sample_limit:
                pose_samples.append(
                    (
                        t,
                        msg.header.frame_id,
                        int(msg.trajectory_generation_mode.mode),
                        target,
                        _quat_to_xyzw(msg.pose),
                        pose_delta,
                    )
                )
        elif topic == "/aic_controller/joint_commands":
            positions = np.array(msg.target_state.positions[:6], dtype=np.float64)
            if len(joint_command_samples) < sample_limit:
                joint_command_samples.append((t, positions))
        elif topic == "/scoring/tf":
            for transform in msg.transforms:
                child = transform.child_frame_id
                translation = transform.transform.translation
                xyz = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
                if child not in tf_first:
                    tf_first[child] = xyz.copy()
                tf_last[child] = xyz.copy()

    print(f"bag: {bag_uri}")
    for topic in sorted(infos):
        info = infos[topic]
        print(f"{topic}: count={info.count} type={info.msg_type}")

    if first_tcp is not None and last_tcp is not None:
        print(f"tcp_first: {_format_vec(first_tcp)}")
        print(f"tcp_last:  {_format_vec(last_tcp)}")
        print(f"tcp_net_delta: {_format_vec(last_tcp - first_tcp)} norm={_norm(last_tcp - first_tcp):.4f}")
        print(f"tcp_bounds_min: {_format_vec(tcp_bounds_min)}")
        print(f"tcp_bounds_max: {_format_vec(tcp_bounds_max)}")
    if first_joint is not None and last_joint is not None:
        print(f"joint_first: {_format_vec(first_joint)}")
        print(f"joint_last:  {_format_vec(last_joint)}")
        print(f"joint_net_delta: {_format_vec(last_joint - first_joint)}")

    if pose_delta_norms:
        delta_norms = np.array(pose_delta_norms)
        print(
            "pose_command_delta_from_latest_tcp_norm: "
            f"min={delta_norms.min():.5f} mean={delta_norms.mean():.5f} max={delta_norms.max():.5f}"
        )
        delta_vectors = np.array(pose_command_delta_from_tcp)
        print(
            "pose_command_delta_from_latest_tcp_mean: "
            f"{_format_vec(delta_vectors.mean(axis=0), precision=5)}"
        )

    if joint_command_samples:
        print("joint_command_samples:")
        for t, positions in joint_command_samples:
            print(f"  t={t:.3f} positions={_format_vec(positions)}")

    if pose_samples:
        print("pose_command_samples:")
        for t, frame_id, mode, target, quat, pose_delta in pose_samples:
            suffix = ""
            if pose_delta is not None:
                suffix = f" delta_from_tcp={_format_vec(pose_delta, precision=5)}"
            print(
                f"  t={t:.3f} frame={frame_id!r} mode={mode} "
                f"target={_format_vec(target)} quat_xyzw={_format_vec(quat)}{suffix}"
            )

    if include_scoring_tf and tf_last:
        interesting = sorted(
            name
            for name in tf_last
            if any(token in name.lower() for token in ("sfp", "sc", "plug", "port", "tip"))
        )
        print("scoring_tf_interesting_frames:")
        for name in interesting:
            first = tf_first[name]
            last = tf_last[name]
            print(
                f"  {name}: first={_format_vec(first)} last={_format_vec(last)} "
                f"delta={_format_vec(last - first)}"
            )
        for plug_name in interesting:
            if "plug" not in plug_name.lower() and "tip" not in plug_name.lower():
                continue
            for port_name in interesting:
                if "port" not in port_name.lower():
                    continue
                distance = _norm(tf_last[plug_name] - tf_last[port_name])
                if math.isfinite(distance) and distance < 1.0:
                    print(f"  final_distance {plug_name} -> {port_name}: {distance:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_uri", help="Path to a rosbag directory")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=12,
        help="Number of command samples to print per command topic",
    )
    parser.add_argument(
        "--include-scoring-tf",
        action="store_true",
        help="Also summarize /scoring/tf frames. Use for offline diagnostics only.",
    )
    args = parser.parse_args()
    analyze_bag(args.bag_uri, max(0, args.sample_limit), args.include_scoring_tf)


if __name__ == "__main__":
    main()
