#!/usr/bin/env python3
"""Extract arm joint traces from successful Gazebo eval bags.

This is an offline calibration helper. It reads `/joint_states` from one or more
rosbag directories and writes a JSON object compatible with
`AIC_RSLRL_PUBLIC_SCRIPTED_JOINT_TRACE_PATH`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _reader(bag_uri: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def _parse_mapping(value: str) -> tuple[str, str]:
    key, sep, bag = value.partition("=")
    if not sep or not key or not bag:
        raise argparse.ArgumentTypeError("expected TARGET_KEY=/path/to/bag")
    return key, bag


def extract_trace(bag_uri: str, sample_dt: float) -> list[list[float]]:
    reader = _reader(bag_uri)
    trace: list[list[float]] = []
    first_t: float | None = None
    last_sample_t: float | None = None

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic != "/joint_states":
            continue
        msg = deserialize_message(data, JointState)
        by_name = dict(zip(msg.name, msg.position, strict=False))
        if not all(name in by_name for name in ARM_JOINT_NAMES):
            continue
        t = timestamp_ns * 1.0e-9
        if first_t is None:
            first_t = t
        if last_sample_t is not None and t - last_sample_t < sample_dt:
            continue
        last_sample_t = t
        trace.append([float(by_name[name]) for name in ARM_JOINT_NAMES])

    if not trace:
        raise RuntimeError(f"no arm joint states extracted from {bag_uri}")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.05,
        help="Minimum seconds between saved joint samples.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "mapping",
        nargs="+",
        type=_parse_mapping,
        help="TARGET_KEY=/path/to/rosbag directory.",
    )
    args = parser.parse_args()

    traces = {
        key: extract_trace(bag, max(0.001, args.sample_dt))
        for key, bag in args.mapping
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    for key, trace in traces.items():
        print(f"{key}: {len(trace)} samples")
    print(output)


if __name__ == "__main__":
    main()
