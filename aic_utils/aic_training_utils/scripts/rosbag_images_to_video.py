#!/usr/bin/env python3
"""Convert a ROS 2 image-topic bag into an MP4 review video."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _storage_id(bag_dir: Path) -> str:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        return "sqlite3"
    match = re.search(
        r"storage_identifier:\s*([A-Za-z0-9_+-]+)",
        metadata_path.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else "sqlite3"


def _image_array(image_msg) -> np.ndarray:
    height = int(image_msg.height)
    width = int(image_msg.width)
    encoding = image_msg.encoding.lower()
    if height <= 0 or width <= 0:
        raise ValueError("empty image")

    if encoding == "mono8":
        channels = 1
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    else:
        channels = 3

    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    row_bytes = width * channels
    step = int(image_msg.step) if int(image_msg.step) > 0 else row_bytes
    expected = height * step
    if raw.size < expected:
        raise ValueError(
            f"image buffer too small for {width}x{height} step={step}: {raw.size}"
        )

    image = raw[:expected].reshape(height, step)[:, :row_bytes]
    image = image.reshape(height, width, channels)
    if encoding == "mono8":
        image = np.repeat(image, 3, axis=2)
    elif encoding in {"bgr8", "bgra8"}:
        image = image[..., [2, 1, 0]]
    else:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def _write_video(frames: list[np.ndarray], output: Path, fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    try:
        import cv2

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except Exception:
        import imageio.v2 as imageio

        imageio.mimsave(output, frames, fps=fps)


class _StreamingVideoWriter:
    def __init__(self, output: Path, fps: float):
        self.output = output
        self.fps = fps
        self._cv2 = None
        self._writer = None
        self._frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None and self._cv2 is None:
            try:
                import cv2

                self._cv2 = cv2
                self.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not self._writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter failed to open")
            except Exception:
                self._cv2 = False

        if self._writer is not None:
            self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))
        else:
            self._frames.append(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            return
        _write_video(self._frames, self.output, self.fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one ROS 2 sensor_msgs/Image topic from a bag to MP4."
    )
    parser.add_argument("bag_dir", type=Path, help="ROS 2 bag directory")
    parser.add_argument(
        "--topic",
        default="/observations",
        help=(
            "Topic to convert. Supports sensor_msgs/msg/Image directly and "
            "aic_model_interfaces/msg/Observation when --image-field is set."
        ),
    )
    parser.add_argument(
        "--image-field",
        choices=("left_image", "center_image", "right_image"),
        default="center_image",
        help=(
            "Image field to extract when --topic is an Observation message. "
            "Ignored for direct sensor_msgs/msg/Image topics."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to write",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = rosbag2_py.StorageOptions(
        uri=str(args.bag_dir),
        storage_id=_storage_id(args.bag_dir),
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    if args.topic not in topic_types:
        available = ", ".join(sorted(topic_types))
        raise SystemExit(f"Topic {args.topic!r} not found. Available topics: {available}")

    topic_type = topic_types[args.topic]
    msg_type = get_message(topic_type)
    writer = _StreamingVideoWriter(args.output, args.fps)
    frame_count = 0
    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, msg_type)
        if topic_type == "sensor_msgs/msg/Image":
            image_msg = msg
        elif topic_type == "aic_model_interfaces/msg/Observation":
            image_msg = getattr(msg, args.image_field)
        else:
            raise SystemExit(
                f"Topic {args.topic!r} has unsupported type {topic_type!r}."
            )
        writer.write(_image_array(image_msg))
        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    writer.close()
    print(f"wrote {frame_count} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
