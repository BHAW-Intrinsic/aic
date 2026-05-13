#!/usr/bin/env python3
"""Export a simple RSL-RL MLP actor checkpoint to TorchScript."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _activation(name: str) -> torch.nn.Module:
    if name == "elu":
        return torch.nn.ELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _actor_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "actor_state_dict" in checkpoint:
        return checkpoint["actor_state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError(
        "Checkpoint does not contain actor_state_dict, model_state_dict, or state_dict"
    )


def _linear_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    indices = []
    for key in state_dict:
        if not key.startswith("mlp.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) == 3 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return sorted(indices)


def _build_actor(
    state_dict: dict[str, torch.Tensor],
    activation: str,
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    indices = _linear_indices(state_dict)
    if not indices:
        raise ValueError("No mlp.<index>.weight tensors found in actor state dict")

    for order, index in enumerate(indices):
        weight = state_dict[f"mlp.{index}.weight"]
        bias = state_dict[f"mlp.{index}.bias"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layers.append(layer)
        if order != len(indices) - 1:
            layers.append(_activation(activation))

    actor = torch.nn.Sequential(*layers)
    actor.eval()
    return actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a simple RSL-RL actor_state_dict with mlp.* tensors to "
            "TorchScript without launching Isaac Sim."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation", default="elu", choices=("elu", "relu", "tanh"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _actor_state_dict(checkpoint)
    actor = _build_actor(state_dict, args.activation)

    input_dim = next(module for module in actor if isinstance(module, torch.nn.Linear))
    example = torch.zeros(1, input_dim.in_features, dtype=torch.float32)
    traced = torch.jit.trace(actor, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)
    print(f"exported TorchScript actor to {args.output}")
    print(f"input_dim: {input_dim.in_features}")
    print(f"output_dim: {actor[-1].out_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
