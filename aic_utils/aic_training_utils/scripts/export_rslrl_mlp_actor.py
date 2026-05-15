#!/usr/bin/env python3
"""Export a simple RSL-RL MLP actor checkpoint to TorchScript."""

from __future__ import annotations

import argparse
import re
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


def _strip_actor_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return actor MLP tensors keyed as ``mlp.<index>.*``."""
    prefix_scores: dict[str, int] = {}
    pattern = re.compile(r"^(?P<prefix>.*?)(?P<mlp>mlp\.\d+\.weight)$")
    for key in state_dict:
        match = pattern.match(key)
        if match is None:
            continue
        prefix = match.group("prefix")
        if "critic" in prefix.lower():
            continue
        prefix_scores[prefix] = prefix_scores.get(prefix, 0) + 1

    if not prefix_scores:
        return state_dict

    priority = ("", "actor.", "actor_critic.actor.", "module.actor.", "model.actor.")
    for prefix in priority:
        if prefix in prefix_scores:
            chosen_prefix = prefix
            break
    else:
        chosen_prefix = max(prefix_scores, key=prefix_scores.get)

    if not chosen_prefix:
        return state_dict
    return {
        key[len(chosen_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(chosen_prefix)
    }


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
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=0,
        help="Optional expected actor input dimension for validation.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=0,
        help="Optional expected actor output dimension for validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _strip_actor_prefix(_actor_state_dict(checkpoint))
    actor = _build_actor(state_dict, args.activation)

    input_dim = next(module for module in actor if isinstance(module, torch.nn.Linear))
    output_dim = actor[-1]
    if not isinstance(output_dim, torch.nn.Linear):
        raise TypeError("Expected final actor module to be torch.nn.Linear")
    if args.obs_dim and input_dim.in_features != args.obs_dim:
        raise ValueError(
            f"Actor input dimension {input_dim.in_features} does not match --obs-dim {args.obs_dim}"
        )
    if args.action_dim and output_dim.out_features != args.action_dim:
        raise ValueError(
            f"Actor output dimension {output_dim.out_features} does not match "
            f"--action-dim {args.action_dim}"
        )
    example = torch.zeros(1, input_dim.in_features, dtype=torch.float32)
    traced = torch.jit.trace(actor, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)
    print(f"exported TorchScript actor to {args.output}")
    print(f"input_dim: {input_dim.in_features}")
    print(f"output_dim: {output_dim.out_features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
