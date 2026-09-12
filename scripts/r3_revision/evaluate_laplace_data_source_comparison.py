"""Evaluate a source-free Laplace DeepONet on MAD1 and solver test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_original_protocol import LinearBranchDeepONet, TextDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-set-1", type=Path, required=True)
    parser.add_argument("--test-set-2", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def predict_grid(
    model: LinearBranchDeepONet,
    boundary: torch.Tensor,
    trunk: torch.Tensor,
) -> torch.Tensor:
    branch_features = model.branch_net(boundary) * model.last_layer_weights
    trunk_features = model.trunk_net(trunk)
    return branch_features @ trunk_features.transpose(0, 1)


@torch.inference_mode()
def evaluate(
    model: LinearBranchDeepONet,
    path: Path,
    trunk: torch.Tensor,
    num_points: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(TextDataset(path), batch_size=batch_size, shuffle=False)
    boundary_count = 4 * (num_points - 1)
    sample_errors: list[torch.Tensor] = []
    error_squared = 0.0
    reference_squared = 0.0
    for inputs in loader:
        inputs = inputs.to(device)
        true = inputs[:, boundary_count:]
        prediction = predict_grid(model, inputs[:, :boundary_count], trunk)
        difference = prediction - true
        sample_errors.append(
            (
                torch.linalg.vector_norm(difference, dim=1)
                / torch.linalg.vector_norm(true, dim=1).clamp_min(1.0e-12)
            ).cpu()
        )
        error_squared += float(torch.sum(difference.double() ** 2).cpu())
        reference_squared += float(torch.sum(true.double() ** 2).cpu())

    errors = torch.cat(sample_errors).numpy()
    return {
        "per_sample_mean": float(np.mean(errors)),
        "per_sample_std": float(np.std(errors, ddof=1)),
        "per_sample_median": float(np.median(errors)),
        "aggregate_relative_l2": float(np.sqrt(error_squared / reference_squared)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = 1.0 / (args.num_points - 1)
    axis = np.arange(0.0, 1.0 + spacing, spacing)
    xx, yy = np.meshgrid(axis, axis)
    trunk = torch.tensor(
        np.column_stack((xx.ravel(), yy.ravel())),
        dtype=torch.float32,
        device=device,
    )
    model = LinearBranchDeepONet(
        4 * (args.num_points - 1), 2, 110
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    results = {
        name: evaluate(model, path, trunk, args.num_points, args.batch_size, device)
        for name, path in (
            ("MAD1 test distribution", args.test_set_1),
            ("GRF-boundary solver test distribution", args.test_set_2),
        )
    }
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
