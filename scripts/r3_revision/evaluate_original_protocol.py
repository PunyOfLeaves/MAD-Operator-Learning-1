"""Evaluate an original-protocol DeepONet on both repository test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_original_protocol import PoissonONet, TextDataset


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--test-set-1",
        type=Path,
        default=repository / "data" / "MAD1helmholtz2D_(200, 51).txt",
    )
    parser.add_argument(
        "--test-set-2",
        type=Path,
        default=repository / "data" / "TNM1helmholtz2D_(200, 51).txt",
    )
    parser.add_argument("--num-points", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def predict_grid(
    model: PoissonONet,
    boundary: torch.Tensor,
    source: torch.Tensor,
    trunk: torch.Tensor,
) -> torch.Tensor:
    boundary_branch = model.net1.branch_net(boundary) * model.net1.last_layer_weights
    boundary_trunk = model.net1.trunk_net(trunk)
    source_branch = model.net2.branch_net(source) * model.net2.last_layer_weights
    source_trunk = model.net2.trunk_net(trunk)
    return boundary_branch @ boundary_trunk.transpose(0, 1) + source_branch @ source_trunk.transpose(0, 1)


@torch.inference_mode()
def evaluate(
    model: PoissonONet,
    path: Path,
    trunk: torch.Tensor,
    num_points: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(TextDataset(path), batch_size=batch_size, shuffle=False)
    boundary_count = 4 * (num_points - 1)
    point_count = num_points * num_points
    sample_errors: list[torch.Tensor] = []
    error_squared = 0.0
    reference_squared = 0.0
    for inputs in loader:
        inputs = inputs.to(device)
        true = inputs[:, boundary_count + point_count :]
        prediction = predict_grid(
            model,
            inputs[:, :boundary_count],
            inputs[:, boundary_count : boundary_count + point_count],
            trunk,
        )
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
    x_grid, y_grid = np.meshgrid(axis, axis)
    trunk = torch.tensor(
        np.vstack((x_grid.ravel(), y_grid.ravel())).T,
        dtype=torch.float32,
        device=device,
    )
    model = PoissonONet(
        4 * (args.num_points - 1),
        args.num_points * args.num_points,
        2,
        90,
        110,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    results = {
        name: evaluate(model, path, trunk, args.num_points, args.batch_size, device)
        for name, path in (("Test Set 1", args.test_set_1), ("Test Set 2", args.test_set_2))
    }
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
