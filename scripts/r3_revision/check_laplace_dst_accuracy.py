"""Check the Laplace DST solver against smooth analytical harmonic modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_helmholtz_datasets import FastHelmholtzSolver
from generate_laplace_mad1_solver_datasets import square_boundary


def harmonic_mode(points: np.ndarray, mode: int) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    frequency = mode * np.pi
    return np.sin(frequency * x) * np.sinh(frequency * y) / np.sinh(frequency)


def evaluate_grid(
    num_points: int,
    output_points: int,
    modes: tuple[int, ...],
    workers: int,
) -> dict[str, object]:
    if (num_points - 1) % (output_points - 1) != 0:
        raise ValueError("The solver grid must align with the output grid.")
    downsample = (num_points - 1) // (output_points - 1)
    axis = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    boundary_points = square_boundary(num_points)

    exact = np.stack(
        [harmonic_mode(points, mode).reshape(num_points, num_points) for mode in modes]
    )
    boundary = np.stack([harmonic_mode(boundary_points, mode) for mode in modes])
    zeros = np.zeros_like(exact)
    numerical = FastHelmholtzSolver(
        num_points, coefficient=0.0, workers=workers
    ).solve(boundary, zeros)

    cases: dict[str, dict[str, float]] = {}
    for index, mode in enumerate(modes):
        difference = numerical[index] - exact[index]
        fine_error = np.linalg.norm(difference) / np.linalg.norm(exact[index])
        numerical_output = numerical[index, ::downsample, ::downsample]
        exact_output = exact[index, ::downsample, ::downsample]
        output_error = np.linalg.norm(numerical_output - exact_output) / np.linalg.norm(
            exact_output
        )
        cases[str(mode)] = {
            "fine_grid_relative_l2": float(fine_error),
            "output_grid_relative_l2": float(output_error),
        }

    return {
        "num_points": num_points,
        "spacing": 1.0 / (num_points - 1),
        "output_points": output_points,
        "modes": cases,
    }


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=(501, 1001))
    parser.add_argument("--output-points", type=int, default=51)
    parser.add_argument("--modes", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "artifacts"
        / "r3_revision"
        / "laplace_data_source"
        / "dst_accuracy.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        str(num_points): evaluate_grid(
            num_points, args.output_points, tuple(args.modes), args.workers
        )
        for num_points in args.grids
    }
    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
