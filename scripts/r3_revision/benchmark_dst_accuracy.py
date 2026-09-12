"""Check the five-point/DST Helmholtz discretization against an exact field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_helmholtz_datasets import FastHelmholtzSolver


def perimeter_values(axis: np.ndarray) -> np.ndarray:
    bottom = np.cos(6.0 * axis) * np.sin(0.0)
    right = np.cos(6.0) * np.sin(8.0 * axis)
    top = np.cos(6.0 * axis[::-1]) * np.sin(8.0)
    left = np.cos(0.0) * np.sin(8.0 * axis[::-1])
    return np.concatenate((bottom[:-1], right[:-1], top[:-1], left[:-1]))


def relative_error(spacing: float, coefficient: float, workers: int) -> float:
    intervals = round(1.0 / spacing)
    if not np.isclose(intervals * spacing, 1.0):
        raise ValueError(f"Spacing {spacing} does not partition [0, 1].")
    num_points = intervals + 1
    axis = np.linspace(0.0, 1.0, num_points)
    xx, yy = np.meshgrid(axis, axis)
    exact = np.cos(6.0 * xx) * np.sin(8.0 * yy)
    source = (coefficient - 100.0) * exact
    solver = FastHelmholtzSolver(num_points, coefficient, workers)
    numerical = solver.solve(
        perimeter_values(axis)[None, :], source[None, :, :]
    )[0]
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficient", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--spacings", type=float, nargs="+", default=(0.02, 0.01, 0.002, 0.001)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {
        str(spacing): relative_error(spacing, args.coefficient, args.workers)
        for spacing in args.spacings
    }
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
