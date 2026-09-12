"""Generate paired MAD1 and solver-labeled Laplace datasets for R3.

Each sample has the original source-free layout

    [boundary values g, solution values u].

MAD1 fields are finite sums of two-dimensional Laplace fundamental solutions
whose singularities lie outside the unit square.  The comparison fields use
the repository's detrended RBF Gaussian-process boundary distribution and a
five-point finite-difference solve diagonalized by a two-dimensional DST-I.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from generate_helmholtz_datasets import FastHelmholtzSolver, sample_boundary_paths


def square_points(num_points: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel()))


def square_boundary(num_points: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    edge_axis = axis[:-1]
    zeros = np.zeros_like(edge_axis)
    ones = np.ones_like(edge_axis)
    return np.concatenate(
        (
            np.column_stack((edge_axis, zeros)),
            np.column_stack((ones, edge_axis)),
            np.column_stack((axis[::-1][:-1], ones)),
            np.column_stack((zeros, axis[::-1][:-1])),
        ),
        axis=0,
    )


def normalize_pairs(boundary: np.ndarray, solution: np.ndarray) -> np.ndarray:
    rows = np.concatenate((boundary, solution), axis=1)
    scales = np.max(np.abs(rows), axis=1, keepdims=True)
    if not np.all(np.isfinite(scales)) or np.any(scales == 0.0):
        raise ValueError("Encountered a non-finite or zero normalization scale.")
    return rows / scales


def shift_outside(values: np.ndarray, minimum_distance: float) -> np.ndarray:
    return np.where(
        values <= 0.0,
        values - minimum_distance,
        values + 1.0 + minimum_distance,
    )


def fundamental_solution_fields(
    rng: np.random.Generator,
    points: np.ndarray,
    num_samples: int,
    num_sources: int,
    minimum_distance: float,
) -> np.ndarray:
    """Evaluate random sums of log fundamental solutions at ``points``."""

    # Cases 0, 1, and 2 place the source outside in x, in y, or in both
    # coordinates, with probabilities matching the original MAD1 generator.
    cases = rng.choice(3, size=(num_samples, num_sources), p=(0.25, 0.25, 0.50))
    normal_x = rng.standard_normal((num_samples, num_sources))
    normal_y = rng.standard_normal((num_samples, num_sources))
    uniform_x = rng.random((num_samples, num_sources))
    uniform_y = rng.random((num_samples, num_sources))
    outside_x = shift_outside(normal_x, minimum_distance)
    outside_y = shift_outside(normal_y, minimum_distance)
    source_x = np.where(cases == 1, uniform_x, outside_x)
    source_y = np.where(cases == 0, uniform_y, outside_y)
    weights = rng.standard_normal((num_samples, num_sources))

    dx = points[None, None, :, 0] - source_x[:, :, None]
    dy = points[None, None, :, 1] - source_y[:, :, None]
    kernels = 0.5 * np.log(dx * dx + dy * dy)
    return np.einsum("bs,bsp->bp", weights, kernels, optimize=True)


def generate_mad1(
    output_path: Path,
    num_samples: int,
    num_points: int,
    seed: int,
    batch_size: int,
    num_sources: int,
    minimum_distance: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    grid = square_points(num_points)
    boundary_points = square_boundary(num_points)
    row_length = 4 * (num_points - 1) + num_points * num_points
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float64, shape=(num_samples, row_length)
    )

    start_time = time.perf_counter()
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        current = stop - start

        # Draw one source ensemble and evaluate it at both sensor sets.  The
        # random state is captured and restored so both evaluations use the
        # same source locations and weights without storing generator details.
        state = rng.bit_generator.state
        solution = fundamental_solution_fields(
            rng, grid, current, num_sources, minimum_distance
        )
        rng.bit_generator.state = state
        boundary = fundamental_solution_fields(
            rng, boundary_points, current, num_sources, minimum_distance
        )
        output[start:stop] = normalize_pairs(boundary, solution)

    output.flush()
    elapsed = time.perf_counter() - start_time
    return {"total_seconds": elapsed, "seconds_per_sample": elapsed / num_samples}


def generate_solver_data(
    output_path: Path,
    num_samples: int,
    coarse_points: int,
    fine_points: int,
    seed: int,
    batch_size: int,
    boundary_length_scale: float,
    workers: int,
) -> dict[str, float]:
    if (fine_points - 1) % (coarse_points - 1) != 0:
        raise ValueError("The fine grid must align with the coarse grid.")
    downsample = (fine_points - 1) // (coarse_points - 1)
    rng = np.random.default_rng(seed)
    row_length = 4 * (coarse_points - 1) + coarse_points * coarse_points
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float64, shape=(num_samples, row_length)
    )

    total_start = time.perf_counter()
    boundary_start = time.perf_counter()
    boundary_paths = sample_boundary_paths(
        rng, num_samples, fine_points, boundary_length_scale
    )
    boundary_seconds = time.perf_counter() - boundary_start

    solver = FastHelmholtzSolver(fine_points, coefficient=0.0, workers=workers)
    solve_seconds = 0.0
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        current = stop - start
        zeros = np.zeros((current, fine_points, fine_points), dtype=np.float64)
        solve_start = time.perf_counter()
        solution = solver.solve(boundary_paths[start:stop], zeros)
        solve_seconds += time.perf_counter() - solve_start

        boundary_coarse = boundary_paths[start:stop, ::downsample]
        solution_coarse = solution[:, ::downsample, ::downsample].reshape(current, -1)
        output[start:stop] = normalize_pairs(boundary_coarse, solution_coarse)

    output.flush()
    total_seconds = time.perf_counter() - total_start
    return {
        "total_seconds": total_seconds,
        "seconds_per_sample": total_seconds / num_samples,
        "boundary_sampling_seconds": boundary_seconds,
        "dst_solve_seconds": solve_seconds,
    }


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=repository / "data" / "r3_revision"
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--coarse-points", type=int, default=51)
    parser.add_argument("--fine-points", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mad-batch-size", type=int, default=50)
    parser.add_argument("--num-sources", type=int, default=10)
    parser.add_argument("--minimum-distance", type=float, default=0.001)
    parser.add_argument("--boundary-length-scale", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--skip-mad", action="store_true")
    parser.add_argument("--skip-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.samples}_p{args.coarse_points}_seed{args.seed}"
    metadata: dict[str, object] = {"parameters": vars(args).copy()}
    metadata["parameters"]["output_dir"] = str(args.output_dir)

    if not args.skip_mad:
        mad_path = args.output_dir / f"mad1_laplace_{tag}.npy"
        print(f"Generating MAD1 data: {mad_path}", flush=True)
        metadata["mad1"] = generate_mad1(
            mad_path,
            args.samples,
            args.coarse_points,
            args.seed,
            args.mad_batch_size,
            args.num_sources,
            args.minimum_distance,
        )
        print(json.dumps(metadata["mad1"], indent=2), flush=True)

    if not args.skip_solver:
        solver_path = args.output_dir / f"dst_fd_laplace_{tag}.npy"
        print(f"Generating GRF-boundary solver data: {solver_path}", flush=True)
        metadata["dst_fd"] = generate_solver_data(
            solver_path,
            args.samples,
            args.coarse_points,
            args.fine_points,
            args.seed + 1,
            args.batch_size,
            args.boundary_length_scale,
            args.workers,
        )
        print(json.dumps(metadata["dst_fd"], indent=2), flush=True)

    metadata_path = args.output_dir / f"generation_laplace_{tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
