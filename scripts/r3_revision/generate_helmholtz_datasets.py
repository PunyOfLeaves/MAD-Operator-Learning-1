"""Generate paired MAD0 and fast-solver Helmholtz datasets for the R3 study.

The sample layout follows the original repository:

    [boundary values g, source values f, solution values u]

MAD0 samples are evaluated in float64 and solver samples use a five-point
finite-difference discretization diagonalized by a two-dimensional DST-I.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dstn
from scipy.ndimage import gaussian_filter
from torch import nn


class SineGenerator(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 50, dtype=dtype)
        self.fc2 = nn.Linear(50, 50, dtype=dtype)
        self.fc3 = nn.Linear(50, 1, dtype=dtype)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
                nn.init.normal_(module.bias, mean=0.0, std=1.0)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = torch.sin(self.fc1(coordinates))
        hidden = torch.sin(self.fc2(hidden))
        return self.fc3(hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def square_grid(num_points: int, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, num_points, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def square_boundary(num_points: int, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, num_points, dtype=dtype)
    edge_axis = axis[:-1]
    zeros = torch.zeros_like(edge_axis)
    ones = torch.ones_like(edge_axis)
    edges = (
        torch.stack((edge_axis, zeros), dim=1),
        torch.stack((ones, edge_axis), dim=1),
        torch.stack((torch.flip(axis, dims=(0,))[:-1], ones), dim=1),
        torch.stack((zeros, torch.flip(axis, dims=(0,))[:-1]), dim=1),
    )
    return torch.cat(edges, dim=0)


def normalize_row(
    boundary: np.ndarray,
    source: np.ndarray,
    solution: np.ndarray,
    normalization: str,
) -> np.ndarray:
    if normalization == "all":
        scale = max(
            float(np.max(np.abs(boundary))),
            float(np.max(np.abs(source))),
            float(np.max(np.abs(solution))),
        )
    elif normalization == "input":
        scale = max(float(np.max(np.abs(boundary))), float(np.max(np.abs(source))))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return np.concatenate((boundary, source, solution)) / scale


def generate_mad0(
    output_path: Path,
    num_samples: int,
    num_points: int,
    coefficient: float,
    seed: int,
    normalization: str,
) -> dict[str, float]:
    set_seed(seed)
    dtype = torch.float64
    points = square_grid(num_points, dtype)
    boundary_points = square_boundary(num_points, dtype)
    row_length = 4 * (num_points - 1) + 2 * num_points * num_points
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float64, shape=(num_samples, row_length)
    )

    start = time.perf_counter()
    for index in range(num_samples):
        generator = SineGenerator(dtype=dtype).eval()
        x = points[:, 0:1].clone().requires_grad_(True)
        y = points[:, 1:2].clone().requires_grad_(True)
        solution = generator(torch.cat((x, y), dim=1))
        ux, uy = torch.autograd.grad(
            solution, (x, y), torch.ones_like(solution), create_graph=True
        )
        uxx = torch.autograd.grad(
            ux, x, torch.ones_like(ux), retain_graph=True
        )[0]
        uyy = torch.autograd.grad(uy, y, torch.ones_like(uy))[0]
        source = uxx + uyy + coefficient * solution
        boundary = generator(boundary_points)

        output[index] = normalize_row(
            boundary.detach().cpu().numpy().reshape(-1),
            source.detach().cpu().numpy().reshape(-1),
            solution.detach().cpu().numpy().reshape(-1),
            normalization,
        )
    output.flush()
    elapsed = time.perf_counter() - start
    return {"total_seconds": elapsed, "seconds_per_sample": elapsed / num_samples}


def sample_boundary_paths(
    rng: np.random.Generator,
    num_samples: int,
    fine_points: int,
    length_scale: float,
    jitter: float = 1.0e-13,
) -> np.ndarray:
    """Sample the repository's detrended RBF-GP perimeter distribution.

    The original Test Set 2 generator samples the Gaussian process directly on
    the fine perimeter used by the h=0.001 solve and then retains every
    ``downsample``-th value. We reproduce that order here; sampling first on the
    coarse sensors and interpolating would define a different joint law.
    """

    fine_boundary_count = 4 * (fine_points - 1)
    fine_parameter = np.linspace(0.0, 1.0, fine_boundary_count + 1)
    distance = fine_parameter[:, None] - fine_parameter[None, :]
    covariance = np.exp(-0.5 * (distance / length_scale) ** 2)
    covariance.flat[:: covariance.shape[0] + 1] += jitter
    cholesky = np.linalg.cholesky(covariance)
    samples = (
        cholesky
        @ rng.standard_normal((fine_boundary_count + 1, num_samples))
    ).T
    trend = samples[:, :1] + (samples[:, -1:] - samples[:, :1]) * fine_parameter
    detrended = samples - trend
    return detrended[:, :-1]


def perimeter_edges(paths: np.ndarray, num_points: int) -> tuple[np.ndarray, ...]:
    """Return bottom, right, top, and left edges in increasing coordinates."""

    bottom = paths[:, 0:num_points]
    right = paths[:, num_points - 1 : 2 * num_points - 1]
    top = paths[:, 2 * num_points - 2 : 3 * num_points - 2][:, ::-1]
    left_top_to_near_bottom = paths[:, 3 * num_points - 3 : 4 * num_points - 4]
    left = np.concatenate((paths[:, 0:1], left_top_to_near_bottom[:, ::-1]), axis=1)
    return bottom, right, top, left


class FastHelmholtzSolver:
    def __init__(self, num_points: int, coefficient: float, workers: int) -> None:
        self.num_points = num_points
        self.coefficient = coefficient
        self.workers = workers
        self.spacing = 1.0 / (num_points - 1)
        interior = num_points - 2
        modes = np.arange(1, interior + 1, dtype=np.float64)
        one_dimensional = 2.0 * np.cos(np.pi * modes / (interior + 1))
        self.eigenvalues = (
            -4.0
            + coefficient * self.spacing**2
            + one_dimensional[:, None]
            + one_dimensional[None, :]
        )
        minimum = float(np.min(np.abs(self.eigenvalues)))
        if minimum < 1.0e-14:
            raise ValueError("The discrete Helmholtz operator is singular or nearly singular.")

    def solve(self, boundary_paths: np.ndarray, source: np.ndarray) -> np.ndarray:
        bottom, right, top, left = perimeter_edges(boundary_paths, self.num_points)
        rhs = self.spacing**2 * source[:, 1:-1, 1:-1].copy()
        rhs[:, 0, :] -= bottom[:, 1:-1]
        rhs[:, -1, :] -= top[:, 1:-1]
        rhs[:, :, 0] -= left[:, 1:-1]
        rhs[:, :, -1] -= right[:, 1:-1]

        transformed = dstn(
            rhs, type=1, axes=(-2, -1), norm="ortho", workers=self.workers
        )
        transformed /= self.eigenvalues
        interior_solution = dstn(
            transformed, type=1, axes=(-2, -1), norm="ortho", workers=self.workers
        )

        solution = np.zeros_like(source)
        solution[:, 1:-1, 1:-1] = interior_solution
        solution[:, 0, :] = bottom
        solution[:, -1, :] = top
        solution[:, :, 0] = left
        solution[:, :, -1] = right
        return solution


def generate_solver_data(
    output_path: Path,
    num_samples: int,
    coarse_points: int,
    fine_points: int,
    coefficient: float,
    seed: int,
    batch_size: int,
    source_sigma: float,
    boundary_length_scale: float,
    normalization: str,
    workers: int,
) -> dict[str, float]:
    if (fine_points - 1) % (coarse_points - 1) != 0:
        raise ValueError("The fine grid must align with the coarse grid.")
    downsample = (fine_points - 1) // (coarse_points - 1)
    rng = np.random.default_rng(seed)
    row_length = 4 * (coarse_points - 1) + 2 * coarse_points * coarse_points
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float64, shape=(num_samples, row_length)
    )

    total_start = time.perf_counter()
    boundary_start = time.perf_counter()
    boundary_paths = sample_boundary_paths(
        rng,
        num_samples,
        fine_points,
        boundary_length_scale,
    )
    boundary_seconds = time.perf_counter() - boundary_start
    solver = FastHelmholtzSolver(fine_points, coefficient, workers)
    solve_seconds = 0.0
    source_seconds = 0.0

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        current_batch = stop - start
        source_start = time.perf_counter()
        white_noise = rng.standard_normal((current_batch, fine_points, fine_points))
        source = gaussian_filter(
            white_noise, sigma=(0.0, source_sigma, source_sigma), mode="reflect"
        )
        source_seconds += time.perf_counter() - source_start

        solve_start = time.perf_counter()
        solution = solver.solve(boundary_paths[start:stop], source)
        solve_seconds += time.perf_counter() - solve_start

        boundary_coarse = boundary_paths[start:stop, ::downsample]
        source_coarse = source[:, ::downsample, ::downsample].reshape(current_batch, -1)
        solution_coarse = solution[:, ::downsample, ::downsample].reshape(current_batch, -1)
        for local_index in range(current_batch):
            output[start + local_index] = normalize_row(
                boundary_coarse[local_index],
                source_coarse[local_index],
                solution_coarse[local_index],
                normalization,
            )

    output.flush()
    total_seconds = time.perf_counter() - total_start
    return {
        "total_seconds": total_seconds,
        "seconds_per_sample": total_seconds / num_samples,
        "boundary_sampling_seconds": boundary_seconds,
        "source_sampling_seconds": source_seconds,
        "dst_solve_seconds": solve_seconds,
    }


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repository / "data" / "r3_revision")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--coarse-points", type=int, default=51)
    parser.add_argument("--fine-points", type=int, default=1001)
    parser.add_argument("--coefficient", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-sigma", type=float, default=5.0)
    parser.add_argument("--boundary-length-scale", type=float, default=0.1)
    parser.add_argument("--normalization", choices=("all", "input"), default="all")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--skip-mad", action="store_true")
    parser.add_argument("--skip-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"k{args.coefficient:g}_n{args.samples}_p{args.coarse_points}_seed{args.seed}"
    metadata: dict[str, object] = {"parameters": vars(args).copy()}
    metadata["parameters"]["output_dir"] = str(args.output_dir)

    if not args.skip_mad:
        mad_path = args.output_dir / f"mad0_{tag}.npy"
        print(f"Generating MAD0 data: {mad_path}", flush=True)
        metadata["mad0"] = generate_mad0(
            mad_path,
            args.samples,
            args.coarse_points,
            args.coefficient,
            args.seed,
            args.normalization,
        )
        print(json.dumps(metadata["mad0"], indent=2), flush=True)

    if not args.skip_solver:
        solver_path = args.output_dir / f"dst_fd_{tag}.npy"
        print(f"Generating DST-FD data: {solver_path}", flush=True)
        metadata["dst_fd"] = generate_solver_data(
            solver_path,
            args.samples,
            args.coarse_points,
            args.fine_points,
            args.coefficient,
            args.seed + 1,
            args.batch_size,
            args.source_sigma,
            args.boundary_length_scale,
            args.normalization,
            args.workers,
        )
        print(json.dumps(metadata["dst_fd"], indent=2), flush=True)

    metadata_path = args.output_dir / f"generation_{tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
