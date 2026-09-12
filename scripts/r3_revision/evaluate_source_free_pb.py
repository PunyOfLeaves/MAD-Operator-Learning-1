"""Evaluate the published MAD0-FNO on genuinely two-dimensional source-free PB data.

The reference solutions solve

    -Delta u + sinh(u) = 0  in (0, 1)^2

with smooth Dirichlet traces independently sampled from the MAD0 generator
family.  A float64 finite-difference Picard iteration is accelerated by a
two-dimensional discrete sine transform (DST).  Reference-grid convergence is
checked before the existing 101-by-101 FNO is evaluated without retraining.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    @staticmethod
    def compl_mul2d(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", values, torch.view_as_complex(weights))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = values.shape
        values_ft = torch.fft.rfft2(values)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=values.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            values_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNO2d(nn.Module):
    def __init__(self, in_dim: int = 4, modes1: int = 16, modes2: int = 16, width: int = 64):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_dim, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = inputs.shape
        values = self.fc0(inputs.reshape(batch, height * width, channels))
        values = values.reshape(batch, height, width, self.width).permute(0, 3, 1, 2)
        values = F.relu(self.conv0(values) + self.w0(values))
        values = F.relu(self.conv1(values) + self.w1(values))
        values = F.relu(self.conv2(values) + self.w2(values))
        values = self.conv3(values) + self.w3(values)
        values = values.permute(0, 2, 3, 1)
        return self.fc2(F.relu(self.fc1(values))).squeeze(-1)


class BoundaryToField(nn.Module):
    def __init__(self, boundary_size: int, grid_size: int, width: int = 256):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(boundary_size, width),
            nn.ReLU(),
            nn.Linear(width, grid_size * grid_size),
        )

    def forward(self, boundary: torch.Tensor) -> torch.Tensor:
        return self.net(boundary).view(boundary.shape[0], self.grid_size, self.grid_size)


@dataclass(frozen=True)
class BoundaryRealization:
    frequencies: np.ndarray
    sine_coefficients: np.ndarray
    cosine_coefficients: np.ndarray
    offset: float
    scale: float

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        values = np.full(np.broadcast(x, y).shape, self.offset, dtype=np.float64)
        for (px, py), sine, cosine in zip(
            self.frequencies, self.sine_coefficients, self.cosine_coefficients
        ):
            phase = math.pi * (px * x + py * y)
            decay = (1.0 + px * px + py * py) ** 0.75
            values += (sine * np.sin(phase) + cosine * np.cos(phase)) / decay
        return self.scale * values


@dataclass(frozen=True)
class SineNetworkBoundaryRealization:
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: np.ndarray
    weight3: np.ndarray
    bias3: float
    scale: float

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        shape = np.broadcast(x, y).shape
        coordinates = np.stack(
            [np.broadcast_to(x, shape).ravel(), np.broadcast_to(y, shape).ravel()], axis=1
        )
        hidden1 = np.sin(coordinates @ self.weight1.T + self.bias1)
        hidden2 = np.sin(hidden1 @ self.weight2.T + self.bias2)
        values = hidden2 @ self.weight3 + self.bias3
        return (self.scale * values).reshape(shape)


@dataclass
class SolveDiagnostics:
    iterations: int
    relative_change: float
    relative_residual: float


def sample_fourier_boundary(rng: np.random.Generator, terms: int = 12) -> BoundaryRealization:
    frequencies = rng.integers(1, 5, size=(terms, 2), endpoint=False)
    sine = rng.normal(size=terms)
    cosine = rng.normal(size=terms)
    offset = float(rng.normal(scale=0.2))
    provisional = BoundaryRealization(frequencies, sine, cosine, offset, 1.0)

    probe = np.linspace(0.0, 1.0, 401)
    perimeter = np.concatenate(
        [
            provisional.evaluate(probe[:-1], np.zeros(400)),
            provisional.evaluate(np.ones(400), probe[:-1]),
            provisional.evaluate(probe[:0:-1], np.ones(400)),
            provisional.evaluate(np.zeros(400), probe[:0:-1]),
        ]
    )
    target_max = float(rng.uniform(0.35, 0.80))
    scale = target_max / max(float(np.max(np.abs(perimeter))), 1.0e-12)
    return BoundaryRealization(frequencies, sine, cosine, offset, scale)


def sample_sine_network_boundary(rng: np.random.Generator) -> SineNetworkBoundaryRealization:
    realization = SineNetworkBoundaryRealization(
        weight1=rng.normal(size=(50, 2)),
        bias1=rng.normal(size=50),
        weight2=rng.normal(size=(50, 50)),
        bias2=rng.normal(size=50),
        weight3=rng.normal(size=50),
        bias3=float(rng.normal()),
        scale=1.0,
    )
    coordinates = np.linspace(0.0, 1.0, 101)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    maximum = float(np.max(np.abs(realization.evaluate(x, y))))
    denominator = maximum * (1.0 + float(rng.random()))
    return SineNetworkBoundaryRealization(
        realization.weight1,
        realization.bias1,
        realization.weight2,
        realization.bias2,
        realization.weight3,
        realization.bias3,
        1.0 / max(denominator, 1.0e-12),
    )


BoundaryGenerator = BoundaryRealization | SineNetworkBoundaryRealization


def boundary_grid(realization: BoundaryGenerator, grid_size: int) -> np.ndarray:
    coordinates = np.linspace(0.0, 1.0, grid_size)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    values = np.zeros((grid_size, grid_size), dtype=np.float64)
    values[0, :] = realization.evaluate(x[0, :], y[0, :])
    values[-1, :] = realization.evaluate(x[-1, :], y[-1, :])
    values[:, 0] = realization.evaluate(x[:, 0], y[:, 0])
    values[:, -1] = realization.evaluate(x[:, -1], y[:, -1])
    return values


def pack_boundary(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            values[0, :-1],
            values[:-1, -1],
            values[-1, :0:-1],
            values[:0:-1, 0],
        ]
    )


def apply_negative_laplacian(interior: np.ndarray, spacing: float) -> np.ndarray:
    transformed = dstn(interior, type=1, norm="ortho")
    count = interior.shape[0]
    indices = np.arange(1, count + 1, dtype=np.float64)
    eigen_1d = 4.0 * np.sin(math.pi * indices / (2.0 * (count + 1))) ** 2 / spacing**2
    eigenvalues = eigen_1d[:, None] + eigen_1d[None, :]
    return idstn(transformed * eigenvalues, type=1, norm="ortho")


def solve_linear_dirichlet(rhs: np.ndarray, spacing: float) -> np.ndarray:
    transformed = dstn(rhs, type=1, norm="ortho")
    count = rhs.shape[0]
    indices = np.arange(1, count + 1, dtype=np.float64)
    eigen_1d = 4.0 * np.sin(math.pi * indices / (2.0 * (count + 1))) ** 2 / spacing**2
    eigenvalues = eigen_1d[:, None] + eigen_1d[None, :]
    return idstn(transformed / eigenvalues, type=1, norm="ortho")


def solve_source_free_pb(
    boundary: np.ndarray,
    tolerance: float = 1.0e-12,
    max_iterations: int = 100,
) -> tuple[np.ndarray, SolveDiagnostics]:
    grid_size = boundary.shape[0]
    if boundary.shape != (grid_size, grid_size) or grid_size < 3:
        raise ValueError("boundary must be a square grid with at least three nodes per axis")

    spacing = 1.0 / (grid_size - 1)
    forcing = np.zeros((grid_size - 2, grid_size - 2), dtype=np.float64)
    forcing[0, :] += boundary[0, 1:-1] / spacing**2
    forcing[-1, :] += boundary[-1, 1:-1] / spacing**2
    forcing[:, 0] += boundary[1:-1, 0] / spacing**2
    forcing[:, -1] += boundary[1:-1, -1] / spacing**2

    interior = solve_linear_dirichlet(forcing, spacing)
    relative_change = math.inf
    for iteration in range(1, max_iterations + 1):
        updated = solve_linear_dirichlet(forcing - np.sinh(interior), spacing)
        relative_change = float(
            np.linalg.norm(updated - interior) / max(np.linalg.norm(updated), 1.0e-15)
        )
        interior = updated
        if relative_change < tolerance:
            break
    else:
        raise RuntimeError(f"nonlinear iteration failed to converge on N={grid_size}")

    residual = apply_negative_laplacian(interior, spacing) + np.sinh(interior) - forcing
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(forcing), np.linalg.norm(interior), 1.0e-15)
    )

    solution = boundary.copy()
    solution[1:-1, 1:-1] = interior
    return solution, SolveDiagnostics(iteration, relative_change, relative_residual)


def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1.0e-15))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def plot_representative_sample(
    boundary: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    error: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_error = np.abs(prediction - reference)
    field_min = min(float(reference.min()), float(prediction.min()))
    field_max = max(float(reference.max()), float(prediction.max()))
    perimeter = np.linspace(0.0, 1.0, boundary.size, endpoint=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.45), constrained_layout=True)
    axes[0].plot(perimeter, boundary, color="#245a73", linewidth=1.4)
    for location in (0.25, 0.5, 0.75):
        axes[0].axvline(location, color="0.75", linewidth=0.7, linestyle="--")
    axes[0].set_xlabel("Normalized perimeter coordinate")
    axes[0].set_ylabel(r"Dirichlet data $g$")
    axes[0].set_title("Boundary input")
    axes[0].grid(alpha=0.18)

    images = []
    for axis, values, title in zip(
        axes[1:3], (reference, prediction), ("Numerical reference", "MAD0-FNO prediction")
    ):
        image = axis.imshow(
            values,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(title)
        images.append(image)
    figure.colorbar(images[-1], ax=axes[1:3], shrink=0.86, pad=0.02)

    error_image = axes[3].imshow(
        absolute_error,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    axes[3].set_xlabel(r"$x$")
    axes[3].set_ylabel(r"$y$")
    axes[3].set_title(rf"Absolute error ($L^2_{{rel}}={error:.2e}$)")
    figure.colorbar(error_image, ax=axes[3], shrink=0.86, pad=0.02)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_model_and_statistics(
    training_data_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    grid_size: int = 101,
    training_count: int = 1800,
) -> tuple[nn.ModuleDict, dict[str, float]]:
    data = np.load(training_data_path, mmap_mode="r")
    boundary_size = 4 * (grid_size - 1)
    field_size = grid_size * grid_size
    expected_width = boundary_size + 2 * field_size
    if data.shape[0] < training_count or data.shape[1] != expected_width:
        raise ValueError(f"unexpected training-data shape {data.shape}; expected (*, {expected_width})")

    train = data[:training_count]
    boundary = train[:, :boundary_size]
    source = train[:, boundary_size : boundary_size + field_size]
    solution = train[:, boundary_size + field_size :]
    statistics = {
        "boundary_mean": float(boundary.mean()),
        "boundary_std": float(boundary.std(ddof=1)),
        "source_mean": float(source.mean()),
        "source_std": float(source.std(ddof=1)),
        "solution_mean": float(solution.mean()),
        "solution_std": float(solution.std(ddof=1)),
    }

    model = nn.ModuleDict(
        {
            "lift": BoundaryToField(boundary_size, grid_size, width=256),
            "fno": FNO2d(in_dim=4, modes1=16, modes2=16, width=64),
        }
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, statistics


def predict(
    model: nn.ModuleDict,
    statistics: dict[str, float],
    boundaries: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    grid_size = 101
    x = torch.linspace(0.0, 1.0, grid_size, device=device)
    y = torch.linspace(0.0, 1.0, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coordinates = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    predictions: list[np.ndarray] = []

    for start in range(0, boundaries.shape[0], batch_size):
        stop = min(start + batch_size, boundaries.shape[0])
        boundary = torch.as_tensor(boundaries[start:stop], dtype=torch.float32, device=device)
        source = torch.zeros((stop - start, grid_size * grid_size), device=device)
        boundary_n = (boundary - statistics["boundary_mean"]) / statistics["boundary_std"]
        source_n = (source - statistics["source_mean"]) / statistics["source_std"]
        with torch.no_grad():
            lifted = model["lift"](boundary_n)
            inputs = torch.cat(
                [
                    lifted.unsqueeze(-1),
                    source_n.view(-1, grid_size, grid_size, 1),
                    coordinates.expand(stop - start, -1, -1, -1),
                ],
                dim=-1,
            )
            prediction_n = model["fno"](inputs)
            prediction = (
                prediction_n * statistics["solution_std"] + statistics["solution_mean"]
            )
        predictions.append(prediction.cpu().numpy().astype(np.float64))
    return np.concatenate(predictions, axis=0)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.reference_grid % 100 != 1 or args.reference_grid < 101:
        raise ValueError("reference-grid must be 101 + 100*k so it nests the 101 grid")
    if args.coarse_grid % 100 != 1 or args.coarse_grid >= args.reference_grid:
        raise ValueError("coarse-grid must be 101 + 100*k and smaller than reference-grid")

    rng = np.random.default_rng(args.seed)
    if args.boundary_family == "sine-network":
        realizations: list[BoundaryGenerator] = [
            sample_sine_network_boundary(rng) for _ in range(args.samples)
        ]
        boundary_description = {
            "family": "independent randomized sine-network traces matching the MAD0 generator family",
            "architecture": [2, 50, 50, 1],
            "normalization": "divide by max(abs(field)) * (1 + U(0,1))",
        }
    else:
        realizations = [
            sample_fourier_boundary(rng, terms=args.boundary_terms) for _ in range(args.samples)
        ]
        boundary_description = {
            "family": "independent smooth two-dimensional random Fourier traces",
            "terms": args.boundary_terms,
            "integer_frequencies_per_axis": [1, 4],
            "target_boundary_max_abs_range": [0.35, 0.80],
        }
    model_boundaries = np.stack(
        [pack_boundary(boundary_grid(realization, 101)) for realization in realizations]
    )

    references: list[np.ndarray] = []
    coarse_references: list[np.ndarray] = []
    solver_diagnostics: list[SolveDiagnostics] = []
    coarse_diagnostics: list[SolveDiagnostics] = []
    start = time.perf_counter()
    for index, realization in enumerate(realizations, start=1):
        fine, fine_diag = solve_source_free_pb(
            boundary_grid(realization, args.reference_grid), tolerance=args.solver_tolerance
        )
        coarse, coarse_diag = solve_source_free_pb(
            boundary_grid(realization, args.coarse_grid), tolerance=args.solver_tolerance
        )
        references.append(fine[:: (args.reference_grid - 1) // 100, :: (args.reference_grid - 1) // 100])
        coarse_references.append(
            coarse[:: (args.coarse_grid - 1) // 100, :: (args.coarse_grid - 1) // 100]
        )
        solver_diagnostics.append(fine_diag)
        coarse_diagnostics.append(coarse_diag)
        if index % max(1, args.samples // 5) == 0:
            print(f"solved {index}/{args.samples} boundary-value problems", flush=True)
    solver_seconds = time.perf_counter() - start

    reference = np.stack(references)
    coarse_reference = np.stack(coarse_references)
    refinement_errors = np.array(
        [relative_l2(coarse, fine) for coarse, fine in zip(coarse_reference, reference)]
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, statistics = load_model_and_statistics(
        Path(args.training_data), Path(args.checkpoint), device
    )
    inference_start = time.perf_counter()
    predictions = predict(model, statistics, model_boundaries, device, args.batch_size)
    inference_seconds = time.perf_counter() - inference_start

    full_errors = np.array(
        [relative_l2(prediction, truth) for prediction, truth in zip(predictions, reference)]
    )
    interior_errors = np.array(
        [
            relative_l2(prediction[1:-1, 1:-1], truth[1:-1, 1:-1])
            for prediction, truth in zip(predictions, reference)
        ]
    )
    boundary_errors = np.array(
        [
            relative_l2(pack_boundary(prediction), boundary)
            for prediction, boundary in zip(predictions, model_boundaries)
        ]
    )
    aggregate_error = relative_l2(predictions, reference)

    output = {
        "equation": "-Delta u + sinh(u) = 0",
        "samples": args.samples,
        "seed": args.seed,
        "boundary_distribution": boundary_description,
        "training_data": str(Path(args.training_data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "reference_grid": args.reference_grid,
        "coarse_grid": args.coarse_grid,
        "solver_tolerance": args.solver_tolerance,
        "solver_seconds": solver_seconds,
        "inference_seconds": inference_seconds,
        "solver_iterations": summarize(
            np.array([item.iterations for item in solver_diagnostics], dtype=np.float64)
        ),
        "solver_relative_residual": summarize(
            np.array([item.relative_residual for item in solver_diagnostics])
        ),
        "coarse_to_reference_relative_l2": summarize(refinement_errors),
        "fno_relative_l2": summarize(full_errors),
        "fno_interior_relative_l2": summarize(interior_errors),
        "fno_boundary_relative_l2": summarize(boundary_errors),
        "fno_aggregate_relative_l2": aggregate_error,
        "training_statistics": statistics,
        "per_sample": [
            {
                "index": index,
                "fno_relative_l2": float(full_errors[index]),
                "fno_interior_relative_l2": float(interior_errors[index]),
                "fno_boundary_relative_l2": float(boundary_errors[index]),
                "coarse_to_reference_relative_l2": float(refinement_errors[index]),
                "solver": asdict(solver_diagnostics[index]),
                "coarse_solver": asdict(coarse_diagnostics[index]),
            }
            for index in range(args.samples)
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    family_tag = args.boundary_family.replace("-", "_")
    stem = (
        f"source_free_pb_{family_tag}_seed{args.seed}_n{args.samples}_grid{args.reference_grid}"
    )
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        boundaries=model_boundaries,
        references=reference,
        predictions=predictions,
        errors=full_errors,
        refinement_errors=refinement_errors,
    )
    if args.figure_path:
        median_index = int(np.argsort(full_errors)[len(full_errors) // 2])
        plot_representative_sample(
            model_boundaries[median_index],
            reference[median_index],
            predictions[median_index],
            float(full_errors[median_index]),
            Path(args.figure_path),
        )
        output["representative_sample"] = {
            "index": median_index,
            "relative_l2": float(full_errors[median_index]),
            "figure": str(Path(args.figure_path).resolve()),
        }
        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "per_sample"}, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/r3_revision/source_free_pb")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--boundary-family", choices=("sine-network", "fourier"), default="sine-network"
    )
    parser.add_argument("--boundary-terms", type=int, default=12)
    parser.add_argument("--coarse-grid", type=int, default=201)
    parser.add_argument("--reference-grid", type=int, default=401)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--figure-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
