"""Generate MAD1 and independent FEM data on a flower domain with three holes.

The geometry and P1 finite-element stiffness matrix are reused from an
existing mesh. Training pairs are generated only from harmonic fundamental-
solution sums. The independent test set uses Fourier boundary data on all
four boundary components and discrete finite-element solutions. The optional
``train`` and ``evaluate`` stages retain an early diagnostic model; the model
reported in the R3 revision is trained by ``train_flower_holes_deeponet.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import factorized
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_original_protocol import LinearBranchDeepONet, set_seed


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPOSITORY
    / "reproducibility"
    / "r3"
    / "inputs"
    / "flower_geometry.mat"
)


@dataclass(frozen=True)
class Hole:
    center: np.ndarray
    major_radius: float
    minor_radius: float
    rotation: float


@dataclass(frozen=True)
class FlowerGeometry:
    nodes: np.ndarray
    triangles: np.ndarray
    stiffness: csc_matrix
    free_ids: np.ndarray
    outer_ids: np.ndarray
    inner_ids: np.ndarray
    hole_ids: np.ndarray
    outer_curve: np.ndarray
    inner_curves: tuple[np.ndarray, ...]
    holes: tuple[Hole, ...]
    center: np.ndarray
    base_radius: float
    petal_amplitude: float
    outer_rotation: float
    mesh_size: float

    @property
    def boundary_ids(self) -> np.ndarray:
        return np.concatenate((self.outer_ids, self.inner_ids))

    @property
    def boundary_nodes(self) -> np.ndarray:
        return self.nodes[self.boundary_ids]


class ArrayDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.data = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(np.asarray(self.data[index]), dtype=torch.float32)


def _curve_from_cell(values: np.ndarray, index: int) -> np.ndarray:
    item = np.asarray(values).reshape(-1)[index]
    return np.asarray(item, dtype=np.float64).reshape(-1)


def load_geometry(path: Path) -> FlowerGeometry:
    model = loadmat(path, squeeze_me=True, struct_as_record=False)
    stiffness = model["K"]
    if not issparse(stiffness):
        stiffness = np.asarray(stiffness, dtype=np.float64)
    nodes = np.asarray(model["p"], dtype=np.float64).T
    triangles = np.asarray(model["tri"], dtype=np.int64).T - 1
    free_ids = np.asarray(model["freeIds"], dtype=np.int64).reshape(-1) - 1
    outer_ids = np.asarray(model["bndIds_outer"], dtype=np.int64).reshape(-1) - 1
    inner_ids = np.asarray(model["bndIds_inner"], dtype=np.int64).reshape(-1) - 1
    hole_ids = np.asarray(model["hole_id"], dtype=np.int64).reshape(-1) - 1
    hole_structs = np.atleast_1d(model["holes"])
    holes = tuple(
        Hole(
            center=np.asarray(item.c, dtype=np.float64),
            major_radius=float(item.a),
            minor_radius=float(item.b),
            rotation=float(item.rot),
        )
        for item in hole_structs
    )
    inner_curves = tuple(
        np.column_stack(
            (
                _curve_from_cell(model["xi"], index),
                _curve_from_cell(model["yi"], index),
            )
        )
        for index in range(len(holes))
    )
    geometry = FlowerGeometry(
        nodes=nodes,
        triangles=triangles,
        stiffness=csc_matrix(stiffness, dtype=np.float64),
        free_ids=free_ids,
        outer_ids=outer_ids,
        inner_ids=inner_ids,
        hole_ids=hole_ids,
        outer_curve=np.column_stack(
            (
                np.asarray(model["xo"], dtype=np.float64).reshape(-1),
                np.asarray(model["yo"], dtype=np.float64).reshape(-1),
            )
        ),
        inner_curves=inner_curves,
        holes=holes,
        center=np.array([0.5, 0.5], dtype=np.float64),
        base_radius=float(model["R0"]),
        petal_amplitude=float(model["epsF"]),
        outer_rotation=float(model["rot_out"]),
        mesh_size=float(model["Hmax"]),
    )
    all_ids = np.concatenate((geometry.boundary_ids, geometry.free_ids))
    if len(np.unique(all_ids)) != len(nodes) or set(all_ids) != set(range(len(nodes))):
        raise ValueError("Boundary and free-node indices do not partition the mesh.")
    if len(geometry.inner_ids) != len(geometry.hole_ids):
        raise ValueError("Inner-boundary component labels are inconsistent.")
    return geometry


def normalize_pair(boundary: np.ndarray, solution: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(boundary)))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError(f"Invalid boundary normalization scale: {scale}")
    return np.concatenate((boundary, solution)) / scale


def source_contours(
    geometry: FlowerGeometry,
    outer_sources: int,
    sources_per_hole: int,
) -> np.ndarray:
    locations: list[np.ndarray] = []
    for theta in np.linspace(0.0, 2.0 * np.pi, outer_sources, endpoint=False):
        boundary_radius = geometry.base_radius * (
            1.0
            + geometry.petal_amplitude
            * np.cos(5.0 * (theta - geometry.outer_rotation))
        )
        radius = boundary_radius + 0.02
        locations.append(
            geometry.center + radius * np.array([np.cos(theta), np.sin(theta)])
        )
    for hole in geometry.holes:
        cosine = np.cos(hole.rotation)
        sine = np.sin(hole.rotation)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        for theta in np.linspace(
            0.0, 2.0 * np.pi, sources_per_hole, endpoint=False
        ):
            local = 0.90 * np.array(
                [hole.major_radius * np.cos(theta), hole.minor_radius * np.sin(theta)]
            )
            locations.append(hole.center + rotation @ local)
    return np.asarray(locations, dtype=np.float64)


def evaluate_fundamental_sum(
    coordinates: np.ndarray,
    locations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    difference = coordinates[:, None, :] - locations[None, :, :]
    radius_squared = np.sum(difference * difference, axis=2)
    if np.any(radius_squared <= 0.0):
        raise ValueError("A fundamental-solution center intersects an evaluation point.")
    return (0.5 * np.log(radius_squared)) @ weights


def generate_mad1(
    path: Path,
    geometry: FlowerGeometry,
    num_samples: int,
    seed: int,
    outer_sources: int,
    sources_per_hole: int,
) -> float:
    rng = np.random.default_rng(seed)
    locations = source_contours(geometry, outer_sources, sources_per_hole)
    kernel = np.column_stack(
        [
            evaluate_fundamental_sum(
                geometry.nodes,
                locations[index : index + 1],
                np.ones(1, dtype=np.float64),
            )
            for index in range(len(locations))
        ]
    )
    data = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, len(geometry.boundary_ids) + len(geometry.nodes)),
    )
    start = time.perf_counter()
    chunk_size = 100
    for start_index in range(0, num_samples, chunk_size):
        end_index = min(start_index + chunk_size, num_samples)
        weights = rng.standard_normal((end_index - start_index, len(locations)))
        weights /= np.sqrt(len(locations))
        solution = weights @ kernel.T
        boundary = solution[:, geometry.boundary_ids]
        scales = np.max(np.abs(boundary), axis=1)
        if np.any(~np.isfinite(scales)) or np.any(scales <= 1.0e-12):
            raise ValueError("Invalid normalization scale in MAD1 generation.")
        data[start_index:end_index] = np.concatenate(
            (boundary / scales[:, None], solution / scales[:, None]), axis=1
        )
    data.flush()
    return time.perf_counter() - start


def _component_parameter(coordinates: np.ndarray, hole: Hole | None) -> np.ndarray:
    if hole is None:
        shifted = coordinates - np.array([0.5, 0.5])
        return np.arctan2(shifted[:, 1], shifted[:, 0])
    shifted = coordinates - hole.center
    cosine = np.cos(hole.rotation)
    sine = np.sin(hole.rotation)
    local_x = cosine * shifted[:, 0] + sine * shifted[:, 1]
    local_y = -sine * shifted[:, 0] + cosine * shifted[:, 1]
    return np.arctan2(local_y / hole.minor_radius, local_x / hole.major_radius)


def smooth_fourier_values(
    rng: np.random.Generator,
    parameter: np.ndarray,
    modes: int,
) -> np.ndarray:
    values = np.full(len(parameter), rng.normal(scale=0.35), dtype=np.float64)
    for mode in range(1, modes + 1):
        scale = 1.0 / mode**2
        values += scale * (
            rng.normal() * np.cos(mode * parameter)
            + rng.normal() * np.sin(mode * parameter)
        )
    return values


def sample_independent_boundary(
    rng: np.random.Generator,
    geometry: FlowerGeometry,
    modes: int,
) -> np.ndarray:
    outer_coordinates = geometry.nodes[geometry.outer_ids]
    outer_parameter = _component_parameter(outer_coordinates, None)
    values = [smooth_fourier_values(rng, outer_parameter, modes)]
    inner_coordinates = geometry.nodes[geometry.inner_ids]
    for index, hole in enumerate(geometry.holes):
        selection = geometry.hole_ids == index
        parameter = _component_parameter(inner_coordinates[selection], hole)
        values.append(smooth_fourier_values(rng, parameter, modes))
    boundary = np.empty(len(geometry.boundary_ids), dtype=np.float64)
    boundary[: len(geometry.outer_ids)] = values[0]
    offset = len(geometry.outer_ids)
    for index, component_values in enumerate(values[1:]):
        selection = geometry.hole_ids == index
        component_positions = np.flatnonzero(selection)
        boundary[offset + component_positions] = component_values
    return boundary


def generate_fem_test(
    path: Path,
    geometry: FlowerGeometry,
    num_samples: int,
    seed: int,
    modes: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boundary_ids = geometry.boundary_ids
    free_ids = geometry.free_ids
    stiffness_ff = geometry.stiffness[free_ids][:, free_ids]
    stiffness_fb = geometry.stiffness[free_ids][:, boundary_ids]
    solve = factorized(csc_matrix(stiffness_ff))
    data = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, len(boundary_ids) + len(geometry.nodes)),
    )
    largest_relative_residual = 0.0
    start = time.perf_counter()
    for sample in range(num_samples):
        boundary = sample_independent_boundary(rng, geometry, modes)
        right_hand_side = -(stiffness_fb @ boundary)
        free_solution = solve(right_hand_side)
        solution = np.empty(len(geometry.nodes), dtype=np.float64)
        solution[boundary_ids] = boundary
        solution[free_ids] = free_solution
        residual = stiffness_ff @ free_solution + stiffness_fb @ boundary
        relative_residual = np.linalg.norm(residual) / max(
            np.linalg.norm(right_hand_side), 1.0e-15
        )
        largest_relative_residual = max(largest_relative_residual, relative_residual)
        data[sample] = normalize_pair(boundary, solution)
    data.flush()
    return time.perf_counter() - start, largest_relative_residual


def predict(
    model: LinearBranchDeepONet,
    boundary: torch.Tensor,
    trunk: torch.Tensor,
) -> torch.Tensor:
    branch_features = model.branch_net(boundary) * model.last_layer_weights
    trunk_features = model.trunk_net(trunk)
    return branch_features @ trunk_features.transpose(0, 1)


def train(
    data_path: Path,
    output_dir: Path,
    geometry: FlowerGeometry,
    epochs: int,
    seed: int,
    batch_size: int,
) -> Path:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = torch.tensor(geometry.nodes, dtype=torch.float32, device=device)
    model = LinearBranchDeepONet(len(geometry.boundary_ids), 2, 110).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.MSELoss()
    dataset = ArrayDataset(data_path)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model_path = output_dir / "flower_holes_laplace_best.pth"
    history_path = output_dir / "flower_holes_laplace_history.json"
    minimum_epoch_mse = float("inf")
    history: list[float] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        squared_loss_sum = 0.0
        row_count = 0
        for rows in loader:
            rows = rows.to(device)
            boundary = rows[:, : len(geometry.boundary_ids)]
            truth = rows[:, len(geometry.boundary_ids) :]
            loss = criterion(predict(model, boundary, trunk), truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_loss_sum += float(loss.detach().cpu()) * len(rows)
            row_count += len(rows)
        epoch_mse = squared_loss_sum / row_count
        history.append(epoch_mse)
        if epoch_mse < minimum_epoch_mse:
            minimum_epoch_mse = epoch_mse
            torch.save(model.state_dict(), model_path)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}: best epoch MSE={minimum_epoch_mse:.6e}, "
                f"epoch MSE={epoch_mse:.6e}",
                flush=True,
            )
    history_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "seed": seed,
                "batch_size": batch_size,
                "best_epoch_mse": minimum_epoch_mse,
                "elapsed_seconds": time.perf_counter() - start,
                "epoch_mse": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


@torch.inference_mode()
def evaluate(
    model_path: Path,
    test_paths: list[Path],
    geometry: FlowerGeometry,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = torch.tensor(geometry.nodes, dtype=torch.float32, device=device)
    model = LinearBranchDeepONet(len(geometry.boundary_ids), 2, 110).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for path in test_paths:
        errors: list[np.ndarray] = []
        difference_squared = 0.0
        reference_squared = 0.0
        loader = DataLoader(ArrayDataset(path), batch_size=batch_size, shuffle=False)
        for rows in loader:
            rows = rows.to(device)
            truth = rows[:, len(geometry.boundary_ids) :]
            prediction = predict(
                model, rows[:, : len(geometry.boundary_ids)], trunk
            )
            difference = prediction - truth
            errors.append(
                (
                    torch.linalg.vector_norm(difference, dim=1)
                    / torch.linalg.vector_norm(truth, dim=1).clamp_min(1.0e-12)
                )
                .cpu()
                .numpy()
            )
            difference_squared += float(torch.sum(difference.double() ** 2).cpu())
            reference_squared += float(torch.sum(truth.double() ** 2).cpu())
        sample_errors = np.concatenate(errors)
        results[path.stem] = {
            "mean_relative_l2": float(np.mean(sample_errors)),
            "std_relative_l2": float(np.std(sample_errors, ddof=1)),
            "median_relative_l2": float(np.median(sample_errors)),
            "maximum_relative_l2": float(np.max(sample_errors)),
            "aggregate_relative_l2": float(
                np.sqrt(difference_squared / reference_squared)
            ),
            "representative_median_sample_index": int(
                np.argmin(np.abs(sample_errors - np.median(sample_errors)))
            ),
        }
    return results


def _plot_boundaries(axis: plt.Axes, geometry: FlowerGeometry) -> None:
    outer = np.vstack((geometry.outer_curve, geometry.outer_curve[0]))
    axis.plot(outer[:, 0], outer[:, 1], color="black", linewidth=1.0)
    for curve in geometry.inner_curves:
        closed = np.vstack((curve, curve[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0)


@torch.inference_mode()
def plot_example(
    model_path: Path,
    test_path: Path,
    output_path: Path,
    geometry: FlowerGeometry,
    sample_index: int,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = torch.tensor(geometry.nodes, dtype=torch.float32, device=device)
    model = LinearBranchDeepONet(len(geometry.boundary_ids), 2, 110).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    row = np.asarray(np.load(test_path, mmap_mode="r")[sample_index])
    boundary = torch.tensor(
        row[: len(geometry.boundary_ids)], dtype=torch.float32, device=device
    ).unsqueeze(0)
    truth = row[len(geometry.boundary_ids) :]
    prediction = predict(model, boundary, trunk).squeeze(0).cpu().numpy()
    error = np.abs(prediction - truth)
    relative_error = float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
    triangulation = mtri.Triangulation(
        geometry.nodes[:, 0], geometry.nodes[:, 1], geometry.triangles
    )
    shared_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))))
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.15), constrained_layout=True)
    axes[0].triplot(triangulation, color="0.72", linewidth=0.25)
    _plot_boundaries(axes[0], geometry)
    axes[0].set_title("Fixed computational domain")
    for axis, field, title in (
        (axes[1], truth, "Finite-element reference"),
        (axes[2], prediction, "MAD1--DeepONet prediction"),
    ):
        image = axis.tripcolor(
            triangulation,
            field,
            shading="gouraud",
            cmap="coolwarm",
            vmin=-shared_limit,
            vmax=shared_limit,
        )
        _plot_boundaries(axis, geometry)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    error_image = axes[3].tripcolor(
        triangulation,
        error,
        shading="gouraud",
        cmap="magma",
        vmin=0.0,
    )
    _plot_boundaries(axes[3], geometry)
    axes[3].set_title(f"Absolute error (relative $L^2$={relative_error:.2e})")
    figure.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_aspect("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return relative_error


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "train", "evaluate"),
        default="generate",
    )
    parser.add_argument("--model-mat", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repository / "data" / "r3_revision" / "flower_holes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "artifacts" / "r3_revision" / "flower_holes",
    )
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--outer-sources", type=int, default=100)
    parser.add_argument("--sources-per-hole", type=int, default=40)
    parser.add_argument("--fourier-modes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    geometry = load_geometry(args.model_mat)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "mad1_train_n2000.npy"
    mad_test_path = args.data_dir / "mad1_test_n200.npy"
    fem_test_path = args.data_dir / "fem_fourier_test_n200.npy"
    model_path = args.output_dir / "flower_holes_laplace_best.pth"

    if args.stage in ("all", "generate"):
        train_seconds = generate_mad1(
            train_path,
            geometry,
            args.train_samples,
            args.seed,
            args.outer_sources,
            args.sources_per_hole,
        )
        mad_test_seconds = generate_mad1(
            mad_test_path,
            geometry,
            args.test_samples,
            args.seed + 1,
            args.outer_sources,
            args.sources_per_hole,
        )
        fem_test_seconds, maximum_residual = generate_fem_test(
            fem_test_path,
            geometry,
            args.test_samples,
            args.seed + 2,
            args.fourier_modes,
        )
        generation = {
            "geometry": {
                "nodes": len(geometry.nodes),
                "triangles": len(geometry.triangles),
                "boundary_nodes": len(geometry.boundary_ids),
                "outer_boundary_nodes": len(geometry.outer_ids),
                "inner_boundary_nodes": len(geometry.inner_ids),
                "holes": len(geometry.holes),
                "mesh_size": geometry.mesh_size,
            },
            "settings": {
                "train_samples": args.train_samples,
                "test_samples": args.test_samples,
                "outer_sources": args.outer_sources,
                "sources_per_hole": args.sources_per_hole,
                "fourier_modes": args.fourier_modes,
                "seed": args.seed,
            },
            "timings_seconds": {
                "mad1_train": train_seconds,
                "mad1_test": mad_test_seconds,
                "fem_test": fem_test_seconds,
            },
            "fem_maximum_relative_algebraic_residual": maximum_residual,
        }
        (args.data_dir / "generation.json").write_text(
            json.dumps(generation, indent=2), encoding="utf-8"
        )
        print(json.dumps(generation, indent=2))

    if args.stage in ("all", "train"):
        model_path = train(
            train_path,
            args.output_dir,
            geometry,
            args.epochs,
            args.seed,
            args.batch_size,
        )

    if args.stage in ("all", "evaluate"):
        results = evaluate(
            model_path,
            [mad_test_path, fem_test_path],
            geometry,
            args.batch_size,
        )
        sample_index = int(
            results[fem_test_path.stem]["representative_median_sample_index"]
        )
        illustrated_error = plot_example(
            model_path,
            fem_test_path,
            args.output_dir / "flower_holes_laplace_fem_example.pdf",
            geometry,
            sample_index,
        )
        results["illustrated_fem_sample"] = {
            "sample_index": sample_index,
            "relative_l2": illustrated_error,
        }
        (args.output_dir / "flower_holes_laplace_evaluation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
