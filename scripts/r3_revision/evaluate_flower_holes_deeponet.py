"""Evaluate the trained flower-hole DeepONet on independent MAD1 and FEM tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from flower_holes_laplace_experiment import DEFAULT_MODEL, load_geometry
from train_flower_holes_deeponet import DeepONetNodes


def evaluate(
    model: DeepONetNodes,
    rows: torch.Tensor,
    coordinates: torch.Tensor,
    boundary_count: int,
    input_mean: float,
    input_std: float,
    output_mean: float,
    output_std: float,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    errors: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    difference_squared = 0.0
    reference_squared = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            boundary = (batch[:, :boundary_count] - input_mean) / input_std
            truth = batch[:, boundary_count:]
            prediction_normalized = model(boundary, coordinates)
            prediction = prediction_normalized * output_std + output_mean
            difference = prediction - truth
            batch_errors = (
                torch.linalg.vector_norm(difference, dim=1)
                / torch.linalg.vector_norm(truth, dim=1).clamp_min(1.0e-12)
            )
            errors.append(batch_errors.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            difference_squared += float(torch.sum(difference.double() ** 2).cpu())
            reference_squared += float(torch.sum(truth.double() ** 2).cpu())
    error_array = np.concatenate(errors)
    prediction_array = np.concatenate(predictions)
    statistics = {
        "mean_relative_l2": float(np.mean(error_array)),
        "std_relative_l2": float(np.std(error_array, ddof=1)),
        "median_relative_l2": float(np.median(error_array)),
        "maximum_relative_l2": float(np.max(error_array)),
        "aggregate_relative_l2": float(
            np.sqrt(difference_squared / reference_squared)
        ),
    }
    return statistics, error_array, prediction_array


def plot_example(
    geometry,
    truth: np.ndarray,
    prediction: np.ndarray,
    error: float,
    output: Path,
) -> None:
    triangulation = mtri.Triangulation(
        geometry.nodes[:, 0], geometry.nodes[:, 1], geometry.triangles
    )
    difference = np.abs(prediction - truth)
    bound = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))))
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.1), constrained_layout=True)
    axes[0].triplot(triangulation, color="0.72", linewidth=0.23)
    axes[0].set_title("Fixed computational domain")
    for curve in (geometry.outer_curve, *geometry.inner_curves):
        closed = np.vstack((curve, curve[0]))
        axes[0].plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0)
    for axis, field, title in (
        (axes[1], truth, "Finite-element reference"),
        (axes[2], prediction, "MAD1--DeepONet prediction"),
    ):
        image = axis.tripcolor(
            triangulation,
            field,
            shading="gouraud",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
    error_image = axes[3].tripcolor(
        triangulation, difference, shading="gouraud", cmap="magma"
    )
    figure.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title(f"Absolute error (rel. $L^2$={error:.3f})")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-mat", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repository / "data" / "r3_revision" / "flower_holes",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repository
        / "artifacts"
        / "r3_revision"
        / "flower_holes"
        / "deeponet_nodes",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = load_geometry(args.model_mat)
    coordinates = torch.tensor(geometry.nodes, dtype=torch.float32, device=device)
    checkpoint_path = (
        args.model_dir / "flower_holes_mad1_fixed1800_200_e5000_checkpoint2.pt"
    )
    best_path = args.model_dir / "flower_holes_mad1_fixed1800_200_e5000_best2.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DeepONetNodes(Nb=checkpoint["Nb"], **checkpoint["arch"]).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    results: dict[str, dict[str, float]] = {}
    saved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, filename in (
        ("independent_mad1", "mad1_test_n200.npy"),
        ("independent_fem_fourier", "fem_fourier_test_n200.npy"),
    ):
        rows_numpy = np.asarray(np.load(args.data_dir / filename), dtype=np.float32)
        rows = torch.from_numpy(rows_numpy).to(device)
        statistics, errors, predictions = evaluate(
            model,
            rows,
            coordinates,
            checkpoint["Nb"],
            checkpoint["x_mean"],
            checkpoint["x_std"] + 1.0e-8,
            checkpoint["y_mean"],
            checkpoint["y_std"] + 1.0e-8,
            args.batch_size,
        )
        results[name] = statistics
        saved[name] = (errors, predictions)

    fem_rows = np.asarray(
        np.load(args.data_dir / "fem_fourier_test_n200.npy"), dtype=np.float32
    )
    fem_errors, fem_predictions = saved["independent_fem_fourier"]
    median_index = int(np.argmin(np.abs(fem_errors - np.median(fem_errors))))
    plot_example(
        geometry,
        fem_rows[median_index, checkpoint["Nb"] :],
        fem_predictions[median_index],
        float(fem_errors[median_index]),
        args.model_dir / "flower_holes_fem_median_example.pdf",
    )
    report = {
        "checkpoint_selection": {
            "validation_samples": 200,
            "best_mean_relative_l2": checkpoint["best_test_rel"],
        },
        "training": {
            "samples": 1800,
            "epochs": checkpoint["epoch"],
            "seed": checkpoint["seed"],
            "batch_size": checkpoint["batch_size"],
            "architecture": checkpoint["arch"],
            "parameter_count": checkpoint["param_count"],
            "wall_time_seconds": checkpoint["total_train_time"],
        },
        "tests": results,
        "illustrated_fem_sample": {
            "index": median_index,
            "relative_l2": float(fem_errors[median_index]),
        },
    }
    (args.model_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
