"""Train the DeepONet reported for the R3 flower-domain experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_holes_laplace_experiment import DEFAULT_MODEL, load_geometry
from scalar_normalizer import ScalarNormalizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_l2_loss(
    prediction_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    normalizer: ScalarNormalizer,
) -> torch.Tensor:
    prediction = normalizer.decode(prediction_normalized)
    truth = normalizer.decode(truth_normalized)
    numerator = torch.linalg.vector_norm(prediction - truth, dim=1)
    denominator = torch.linalg.vector_norm(truth, dim=1).clamp_min(1.0e-12)
    return torch.mean(numerator / denominator)


def adamw_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name.lower() or "scale" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        width: int = 256,
        depth: int = 5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError("activation must be tanh, relu, or gelu")
        activation_layer = activations[activation]
        layers: list[nn.Module] = [
            nn.Linear(input_dimension, width),
            activation_layer(),
        ]
        for _ in range(depth - 2):
            layers.extend((nn.Linear(width, width), activation_layer()))
        layers.append(nn.Linear(width, output_dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class DeepONetNodes(nn.Module):
    def __init__(
        self,
        Nb: int,
        p: int = 1024,
        branch_width: int = 256,
        branch_depth: int = 5,
        trunk_width: int = 256,
        trunk_depth: int = 5,
        act: str = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.branch = MLP(Nb, p, branch_width, branch_depth, act)
        self.trunk = MLP(2, p, trunk_width, trunk_depth, act)
        self.use_bias = bool(use_bias)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, boundary_values: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        output = self.branch(boundary_values) @ self.trunk(coordinates).T
        if self.use_bias:
            output = output + self.bias
        return output


def validation_error(
    model: DeepONetNodes,
    loader: DataLoader,
    coordinates: torch.Tensor,
    normalizer: ScalarNormalizer,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for boundary, truth in loader:
            boundary = boundary.to(device)
            truth = truth.to(device)
            loss = relative_l2_loss(model(boundary, coordinates), truth, normalizer)
            total += float(loss) * len(boundary)
            count += len(boundary)
    return total / count


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=repository
        / "data"
        / "r3_revision"
        / "flower_holes"
        / "mad1_train_n2000.npy",
    )
    parser.add_argument("--model-mat", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "artifacts"
        / "r3_revision"
        / "flower_holes"
        / "deeponet_nodes",
    )
    parser.add_argument("--training-samples", type=int, default=1800)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--save-frequency", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = load_geometry(args.model_mat)
    rows = torch.tensor(np.load(args.data), dtype=torch.float32)
    boundary_count = len(geometry.boundary_ids)
    required = args.training_samples + args.validation_samples
    if len(rows) < required:
        raise ValueError(f"dataset contains {len(rows)} rows; {required} are required")
    if rows.shape[1] != boundary_count + len(geometry.nodes):
        raise ValueError("dataset width does not match the geometry")

    training = rows[: args.training_samples]
    validation = rows[args.training_samples : required]
    input_normalizer = ScalarNormalizer(training[:, :boundary_count])
    output_normalizer = ScalarNormalizer(training[:, boundary_count:])
    training_loader = DataLoader(
        TensorDataset(
            input_normalizer.encode(training[:, :boundary_count]),
            output_normalizer.encode(training[:, boundary_count:]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            input_normalizer.encode(validation[:, :boundary_count]),
            output_normalizer.encode(validation[:, boundary_count:]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    coordinates = torch.tensor(geometry.nodes, dtype=torch.float32, device=device)
    architecture = {
        "p": 1024,
        "branch_width": 256,
        "branch_depth": 5,
        "trunk_width": 256,
        "trunk_depth": 5,
        "act": "tanh",
        "use_bias": True,
    }
    model = DeepONetNodes(Nb=boundary_count, **architecture).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay), lr=args.learning_rate
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "flower_holes_mad1_fixed1800_200_e5000"
    best_path = args.output_dir / f"{prefix}_best2.pt"
    checkpoint_path = args.output_dir / f"{prefix}_checkpoint2.pt"
    best_validation = float("inf")
    history_training: list[float] = []
    history_validation: list[float] = []
    history_epoch_seconds: list[float] = []
    total_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for boundary, truth in training_loader:
            boundary = boundary.to(device)
            truth = truth.to(device)
            optimizer.zero_grad()
            loss = relative_l2_loss(
                model(boundary, coordinates), truth, output_normalizer
            )
            loss.backward()
            optimizer.step()
            total += float(loss) * len(boundary)
            count += len(boundary)
        training_error = total / count
        current_validation = validation_error(
            model,
            validation_loader,
            coordinates,
            output_normalizer,
            device,
        )
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        history_training.append(training_error)
        history_validation.append(current_validation)
        history_epoch_seconds.append(elapsed)

        if current_validation < best_validation:
            best_validation = current_validation
            torch.save(model.state_dict(), best_path)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch={epoch} train={training_error:.6e} "
                f"validation={current_validation:.6e} elapsed={total_seconds:.1f}s"
            )
        if epoch % args.save_frequency == 0 or epoch == args.epochs:
            checkpoint = {
                "epoch": epoch,
                "best_test_rel": best_validation,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_name": "flower_holes_mad1_deeponet_nodes",
                "optimizer_name": "AdamW",
                "param_count": parameter_count,
                "data_path": str(args.data),
                "xy_nodes_path": str(args.model_mat),
                "Nb": boundary_count,
                "Np": len(geometry.nodes),
                "seed": args.seed,
                "shuffle": True,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "lr_init": args.learning_rate,
                "epochs": args.epochs,
                "split": {
                    "mode": "fixed",
                    "ntrain": args.training_samples,
                    "ntest": args.validation_samples,
                },
                "arch": architecture,
                "x_mean": input_normalizer.mean,
                "x_std": input_normalizer.std,
                "y_mean": output_normalizer.mean,
                "y_std": output_normalizer.std,
                "history_train_rel": history_training,
                "history_test_rel": history_validation,
                "history_epoch_time": history_epoch_seconds,
                "total_train_time": total_seconds,
            }
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)

    summary = {
        "best_validation_mean_relative_l2": best_validation,
        "parameter_count": parameter_count,
        "training_wall_time_seconds": total_seconds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
