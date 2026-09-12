"""Run the original 2D Helmholtz DeepONet protocol on a selected dataset.

Only the dataset path, random seed, epoch count, and output paths are exposed
as arguments. The network, loss, optimizer, batching, and best-model rule match
scripts/model_training/trainMADhelmholtz2D.py.
"""

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
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, file_path: Path) -> None:
        if file_path.suffix.lower() == ".npy":
            self.data = np.load(file_path, mmap_mode="r")
        else:
            with file_path.open("r", encoding="utf-8") as file:
                self.data = np.asarray(
                    [
                        [float(value) for value in line.strip().split()]
                        for line in file
                        if line.strip()
                    ],
                    dtype=np.float32,
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(np.asarray(self.data[index]), dtype=torch.float32)


class LinearBranchDeepONet(nn.Module):
    def __init__(self, branch_features: int, trunk_features: int, common_features: int) -> None:
        super().__init__()
        self.branch_net = nn.Sequential(nn.Linear(branch_features, 1000, bias=False))
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, 1000),
        )
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(
            branch_output * trunk_output * self.last_layer_weights,
            dim=1,
            keepdim=True,
        )


class PoissonONet(nn.Module):
    def __init__(
        self,
        branch1_features: int,
        branch2_features: int,
        trunk_features: int,
        common_features1: int,
        common_features2: int,
    ) -> None:
        super().__init__()
        self.net1 = LinearBranchDeepONet(
            branch1_features, trunk_features, common_features1
        )
        self.net2 = LinearBranchDeepONet(
            branch2_features, trunk_features, common_features2
        )

    def forward(
        self,
        branch1_input: torch.Tensor,
        branch2_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.net1(branch1_input, trunk_input) + self.net2(
            branch2_input, trunk_input
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-functions", type=int, default=2000)
    parser.add_argument("--num-points", type=int, default=51)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.run_name}_best.pth"
    checkpoint_path = args.output_dir / f"{args.run_name}_checkpoint.pth"
    history_path = args.output_dir / f"{args.run_name}_history.json"

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = 1.0 / (args.num_points - 1)
    axis = np.arange(0.0, 1.0 + spacing, spacing)
    x_grid, y_grid = np.meshgrid(axis, axis)
    grid_points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

    model = PoissonONet(
        4 * (args.num_points - 1),
        args.num_points * args.num_points,
        2,
        90,
        110,
    ).to(device)
    model = torch.nn.DataParallel(model)
    criterion = nn.MSELoss().to(device)

    dataset = TextDataset(args.data)
    if len(dataset) != args.num_functions:
        raise ValueError(
            f"Expected {args.num_functions} samples, found {len(dataset)} in {args.data}"
        )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=50, factor=0.95
    )

    start_epoch = 0
    minimum_loss = 1.0
    loss_records: list[float] = []
    elapsed_time = 0.0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        minimum_loss = float(checkpoint["min_loss"])
        loss_records = list(checkpoint["loss_records"])
        elapsed_time = float(checkpoint["elapsed_time"])
        if "loader_generator_state" in checkpoint:
            loader_generator.set_state(checkpoint["loader_generator_state"].cpu())

    def label_loss(inputs: torch.Tensor) -> torch.Tensor:
        num_rows = inputs.size(0)
        point_count = args.num_points * args.num_points
        boundary_count = 4 * (args.num_points - 1)
        trunk_batch = trunk.repeat(num_rows, 1)
        true_batch = inputs[:, point_count + boundary_count :].reshape(-1, 1).to(device)
        branch1_batch = (
            inputs[:, :boundary_count]
            .unsqueeze(1)
            .repeat(1, point_count, 1)
            .view(-1, boundary_count)
            .to(device)
        )
        branch2_batch = (
            inputs[:, boundary_count : point_count + boundary_count]
            .unsqueeze(1)
            .repeat(1, point_count, 1)
            .view(-1, point_count)
            .to(device)
        )
        prediction = model(branch1_batch, branch2_batch, trunk_batch)
        return criterion(prediction, true_batch)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        last_loss = float("nan")
        for batch in dataloader:
            inputs = batch.to(device)
            loss_value = label_loss(inputs)
            last_loss = loss_value.item()
            if last_loss < minimum_loss:
                torch.save(model.module.state_dict(), model_path)
                minimum_loss = last_loss
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}], Minimum Loss: {minimum_loss:.6e}, "
            f"Last-batch Loss: {last_loss:.6e}",
            flush=True,
        )
        loss_records.append(last_loss)
        if (epoch + 1) % 10 == 0:
            end_time = time.time()
            elapsed_time += end_time - start_time
            start_time = end_time
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "min_loss": minimum_loss,
                "loss_records": loss_records,
                "elapsed_time": elapsed_time,
                "loader_generator_state": loader_generator.get_state(),
            }
            torch.save(checkpoint, checkpoint_path)
            history_path.write_text(
                json.dumps(
                    {
                        "data": str(args.data),
                        "seed": args.seed,
                        "epoch": epoch + 1,
                        "minimum_batch_loss": minimum_loss,
                        "last_batch_losses": loss_records,
                        "elapsed_time": elapsed_time,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    end_time = time.time()
    total_time = elapsed_time + (end_time - start_time)
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
