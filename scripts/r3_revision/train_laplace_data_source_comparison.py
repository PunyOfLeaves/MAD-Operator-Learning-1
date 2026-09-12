"""Train the original source-free Laplace DeepONet on a selected dataset.

The architecture, supervised loss, optimizer, batch size, and minimum-batch-
loss checkpoint rule match ``trainMADlaplace2D.py``.  Repeated evaluation of
identical branch and trunk features is replaced by an algebraically equivalent
matrix product so the controlled R3 runs finish in practical time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_original_protocol import LinearBranchDeepONet, TextDataset, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-functions", type=int, default=2000)
    parser.add_argument("--num-points", type=int, default=51)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def predict_grid(
    model: LinearBranchDeepONet,
    boundary: torch.Tensor,
    trunk: torch.Tensor,
) -> torch.Tensor:
    branch_features = model.branch_net(boundary) * model.last_layer_weights
    trunk_features = model.trunk_net(trunk)
    return branch_features @ trunk_features.transpose(0, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.run_name}_best.pth"
    checkpoint_path = args.output_dir / f"{args.run_name}_checkpoint.pth"
    history_path = args.output_dir / f"{args.run_name}_history.json"
    if args.resume and (not checkpoint_path.is_file() or not model_path.is_file()):
        raise FileNotFoundError("Resume requires both the latest checkpoint and retained best model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = 1.0 / (args.num_points - 1)
    axis = np.arange(0.0, 1.0 + spacing, spacing)
    xx, yy = np.meshgrid(axis, axis)
    trunk = torch.tensor(
        np.column_stack((xx.ravel(), yy.ravel())),
        dtype=torch.float32,
        device=device,
    )

    model = LinearBranchDeepONet(
        4 * (args.num_points - 1), 2, 110
    ).to(device)
    model = torch.nn.DataParallel(model)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=50, factor=0.95
    )

    dataset = TextDataset(args.data)
    if len(dataset) != args.num_functions:
        raise ValueError(
            f"Expected {args.num_functions} samples, found {len(dataset)} in {args.data}"
        )
    expected_columns = 4 * (args.num_points - 1) + args.num_points * args.num_points
    if int(np.asarray(dataset.data).shape[1]) != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {args.data}, "
            f"found {np.asarray(dataset.data).shape[1]}"
        )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
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

    boundary_count = 4 * (args.num_points - 1)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        last_loss = float("nan")
        for inputs in dataloader:
            inputs = inputs.to(device)
            prediction = predict_grid(model.module, inputs[:, :boundary_count], trunk)
            true = inputs[:, boundary_count:]
            loss_value = criterion(prediction, true)
            last_loss = loss_value.item()
            if not math.isfinite(last_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            if last_loss < minimum_loss:
                torch.save(model.module.state_dict(), model_path)
                minimum_loss = last_loss
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        if (epoch + 1) % args.print_every == 0 or epoch == start_epoch:
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

    total_time = elapsed_time + (time.time() - start_time)
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
