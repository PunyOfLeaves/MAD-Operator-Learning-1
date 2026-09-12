"""Paired MAD/physics-informed DeepONet with common boundary-only scaling."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from molecular_laplace_experiment import digest, write_json
from scalar_normalizer import ScalarNormalizer
from train_molecular_laplace import MolecularDeepONet, atomic_save, evaluate
from train_original_protocol import set_seed


def trunk_laplacian(trunk, coordinates):
    """Exact chain rule for this pointwise Linear/Tanh trunk; no FD or sampling.

    Propagating all three first derivatives and the Laplacian avoids repeating
    trunk evaluation for every boundary function. Parameter gradients still use
    PyTorch autograd. Tests compare both residuals and parameter gradients to AD.
    """
    value = coordinates
    first = torch.eye(3, dtype=value.dtype, device=value.device).expand(len(value), -1, -1)
    lap = torch.zeros_like(value)
    layers = list(trunk)
    for index, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            lap = F.linear(lap, layer.weight, None)
            if index == len(layers)-1:
                return lap
            value = layer(value)
            first = F.linear(first, layer.weight, None)
        elif isinstance(layer, nn.Tanh):
            value = torch.tanh(value)
            slope = 1-value.square()
            lap = slope*lap-2*value*slope*first.square().sum(dim=1)
            first = slope[:, None, :]*first
        else:
            raise TypeError("Unsupported trunk layer: " + type(layer).__name__)
    raise ValueError("Expected a final linear layer")


def objective(model, boundary, coordinates, boundary_coordinates, method, truth=None):
    coefficients = model.branch_net(boundary)*model.last_layer_weights
    boundary_prediction = coefficients @ model.trunk_net(boundary_coordinates).T
    boundary_mse = F.mse_loss(boundary_prediction, boundary)
    if method == "pi":
        residual = coefficients @ trunk_laplacian(model.trunk_net, coordinates).T
        interior_mse = residual.square().mean()
        total = .9*interior_mse+.1*boundary_mse
    elif method == "mad":
        if truth is None:
            raise ValueError("MAD needs interior labels")
        interior_mse = F.mse_loss(coefficients @ model.trunk_net(coordinates).T, truth)
        # Ordinary MSE over the concatenated point set, without tuned weights.
        total = (len(coordinates)*interior_mse+len(boundary_coordinates)*boundary_mse) / (
            len(coordinates)+len(boundary_coordinates))
    else:
        raise ValueError(method)
    return total, interior_mse, boundary_mse


def load_fields(data, method):
    arrays = {}
    for split in ("train", "validation"):
        with np.load(data/(split+".npz")) as raw:
            arrays[split] = {"boundary": torch.from_numpy(raw["boundary"])}
            if method == "mad":
                arrays[split]["solution"] = torch.from_numpy(raw["solution"])
    return arrays


def train(data, run, method, paired_protocol, resume=False, stop_after=None, require_cuda=False):
    protocol = json.loads(paired_protocol.read_text())
    if method not in ("mad", "pi"):
        raise ValueError(method)
    if run.exists() and not resume:
        raise FileExistsError("Use a fresh run; preserve previous evidence")
    if resume and (not (run/"checkpoint.pt").exists() or (run/"freeze.json").exists()):
        raise ValueError("Resume requires an unfinished checkpoint")
    run.mkdir(parents=True, exist_ok=resume)
    set_seed(protocol["model_seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and (device.type != "cuda" or "3090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Formal run requires an RTX 3090")
    raw = np.load(data/"geometry.npz")
    coordinates = torch.tensor((raw["train_queries"]-raw["center"])/raw["scale"], dtype=torch.float32, device=device)
    bc = torch.tensor((raw["nodes"][raw["boundary_ids"]]-raw["center"])/raw["scale"], dtype=torch.float32, device=device)
    arrays = load_fields(data, method)
    normalizer = ScalarNormalizer(arrays["train"]["boundary"])
    x = normalizer.encode(arrays["train"]["boundary"])
    vx = normalizer.encode(arrays["validation"]["boundary"]).to(device)
    vy = normalizer.encode(arrays["validation"]["solution"]).to(device) if method == "mad" else None
    fields = [x]
    if method == "mad":
        fields.append(normalizer.encode(arrays["train"]["solution"]))
    loader = DataLoader(TensorDataset(*fields), batch_size=protocol["batch_size"], shuffle=True)
    hidden_width = protocol.get("hidden_width", 110)
    model = MolecularDeepONet(x.shape[1], hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])
    meta = {"method": method, "epochs": protocol["epochs"], "model_seed": protocol["model_seed"],
            "pilot": False, "device": str(device), "hidden_width": hidden_width,
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "x_mean": normalizer.mean, "x_std": normalizer.std,
            "y_mean": normalizer.mean, "y_std": normalizer.std, "normalizer_eps": normalizer.eps,
            "normalization_statistics_source": "training boundary inputs only, identical for both arms",
            "interior_training_and_validation_labels_loaded": method == "mad",
            "boundary_count": x.shape[1], "protocol_sha256": digest(data/"protocol.json"),
            "paired_protocol_sha256": digest(paired_protocol), "training_script_sha256": digest(__file__),
            "data_hashes": {name: digest(data/name) for name in ("geometry.npz", "train.npz", "validation.npz")},
            "started_utc": datetime.now(timezone.utc).isoformat(), "test_data_loaded_during_training": False,
            "selection": "Minimum held-out validation objective of this method; PI uses PDE residual and boundary data only",
            "timing_scope": "Training loop including validation and checkpoint I/O; excludes setup and testing",
            "pde_coordinates": "xi=(x-center)/largest_span; residual is Laplacian_xi of boundary-standardized output",
            "mad_loss": "Equal pointwise MSE on concatenated interior and boundary points",
            "pi_loss": "0.9*MSE(Laplacian_xi(z),0)+0.1*MSE(z_boundary,g_standardized)"}
    history, best_model = [], None
    best, best_epoch, first_epoch, previous_wall = float("inf"), 0, 1, 0.
    if resume:
        saved = torch.load(run/"checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ("method", "paired_protocol_sha256", "protocol_sha256", "data_hashes", "epochs",
                    "training_script_sha256", "torch_version", "numpy_version", "device_name"):
            if saved["metadata"][key] != meta[key]:
                raise ValueError("Resume mismatch: " + key)
        meta = saved["metadata"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_model = saved["history"], saved["best_model"]
        best, best_epoch = saved["best_validation_mse"], saved["best_epoch"]
        first_epoch, previous_wall = saved["epoch"]+1, history[-1]["wall_seconds"]
        atomic_save(best_model, run/"best.pt")
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        write_json(run/"run.json", meta)
        atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run/"initial.pt")
    last_epoch = min(protocol["epochs"], stop_after) if stop_after is not None else protocol["epochs"]
    if last_epoch < first_epoch:
        raise ValueError("Stopping epoch precedes checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(first_epoch, last_epoch+1):
        model.train()
        totals = np.zeros(3)
        for batch in loader:
            bx = batch[0].to(device)
            by = batch[1].to(device) if method == "mad" else None
            optimizer.zero_grad(set_to_none=True)
            losses = objective(model, bx, coordinates, bc, method, by)
            if not torch.isfinite(losses[0]):
                raise ValueError("Nonfinite training loss")
            losses[0].backward()
            optimizer.step()
            totals += np.array([float(v.detach()) for v in losses])*len(bx)
        model.eval()
        with torch.no_grad():
            validation = tuple(float(v) for v in objective(model, vx, coordinates, bc, method, vy))
        if not np.isfinite(validation).all():
            raise ValueError("Nonfinite validation objective")
        totals /= len(x)
        history.append({"epoch": epoch, "train_objective": float(totals[0]),
                        "train_interior_term": float(totals[1]), "train_boundary_mse": float(totals[2]),
                        "validation_objective": validation[0], "validation_interior_term": validation[1],
                        "validation_boundary_mse": validation[2],
                        "wall_seconds": previous_wall+time.perf_counter()-start})
        if validation[0] < best:
            best, best_epoch = validation[0], epoch
            best_model = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "epoch": epoch, "validation_mse": best, "metadata": meta}
            atomic_save(best_model, run/"best.pt")
        if epoch % 50 == 0 or epoch in (1, 10, last_epoch):
            write_json(run/"history.json", history)
            status = {**history[-1], "best_validation_objective": best, "best_epoch": best_epoch,
                      "completed": epoch == protocol["epochs"]}
            write_json(run/"status.json", status)
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best_validation_mse": best, "best_epoch": best_epoch,
                         "history": history, "best_model": best_model, "metadata": meta,
                         "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                         "python_rng": random.getstate(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, run/"checkpoint.pt")
            print(json.dumps(status), flush=True)
    if last_epoch == protocol["epochs"]:
        write_json(run/"freeze.json", {"best_sha256": digest(run/"best.pt"), "best_epoch": best_epoch,
                                      "test_labels_used": False, "run_sha256": digest(run/"run.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", choices=["mad", "pi"], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    train(args.data, args.run, args.method, args.protocol, args.resume, args.stop_after)
