"""Original Laplace DeepONet, with 3D coordinates and factored evaluations."""

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
from torch.utils.data import DataLoader, TensorDataset

from molecular_laplace_experiment import DEFAULT_OUT, ROOT, digest, write_json
from train_original_protocol import LinearBranchDeepONet, set_seed
from scalar_normalizer import ScalarNormalizer

DEFAULT_RUN = ROOT / "artifacts/r3_revision/molecular_3sgs_r1"


class MolecularDeepONet(LinearBranchDeepONet):
    def __init__(self, boundary_count, hidden_width=110):
        super().__init__(boundary_count, 3, hidden_width)

    def forward(self, boundary, coordinates):
        return (self.branch_net(boundary) * self.last_layer_weights) @ self.trunk_net(coordinates).T


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(data, run, epochs=None, resume=False, stop_after=None, require_cuda=False):
    protocol = json.loads((data/"protocol.json").read_text())
    if run.exists() and not resume:
        raise FileExistsError("A fresh run directory is required")
    if resume and (not (run/"checkpoint.pt").exists() or (run/"freeze.json").exists()):
        raise ValueError("Resume requires an unfinished checkpoint")
    run.mkdir(parents=True, exist_ok=resume)
    set_seed(protocol["model_seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Formal remote training requires CUDA")
    geometry = np.load(data/"geometry.npz")
    training = np.load(data/"train.npz")
    validation = np.load(data/"validation.npz")
    x = torch.from_numpy(training["boundary"])
    y = torch.from_numpy(training["solution"])
    xn, yn = ScalarNormalizer(x), ScalarNormalizer(y)
    train_loader = DataLoader(TensorDataset(xn.encode(x), yn.encode(y)),
                              batch_size=protocol["batch_size"], shuffle=True)
    vx = xn.encode(torch.from_numpy(validation["boundary"])).to(device)
    vy = yn.encode(torch.from_numpy(validation["solution"])).to(device)
    coords = torch.tensor((geometry["train_queries"]-geometry["center"])/geometry["scale"],
                          dtype=torch.float32, device=device)
    model = MolecularDeepONet(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])
    criterion = torch.nn.MSELoss()
    actual_epochs = epochs or protocol["epochs"]
    meta = {"model_seed": protocol["model_seed"], "epochs": actual_epochs,
            "pilot": actual_epochs != protocol["epochs"], "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__, "parameter_count": sum(p.numel() for p in model.parameters()),
            "numpy_version": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "training_script_sha256": digest(__file__),
            "timing_scope": "Training loop including validation and periodic checkpoint I/O; excludes data generation, model setup, reference solves and final testing",
            "x_mean": xn.mean, "x_std": xn.std, "y_mean": yn.mean, "y_std": yn.std,
            "normalizer_eps": xn.eps, "boundary_count": x.shape[1],
            "protocol_sha256": digest(data/"protocol.json"),
            "data_hashes": {name: digest(data/name) for name in ["geometry.npz", "train.npz", "validation.npz"]},
            "test_data_loaded_during_training": False,
            "selection": "minimum validation normalized MSE at end of epoch"}
    best, best_epoch = float("inf"), 0
    history = []
    first_epoch, previous_wall, best_model = 1, 0.0, None
    if resume:
        saved = torch.load(run/"checkpoint.pt", map_location="cpu", weights_only=False)
        old = saved["metadata"]
        for key in ("protocol_sha256", "data_hashes", "epochs", "model_seed", "torch_version",
                    "numpy_version", "device_name", "training_script_sha256"):
            if meta[key] != old[key]:
                raise ValueError("Resume provenance mismatch: " + key)
        meta = old
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        best, best_epoch = saved["best_validation_mse"], saved["best_epoch"]
        best_model = saved["best_model"]
        first_epoch, previous_wall = saved["epoch"]+1, history[-1]["wall_seconds"]
        # Restore the best state paired with this checkpoint, not a later partial epoch.
        atomic_save(best_model, run/"best.pt")
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        write_json(run/"run.json", meta)
    last_epoch = min(actual_epochs, stop_after) if stop_after is not None else actual_epochs
    if last_epoch < first_epoch:
        raise ValueError("Requested stopping epoch precedes the checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(first_epoch, last_epoch+1):
        model.train(); total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx, coords), by)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(bx)
        model.eval()
        with torch.inference_mode():
            v = float(criterion(model(vx, coords), vy))
        if not np.isfinite(v):
            raise ValueError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_mse": total/len(x), "validation_mse": v,
                        "wall_seconds": previous_wall+time.perf_counter()-start})
        if v < best:
            best, best_epoch = v, epoch
            best_model = {"model": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                          "epoch": epoch, "validation_mse": v, "metadata": meta}
            atomic_save(best_model, run/"best.pt")
        if epoch % 50 == 0 or epoch in (1, 10, last_epoch):
            write_json(run/"history.json", history)
            write_json(run/"status.json", {**history[-1], "best_validation_mse": best,
                                            "best_epoch": best_epoch, "completed": epoch == actual_epochs})
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best_validation_mse": best, "best_epoch": best_epoch,
                         "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                         "python_rng": random.getstate(), "history": history, "best_model": best_model,
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                         "metadata": meta}, run/"checkpoint.pt")
            print(json.dumps({**history[-1], "best_epoch": best_epoch}), flush=True)
    if last_epoch < actual_epochs:
        return
    write_json(run/"freeze.json", {"best_sha256": digest(run/"best.pt"),
                                   "best_epoch": best_epoch, "test_labels_used": False,
                                   "run_sha256": digest(run/"run.json")})


def metrics(predicted, true):
    per = np.linalg.norm(predicted-true, axis=1)/np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return {"aggregate": float(np.linalg.norm(predicted-true)/np.linalg.norm(true)),
            "per_sample_mean": float(per.mean()), "per_sample_sd": float(per.std(ddof=1)),
            "median": float(np.median(per)), "p90": float(np.quantile(per, 0.9)),
            "max": float(per.max()), "per_sample": per.tolist(),
            "representative_index": int(np.argmin(np.abs(per-np.median(per))))}


def evaluate(data, run):
    target = run/"evaluation"
    if target.exists():
        raise FileExistsError("Preserve previous evaluation; do not overwrite it")
    freeze = json.loads((run/"freeze.json").read_text())
    if digest(run/"best.pt") != freeze["best_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if digest(run/"run.json") != freeze["run_sha256"]:
        raise ValueError("Frozen metadata changed")
    reference_report = json.loads((data/"reference/report.json").read_text())
    if digest(data/"reference/level2_solutions.npy") != reference_report["test_solution_sha256"]:
        raise ValueError("FEM reference changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run/"best.pt", map_location=device, weights_only=False)
    meta = checkpoint["metadata"]
    if digest(data/"protocol.json") != meta["protocol_sha256"]:
        raise ValueError("Evaluation protocol mismatch")
    for name, checksum in meta["data_hashes"].items():
        if digest(data/name) != checksum:
            raise ValueError("Evaluation data mismatch: " + name)
    geometry = np.load(data/"geometry.npz")
    model = MolecularDeepONet(meta["boundary_count"], meta.get("hidden_width", 110)).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    coords = torch.tensor((geometry["test_queries"]-geometry["center"])/geometry["scale"],
                          dtype=torch.float32, device=device)
    bcoords = torch.tensor((geometry["nodes"][geometry["boundary_ids"]]-geometry["center"])/geometry["scale"],
                           dtype=torch.float32, device=device)
    target.mkdir()
    report = {"checkpoint_sha256": freeze["best_sha256"], "epoch": checkpoint["epoch"],
              "test_size": 100, "query_count": len(coords), "single_training_seed": True,
              "metric": "uniform-volume-query empirical relative L2; sample SD is over cases, not seeds",
              "fem_reference_qualified": reference_report["refinement_target_pass"],
              "fem_result_status": "qualified refinement check" if reference_report["refinement_target_pass"]
                                   else "provisional: reference refinement target not met; refine before publication",
              "sets": {}}
    for name in ("mad_test", "fem_test"):
        inputs = np.load(data/(name+".npz" if name == "mad_test" else "fem_test_inputs.npz"))
        g = inputs["boundary"]
        b = torch.tensor((g-meta["x_mean"])/(meta["x_std"]+meta["normalizer_eps"]), device=device)
        with torch.inference_mode():
            predicted = (model(b, coords)*(meta["y_std"]+meta["normalizer_eps"])+meta["y_mean"]).cpu().numpy()
            boundary = (model(b, bcoords)*(meta["y_std"]+meta["normalizer_eps"])+meta["y_mean"]).cpu().numpy()
        # Reference targets are accessed only after the frozen model has predicted.
        true = inputs["solution"].astype(np.float64) if name == "mad_test" else np.load(data/"reference/level2_solutions.npy")
        result = metrics(predicted, true)
        result["boundary"] = metrics(boundary, g)
        report["sets"][name] = result
        np.savez(target/(name+".npz"), predicted=predicted, true=true,
                 boundary_predicted=boundary, boundary_true=g)
    report["reference_check"] = reference_report
    write_json(target/"report.json", report)
    print(json.dumps({name: {k:v for k,v in r.items() if k not in ("per_sample", "boundary")}
                      for name, r in report["sets"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate"])
    parser.add_argument("--data", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.stage == "train":
        train(args.data, args.run, args.epochs, args.resume, args.stop_after, args.require_cuda)
    else:
        evaluate(args.data, args.run)
