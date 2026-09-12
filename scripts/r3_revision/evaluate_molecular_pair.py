"""Evaluate a freshly trained MAD/PI pair on independent same-law functions.

Use only locally generated, trusted checkpoints. This is a new reproduction;
confirm_molecular_frozen.py remains the immutable archived-checkpoint replay.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from molecular_geometry_sources import MolecularGeometry
from molecular_laplace_experiment import digest, write_json
from molecular_laplace_true_exterior import generate_samples
from train_molecular_laplace import MolecularDeepONet


def main(args):
    if args.samples < 2:
        raise ValueError("At least two test functions are required")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh evaluation directory")
    paths = {"mad": args.mad_checkpoint, "pi": args.pi_checkpoint}
    hashes = {name: digest(path) for name, path in paths.items()}
    data_hashes = {name: digest(args.data / name)
                   for name in ("geometry.npz", "train.npz", "validation.npz")}
    saved = {}
    for name, path in paths.items():
        saved[name] = torch.load(path, map_location="cpu", weights_only=False)
        meta = saved[name]["metadata"]
        if meta["method"] != name or meta["data_hashes"] != data_hashes:
            raise ValueError("Checkpoint method or training inputs do not match")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol_before_generation.json", {
        "count": args.samples, "seed": args.seed, "relative_clearance": .01,
        "checkpoints": hashes, "data_hashes": data_hashes,
        "scope": "Fresh same-law functions, fixed spatial queries; all samples reported",
    })
    with np.load(args.data / "geometry.npz", allow_pickle=False) as raw:
        geo = {key: raw[key] for key in raw.files}
    geometry = MolecularGeometry(geo["nodes"], geo["tetra"])
    boundary = geo["nodes"][geo["boundary_ids"]]
    samples, _ = generate_samples(geometry, boundary, geo["test_queries"],
                                  args.samples, args.seed, relative_clearance=.01)
    new_sources = {a.tobytes() for a in samples["sources"]}
    for split in ("train", "validation", "mad_test"):
        with np.load(args.data / (split + ".npz"), allow_pickle=False) as old:
            if new_sources & {a.tobytes() for a in old["sources"]}:
                raise ValueError("Test source configurations overlap an earlier split")
    np.savez(args.output_dir / "fresh_test.npz", **samples)
    torch.set_num_threads(2)
    report = {}
    for name, checkpoint in saved.items():
        meta = checkpoint["metadata"]
        model = MolecularDeepONet(meta["boundary_count"], meta["hidden_width"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        report[name] = {"epoch": checkpoint["epoch"], "checkpoint_sha256": hashes[name]}
        for region, points, key in (("interior", geo["test_queries"], "solution"),
                                    ("boundary", boundary, "boundary")):
            q = torch.tensor((points - geo["center"]) / geo["scale"], dtype=torch.float32)
            predictions = []
            with torch.inference_mode():
                for start in range(0, args.samples, 32):
                    b = (samples["boundary"][start:start+32] - meta["x_mean"])
                    b = torch.tensor(b / (meta["x_std"] + meta["normalizer_eps"]), dtype=torch.float32)
                    pred = model(b, q) * (meta["y_std"] + meta["normalizer_eps"]) + meta["y_mean"]
                    predictions.append(pred.numpy())
            pred = np.concatenate(predictions).astype(np.float64)
            truth = samples[key].astype(np.float64)
            norms = np.linalg.norm(truth, axis=1)
            if not np.isfinite(pred).all() or np.any(norms == 0):
                raise ValueError("Nonfinite predictions or undefined relative errors")
            errors = np.linalg.norm(pred - truth, axis=1) / norms
            report[name][region] = {
                "count": args.samples,
                "aggregate": float(np.linalg.norm(pred-truth) / np.linalg.norm(truth)),
                "mean": float(errors.mean()), "sd": float(errors.std(ddof=1)),
                "median": float(np.median(errors)), "p90": float(np.quantile(errors, .9)),
                "maximum": float(errors.max()),
            }
            np.savez(args.output_dir / (name + "_" + region + ".npz"),
                     prediction=pred, per_function_error=errors)
        if digest(paths[name]) != hashes[name]:
            raise ValueError("Checkpoint changed during evaluation")
    write_json(args.output_dir / "results.json", report)
    print(args.output_dir / "results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mad-checkpoint", type=Path, required=True)
    parser.add_argument("--pi-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026091107)
    main(parser.parse_args())
