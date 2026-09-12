"""Reuse the LNF-NO PB implementation for a fixed-data source-free comparison.

The original training function, architecture, loss, and best-model selection
are imported unchanged. Only its dataset path is redirected to the original
MAD0-FNO data. The 200 held-out source-driven rows are validation data; the
independent source-free cases are evaluated only after training is finished.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import evaluate_source_free_pb as baseline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_training(path: Path):
    spec = importlib.util.spec_from_file_location("lnfno_pb_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    if prediction.shape != reference.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or non-finite values")
    full = np.array([baseline.relative_l2(p, r) for p, r in zip(prediction, reference)])
    interior = np.array([
        baseline.relative_l2(p[1:-1, 1:-1], r[1:-1, 1:-1])
        for p, r in zip(prediction, reference)
    ])
    boundary = np.array([
        baseline.relative_l2(baseline.pack_boundary(p), baseline.pack_boundary(r))
        for p, r in zip(prediction, reference)
    ])
    return {
        "full": baseline.summarize(full),
        "interior": baseline.summarize(interior),
        "boundary": baseline.summarize(boundary),
        "aggregate": baseline.relative_l2(prediction, reference),
        "per_sample_full": full.tolist(),
        "per_sample_interior": interior.tolist(),
        "per_sample_boundary": boundary.tolist(),
    }


def lnf_predict(module, model, normalizers, boundaries, sources, device, batch_size):
    x_norm, y_norm = normalizers
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(boundaries), batch_size):
            boundary = torch.as_tensor(
                boundaries[start:start + batch_size], dtype=torch.float32, device=device
            )
            source = torch.as_tensor(
                sources[start:start + batch_size], dtype=torch.float32, device=device
            )
            boundary_n, source_n = x_norm.encode(boundary, source)
            prediction = y_norm.decode(model(boundary_n, source_n))
            predictions.append(prediction.cpu().numpy().reshape(-1, 101, 101))
    return np.concatenate(predictions).astype(np.float64)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_comparison_inputs(
    training_data: Path,
    test_json: Path,
    fno_checkpoint: Path | None = None,
    training_data_sha256: str | None = None,
) -> tuple[Path, str]:
    metadata = read_json(test_json)

    def metadata_path(key):
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else test_json.resolve().parent / path

    expected_hashes = []
    for value in (training_data_sha256, metadata.get("training_data_sha256")):
        if value is not None:
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in value)):
                raise ValueError("Expected training-data SHA-256 must be 64 hexadecimal characters")
            expected_hashes.append(value.lower())
    if len(set(expected_hashes)) > 1:
        raise ValueError("Conflicting expected training-data SHA-256 values")
    if expected_hashes:
        expected_hash = expected_hashes[0]
    else:
        original_data = metadata_path("training_data")
        if original_data is None or not original_data.is_file():
            raise ValueError(
                "Cannot verify the original training data: archived file is unavailable and "
                "metadata has no training_data_sha256. Supply --training-data-sha256 "
                "from a trusted original-data record."
            )
        expected_hash = sha256(original_data)

    actual_hash = sha256(training_data)
    if actual_hash != expected_hash:
        raise ValueError("FNO and LNF-NO training arrays differ (training-data SHA-256 mismatch)")

    checkpoint = fno_checkpoint if fno_checkpoint is not None else metadata_path("checkpoint")
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError("FNO checkpoint is unavailable; supply --fno-checkpoint")
    return checkpoint.resolve(), actual_hash


def main(args):
    # Read provenance only; source-free arrays remain unread until after training.
    fno_checkpoint, training_data_hash = resolve_comparison_inputs(
        args.training_data, args.test_json, args.fno_checkpoint, args.training_data_sha256,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = import_training(args.lnfno_script.resolve())
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")

    train_array = np.load(args.training_data, mmap_mode="r")
    if train_array.shape != (2000, 20802):
        raise ValueError(f"Unexpected training shape {train_array.shape}")
    boundary_mismatch = max(
        float(np.max(np.abs(baseline.pack_boundary(row[10601:].reshape(101, 101)) - row[:400])))
        for row in train_array
    )
    if boundary_mismatch > 1e-5:
        raise ValueError(f"Boundary/grid layout mismatch: {boundary_mismatch}")

    manifest = {
        "training_data": str(args.training_data.resolve()),
        "training_data_sha256": training_data_hash,
        "lnfno_script": str(args.lnfno_script.resolve()),
        "lnfno_script_sha256": sha256(args.lnfno_script),
        "protocol": {
            "ntrain": 1800, "nvalidation": 200,
            "split": "contiguous rows [0:1800] / [1800:2000]",
            "epochs": args.epochs, "seed": args.seed, "batch_size": 32,
            "lr": 1e-4, "weight_decay": 1e-4,
            "selection": "minimum source-driven validation mean relative L2",
            "zero_source_training_or_selection": False,
            "normalization": "original LNF-NO scalar statistics from training rows only",
        },
        "source_convention": "Delta u - sinh(u) = f; source-free test has f = 0",
        "boundary_grid_max_mismatch": boundary_mismatch,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "threads": args.threads,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Existing run has a different manifest; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The upstream function fixes a filename; bind its dataset to the verified
    # original array without changing the upstream source or copying the data.
    original_dataset = module.PBWithSourceDataset

    class OriginalMAD0Dataset(original_dataset):
        def __init__(self, npy_path, num_points=101):
            super().__init__(str(args.training_data.resolve()), num_points=num_points)

    module.PBWithSourceDataset = OriginalMAD0Dataset
    module.train_lnfno_pb_with_source(
        device=str(device), data_dir=str(args.training_data.parent),
        save_dir=str(output / "models"), epochs=args.epochs, seed=args.seed,
    )

    final_path = output / "models/LNFNO_PB_k1.0_withsource_checkpoint.pt"
    best_path = output / "models/LNFNO_PB_k1.0_withsource_best.pt"
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != args.epochs:
        raise RuntimeError("Training did not complete the requested schedule")
    history = np.asarray(checkpoint["history_test_rel"])
    best_epoch = int(np.argmin(history)) + 1

    train_tensor = torch.tensor(np.asarray(train_array[:1800]), dtype=torch.float32)
    normalizers = (
        module.TwoPartNormalizer(train_tensor[:, :400], train_tensor[:, 400:10601]),
        module.GaussianNormalizer(train_tensor[:, 10601:]),
    )
    model = module.LNFNO_PB_WithSource(N=101, Nb=400).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation = np.asarray(train_array[1800:2000]).copy()
    validation_predictions = lnf_predict(
        module, model, normalizers, validation[:, :400], validation[:, 400:10601],
        device, args.eval_batch_size,
    )
    validation_metrics = metrics(validation_predictions, validation[:, 10601:].reshape(-1, 101, 101))
    if not np.isclose(validation_metrics["full"]["mean"], history.min(), rtol=2e-4, atol=2e-6):
        raise RuntimeError("Reloaded best checkpoint does not reproduce validation score")

    # Source-free reference data enter only here, after validation-based selection.
    with np.load(args.test_npz) as test:
        boundaries = test["boundaries"].copy()
        references = test["references"].copy()
        saved_fno = test["predictions"].copy()
    if boundaries.shape != (100, 400) or references.shape != (100, 101, 101):
        raise ValueError("Unexpected source-free test shapes")
    expected_boundary = np.stack([baseline.pack_boundary(r) for r in references])
    if not np.allclose(boundaries, expected_boundary, rtol=0, atol=1e-12):
        raise ValueError("Source-free reference/boundary mismatch")
    fno, fno_stats = baseline.load_model_and_statistics(
        args.training_data, fno_checkpoint, device,
    )
    fno_prediction = baseline.predict(fno, fno_stats, boundaries, device, args.eval_batch_size)
    fno_replay = float(np.max(np.abs(fno_prediction - saved_fno)))
    if not np.allclose(fno_prediction, saved_fno, rtol=1e-4, atol=2e-6):
        raise RuntimeError(f"FNO replay differs from original test: {fno_replay}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    lnf_prediction = lnf_predict(
        module, model, normalizers, boundaries, np.zeros((100, 10201), dtype=np.float32),
        device, args.eval_batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    fno_metrics = metrics(fno_prediction, references)
    lnf_metrics = metrics(lnf_prediction, references)
    paired = np.asarray(lnf_metrics["per_sample_full"]) - np.asarray(fno_metrics["per_sample_full"])

    result = {
        "manifest": manifest,
        "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256(best_path),
        "best_epoch": best_epoch, "best_source_validation_error": float(history.min()),
        "training_seconds": checkpoint["total_train_time"],
        "parameter_count": checkpoint["param_count"],
        "source_validation": validation_metrics,
        "test_npz": str(args.test_npz.resolve()), "test_npz_sha256": sha256(args.test_npz),
        "test_json_sha256": sha256(args.test_json),
        "fno_checkpoint_sha256": sha256(fno_checkpoint),
        "fno_replay_max_absolute_difference": fno_replay,
        "source_free_fno": fno_metrics, "source_free_lnfno": lnf_metrics,
        "paired_lnfno_minus_fno": baseline.summarize(paired),
        "lnfno_lower_error_count": int(np.sum(paired < 0)),
        "lnfno_inference_seconds": inference_seconds,
        "evaluation_note": "Single fixed-seed architecture comparison; no source-free fine-tuning or test selection",
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz", boundaries=boundaries, references=references,
        fno_predictions=fno_prediction, lnfno_predictions=lnf_prediction,
        validation_predictions=validation_predictions,
    )
    compact = {
        "best_epoch": best_epoch, "validation_mean": validation_metrics["full"]["mean"],
        "training_seconds": checkpoint["total_train_time"],
        "FNO": {k: v for k, v in fno_metrics.items() if not k.startswith("per_sample")},
        "LNFNO": {k: v for k, v in lnf_metrics.items() if not k.startswith("per_sample")},
        "LNFNO_lower_error_count": int(np.sum(paired < 0)),
        "output": str(output / "comparison.json"),
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lnfno-script", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument(
        "--training-data-sha256",
        help="Expected SHA-256 from a trusted original-data record; otherwise use archived metadata or its original file",
    )
    parser.add_argument(
        "--fno-checkpoint", type=Path,
        help="Override the archived FNO checkpoint path (relative CLI paths use the working directory)",
    )
    parser.add_argument("--test-npz", required=True, type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", default=4, type=int)
    main(parser.parse_args())
