"""Verify saved paired predictions and render the fixed representative case."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_lnfno_source_free_pb import metrics


def main(args):
    folder = args.output_dir
    result = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    with np.load(folder / "predictions.npz") as data:
        boundary = data["boundaries"].copy()
        reference = data["references"].copy()
        fno = data["fno_predictions"].copy()
        lnfno = data["lnfno_predictions"].copy()
    for key, values in (("source_free_fno", fno), ("source_free_lnfno", lnfno)):
        computed = metrics(values, reference)
        assert np.allclose(computed["per_sample_full"], result[key]["per_sample_full"])
        assert np.isclose(computed["aggregate"], result[key]["aggregate"])

    # Keep the representative sample from the original FNO report. Selection
    # does not depend on whether LNF-NO improves this particular sample.
    original = json.loads(args.original_test_json.read_text(encoding="utf-8"))
    fno_training = torch.load(args.fno_training_checkpoint, map_location="cpu", weights_only=True)
    assert fno_training["split"] == {"mode": "fixed", "ntrain": 1800, "ntest": 200}
    assert fno_training["seed"] == result["manifest"]["protocol"]["seed"]
    assert fno_training["epochs"] == result["manifest"]["protocol"]["epochs"]
    assert fno_training["lr_init"] == 1e-3
    index = original["representative_sample"]["index"]
    fno_error = result["source_free_fno"]["per_sample_full"][index]
    lnf_error = result["source_free_lnfno"]["per_sample_full"][index]
    fields = (reference[index], fno[index], lnfno[index])
    field_min = min(float(a.min()) for a in fields)
    field_max = max(float(a.max()) for a in fields)
    errors = (np.abs(fno[index] - reference[index]), np.abs(lnfno[index] - reference[index]))
    error_max = max(float(a.max()) for a in errors)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.8), constrained_layout=True)
    for axis, field, title in zip(
        axes[0], fields,
        ("FD/DST reference", f"MAD0-FNO ({100*fno_error:.2f}%)", f"MAD0-LNF-NO ({100*lnf_error:.2f}%)"),
    ):
        im = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                         vmin=field_min, vmax=field_max, interpolation="nearest")
        axis.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(im, ax=list(axes[0]), shrink=.82, label="u")
    axes[1, 0].plot(np.linspace(0, 1, 400, endpoint=False), boundary[index], color="#256a78")
    axes[1, 0].set(title="Dirichlet input", xlabel="Normalized perimeter", ylabel="g")
    axes[1, 0].grid(alpha=.2)
    for axis, field, title in zip(axes[1, 1:], errors, ("FNO absolute error", "LNF-NO absolute error")):
        im = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="magma",
                         vmin=0, vmax=error_max, interpolation="nearest")
        axis.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(im, ax=list(axes[1, 1:]), shrink=.82, label="Absolute error")
    fig.suptitle(f"Source-free PB: original representative sample {index + 1} of 100")
    fig.savefig(folder / "representative_comparison.png", dpi=180)
    fig.savefig(folder / "representative_comparison.pdf")
    plt.close(fig)

    fno_errors = np.asarray(result["source_free_fno"]["per_sample_full"]) * 100
    lnf_errors = np.asarray(result["source_free_lnfno"]["per_sample_full"]) * 100
    low = min(float(fno_errors.min()), float(lnf_errors.min())) * .85
    high = max(float(fno_errors.max()), float(lnf_errors.max())) * 1.15
    fig, axis = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    axis.scatter(fno_errors, lnf_errors, s=25, alpha=.75, color="#256a78")
    axis.plot([low, high], [low, high], linestyle="--", color="#a34742", linewidth=1)
    axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
             xlabel="FNO relative L2 error (%)", ylabel="LNF-NO relative L2 error (%)",
             title="All 100 source-free cases")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2, which="both")
    fig.savefig(folder / "paired_errors.png", dpi=180)
    plt.close(fig)

    lines = [
        "# MAD0-LNF-NO Source-Free PB Comparison", "",
        "Fixed-data neural-operator configuration comparison for Supplementary Table S.4. All 100 source-free cases are retained.", "",
        "## Protocol", "",
        "- The original LNF-NO source-driven PB training implementation was imported without editing it.",
        "- Training used the original MAD0-FNO array, rows 0-1799 for training and 1800-1999 for validation.",
        "- Seed 0, 500 epochs, batch size 32; checkpoint selected only by source-driven validation error.",
        "- Both inputs were standardized with training statistics. Physical f=0 was set before normalization.",
        "- All 100 existing independent source-free cases were tested without retraining or fine-tuning.",
        "- This retains each implementation's original learning rate: FNO 1e-3, LNF-NO 1e-4.",
        "- The original FNO predictions were replayed and checked against the stored R3 results.",
        "- Numerical references and test samples were unchanged. Full hashes and per-sample errors are in comparison.json.",
        "", "## Results", "",
        "| Metric | FNO | LNF-NO |", "|---|---:|---:|",
        f"| Source-driven validation mean relative L2 | {100*fno_training['best_test_rel']:.4f}% | {100*result['source_validation']['full']['mean']:.4f}% |",
    ]
    for key, label in (("full", "Mean full-grid relative L2"),
                       ("interior", "Mean interior relative L2"),
                       ("boundary", "Mean boundary relative L2")):
        a, b = result["source_free_fno"][key]["mean"], result["source_free_lnfno"][key]["mean"]
        lines.append(f"| {label} | {100*a:.4f}% | {100*b:.4f}% |")
    lines.extend([
        f"| Aggregate relative L2 | {100*result['source_free_fno']['aggregate']:.4f}% | {100*result['source_free_lnfno']['aggregate']:.4f}% |",
        "",
        f"LNF-NO has lower full-grid error on {result['lnfno_lower_error_count']} of the 100 paired cases.",
        f"Its source-driven validation mean error is {100*result['source_validation']['full']['mean']:.4f}%, at epoch {result['best_epoch']}.",
        f"The training loop took {result['training_seconds']:.2f} seconds on an RTX 4070 Laptop GPU (includes validation, excludes checkpoint writes).",
        "",
        "## Interpretation", "",
        "The comparison measures the effect of using the existing LNF-NO configuration with unchanged MAD0 training data on this specified source-free distribution. It does not isolate architecture from optimizer settings or seed variability, and does not establish general out-of-distribution guarantees.",
        "",
        f"The displayed example is original FNO representative index {index} (zero-based), chosen before inspecting LNF-NO results. All cases remain in the aggregate statistics.",
        "", "![Paired errors](paired_errors.png)", "",
        "![Original representative case](representative_comparison.png)", "",
    ])
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Prediction checks passed; report and figures written to {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--original-test-json", required=True, type=Path)
    parser.add_argument("--fno-training-checkpoint", required=True, type=Path)
    main(parser.parse_args())
