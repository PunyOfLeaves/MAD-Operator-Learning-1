"""Generate the final 0.01L MAD1 distribution on a supplied fixed 3D geometry.

The geometry NPZ must contain nodes, tetra, boundary_ids, train_queries,
test_queries, center and scale. No FEM reference or historical run folder is
needed. Supplying a different geometry defines a new experiment.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from molecular_geometry_sources import MolecularGeometry, triangle_distances
from molecular_laplace_experiment import digest, write_json
from molecular_laplace_true_exterior import generate_samples


def main(args):
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh data directory")
    if min(args.train_samples, args.validation_samples, args.test_samples) <= 0:
        raise ValueError("Sample counts must be positive")
    with np.load(args.geometry, allow_pickle=False) as raw:
        required = ("nodes", "tetra", "boundary_ids", "train_queries",
                    "test_queries", "center", "scale")
        a = {key: raw[key] for key in required}
    geo = MolecularGeometry(a["nodes"], a["tetra"])
    if not np.isclose(a["scale"], geo.scale):
        raise ValueError("Geometry scale must equal the largest bounding-box span")
    protocol = {
        "equation": "Delta u = 0", "relative_clearance": 0.01,
        "geometry_sha256": digest(args.geometry),
        "source_rule": "Ten independent exterior sources; 50/30/20 mixture of box exterior, box-interior material exterior and reentrant exterior",
        "weights": "Independent standard normal",
        "normalization": "Joint maximum absolute value on boundary and interior queries per function",
        "ntrain": args.train_samples, "nvalidation": args.validation_samples,
        "ntest": args.test_samples, "train_seed": 2026091052,
        "validation_seed": 2026091053, "mad_test_seed": 2026091054,
        "scope": "Fixed-domain harmonic data; no claim of unseen-geometry transfer",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    shutil.copyfile(args.geometry, args.output_dir / "geometry.npz")
    boundary = a["nodes"][a["boundary_ids"]]
    report = {}
    for name, count, seed, query_key in (
        ("train", args.train_samples, 2026091052, "train_queries"),
        ("validation", args.validation_samples, 2026091053, "train_queries"),
        ("mad_test", args.test_samples, 2026091054, "test_queries"),
    ):
        samples, stats = generate_samples(geo, boundary, a[query_key],
                                           count, seed, relative_clearance=0.01)
        if not np.all(samples["source_clearance"] > 0.01 * geo.scale):
            raise ValueError("Source clearance check failed")
        flat = samples["sources"].reshape(-1, 3)
        for point in flat[::max(1, len(flat)//100)]:
            if geo.contains(point) or triangle_distances(point, geo.triangles).min() <= 0.01 * geo.scale:
                raise ValueError("Independent source-geometry check failed")
        np.savez(args.output_dir / (name + ".npz"), **samples)
        report[name] = {"count": count, "seed": seed, "rejections": stats,
                        "minimum_clearance": float(samples["source_clearance"].min())}
    write_json(args.output_dir / "generation.json", report)
    write_json(args.output_dir / "input_manifest.json", {
        p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=100)
    main(parser.parse_args())
