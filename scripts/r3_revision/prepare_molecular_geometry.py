"""Extract the molecular PL domain and fixed query points from a Gmsh mesh."""

import argparse
from pathlib import Path

import numpy as np

from molecular_laplace_experiment import (
    audit_geometry, digest, extract_domain, uniform_points, write_json,
)


def prepare(mesh, output):
    if output.exists():
        raise FileExistsError("Use a fresh geometry directory")
    nodes, tetra, surface, original_ids = extract_domain(mesh)
    surface, _, _, _, report = audit_geometry(nodes, tetra, surface)
    train, _, _ = uniform_points(nodes, tetra, 2048, 2026091011)
    test, cells, bary = uniform_points(nodes, tetra, 4096, 2026091036)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "geometry.npz", nodes=nodes, tetra=tetra,
             surface=surface, boundary_ids=np.unique(surface),
             original_node_ids=original_ids, train_queries=train,
             test_queries=test, test_cells=cells, test_barycentric=bary,
             center=(nodes.min(0) + nodes.max(0)) / 2,
             scale=float(np.ptp(nodes, axis=0).max()))
    report.update(mesh_sha256=digest(mesh),
                  geometry_sha256=digest(output / "geometry.npz"),
                  training_query_seed=2026091011, test_query_seed=2026091036)
    write_json(output / "geometry_report.json", report)
    print(output / "geometry.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.mesh, args.output_dir)
