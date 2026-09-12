"""Isolated 3D MAD1 experiment on a fixed PQR-derived molecular PL domain.

Only geometry is read from the supplied mesh. FEM labels never enter MAD training
or checkpoint selection. scikit-fem supplies tetrahedral P1 assembly/refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/r3_revision/molecular_3sgs_r1"
DEFAULT_MESH = ROOT / "reproducibility/r3/inputs/molecular.msh"
DEFAULT_PQR = ROOT / "reproducibility/r3/inputs/molecular.pqr"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def volumes(x, t):
    p = x[t]
    return np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6


def extract_domain(path):
    import meshio

    raw = meshio.read(path)
    pairs = list(zip(raw.cells, raw.cell_data["gmsh:physical"]))
    t = np.concatenate([b.data[p == 1] for b, p in pairs if b.type == "tetra"])
    f = np.concatenate([b.data[p == 11] for b, p in pairs if b.type == "triangle"])
    ids = np.unique(t)
    remap = np.full(len(raw.points), -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    if np.any(remap[f] < 0):
        raise ValueError("Interface is not a boundary of the selected molecular volume")
    return raw.points[ids, :3], remap[t], remap[f], ids


def audit_geometry(x, t, surface):
    faces = np.concatenate([np.delete(t, i, axis=1) for i in range(4)])
    face_keys, first, count = np.unique(np.sort(faces, axis=1), axis=0,
                                        return_index=True, return_counts=True)
    boundary = face_keys[count == 1]
    if count.max() != 2 or set(map(tuple, boundary)) != set(map(tuple, np.sort(surface, axis=1))):
        raise ValueError("Volume face incidence / interface boundary mismatch")
    owners = np.tile(np.arange(len(t)), 4)[first[count == 1]]
    centers = x[boundary].mean(axis=1)
    normal = np.cross(x[boundary[:, 1]] - x[boundary[:, 0]],
                      x[boundary[:, 2]] - x[boundary[:, 0]])
    area = np.linalg.norm(normal, axis=1) / 2
    if np.any(area <= 0) or np.any(volumes(x, t) <= 1e-15):
        raise ValueError("Degenerate geometry")
    normal /= (2 * area[:, None])
    inward = np.sum(normal * (x[t[owners]].mean(axis=1) - centers), axis=1) > 0
    normal[inward] *= -1
    edge = np.sort(np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]],
                                   boundary[:, [2, 0]]]), axis=1)
    unique_edge, edge_count = np.unique(edge, axis=0, return_counts=True)
    if not np.all(edge_count == 2):
        raise ValueError("Boundary edge incidence is not closed")
    adjacency = {int(k): set() for k in np.unique(boundary)}
    for a, b in unique_edge:
        adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    left = set(adjacency)
    components = 0
    while left:
        queue = [left.pop()]; components += 1
        while queue:
            fresh = adjacency[queue.pop()] & left
            left.difference_update(fresh); queue.extend(fresh)
    report = {"nodes": len(x), "tetrahedra": len(t), "boundary_triangles": len(boundary),
              "boundary_nodes": len(adjacency), "boundary_components": components,
              "boundary_euler_characteristic": len(adjacency)-len(unique_edge)+len(boundary),
              "volume_A3": float(volumes(x, t).sum()), "boundary_matches_tag11": True,
              "closed_edge_incidence": True,
              "scope": "Fixed polyhedral domain; not a certification of a continuous molecular surface"}
    return boundary, centers, normal, area, report


def contained(points, x, t):
    """Exhaustive barycentric test, including a tolerance around the PL boundary."""
    origin = x[t[:, 0]]
    inverse = np.linalg.inv((x[t[:, 1:]] - origin[:, None]).transpose(0, 2, 1))
    result = []
    for start in range(0, len(points), 32):
        delta = points[start:start+32, None] - origin[None]
        bary = np.einsum("tij,btj->bti", inverse, delta)
        inside = np.all(bary >= -1e-9, axis=2) & (bary.sum(axis=2) <= 1+1e-9)
        result.extend(np.any(inside, axis=1).tolist())
    return np.asarray(result)


def uniform_points(x, t, n, seed):
    rng = np.random.default_rng(seed)
    v = volumes(x, t)
    cells = rng.choice(len(t), n, p=v/v.sum())
    bary = rng.dirichlet(np.ones(4), n)
    return np.einsum("bi,bij->bj", bary, x[t[cells]]), cells, bary


def kernel(x, sources):
    distance = np.linalg.norm(x[:, None] - sources[None], axis=2)
    if np.min(distance) <= 0:
        raise ValueError("Singular kernel evaluation")
    return 1.0 / (4 * np.pi * distance)


def source_pool(x, t, centers, normal, area, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(centers), 1200, p=area/area.sum())
    offsets = rng.choice([0.5, 1.0, 2.0], len(indices))
    candidates = centers[indices] + offsets[:, None] * normal[indices]
    valid = ~contained(candidates, x, t)
    near = candidates[valid][:288]
    if len(near) != 288:
        raise ValueError("Insufficient geometrically admissible exterior centers")
    center = (x.min(axis=0)+x.max(axis=0))/2
    radius = np.linalg.norm(x-center, axis=1).max()+2.0
    direction = rng.normal(size=(64, 3)); direction /= np.linalg.norm(direction, axis=1)[:, None]
    sources = np.vstack((near, center+radius*direction))
    if np.any(contained(sources, x, t)):
        raise ValueError("A source lies in the closed computational domain")
    return sources, {"near_count": 288, "far_count": 64, "candidate_count": len(candidates),
                     "rejected_inside_count": int((~valid).sum()),
                     "near_offset_A": [0.5, 1.0, 2.0], "far_radius_A": float(radius),
                     "all_sources_outside_all_tetrahedra": True}


def prepare(out, mesh_path, pqr_path):
    if out.exists():
        raise FileExistsError("Use a fresh experiment directory; original evidence is not overwritten")
    out.mkdir(parents=True)
    protocol = {"schema": "MAD_R3_molecular_Laplace_r1", "equation": "-Delta u=0; Dirichlet g on entire boundary",
                "mesh": str(mesh_path), "mesh_sha256": digest(mesh_path),
                "pqr": str(pqr_path), "pqr_sha256": digest(pqr_path),
                "geometry_seed": 2026090901, "train_seed": 2026090902,
                "validation_seed": 2026090903, "mad_test_seed": 2026090904,
                "fem_test_seed": 2026090905, "model_seed": 2026090906,
                "ntrain": 1000, "nvalidation": 100, "ntest": 100,
                "training_query_points": 2048, "test_query_points": 4096,
                "epochs": 5000, "batch_size": 32, "lr": 1e-4,
                "checkpoint": "minimum MAD validation MSE; no test-set selection",
                "architecture": "original Laplace linear-branch DeepONet; 3D trunk input; width110, latent1000",
                "reference": "scikit-fem P1 on fixed PL mesh and uniform nested refinements",
                "reference_levels": [0, 1, 2], "refinement_test_cases": 10,
                "reference_gap_target_max": 0.003,
                "test_boundary": "24 random Fourier features restricted to boundary; continuous field is not a harmonic extension",
                "primary_metric": "aggregate relative L2 on 4096 independent volume-uniform common query points",
                "other_metrics": "per-sample mean, SD, median, p90, max; boundary relative L2",
                "scope": "one fixed PQR-derived polyhedral shape, one training seed; not biological PNP or unseen-geometry transfer",
                "disclosure": "No selection of best test cases; representative plot nearest median error",
                "pqr_provenance": "existing group asset; original processing/source citation must be confirmed before publication"}
    write_json(out/"protocol.json", protocol)
    tick = time.perf_counter()
    x, t, f, original_ids = extract_domain(mesh_path)
    f, centers, normal, area, report = audit_geometry(x, t, f)
    boundary = np.unique(f)
    train_q, _, _ = uniform_points(x, t, protocol["training_query_points"], 2026090911)
    test_q, tc, tb = uniform_points(x, t, protocol["test_query_points"], 2026090912)
    sources, source_report = source_pool(x, t, centers, normal, area, protocol["geometry_seed"])
    center = (x.min(axis=0)+x.max(axis=0))/2
    scale = float(np.ptp(x, axis=0).max())
    np.savez(out/"geometry.npz", nodes=x, tetra=t, surface=f, boundary_ids=boundary,
             original_node_ids=original_ids, train_queries=train_q, test_queries=test_q,
             test_cells=tc, test_barycentric=tb, sources=sources, center=center, scale=scale)
    report["geometry_setup_seconds"] = time.perf_counter()-tick
    report["source_placement"] = source_report
    write_json(out/"geometry_report.json", report)
    for name, n, seed, points in [("train", 1000, protocol["train_seed"], train_q),
                                  ("validation", 100, protocol["validation_seed"], train_q),
                                  ("mad_test", 100, protocol["mad_test_seed"], test_q)]:
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(n, len(sources)))/np.sqrt(len(sources))
        g = weights @ kernel(x[boundary], sources).T
        u = weights @ kernel(points, sources).T
        normalization = np.abs(g).max(axis=1)
        if np.min(normalization) <= 1e-12:
            raise ValueError("Degenerate MAD sample")
        np.savez(out/(name+".npz"), boundary=(g/normalization[:, None]).astype(np.float32),
                 solution=(u/normalization[:, None]).astype(np.float32), weights=weights,
                 normalization=normalization)
        report[name+"_generation_seconds_including_kernel_and_write"] = time.perf_counter()-start
    # Freeze test inputs; FEM labels never enter training or checkpoint selection.
    rng = np.random.default_rng(protocol["fem_test_seed"])
    frequencies = rng.normal(size=(24, 3))/0.35
    phases = rng.uniform(0, 2*np.pi, 24)
    coeff = rng.normal(size=(100, 24))/np.sqrt(24)
    bias = rng.normal(scale=0.3, size=100)
    raw = coeff @ np.cos(((x[boundary]-center)/scale) @ frequencies.T + phases).T + bias[:, None]
    normalization = np.abs(raw).max(axis=1)
    np.savez(out/"fem_test_inputs.npz", frequencies=frequencies, phases=phases,
             coefficients=coeff, bias=bias, normalization=normalization,
             boundary=(raw/normalization[:, None]).astype(np.float32))
    write_json(out/"geometry_report.json", report)
    write_json(out/"input_manifest.json", {p.name: digest(p) for p in out.iterdir() if p.suffix in (".npz", ".json")})
    print(json.dumps(report, indent=2), flush=True)


def boundary_function(points, definition, geometry, ids):
    features = np.cos(((points-geometry["center"])/geometry["scale"]) @ definition["frequencies"].T + definition["phases"])
    return (features @ definition["coefficients"][ids].T + definition["bias"][ids]) / definition["normalization"][ids]


def fem_solve(mesh, boundary_values, queries):
    from scipy.sparse.linalg import splu
    from skfem import Basis, ElementTetP1, asm
    from skfem.models.poisson import laplace

    basis = Basis(mesh, ElementTetP1())
    matrix = asm(laplace, basis).tocsc()
    boundary = mesh.boundary_nodes()
    free = np.setdiff1d(np.arange(mesh.nvertices), boundary)
    rhs = -matrix[free][:, boundary] @ boundary_values
    factor = splu(matrix[free][:, free])
    u = np.zeros((mesh.nvertices, boundary_values.shape[1]))
    u[boundary] = boundary_values
    u[free] = factor.solve(rhs)
    residual = matrix[free] @ u
    relative_residual = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
    # Small query chunks avoid the element finder's all-cell fallback using large temporaries.
    sampled = np.vstack([basis.probes(queries[i:i+64].T) @ u for i in range(0, len(queries), 64)])
    return sampled.T, float(relative_residual.max())


def reference(out):
    from skfem import MeshTet

    target = out/"reference"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    geometry = np.load(out/"geometry.npz")
    inputs = np.load(out/"fem_test_inputs.npz")
    mesh = MeshTet(geometry["nodes"].T, geometry["tetra"].T)
    query = geometry["test_queries"]
    report = {"levels": [], "same_polyhedral_domain": True,
              "scope": "spatial refinement differences, not a certified exact error bound"}
    previous = None
    for level in range(3):
        start = time.perf_counter()
        ids = np.arange(100 if level == 2 else 10)
        bcoords = mesh.p[:, mesh.boundary_nodes()].T
        g = boundary_function(bcoords, inputs, geometry, ids)
        # A linear harmonic field checks assembly, ordering and Dirichlet elimination.
        affine_g = 1 + bcoords @ np.array([0.1, -0.2, 0.3])
        both, residual = fem_solve(mesh, np.column_stack((g, affine_g)), query)
        values, affine = both[:-1], both[-1]
        truth = 1 + query @ np.array([0.1, -0.2, 0.3])
        affine_error = float(np.linalg.norm(affine-truth)/np.linalg.norm(truth))
        if affine_error > 1e-9 or residual > 1e-9:
            raise ValueError("FEM affine patch or linear solve check failed")
        record = {"level": level, "nodes": int(mesh.nvertices), "tetrahedra": int(mesh.nelements),
                  "samples": len(ids), "volume_A3": float(volumes(mesh.p.T, mesh.t.T).sum()),
                  "affine_patch_relative_error": affine_error, "linear_residual_max": residual,
                  "wall_seconds": time.perf_counter()-start}
        if previous is not None:
            delta = np.linalg.norm(values[:10]-previous, axis=1)/np.linalg.norm(values[:10], axis=1)
            record["previous_level_gap_mean"] = float(delta.mean())
            record["previous_level_gap_max"] = float(delta.max())
        report["levels"].append(record)
        np.save(target/f"level{level}_solutions.npy", values)
        write_json(target/"report.json", report)
        print(json.dumps(record), flush=True)
        previous = values[:10]
        if level < 2:
            mesh = mesh.refined()
    report["refinement_target_pass"] = report["levels"][-1]["previous_level_gap_max"] <= 0.003
    report["test_solution_sha256"] = digest(target/"level2_solutions.npy")
    write_json(target/"report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "reference"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--pqr", type=Path, default=DEFAULT_PQR)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.mesh, args.pqr)
    else:
        reference(args.out)
