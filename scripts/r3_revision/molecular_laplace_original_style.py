"""Per-sample ten-source MAD1, following gMADlaplace2D1.py in three dimensions.

This is a new dataset, not an overwrite or continuation of the fixed-pool pilot.
The shared trainer accepts this directory through --data; train from scratch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from molecular_laplace_experiment import (
    DEFAULT_MESH, DEFAULT_PQR, ROOT, audit_geometry, digest, extract_domain,
    reference, uniform_points, write_json,
)

DEFAULT_DATA = ROOT / "data/r3_revision/molecular_3sgs_original_style_r2"
ORIGINAL = ROOT / "scripts/data_generation/gMADlaplace2D1.py"


def sample_sources(rng, count, low, high, margin=0.001):
    """Half face-exterior, half all-coordinate-exterior, in bounding-box units."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.ndim != 1 or low.shape != high.shape or len(low) not in (2, 3):
        raise ValueError("Expected two- or three-dimensional bounds")
    scale = float(np.max(high-low))
    if count <= 0 or margin <= 0 or np.any(high <= low):
        raise ValueError("Invalid source sampling parameters")
    sources = np.empty((count, len(low)))
    weights = np.empty(count)
    for j in range(count):
        if rng.random() > 0.5:
            axes = np.array([rng.integers(len(low))])
            point = rng.uniform(low, high)
        else:
            axes = np.arange(len(low))
            point = np.empty(len(low))
        z = rng.standard_normal(len(axes))
        point[axes] = np.where(z <= 0, low[axes]+scale*(z-margin),
                             high[axes]+scale*(z+margin))
        sources[j] = point
        weights[j] = rng.standard_normal()
    return sources, weights


def fields(points, sources, weights):
    distance = np.linalg.norm(points[None, :, None, :]-sources[:, None, :, :], axis=-1)
    if np.any(distance <= 0):
        raise ValueError("Singular evaluation")
    return np.sum(weights[:, None, :]/(4*np.pi*distance), axis=-1)


def normalize_pair(g, u):
    normalization = np.maximum(np.max(np.abs(g), axis=1), np.max(np.abs(u), axis=1))
    if np.any(normalization <= 1e-12) or not np.isfinite(normalization).all():
        raise ValueError("Degenerate sample; do not silently redraw it")
    return g/normalization[:, None], u/normalization[:, None], normalization


def generate_samples(nodes, boundary_ids, points, count, seed):
    rng = np.random.default_rng(seed)
    low, high = nodes.min(0), nodes.max(0)
    sources, weights = zip(*(sample_sources(rng, 10, low, high) for _ in range(count)))
    sources, weights = np.asarray(sources), np.asarray(weights)
    g = np.empty((count, len(boundary_ids)), dtype=np.float32)
    u = np.empty((count, len(points)), dtype=np.float32)
    normalization = np.empty(count)
    for start in range(0, count, 16):
        sl = slice(start, start+16)
        raw_g = fields(nodes[boundary_ids], sources[sl], weights[sl])
        raw_u = fields(points, sources[sl], weights[sl])
        a, b, c = normalize_pair(raw_g, raw_u)
        g[sl], u[sl], normalization[sl] = a, b, c
    return {"boundary":g, "solution":u, "sources":sources, "weights":weights,
            "normalization":normalization}


def prepare(out, mesh_path, pqr_path, seed_offset=0):
    if out.exists():
        raise FileExistsError("Use a new data directory; previous evidence is retained")
    out.mkdir(parents=True)
    protocol = {
        "schema":"MAD_R3_molecular_Laplace_original_style_r2",
        "equation":"-Delta u=0; Dirichlet g on the entire boundary",
        "mesh":str(mesh_path), "mesh_sha256":digest(mesh_path),
        "pqr":str(pqr_path), "pqr_sha256":digest(pqr_path),
        "original_generator":str(ORIGINAL), "original_generator_sha256":digest(ORIGINAL),
        "source_rule":"Independent 10 sources per sample; 50% one coordinate outside the bounding box, 50% all three outside; other coordinates uniform inside",
        "source_offset":"Standard normal offset times the largest bounding-box span; additional margin 0.001 times that span",
        "source_weights":"Independent standard normal, resampled with the positions for every sample",
        "normalization":"One maximum absolute value over the concatenated boundary and interior samples, as in the original 2D script",
        "kernel":"1/(4*pi*Euclidean_distance), evaluated in physical 3D coordinates; no softening",
        "train_seed":2026090932, "validation_seed":2026090933, "mad_test_seed":2026090934,
        "fem_test_seed":2026090935, "model_seed":2026090906,
        "training_query_seed":2026090911, "test_query_seed":2026090936,
        "ntrain":2000, "nvalidation":100, "ntest":100,
        "training_query_points":2048, "test_query_points":4096,
        "epochs":50000, "batch_size":32, "lr":1e-4,
        "checkpoint":"Minimum MAD validation MSE; fresh model and optimizer, no pilot checkpoint reuse",
        "architecture":"Linear-branch DeepONet; 3D trunk, hidden width110, latent1000",
        "reference":"scikit-fem P1 on fixed PL mesh and uniform nested refinements",
        "reference_levels":[0,1,2], "refinement_test_cases":10, "reference_gap_target_max":0.003,
        "test_boundary":"Independent 24-feature Fourier traces, frequencies N(0,1)/0.35, uniform phases, coefficients N(0,1)/sqrt(24), bias N(0,0.3^2)",
        "primary_metric":"Aggregate relative L2 on 4096 volume-uniform common test queries",
        "other_metrics":"All 100 cases per set; per-sample mean, SD, median, p90, max; boundary relative L2",
        "development_disclosure":"Protocol changed after fixed-pool pilot evaluation, following the user's request to follow the original 2D generator; fresh test seeds and no reuse of pilot results as confirmation",
        "scope":"One fixed PQR-derived PL molecular interior, one training seed; not molecular electrostatics, PNP, or unseen-geometry transfer",
        "remaining_differences":"3D kernel and geometry; irregular boundary/volume points; 2000 training plus 100 validation cases; float64 evaluation and float32 array storage instead of decimal text",
        "pqr_provenance":"Existing group asset; original processing/source citation still to be confirmed before publication",
    }
    if seed_offset:
        for key in ("train_seed", "validation_seed", "mad_test_seed", "fem_test_seed",
                    "training_query_seed", "test_query_seed"):
            protocol[key] += seed_offset
        protocol["data_seed_offset"] = seed_offset
    protocol["geometry_asset"] = pqr_path.stem
    write_json(out/"protocol.json", protocol)
    start = time.perf_counter()
    nodes, tetra, surface, original_ids = extract_domain(mesh_path)
    surface, _, _, _, geometry_report = audit_geometry(nodes, tetra, surface)
    boundary = np.unique(surface)
    train_q, _, _ = uniform_points(nodes, tetra, 2048, protocol["training_query_seed"])
    test_q, tc, tb = uniform_points(nodes, tetra, 4096, protocol["test_query_seed"])
    center = (nodes.min(0)+nodes.max(0))/2
    scale = float(np.ptp(nodes, axis=0).max())
    np.savez(out/"geometry.npz", nodes=nodes, tetra=tetra, surface=surface, boundary_ids=boundary,
             original_node_ids=original_ids, train_queries=train_q, test_queries=test_q,
             test_cells=tc, test_barycentric=tb, center=center, scale=scale)
    geometry_report["geometry_setup_seconds"] = time.perf_counter()-start
    geometry_report["source_geometry_rule"] = "All sources exterior to the enclosing bounding box, hence exterior to every domain tetrahedron"
    write_json(out/"geometry_report.json", geometry_report)
    generation = {}
    for name, size, seed, q in (("train",2000,protocol["train_seed"],train_q),
                                ("validation",100,protocol["validation_seed"],train_q),
                                ("mad_test",100,protocol["mad_test_seed"],test_q)):
        start = time.perf_counter()
        samples = generate_samples(nodes, boundary, q, size, seed)
        np.savez(out/(name+".npz"), **samples)
        generation[name] = {"count":size,"seconds_including_sampling_evaluation_and_write":time.perf_counter()-start}
    rng = np.random.default_rng(protocol["fem_test_seed"])
    frequencies = rng.normal(size=(24,3))/0.35
    phases = rng.uniform(0,2*np.pi,24)
    coefficients = rng.normal(size=(100,24))/np.sqrt(24)
    bias = rng.normal(scale=.3,size=100)
    raw = coefficients @ np.cos(((nodes[boundary]-center)/scale) @ frequencies.T+phases).T+bias[:,None]
    normalization = np.max(np.abs(raw),axis=1)
    np.savez(out/"fem_test_inputs.npz",frequencies=frequencies,phases=phases,
             coefficients=coefficients,bias=bias,normalization=normalization,
             boundary=(raw/normalization[:,None]).astype(np.float32))
    write_json(out/"generation_report.json",generation)
    write_json(out/"input_manifest.json",{p.name:digest(p) for p in out.iterdir() if p.suffix in (".json",".npz")})
    print(json.dumps(generation,indent=2))


def verify(out):
    target = out/"verification.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((out/"input_manifest.json").read_text())
    for name, checksum in manifest.items():
        if digest(out/name) != checksum:
            raise ValueError("Input changed: "+name)
    geometry = np.load(out/"geometry.npz")
    nodes = geometry["nodes"]
    low, high = nodes.min(0), nodes.max(0)
    report = {"manifest_unchanged":True,"sets":{},"test_error_not_evaluated":True,
              "source_hash":digest(__file__)}
    all_sources = []
    for name, count in (("train",2000),("validation",100),("mad_test",100)):
        a = np.load(out/(name+".npz"))
        sources, weights = a["sources"], a["weights"]
        if sources.shape != (count,10,3) or weights.shape != (count,10):
            raise ValueError("Wrong per-sample source count")
        all_sources.append(sources.reshape(-1,3))
        # Distance to an enclosing box lower-bounds distance to the enclosed domain.
        gap = np.maximum(np.maximum(low-sources,sources-high),0)
        clearance = np.linalg.norm(gap,axis=-1)
        if clearance.min() < .001*float(geometry["scale"])*(1-1e-10):
            raise ValueError("A source lacks the prescribed enclosing-box clearance")
        q = geometry["test_queries"] if name == "mad_test" else geometry["train_queries"]
        errors = []
        for key, points in (("boundary",nodes[geometry["boundary_ids"]]),("solution",q)):
            numerator, denominator = 0., 0.
            for i in range(count):
                # Independent per-source replay rather than the batched generator.
                value = np.zeros(len(points))
                for source, weight in zip(sources[i], weights[i]):
                    r = np.sqrt(np.sum((points-source)**2,axis=1))
                    value += weight/(4*np.pi*r)
                value /= a["normalization"][i]
                numerator += float(np.sum((value-a[key][i])**2))
                denominator += float(np.sum(value**2))
            error = float(np.sqrt(numerator/denominator))
            if error > 1e-6:
                raise ValueError("Stored label replay failed")
            errors.append(error)
        maximum = np.maximum(np.abs(a["boundary"]).max(1),np.abs(a["solution"]).max(1))
        np.testing.assert_allclose(maximum,1,atol=6e-8,rtol=0)
        report["sets"][name] = {"samples":count,"sources_per_sample":10,
                                "min_box_clearance_A":float(clearance.min()),
                                "storage_relative_error_boundary_solution":errors,
                                "joint_max_normalization_checked":True}
    all_sources = np.vstack(all_sources)
    if len(np.unique(all_sources,axis=0)) != len(all_sources):
        raise ValueError("Unexpected repeated source positions")
    report["all_source_positions_distinct_across_splits"] = True
    write_json(target, report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","verify","reference"])
    parser.add_argument("--out",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--mesh",type=Path,default=DEFAULT_MESH)
    parser.add_argument("--pqr",type=Path,default=DEFAULT_PQR)
    parser.add_argument("--seed-offset",type=int,default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out,args.mesh,args.pqr,args.seed_offset)
    elif args.stage == "verify":
        verify(args.out)
    else:
        reference(args.out)
