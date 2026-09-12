"""Fresh true-domain-exterior MAD1 data with unchanged geometry and FEM tests."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from molecular_geometry_sources import GeometrySourceSampler, MolecularGeometry, triangle_distances
from molecular_laplace_experiment import contained, digest, write_json
from molecular_laplace_original_style import fields, normalize_pair


CATEGORIES = ("box_exterior", "inside_box_exterior", "reentrant_exterior")
PRESERVED = ("geometry.npz", "fem_test_inputs.npz", "reference/report.json",
             "reference/level2_solutions.npy")


def generate_samples(geometry, boundary, queries, count, seed, relative_clearance=0.001):
    sampler = GeometrySourceSampler(geometry, seed, relative_clearance=relative_clearance)
    sources, weights, records = [], [], []
    for index in range(count):
        a, b, c = sampler.sample()
        sources.append(a)
        weights.append(b)
        records.append(c)
        if (index+1) % 250 == 0:
            print(json.dumps({"stage": "sampling", "seed": seed, "samples": index+1}), flush=True)
    sources, weights = np.asarray(sources), np.asarray(weights)
    g = np.empty((count, len(boundary)), dtype=np.float32)
    u = np.empty((count, len(queries)), dtype=np.float32)
    normalization = np.empty(count)
    for start in range(0, count, 16):
        sl = slice(start, start+16)
        g[sl], u[sl], normalization[sl] = normalize_pair(
            fields(boundary, sources[sl], weights[sl]), fields(queries, sources[sl], weights[sl]))
    sample = {"boundary": g, "solution": u, "sources": sources, "weights": weights,
              "normalization": normalization,
              "source_category": np.array([[CATEGORIES.index(r["kind"]) for r in row] for row in records], dtype=np.uint8),
              "source_clearance": np.array([[r["clearance"] for r in row] for row in records]),
              "source_inside_box": np.array([[r["inside_box"] for r in row] for row in records]),
              "source_inside_convex_hull": np.array([[r["inside_convex_hull"] for r in row] for row in records]),
              "nearest_boundary_face": np.array([[r["face"] for r in row] for row in records], dtype=np.int32)}
    return sample, dict(sampler.stats)


def prepare(base, out):
    if out.exists():
        raise FileExistsError("Preserve prior evidence; use a fresh data directory")
    for name, checksum in json.loads((base/"input_manifest.json").read_text()).items():
        if digest(base/name) != checksum:
            raise ValueError("Base data changed: " + name)
    report = json.loads((base/"reference/report.json").read_text())
    if digest(base/"reference/level2_solutions.npy") != report["test_solution_sha256"]:
        raise ValueError("Base reference hash mismatch")
    protocol = json.loads((base/"protocol.json").read_text())
    old_protocol_hash = digest(base/"protocol.json")
    out.mkdir(parents=True)
    protocol.update({
        "schema": "MAD_R3_molecular_Laplace_true_exterior_r3",
        "base_data": str(base), "base_protocol_sha256": old_protocol_hash,
        "source_rule": "Ten independently sampled sources per function; mixture: 50% box exterior, 30% box-interior material exterior, 20% convex-hull-interior material exterior; reject material and clearance band",
        "source_offset": "Strict distance to actual PL material domain greater than 0.001 times the largest bounding-box span; normalized threshold, not 0.001 Angstrom",
        "outside_box_proposal": "Standard normal offsets times the largest span plus 0.001 span; 50% one, 25% two, 25% three exterior coordinates; remaining coordinates uniform in box",
        "inside_box_proposal": "Uniform box proposals rejected inside material or clearance band; reentrant component additionally rejects outside convex hull",
        "mixture_fallback": "On convex domains the unavailable hull-interior stratum is assigned to box-interior exterior; on box-filling domains all sources use box exterior",
        "source_category_codes": dict(enumerate(CATEGORIES)),
        "source_geometry": "Eligibility includes concavities, tunnels and sealed cavities outside the material; finite draws need not visit every small component",
        "train_seed": protocol["train_seed"]+20,
        "validation_seed": protocol["validation_seed"]+20,
        "mad_test_seed": protocol["mad_test_seed"]+20,
        "development_disclosure": "Box-only runs stopped on user instruction because source support omitted the material exterior inside the box; change is geometric, not selected by test errors. Sampling probabilities fixed before formal generation; no old checkpoint or optimizer reused",
        "checkpoint": "Minimum MAD validation MSE; fresh model and optimizer, no earlier checkpoint reuse",
        "preserved_inputs": {name: digest(base/name) for name in PRESERVED},
        "reference_status": "Existing independent FEM reference retained unchanged; its refinement qualification flag remains binding",
        "generation_sources": {p.name: digest(p) for p in (
            Path(__file__), Path(__file__).with_name("molecular_geometry_sources.py"),
            Path(__file__).with_name("molecular_laplace_original_style.py"))},
    })
    # Freeze the distribution and seeds before constructing either training or test data.
    write_json(out/"protocol.json", protocol)
    for name in PRESERVED:
        dest = out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/name, dest)
        if digest(dest) != protocol["preserved_inputs"][name]:
            raise ValueError("Preserved file copy failed")
    raw = np.load(out/"geometry.npz")
    start = time.perf_counter()
    geometry = MolecularGeometry(raw["nodes"], raw["tetra"])
    write_json(out/"geometry_report.json", {"topology": geometry.topology,
               "required_clearance": .001*geometry.scale,
               "geometry_setup_seconds": time.perf_counter()-start,
               "geometry_sha256": digest(out/"geometry.npz"),
               "source_geometry_rule": protocol["source_rule"]})
    generation = {}
    for name, count, seed_key, query_key in (
            ("train", protocol["ntrain"], "train_seed", "train_queries"),
            ("validation", protocol["nvalidation"], "validation_seed", "train_queries"),
            ("mad_test", protocol["ntest"], "mad_test_seed", "test_queries")):
        start = time.perf_counter()
        samples, rejections = generate_samples(geometry, raw["nodes"][raw["boundary_ids"]],
                                               raw[query_key], count, protocol[seed_key])
        np.savez(out/(name+".npz"), **samples)
        generation[name] = {"count": count,
            "seconds_including_sampling_evaluation_and_write": time.perf_counter()-start,
            "category_counts": {c: int((samples["source_category"] == i).sum()) for i, c in enumerate(CATEGORIES)},
            "sources_inside_box": int(samples["source_inside_box"].sum()),
            "sources_inside_hull": int(samples["source_inside_convex_hull"].sum()),
            "min_clearance": float(samples["source_clearance"].min()),
            "proposal_rejections": rejections}
        print(json.dumps({name: generation[name]}), flush=True)
    write_json(out/"generation_report.json", generation)
    manifest = {str(p.relative_to(out)).replace("\\", "/"): digest(p)
                for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out/"input_manifest.json", manifest)


def verify(out):
    if (out/"verification.json").exists():
        raise FileExistsError("Preserve the existing verification record")
    for name, checksum in json.loads((out/"input_manifest.json").read_text()).items():
        if digest(out/name) != checksum:
            raise ValueError("Input changed: " + name)
    protocol = json.loads((out/"protocol.json").read_text())
    for name, checksum in protocol["preserved_inputs"].items():
        if digest(out/name) != checksum:
            raise ValueError("Preserved test or geometry changed: " + name)
    raw = np.load(out/"geometry.npz")
    geometry = MolecularGeometry(raw["nodes"], raw["tetra"])
    report = {"manifest_unchanged": True, "preserved_geometry_and_fem_tests": True,
              "test_error_not_evaluated": True, "sets": {}, "source_hash": digest(__file__)}
    all_sources = []
    for name, count in (("train", protocol["ntrain"]), ("validation", protocol["nvalidation"]),
                         ("mad_test", protocol["ntest"])):
        samples = np.load(out/(name+".npz"))
        sources, weights = samples["sources"], samples["weights"]
        if sources.shape != (count, 10, 3) or weights.shape != (count, 10):
            raise ValueError("Wrong source or coefficient shape")
        if not all(np.isfinite(samples[k]).all() for k in samples.files):
            raise ValueError("Nonfinite dataset value")
        flat = sources.reshape(-1, 3)
        all_sources.append(flat)
        # Check every source against all tetrahedra, independently of the tree filter.
        if contained(flat, geometry.nodes, geometry.tetra).any():
            raise ValueError("A source is inside the material")
        distance = np.array([geometry.clearance(p) for p in flat])
        if np.any(distance <= .001*geometry.scale):
            raise ValueError("Source violates global clearance")
        np.testing.assert_allclose(distance, samples["source_clearance"].ravel(), rtol=1e-11)
        for point in flat[::max(1, len(flat)//100)]:
            exact = triangle_distances(point, geometry.triangles).min()
            np.testing.assert_allclose(geometry.clearance(point), exact, rtol=1e-10, atol=1e-11*geometry.scale)
        queries = raw["test_queries"] if name == "mad_test" else raw["train_queries"]
        errors = {}
        for key, points in (("boundary", raw["nodes"][raw["boundary_ids"]]), ("solution", queries)):
            numerator, denominator = 0., 0.
            for i in range(count):
                value = np.zeros(len(points))
                for source, weight in zip(sources[i], weights[i]):
                    value += weight/(4*np.pi*np.sqrt(((points-source)**2).sum(axis=1)))
                value /= samples["normalization"][i]
                numerator += float(((value-samples[key][i])**2).sum())
                denominator += float((value**2).sum())
            errors[key] = float(np.sqrt(numerator/denominator))
            if errors[key] > 1e-6:
                raise ValueError("Independent label replay failed")
        maximum = np.maximum(np.abs(samples["boundary"]).max(1), np.abs(samples["solution"]).max(1))
        np.testing.assert_allclose(maximum, 1, atol=6e-8, rtol=0)
        report["sets"][name] = {"samples": count, "all_sources_outside_material": True,
             "global_clearance_verified_all_sources": True,
             "min_clearance": float(distance.min()), "required_clearance": .001*geometry.scale,
             "full_triangle_distance_spotcheck": True, "label_storage_relative_error": errors,
             "joint_max_normalization_checked": True}
        print(json.dumps({"verified": name, **report["sets"][name]}), flush=True)
    all_sources = np.vstack(all_sources)
    if len(np.unique(all_sources, axis=0)) != len(all_sources):
        raise ValueError("Repeated sources across functions or splits")
    report["all_source_positions_distinct_across_splits"] = True
    write_json(out/"verification.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "verify"])
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.base, args.out)
    else:
        verify(args.out)
