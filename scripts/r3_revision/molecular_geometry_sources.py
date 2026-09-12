"""Geometry-aware, fresh-per-function MAD1 source sampling and read-only audit.

This does not replace the running box-exterior datasets or train any model.
All containment and distance decisions refer to the supplied tetrahedral PL domain.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from molecular_laplace_experiment import ROOT, contained, digest, volumes, write_json
from molecular_laplace_original_style import fields, normalize_pair


def graph_components(count, pairs):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)


def triangle_distances(point, triangles):
    """Distances to planar triangles, using interior projections and three edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b-a, c-a, point-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    d20, d21 = (ap*ab).sum(1), (ap*ac).sum(1)
    denominator = d00*d11-d01*d01
    if np.any(denominator <= 0):
        raise ValueError("Degenerate surface triangle")
    v = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    normal = np.cross(ab, ac)
    plane = np.abs((ap*normal).sum(1))/np.linalg.norm(normal, axis=1)
    distances = []
    for first, second in ((a, b), (b, c), (c, a)):
        direction = second-first
        alpha = np.clip(((point-first)*direction).sum(1)/(direction*direction).sum(1), 0, 1)
        distances.append(np.linalg.norm(point-first-alpha[:, None]*direction, axis=1))
    edge_distance = np.min(distances, axis=0)
    return np.where((v >= 0) & (w >= 0) & (v+w <= 1), plane, edge_distance)


class MolecularGeometry:
    def __init__(self, nodes, tetra):
        self.nodes, self.tetra = np.asarray(nodes), np.asarray(tetra)
        self.low, self.high = self.nodes.min(0), self.nodes.max(0)
        self.scale = float(np.max(self.high-self.low))
        self.volume = float(volumes(self.nodes, self.tetra).sum())
        self.hull = ConvexHull(self.nodes)
        self.has_reentrant_space = 1-self.volume/self.hull.volume > 1e-8
        self.has_box_exterior_space = 1-self.volume/np.prod(self.high-self.low) > 1e-8
        all_faces = np.concatenate([np.delete(self.tetra, i, axis=1) for i in range(4)])
        owners = np.tile(np.arange(len(self.tetra)), 4)
        unique, inverse, counts = np.unique(np.sort(all_faces, axis=1), axis=0,
                                            return_inverse=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Nonmanifold tetrahedral face")
        face_owners = defaultdict(list)
        for index, owner in zip(inverse, owners):
            face_owners[int(index)].append(int(owner))
        self.faces = unique[counts == 1].copy()
        self.owner = np.array([face_owners[int(i)][0] for i in np.flatnonzero(counts == 1)])
        self.centers = self.nodes[self.faces].mean(1)
        normal = np.cross(self.nodes[self.faces[:, 1]]-self.nodes[self.faces[:, 0]],
                          self.nodes[self.faces[:, 2]]-self.nodes[self.faces[:, 0]])
        inward = (normal*(self.nodes[self.tetra[self.owner]].mean(1)-self.centers)).sum(1) > 0
        self.faces[inward] = self.faces[inward][:, [0, 2, 1]]
        normal[inward] *= -1
        norm = np.linalg.norm(normal, axis=1)
        if np.any(norm <= 0):
            raise ValueError("Degenerate surface")
        self.normals, self.area = normal/norm[:, None], norm/2
        self.triangles = self.nodes[self.faces]
        self.surface_tree = cKDTree(self.centers)
        self.surface_radius = float(np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max())
        self.vertex_tree = cKDTree(self.nodes[np.unique(self.faces)])
        self.origin = self.nodes[self.tetra[:, 0]]
        self.inverse = np.linalg.inv((self.nodes[self.tetra[:, 1:]]-self.origin[:, None]).transpose(0, 2, 1))
        centers = self.nodes[self.tetra].mean(1)
        self.tet_tree = cKDTree(centers)
        self.tet_radius = float(np.linalg.norm(self.nodes[self.tetra]-centers[:, None], axis=2).max())
        self.topology = self._topology(face_owners, counts)

    def _topology(self, face_owners, counts):
        edge_faces = defaultdict(list)
        link_edges = defaultdict(list)
        for index, face in enumerate(self.faces):
            for j in range(3):
                a, b = int(face[j]), int(face[(j+1) % 3])
                edge_faces[tuple(sorted((a, b)))].append((index, 1 if a < b else -1))
                other = [int(face[k]) for k in range(3) if k != j]
                link_edges[int(face[j])].append(other)
        if any(len(owners) != 2 or owners[0][1]+owners[1][1] != 0 for owners in edge_faces.values()):
            raise ValueError("Surface is not closed and consistently oriented")
        for vertex, edges in link_edges.items():
            adjacent = defaultdict(set)
            for a, b in edges:
                adjacent[a].add(b)
                adjacent[b].add(a)
            if any(len(v) != 2 for v in adjacent.values()):
                raise ValueError("Nonmanifold vertex link: " + str(vertex))
            remaining = set(adjacent)
            queue = [remaining.pop()]
            while queue:
                fresh = adjacent[queue.pop()] & remaining
                remaining.difference_update(fresh)
                queue.extend(fresh)
            if remaining:
                raise ValueError("Disconnected vertex link: " + str(vertex))
        count, labels = graph_components(len(self.faces), [(p[0][0], p[1][0]) for p in edge_faces.values()])
        self.component_labels = labels
        records = []
        for component in range(count):
            faces = self.faces[labels == component]
            edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
            chi = len(np.unique(faces))-len(edges)+len(faces)
            records.append({"id": component, "vertices": len(np.unique(faces)), "edges": len(edges),
                            "faces": len(faces), "euler_characteristic": int(chi), "genus": int((2-chi)//2)})
        tet_count, _ = graph_components(len(self.tetra), [p for p in face_owners.values() if len(p) == 2])
        edges = np.unique(np.sort(np.concatenate([self.tetra[:, list(pair)] for pair in combinations(range(4), 2)]), axis=1), axis=0)
        volume_chi = len(np.unique(self.tetra))-len(edges)+len(counts)-len(self.tetra)
        if sum(r["euler_characteristic"] for r in records) != 2*volume_chi:
            raise ValueError("Boundary/volume Euler identity failed")
        if tet_count != 1:
            raise ValueError("This experiment expects one connected material domain")
        return {"tetrahedral_components": int(tet_count), "boundary_components": int(count),
                "closed_oriented_edges": True, "all_vertex_links_single_cycles": True,
                "boundary": records, "volume_euler_characteristic": int(volume_chi),
                "bounded_complement_components_inferred": int(count-1),
                "scope": "Combinatorial audit of the supplied conforming PL mesh, not certification of the original continuous molecule"}

    def contains(self, point):
        candidates = self.tet_tree.query_ball_point(point, self.tet_radius*(1+1e-12))
        if not candidates:
            return False
        bary = np.einsum("tij,tj->ti", self.inverse[candidates], point-self.origin[candidates])
        return bool(np.any((bary >= -1e-9).all(1) & (bary.sum(1) <= 1+1e-9)))

    def nearest_boundary(self, point):
        upper, _ = self.vertex_tree.query(point)
        # A triangle improving this vertex bound has its center inside this ball.
        candidates = self.surface_tree.query_ball_point(point, (upper+self.surface_radius)*(1+1e-12))
        distances = triangle_distances(point, self.triangles[candidates])
        index = int(np.argmin(distances))
        return float(distances[index]), int(candidates[index])

    def clearance(self, point):
        return self.nearest_boundary(point)[0]

    def in_hull(self, point):
        return bool(np.all(self.hull.equations[:, :3] @ point+self.hull.equations[:, 3] <= 1e-10*self.scale))

    def in_box(self, point):
        return bool(np.all((point >= self.low) & (point <= self.high)))


class GeometrySourceSampler:
    def __init__(self, geometry, seed, relative_clearance=0.001):
        self.geometry, self.rng = geometry, np.random.default_rng(seed)
        if not np.isfinite(relative_clearance) or relative_clearance <= 0:
            raise ValueError("Positive finite relative clearance required")
        self.relative_clearance = float(relative_clearance)
        self.stats = defaultdict(int)

    def in_box_exterior(self, reentrant=False):
        g = self.geometry
        for _ in range(20000):
            self.stats["inside_box_candidates"] += 1
            point = self.rng.uniform(g.low, g.high)
            if reentrant and not g.in_hull(point):
                self.stats["rejected_outside_hull_for_reentrant_stratum"] += 1
                continue
            if g.contains(point):
                self.stats["rejected_inside_material"] += 1
                continue
            distance, face = g.nearest_boundary(point)
            if distance <= self.relative_clearance*g.scale:
                self.stats["rejected_insufficient_clearance"] += 1
                continue
            return point, face, distance
        raise RuntimeError("Cannot fill the requested geometry stratum; do not silently change the sampling law")

    def outside_box(self):
        g = self.geometry
        # The 3D extension also samples edge sectors (two exterior coordinates).
        count = 1 if self.rng.random() < .5 else int(self.rng.integers(2, 4))
        axes = self.rng.choice(3, count, replace=False)
        point = self.rng.uniform(g.low, g.high)
        z = self.rng.standard_normal(count)
        point[axes] = np.where(z <= 0, g.low[axes]+g.scale*(z-self.relative_clearance),
                              g.high[axes]+g.scale*(z+self.relative_clearance))
        distance, face = g.nearest_boundary(point)
        if g.contains(point) or distance <= self.relative_clearance*g.scale:
            raise ValueError("Invalid box-exterior source")
        return point, face, distance

    def sample(self):
        g = self.geometry
        points, records = [], []
        for _ in range(10):
            draw = self.rng.random()
            if not g.has_box_exterior_space or draw < .5:
                point, face, distance = self.outside_box()
                kind = "box_exterior"
            else:
                reentrant = g.has_reentrant_space and draw >= .8
                point, face, distance = self.in_box_exterior(reentrant=reentrant)
                kind = "reentrant_exterior" if reentrant else "inside_box_exterior"
            points.append(point)
            records.append({"kind": kind, "nearest_boundary_component": int(g.component_labels[face]),
                            "face": face, "clearance": distance,
                            "inside_convex_hull": g.in_hull(point), "inside_box": g.in_box(point)})
        weights = self.rng.standard_normal(10)
        return np.asarray(points), weights, records


def audit(data, output, samples=100):
    output.mkdir(parents=True, exist_ok=False)
    raw = np.load(data/"geometry.npz")
    g = MolecularGeometry(raw["nodes"], raw["tetra"])
    settings = {"geometry_sha256": digest(data/"geometry.npz"), "seed": 2026090942,
                "samples": samples, "sources_per_sample": 10,
                "source_category_selection": "Independent for each of the ten sources, not fixed category counts per function",
                "category_probabilities": {"box_exterior": .5 if g.has_box_exterior_space else 1.,
                                           "inside_box_exterior": (.3 if g.has_reentrant_space else .5) if g.has_box_exterior_space else 0.,
                                           "reentrant_exterior": .2 if g.has_reentrant_space else 0.},
                "inside_box_sampling": "Uniform box proposals rejected inside the material or its clearance band; the reentrant category additionally requires being inside the convex hull",
                "outside_box_sampling": "Original normal exterior offsets; 50% one, 25% two and 25% three exterior coordinates in 3D; remaining coordinates uniform in the box",
                "clearance": "Strictly greater than 0.001 times the largest span, measured to all boundary triangles; 0.001 refers to normalized coordinates, not Angstrom",
                "eligible_space": "All points outside the actual PL domain with the prescribed clearance; no extra near-distance interval",
                "finite_sampling": "No guarantee of sampling every small cavity in each finite ten-source realization",
                "weights": "Independent standard normal, fresh for each sample",
                "purpose": "Geometry and label audit only; no model trained and no test error used",
                "source_sha256": digest(__file__)}
    write_json(output/"protocol.json", settings)
    sampler = GeometrySourceSampler(g, settings["seed"])
    sources, weights, provenance = [], [], []
    for _ in range(samples):
        a, b, c = sampler.sample()
        sources.append(a); weights.append(b); provenance.append(c)
    sources, weights = np.asarray(sources), np.asarray(weights)
    flat = sources.reshape(-1, 3)
    if len(np.unique(flat, axis=0)) != len(flat):
        raise ValueError("Unexpected reused source position")
    # Independent exhaustive test avoids relying solely on the spatial broad phase.
    if contained(flat, g.nodes, g.tetra).any():
        raise ValueError("Independent containment audit failed")
    for point in flat[::max(1, len(flat)//100)]:
        exact = float(triangle_distances(point, g.triangles).min())
        if not np.isclose(exact, g.clearance(point), rtol=1e-10, atol=1e-11*g.scale):
            raise ValueError("Distance broad phase missed a triangle")
    boundary = raw["nodes"][raw["boundary_ids"]]
    queries = raw["train_queries"]
    labels = []
    for begin in range(0, samples, 10):
        s, w = sources[begin:begin+10], weights[begin:begin+10]
        a, b, _ = normalize_pair(fields(boundary, s, w), fields(queries, s, w))
        labels.append((a, b))
    np.savez(output/"audit_samples.npz", sources=sources, weights=weights,
             boundary=np.vstack([x[0] for x in labels]).astype("f4"),
             solution=np.vstack([x[1] for x in labels]).astype("f4"))
    records = [r for sample in provenance for r in sample]
    report = {"topology": g.topology, "samples": samples, "source_count": len(flat),
              "all_sources_distinct": True, "independent_exhaustive_containment_passed": True,
              "distance_broad_phase_crosscheck_passed": True,
              "min_surface_clearance": min(r["clearance"] for r in records),
              "required_clearance": .001*g.scale,
              "inside_box_but_outside_material": sum(r["inside_box"] for r in records),
              "inside_hull_but_outside_material": sum(r["inside_convex_hull"] for r in records),
              "category_counts": {kind: sum(r["kind"] == kind for r in records) for kind in ("box_exterior", "inside_box_exterior", "reentrant_exterior")},
              "nearest_boundary_component_counts": {str(c): sum(r["nearest_boundary_component"] == c for r in records) for c in range(g.topology["boundary_components"])},
              "rejections": dict(sampler.stats),
              "existing_data_unchanged": digest(data/"geometry.npz") == settings["geometry_sha256"],
              "training_started": False, "no_claim_of_optimal_source_distribution": True}
    write_json(output/"source_provenance.json", provenance)
    write_json(output/"report.json", report)
    print(__import__("json").dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    audit(args.data, args.out, args.samples)
