"""Check the flower-hole finite-element labels on a finer independent mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmsh
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import factorized

from flower_holes_laplace_experiment import (
    DEFAULT_MODEL,
    _component_parameter,
    load_geometry,
    sample_independent_boundary,
    smooth_fourier_values,
)


def polygon_curves(coordinates: np.ndarray, mesh_size: float) -> tuple[int, list[int]]:
    points = [
        gmsh.model.geo.addPoint(float(x), float(y), 0.0, mesh_size)
        for x, y in coordinates
    ]
    lines = [
        gmsh.model.geo.addLine(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    return gmsh.model.geo.addCurveLoop(lines), lines


def assemble_stiffness(nodes: np.ndarray, triangles: np.ndarray) -> csc_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for ids in triangles:
        coordinates = nodes[ids]
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0])
        )
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        local = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for local_row, global_row in enumerate(ids):
            for local_column, global_column in enumerate(ids):
                rows.append(int(global_row))
                columns.append(int(global_column))
                values.append(float(local[local_row, local_column]))
    return coo_matrix(
        (values, (rows, columns)), shape=(len(nodes), len(nodes))
    ).tocsc()


def generate_mesh(mesh_size: float):
    coarse = load_geometry(DEFAULT_MODEL)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("flower_holes_refined")
    outer_loop, outer_lines = polygon_curves(coarse.outer_curve, mesh_size)
    hole_loops: list[int] = []
    hole_lines: list[list[int]] = []
    for curve in coarse.inner_curves:
        loop, lines = polygon_curves(curve[::-1], mesh_size)
        hole_loops.append(loop)
        hole_lines.append(lines)
    gmsh.model.geo.addPlaneSurface([outer_loop, *hole_loops])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = coordinates.reshape(-1, 3)[:, :2]
    tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
    element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
    triangle_nodes = None
    for element_type, flattened in zip(element_types, element_nodes):
        if element_type == 2:
            triangle_nodes = flattened.reshape(-1, 3)
            break
    if triangle_nodes is None:
        gmsh.finalize()
        raise RuntimeError("No first-order triangles were generated.")
    triangles = np.vectorize(lambda tag: tag_to_index[int(tag)])(triangle_nodes)

    def nodes_on(lines: list[int]) -> np.ndarray:
        tags: set[int] = set()
        for line in lines:
            line_tags, _, _ = gmsh.model.mesh.getNodes(1, line, includeBoundary=True)
            tags.update(int(tag) for tag in line_tags)
        return np.array(sorted(tag_to_index[tag] for tag in tags), dtype=np.int64)

    outer_ids = nodes_on(outer_lines)
    component_ids = [nodes_on(lines) for lines in hole_lines]
    inner_ids = np.concatenate(component_ids)
    boundary_ids = np.concatenate((outer_ids, inner_ids))
    all_ids = np.arange(len(nodes), dtype=np.int64)
    free_ids = np.setdiff1d(all_ids, boundary_ids, assume_unique=False)
    gmsh.finalize()
    return coarse, nodes, triangles, outer_ids, component_ids, boundary_ids, free_ids


def component_values(
    rng: np.random.Generator,
    coarse,
    outer_coordinates: np.ndarray,
    hole_coordinates: list[np.ndarray],
    modes: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    outer = smooth_fourier_values(
        rng, _component_parameter(outer_coordinates, None), modes
    )
    holes = [
        smooth_fourier_values(
            rng, _component_parameter(coordinates, coarse.holes[index]), modes
        )
        for index, coordinates in enumerate(hole_coordinates)
    ]
    return outer, holes


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "artifacts"
        / "r3_revision"
        / "flower_holes"
        / "fem_refinement.json",
    )
    args = parser.parse_args()

    (
        coarse,
        nodes,
        triangles,
        outer_ids,
        component_ids,
        boundary_ids,
        free_ids,
    ) = generate_mesh(args.mesh_size)
    stiffness = assemble_stiffness(nodes, triangles)
    solve = factorized(stiffness[free_ids][:, free_ids])
    stiffness_fb = stiffness[free_ids][:, boundary_ids]
    coarse_rows = np.load(
        repository
        / "data"
        / "r3_revision"
        / "flower_holes"
        / "fem_fourier_test_n200.npy",
        mmap_mode="r",
    )
    coarse_boundary_count = len(coarse.boundary_ids)
    rng = np.random.default_rng(args.seed)
    differences: list[float] = []
    coarse_regeneration_errors: list[float] = []

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    for sample in range(args.samples):
        outer_values, hole_values = component_values(
            rng,
            coarse,
            nodes[outer_ids],
            [nodes[ids] for ids in component_ids],
            args.modes,
        )
        boundary = np.concatenate((outer_values, *hole_values))
        solution = np.empty(len(nodes), dtype=np.float64)
        solution[boundary_ids] = boundary
        solution[free_ids] = solve(-(stiffness_fb @ boundary))

        stored_boundary = np.asarray(coarse_rows[sample, :coarse_boundary_count])
        stored_solution = np.asarray(coarse_rows[sample, coarse_boundary_count:])
        coarse_rng = np.random.default_rng(args.seed)
        for _ in range(sample + 1):
            regenerated_boundary = sample_independent_boundary(
                coarse_rng, coarse, args.modes
            )
        scale = float(np.max(np.abs(regenerated_boundary)))
        regenerated_boundary /= scale
        coarse_regeneration_errors.append(
            float(np.max(np.abs(regenerated_boundary - stored_boundary)))
        )
        refined_solution = solution / scale
        interpolator = mtri.LinearTriInterpolator(triangulation, refined_solution)
        refined_on_coarse = np.asarray(
            interpolator(coarse.nodes[:, 0], coarse.nodes[:, 1]).filled(np.nan)
        )
        valid = np.isfinite(refined_on_coarse)
        differences.append(
            float(
                np.linalg.norm(refined_on_coarse[valid] - stored_solution[valid])
                / np.linalg.norm(refined_on_coarse[valid])
            )
        )

    report = {
        "coarse": {
            "mesh_size": coarse.mesh_size,
            "nodes": len(coarse.nodes),
            "triangles": len(coarse.triangles),
        },
        "refined": {
            "mesh_size": args.mesh_size,
            "nodes": len(nodes),
            "triangles": len(triangles),
        },
        "samples": args.samples,
        "seed": args.seed,
        "mean_relative_l2_difference": float(np.mean(differences)),
        "maximum_relative_l2_difference": float(np.max(differences)),
        "maximum_boundary_regeneration_difference": float(
            np.max(coarse_regeneration_errors)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
