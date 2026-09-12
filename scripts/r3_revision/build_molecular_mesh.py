# gmsh_build_7gly_conforming_mesh.py
# ============================================================
# Build a conforming tetrahedral mesh for 7_GLY.pqr:
#   outer box + union of atom spheres
#   molecule / solvent conforming volumes
#   molecule-solvent interface physical surface
#   outer box boundary physical surface
#   distance-based local refinement near molecular interface
#
# Coordinates and mesh sizes are in Angstrom.
# ============================================================

import argparse
from pathlib import Path
import math
import json
import hashlib
import gmsh
import numpy as np


def parse_pqr(path: str):
    atoms = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                x, y, z = map(float, parts[-5:-2])
                q = float(parts[-2])
                r = float(parts[-1])
                atoms.append((x, y, z, q, r, line.rstrip()))
            except ValueError:
                continue
    if not atoms:
        raise RuntimeError(f"No atoms parsed from {path}")
    return atoms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--pqr", required=True)
    p.add_argument("--out", default="fem_meshes/7_GLY_conforming_coarse.msh")
    p.add_argument("--report", default="fem_meshes/7_GLY_conforming_coarse_report.json")
    p.add_argument("--padding_A", type=float, default=10.0)
    p.add_argument(
        "--problem_npz",
        default="",
        help="Optional sharp-PNP problem whose anisotropic box bounds are used exactly.",
    )
    p.add_argument("--box_lo_A", type=float, nargs=3, default=None)
    p.add_argument("--box_hi_A", type=float, nargs=3, default=None)

    p.add_argument("--h_interface_A", type=float, default=0.30)
    p.add_argument("--h_outer_A", type=float, default=0.50)
    p.add_argument("--h_bulk_A", type=float, default=1.00)

    p.add_argument("--dist_inner_A", type=float, default=0.40)
    p.add_argument("--dist_outer_A", type=float, default=1.50)
    p.add_argument("--charge_sigma_A", type=float, default=0.0,
                   help="Optional PQR-centered mesh sizing only; 0 preserves the legacy mesh field.")
    p.add_argument("--charge_h_over_sigma", type=float, default=0.5)
    p.add_argument("--charge_core_over_sigma", type=float, default=3.0)
    p.add_argument("--charge_transition_over_sigma", type=float, default=1.0)

    p.add_argument("--min_radius_A", type=float, default=0.50)
    p.add_argument("--radius_scale", type=float, default=1.0)
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--algorithm3d", type=int, default=10)
    p.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Optional Gmsh thread cap (0 keeps the Gmsh default).",
    )
    p.add_argument(
        "--disable_netgen_optimization",
        action="store_true",
        help="Keep Gmsh optimization but skip the extra Netgen optimization pass.",
    )
    p.add_argument(
        "--individual_fragment",
        action="store_true",
        help="Fragment the box against individual atom spheres (robust for large unions).",
    )
    p.add_argument(
        "--sequential_fuse",
        action="store_true",
        help="Fuse atom spheres sequentially before fragmenting the outer box.",
    )
    p.add_argument(
        "--chunk_fuse_size",
        type=int,
        default=0,
        help="Fuse spatially ordered atom-sphere chunks before box fragmentation (0 disables).",
    )
    p.add_argument("--save_geo_unrolled", action="store_true")
    return p


def charge_core_settings(args):
    """Sizing parameters, not new CAD spheres, sources, or embedded points."""
    sigma = float(args.charge_sigma_A)
    ratios = [args.charge_h_over_sigma, args.charge_core_over_sigma,
              args.charge_transition_over_sigma]
    if not math.isfinite(sigma) or sigma < 0 or any(
        not math.isfinite(x) or x <= 0 for x in ratios
    ):
        raise ValueError("Charge sizing needs finite sigma>=0 and positive ratios")
    if sigma == 0:
        return None
    if sigma * ratios[0] > args.h_bulk_A:
        raise ValueError("Charge-core size must not exceed bulk size")
    return {"sigma_A": sigma, "h_core_A": sigma * ratios[0],
            "radius_A": sigma * ratios[1], "transition_A": sigma * ratios[2]}


def main():
    args = build_parser().parse_args()
    charge_settings = charge_core_settings(args)

    out_path = Path(args.out).resolve()
    report_path = Path(args.report).resolve()
    if out_path.exists() or report_path.exists():
        raise FileExistsError("Use fresh mesh and report paths")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    atoms = parse_pqr(args.pqr)

    xyz_array = np.asarray([(a[0], a[1], a[2]) for a in atoms], dtype=np.float64)
    charge_array = np.asarray([a[3] for a in atoms], dtype=np.float64)
    radii_raw_array = np.asarray([a[4] for a in atoms], dtype=np.float64)
    xyz = [tuple(row) for row in xyz_array]
    radii_raw = radii_raw_array.tolist()

    problem_path = None
    problem_box_lo = None
    problem_box_hi = None
    if args.problem_npz:
        problem_path = Path(args.problem_npz).resolve()
        with np.load(problem_path, allow_pickle=False) as problem:
            problem_xyz = np.asarray(problem["atom_xyz"], dtype=np.float64)
            problem_q = np.asarray(problem["atom_q"], dtype=np.float64)
            problem_radius = np.asarray(problem["atom_radius"], dtype=np.float64)
            problem_box_lo = np.asarray(problem["box_lo"], dtype=np.float64)
            problem_box_hi = np.asarray(problem["box_hi"], dtype=np.float64)
        if problem_xyz.shape != xyz_array.shape or not np.allclose(
            problem_xyz, xyz_array, rtol=0.0, atol=1.0e-10
        ):
            raise RuntimeError("PQR coordinates differ from --problem_npz.")
        if problem_q.shape != charge_array.shape or not np.allclose(
            problem_q, charge_array, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError("PQR charges differ from --problem_npz.")
        if problem_radius.shape != radii_raw_array.shape or not np.allclose(
            problem_radius, radii_raw_array, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError("PQR radii differ from --problem_npz.")
        if np.any(problem_box_hi <= problem_box_lo):
            raise RuntimeError("Invalid box bounds in --problem_npz.")
    if args.box_lo_A is not None or args.box_hi_A is not None:
        if args.problem_npz or args.box_lo_A is None or args.box_hi_A is None:
            raise ValueError("Explicit box needs both bounds and no --problem_npz")
        problem_box_lo = np.asarray(args.box_lo_A, dtype=np.float64)
        problem_box_hi = np.asarray(args.box_hi_A, dtype=np.float64)
        if not np.isfinite([problem_box_lo, problem_box_hi]).all() or np.any(problem_box_hi <= problem_box_lo):
            raise ValueError("Invalid explicit box bounds")

    radii = []
    zero_or_small = []
    for i, r in enumerate(radii_raw):
        rr = r * args.radius_scale
        if rr <= 0.0:
            zero_or_small.append(i)
            rr = args.min_radius_A
        radii.append(rr)

    replacement_containment = []
    for index in zero_or_small:
        distances = np.linalg.norm(xyz_array - xyz_array[index], axis=1)
        available = radii_raw_array * float(args.radius_scale) - distances
        available[index] = -np.inf
        covering_atom = int(np.argmax(available))
        margin = float(available[covering_atom] - radii[index])
        replacement_containment.append(
            {
                "atom_index_zero_based": int(index),
                "covering_atom_index_zero_based": covering_atom,
                "containment_margin_A": margin,
            }
        )
        if margin < -1.0e-10:
            raise RuntimeError(
                f"Radius replacement for atom {index} changes the exposed union "
                f"geometry; best containment margin is {margin:.6e} A."
            )

    xmin = min(x - r for (x, y, z), r in zip(xyz, radii))
    xmax = max(x + r for (x, y, z), r in zip(xyz, radii))
    ymin = min(y - r for (x, y, z), r in zip(xyz, radii))
    ymax = max(y + r for (x, y, z), r in zip(xyz, radii))
    zmin = min(z - r for (x, y, z), r in zip(xyz, radii))
    zmax = max(z + r for (x, y, z), r in zip(xyz, radii))

    if problem_box_lo is not None:
        molecule_lo = np.asarray([xmin, ymin, zmin])
        molecule_hi = np.asarray([xmax, ymax, zmax])
        if np.any(molecule_lo < problem_box_lo - 1.0e-10) or np.any(
            molecule_hi > problem_box_hi + 1.0e-10
        ):
            raise RuntimeError(
                "Molecular spheres, including radius replacements, escape the problem box."
            )
        box_lo = problem_box_lo
        box_hi = problem_box_hi
    else:
        molecule_center = np.asarray(
            [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
        )
        half_extent_default = (
            0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin) + args.padding_A
        )
        box_lo = molecule_center - half_extent_default
        box_hi = molecule_center + half_extent_default
    center = 0.5 * (box_lo + box_hi)
    half_extents = 0.5 * (box_hi - box_lo)
    cx, cy, cz = center.tolist()
    bx0, by0, bz0 = box_lo.tolist()
    box_size_x, box_size_y, box_size_z = (box_hi - box_lo).tolist()
    half_extent = float(np.max(half_extents))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("7_GLY_conforming_mesh")
    occ = gmsh.model.occ

    box = occ.addBox(
        bx0,
        by0,
        bz0,
        box_size_x,
        box_size_y,
        box_size_z,
    )

    spheres = []
    for (x, y, z), r in zip(xyz, radii):
        spheres.append((3, occ.addSphere(x, y, z, r)))

    # Fuse all atom spheres into one molecular volume by default.  For larger
    # molecules, OCC may keep the fused solid as a compound that does not
    # participate correctly in the box boolean; individual fragmentation is
    # slower but produces a conforming partition robustly.
    if args.individual_fragment:
        molecule_entities = spheres
    elif args.chunk_fuse_size and args.chunk_fuse_size > 0:
        xyz_arr = np.asarray(xyz, dtype=float)
        order = np.lexsort((xyz_arr[:, 2], xyz_arr[:, 1], xyz_arr[:, 0]))
        molecule_entities = []
        chunk_size = int(args.chunk_fuse_size)
        for start in range(0, len(order), chunk_size):
            batch = [spheres[int(i)] for i in order[start : start + chunk_size]]
            if len(batch) == 1:
                fused_batch = batch
            else:
                fused_step, _ = occ.fuse(
                    [batch[0]], batch[1:], removeObject=True, removeTool=True
                )
                fused_batch = [entity for entity in fused_step if entity[0] == 3]
            if not fused_batch:
                gmsh.finalize()
                raise RuntimeError("Chunk atom-sphere fuse produced no molecular volume.")
            molecule_entities.extend(fused_batch)
            occ.synchronize()
    elif args.sequential_fuse:
        current = [spheres[0]]
        occ.synchronize()
        for sphere in spheres[1:]:
            fused_step, _ = occ.fuse(
                current, [sphere], removeObject=True, removeTool=True
            )
            current = [entity for entity in fused_step if entity[0] == 3]
            if not current:
                gmsh.finalize()
                raise RuntimeError(
                    "Sequential atom-sphere fuse did not produce a molecular volume."
                )
            occ.synchronize()
        # A PQR can contain disconnected atom-sphere components (e.g. due to
        # conservative radii or missing covalent overlap).  Keep all resulting
        # solids; the subsequent box fragment and physical-group assignment
        # handles them as one molecule region.
        molecule_entities = current
    elif len(spheres) == 1:
        molecule_entities = spheres
    else:
        fused, _ = occ.fuse([spheres[0]], spheres[1:], removeObject=True, removeTool=True)
        molecule_entities = [e for e in fused if e[0] == 3]

    if not molecule_entities:
        gmsh.finalize()
        raise RuntimeError("Boolean fuse of atom spheres produced no molecular volume.")

    # Fragment outer box with molecular volume(s), preserving conforming interface.
    occ.fragment([(3, box)], molecule_entities, removeObject=True, removeTool=True)
    occ.synchronize()
    # Some larger unions leave coincident interface faces as separate OCC
    # entities; merge them before extracting physical surfaces so the mesh is
    # truly conforming across the molecule/solvent interface.
    occ.removeAllDuplicates()
    occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    if len(volumes) < 2:
        gmsh.finalize()
        raise RuntimeError(f"Expected at least two volumes after fragment, got {volumes}")

    # Identify molecule as the smaller total region near molecular center.
    volume_data = []
    for dim, tag in volumes:
        mass = occ.getMass(dim, tag)
        com = occ.getCenterOfMass(dim, tag)
        volume_data.append((tag, mass, com))

    # Solvent should be the largest volume.
    solvent_tag = max(volume_data, key=lambda x: x[1])[0]
    molecular_tags = [tag for tag, mass, com in volume_data if tag != solvent_tag]

    gmsh.model.addPhysicalGroup(3, molecular_tags, 1)
    gmsh.model.setPhysicalName(3, 1, "molecule")
    gmsh.model.addPhysicalGroup(3, [solvent_tag], 2)
    gmsh.model.setPhysicalName(3, 2, "solvent")

    mol_surfs = set()
    for tag in molecular_tags:
        mol_surfs |= {
            s for dim, s in gmsh.model.getBoundary([(3, tag)], oriented=False, recursive=False)
        }

    sol_surfs = {
        s for dim, s in gmsh.model.getBoundary([(3, solvent_tag)], oriented=False, recursive=False)
    }

    interface_surfs = sorted(mol_surfs & sol_surfs)
    outer_surfs = sorted(sol_surfs - mol_surfs)

    if not interface_surfs:
        gmsh.finalize()
        raise RuntimeError("No molecule-solvent interface surfaces found.")
    if not outer_surfs:
        gmsh.finalize()
        raise RuntimeError("No outer boundary surfaces found.")

    gmsh.model.addPhysicalGroup(2, interface_surfs, 11)
    gmsh.model.setPhysicalName(2, 11, "molecule_solvent_interface")
    gmsh.model.addPhysicalGroup(2, outer_surfs, 12)
    gmsh.model.setPhysicalName(2, 12, "outer_boundary")

    # Two-stage distance refinement:
    # interface -> h_interface near surface,
    # transition to h_outer,
    # then to h_bulk.
    fdist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fdist, "SurfacesList", interface_surfs)
    gmsh.model.mesh.field.setNumber(fdist, "Sampling", 200)

    fnear = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(fnear, "InField", fdist)
    gmsh.model.mesh.field.setNumber(fnear, "SizeMin", args.h_interface_A)
    gmsh.model.mesh.field.setNumber(fnear, "SizeMax", args.h_outer_A)
    gmsh.model.mesh.field.setNumber(fnear, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(fnear, "DistMax", args.dist_inner_A)

    ffar = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ffar, "InField", fdist)
    gmsh.model.mesh.field.setNumber(ffar, "SizeMin", args.h_outer_A)
    gmsh.model.mesh.field.setNumber(ffar, "SizeMax", args.h_bulk_A)
    gmsh.model.mesh.field.setNumber(ffar, "DistMin", args.dist_inner_A)
    gmsh.model.mesh.field.setNumber(ffar, "DistMax", args.dist_outer_A)

    # Keep the legacy interface/far field exactly unchanged in this extension.
    # In particular, do not silently replace its Min by a concatenated ramp.
    background_fields = [fnear, ffar]
    charged_indices = []
    if charge_settings is not None:
        for i, ((x, y, z), charge) in enumerate(zip(xyz, charge_array)):
            if charge == 0:
                continue
            ball = gmsh.model.mesh.field.add("Ball")
            for key, value in {"XCenter": x, "YCenter": y, "ZCenter": z,
                               "Radius": charge_settings["radius_A"],
                               "Thickness": charge_settings["transition_A"],
                               "VIn": charge_settings["h_core_A"],
                               "VOut": args.h_bulk_A}.items():
                gmsh.model.mesh.field.setNumber(ball, key, value)
            background_fields.append(ball)
            charged_indices.append(i)
    fmin = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", background_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(fmin)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", args.algorithm3d)
    if args.num_threads > 0:
        gmsh.option.setNumber("General.NumThreads", args.num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", args.num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", args.num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", args.num_threads)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber(
        "Mesh.OptimizeNetgen", 0 if args.disable_netgen_optimization else 1
    )
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    gmsh.model.mesh.generate(3)

    if args.order == 2:
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("HighOrder")

    gmsh.write(str(out_path))

    if args.save_geo_unrolled:
        geo_path = out_path.with_suffix(".geo_unrolled")
        gmsh.write(str(geo_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(3)
    n_3d = sum(len(tags) for tags in elem_tags)

    molecular_volume = sum(occ.getMass(3, tag) for tag in molecular_tags)
    solvent_volume = occ.getMass(3, solvent_tag)

    report = {
        "pqr": str(Path(args.pqr).resolve()),
        "problem_npz": None if problem_path is None else str(problem_path),
        "problem_npz_sha256": (
            None if problem_path is None else sha256_file(problem_path)
        ),
        "box_source": ("problem_npz" if problem_path is not None else
                       "explicit_bounds" if args.box_lo_A is not None else "cubic_padding"),
        "n_atoms": len(atoms),
        "zero_or_nonpositive_radius_atom_indices_zero_based": zero_or_small,
        "radius_replacement_containment": replacement_containment,
        "min_radius_replacement_A": args.min_radius_A,
        "radius_scale": args.radius_scale,
        "center_A": [cx, cy, cz],
        "half_extent_A": half_extent,
        "half_extents_A": half_extents.tolist(),
        "box_bounds_A": [
            [bx0, bx0 + box_size_x],
            [by0, by0 + box_size_y],
            [bz0, bz0 + box_size_z],
        ],
        "molecule_volume_tags": molecular_tags,
        "solvent_volume_tag": solvent_tag,
        "interface_surface_tags": interface_surfs,
        "outer_surface_tags": outer_surfs,
        "molecular_volume_A3": molecular_volume,
        "solvent_volume_A3": solvent_volume,
        "n_nodes": len(node_tags),
        "n_3d_elements": n_3d,
        "mesh_parameters": {
            "h_interface_A": args.h_interface_A,
            "h_outer_A": args.h_outer_A,
            "h_bulk_A": args.h_bulk_A,
            "dist_inner_A": args.dist_inner_A,
            "dist_outer_A": args.dist_outer_A,
            "min_radius_A": args.min_radius_A,
            "radius_scale": args.radius_scale,
            "order": args.order,
            "algorithm3d": args.algorithm3d,
            "num_threads": args.num_threads,
            "optimize": True,
            "optimize_netgen": not args.disable_netgen_optimization,
            "charge_core_sizing": charge_settings,
            "charge_core_atom_indices_zero_based": charged_indices,
            "charge_sizing_changes_cad_geometry": False,
        },
        "physical_tags": {
            "molecule_volume": 1,
            "solvent_volume": 2,
            "molecule_solvent_interface": 11,
            "outer_boundary": 12,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("7_GLY conforming mesh generation completed")
    print(f"PQR file              : {Path(args.pqr).resolve()}")
    print(f"Output mesh           : {out_path}")
    print(f"Report                : {report_path}")
    print(f"Atoms                 : {len(atoms)}")
    print(f"Zero-radius replacements: {zero_or_small}")
    print(f"Center [A]            : ({cx:.6f}, {cy:.6f}, {cz:.6f})")
    print(f"Half extent L_A [A]   : {half_extent:.6f}")
    print(f"Molecular volume tags : {molecular_tags}")
    print(f"Solvent volume tag    : {solvent_tag}")
    print(f"Interface surfaces    : {interface_surfs}")
    print(f"Outer surfaces        : {outer_surfs}")
    print(f"Nodes                 : {len(node_tags)}")
    print(f"3D elements           : {n_3d}")
    print("=" * 72)

    gmsh.finalize()


if __name__ == "__main__":
    main()
