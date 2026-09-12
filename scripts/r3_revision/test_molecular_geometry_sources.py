"""Tests for true-exterior sources, including a sealed cavity and a tunnel."""

import unittest

import numpy as np
from skfem import MeshTet

from molecular_geometry_sources import GeometrySourceSampler, MolecularGeometry, triangle_distances
from molecular_laplace_experiment import contained


def cube_with_void(tunnel=False):
    grid = np.linspace(0, 1, 5)
    mesh = MeshTet.init_tensor(grid, grid, grid)
    center = mesh.p[:, mesh.t].mean(1).T
    void = ((center[:, :2] > .25) & (center[:, :2] < .75)).all(1)
    if not tunnel:
        void &= (center[:, 2] > .25) & (center[:, 2] < .75)
    tetra = mesh.t.T[~void]
    ids = np.unique(tetra)
    remap = np.full(mesh.nvertices, -1, dtype=int)
    remap[ids] = np.arange(len(ids))
    return mesh.p.T[ids], remap[tetra]


class GeometrySamplingChecks(unittest.TestCase):
    def test_exact_triangle_distances(self):
        triangle = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
        for point, distance in (([.2, .2, .3], .3), ([2, 0, 0], 1), ([.8, .8, 0], .6/np.sqrt(2))):
            self.assertAlmostEqual(float(triangle_distances(np.array(point), triangle)[0]), distance)

    def test_cavity_and_tunnel_are_distinguished(self):
        hollow = MolecularGeometry(*cube_with_void())
        tunnel = MolecularGeometry(*cube_with_void(tunnel=True))
        self.assertEqual(hollow.topology["boundary_components"], 2)
        self.assertEqual(hollow.topology["bounded_complement_components_inferred"], 1)
        self.assertEqual([x["genus"] for x in hollow.topology["boundary"]], [0, 0])
        self.assertEqual(tunnel.topology["boundary_components"], 1)
        self.assertEqual(tunnel.topology["bounded_complement_components_inferred"], 0)
        self.assertEqual(tunnel.topology["boundary"][0]["genus"], 1)

    def test_sources_are_sampled_in_sealed_cavity(self):
        geometry = MolecularGeometry(*cube_with_void())
        sampler = GeometrySourceSampler(geometry, 48)
        sources, weights, records = sampler.sample()
        inner = sources[[r["inside_box"] for r in records]]
        self.assertGreater(len(inner), 0)
        self.assertTrue(((inner > .251) & (inner < .749)).all())
        self.assertEqual(weights.shape, (10,))
        self.assertTrue(all(r["clearance"] > .001 for r in records))
        self.assertFalse(contained(sources, geometry.nodes, geometry.tetra).any())
        self.assertTrue(any(r["kind"] == "reentrant_exterior" for r in records))

    def test_tunnel_sources_and_containment_broad_phase(self):
        geometry = MolecularGeometry(*cube_with_void(tunnel=True))
        sampler = GeometrySourceSampler(geometry, 49)
        sources, _, records = sampler.sample()
        inner = sources[[r["inside_box"] for r in records]]
        self.assertGreater(len(inner), 0)
        self.assertTrue(((inner[:, :2] > .251) & (inner[:, :2] < .749)).all())
        points = np.random.default_rng(18).uniform(-.1, 1.1, (150, 3))
        np.testing.assert_array_equal([geometry.contains(p) for p in points], contained(points, geometry.nodes, geometry.tetra))
        for point in points[:20]:
            self.assertAlmostEqual(geometry.clearance(point), float(triangle_distances(point, geometry.triangles).min()))

    def test_fresh_sources_reproducibility_and_all_exterior_sectors(self):
        mesh = MeshTet()
        geometry = MolecularGeometry(mesh.p.T, mesh.t.T)
        a, b = GeometrySourceSampler(geometry, 7), GeometrySourceSampler(geometry, 7)
        first = a.sample()
        np.testing.assert_array_equal(first[0], b.sample()[0])
        self.assertFalse(np.array_equal(first[0], a.sample()[0]))
        points = np.array([a.outside_box()[0] for _ in range(200)])
        counts = ((points < geometry.low) | (points > geometry.high)).sum(1)
        self.assertEqual(set(counts), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
