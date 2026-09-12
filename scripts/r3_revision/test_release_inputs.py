"""Validate bundled inputs and the portable final molecular-data entry point."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from molecular_geometry_sources import MolecularGeometry
from molecular_laplace_true_exterior import generate_samples
from prepare_molecular_dataset import main
from test_molecular_geometry_sources import cube_with_void


ROOT = Path(__file__).resolve().parents[2]


class ReleaseInputChecks(unittest.TestCase):
    def test_bundled_checksums_and_test_shape(self):
        base = ROOT / "reproducibility/r3/inputs"
        manifest = json.loads((base / "manifest.json").read_text())
        for name, record in manifest["files"].items():
            content = (base / name).read_bytes()
            self.assertEqual(len(content), record["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])
        test = np.loadtxt(base / "TSLlaplace2D_(200, 51).txt")
        self.assertEqual(test.size, 200 * (200 + 51 * 51))
        self.assertTrue(np.isfinite(test).all())

    def test_portable_generation_matches_final_generator(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            nodes, tetra = cube_with_void()
            geo = MolecularGeometry(nodes, tetra)
            ids = np.unique(geo.faces)
            query = np.array([[.1, .1, .1], [.9, .9, .9]])
            geometry = root / "geometry.npz"
            np.savez(geometry, nodes=nodes, tetra=tetra, boundary_ids=ids,
                     train_queries=query, test_queries=query,
                     center=(nodes.min(axis=0) + nodes.max(axis=0)) / 2,
                     scale=geo.scale)
            args = Namespace(geometry=geometry, output_dir=root / "generated",
                             train_samples=2, validation_samples=1, test_samples=1)
            main(args)
            expected, _ = generate_samples(geo, nodes[ids], query, 2, 2026091052,
                                           relative_clearance=.01)
            with np.load(args.output_dir / "train.npz") as actual:
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()
