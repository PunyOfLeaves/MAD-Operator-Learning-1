"""Small end-to-end data tests, without fitting or selecting a neural operator."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_geometry_sources import MolecularGeometry
from molecular_laplace_experiment import digest, write_json
from molecular_laplace_true_exterior import generate_samples, prepare, verify
from test_molecular_geometry_sources import cube_with_void


class TrueExteriorDataChecks(unittest.TestCase):
    def make_base(self, base):
        base.mkdir()
        (base/"reference").mkdir()
        nodes, tetra = cube_with_void()
        geometry = MolecularGeometry(nodes, tetra)
        np.savez(base/"geometry.npz", nodes=nodes, tetra=tetra,
                 boundary_ids=np.unique(geometry.faces),
                 train_queries=np.array([[.1, .1, .1], [.9, .9, .9]]),
                 test_queries=np.array([[.1, .9, .1], [.9, .1, .9]]))
        np.savez(base/"fem_test_inputs.npz", boundary=np.zeros((2, len(np.unique(geometry.faces))), dtype="f4"))
        np.save(base/"reference/level2_solutions.npy", np.zeros((2, 2)))
        write_json(base/"reference/report.json", {"test_solution_sha256": digest(base/"reference/level2_solutions.npy"),
                                                "refinement_target_pass": False})
        write_json(base/"protocol.json", {"train_seed": 1, "validation_seed": 2, "mad_test_seed": 3,
                                          "ntrain": 3, "nvalidation": 2, "ntest": 2})
        write_json(base/"input_manifest.json", {p.name: digest(p) for p in base.iterdir() if p.is_file()})

    def test_fresh_generation_is_reproducible(self):
        geometry = MolecularGeometry(*cube_with_void())
        boundary = geometry.nodes[np.unique(geometry.faces)]
        points = np.array([[.1, .1, .1], [.9, .9, .9]])
        a, _ = generate_samples(geometry, boundary, points, 3, 88)
        b, _ = generate_samples(geometry, boundary, points, 3, 88)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a["solution"].dtype, np.float32)
        self.assertTrue(a["source_inside_box"].any())

    def test_pipeline_preserves_inputs_and_reference_warning(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            base, out = Path(tmp)/"base", Path(tmp)/"new"
            self.make_base(base)
            hashes = {str(p.relative_to(base)): digest(p) for p in base.rglob("*") if p.is_file()}
            prepare(base, out)
            verify(out)
            report = json.loads((out/"verification.json").read_text())
            self.assertTrue(report["all_source_positions_distinct_across_splits"])
            self.assertTrue(report["preserved_geometry_and_fem_tests"])
            self.assertFalse(json.loads((out/"reference/report.json").read_text())["refinement_target_pass"])
            self.assertEqual(hashes, {str(p.relative_to(base)): digest(p) for p in base.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                prepare(base, out)

    def test_verification_rejects_modified_data(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            base, out = Path(tmp)/"base", Path(tmp)/"new"
            self.make_base(base)
            prepare(base, out)
            (out/"train.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                verify(out)


if __name__ == "__main__":
    unittest.main()
