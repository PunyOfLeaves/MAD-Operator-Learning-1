"""Check portable comparison inputs without loading models or training."""

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compare_lnfno_source_free_pb as comparison


class ComparisonInputChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original_bytes = b"archived source-driven MAD0 training data"
        self.expected_hash = hashlib.sha256(self.original_bytes).hexdigest()
        self.training = self.root / "relocated.npy"
        self.training.write_bytes(self.original_bytes)
        self.checkpoint = self.root / "relocated_fno.pt"
        self.checkpoint.write_bytes(b"not loaded by these tests")
        self.metadata = self.root / "source_free.json"
        self.write_metadata()

    def write_metadata(self, **overrides):
        record = {
            "training_data": str(self.root / "unavailable_original.npy"),
            "checkpoint": str(self.root / "unavailable_original_fno.pt"),
        }
        record.update(overrides)
        self.metadata.write_text(json.dumps(record), encoding="utf-8")

    def test_relocated_inputs_with_explicit_expected_hash(self):
        before = self.metadata.read_bytes()
        checkpoint, digest = comparison.resolve_comparison_inputs(
            self.training, self.metadata, self.checkpoint, self.expected_hash,
        )
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(digest, self.expected_hash)
        self.assertEqual(self.metadata.read_bytes(), before)

    def test_archived_hash_and_manifest_relative_checkpoint(self):
        self.write_metadata(
            training_data_sha256=self.expected_hash.upper(),
            checkpoint=self.checkpoint.name,
        )
        checkpoint, digest = comparison.resolve_comparison_inputs(self.training, self.metadata)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(digest, self.expected_hash)

    def test_mismatched_training_data_is_rejected(self):
        self.training.write_bytes(b"different training data")
        for stored_hash in (False, True):
            with self.subTest(stored_hash=stored_hash):
                self.write_metadata(**({"training_data_sha256": self.expected_hash} if stored_hash else {}))
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    comparison.resolve_comparison_inputs(
                        self.training, self.metadata, self.checkpoint,
                        None if stored_hash else self.expected_hash,
                    )

    def test_missing_original_without_expected_hash_is_rejected_before_training(self):
        args = argparse.Namespace(
            training_data=self.training, test_json=self.metadata,
            fno_checkpoint=self.checkpoint, training_data_sha256=None,
            output_dir=self.root / "must_not_be_created",
        )
        with mock.patch.object(comparison, "import_training") as import_training:
            with mock.patch.object(comparison.np, "load") as load:
                with self.assertRaisesRegex(ValueError, "Supply --training-data-sha256"):
                    comparison.main(args)
                import_training.assert_not_called()
                load.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_legacy_metadata_verifies_original_file(self):
        original = self.root / "original.npy"
        original.write_bytes(self.original_bytes)
        self.write_metadata(training_data=original.name, checkpoint=self.checkpoint.name)
        checkpoint, digest = comparison.resolve_comparison_inputs(self.training, self.metadata)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(digest, self.expected_hash)
        self.training.write_bytes(b"not the original data")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            comparison.resolve_comparison_inputs(self.training, self.metadata)

    def test_explicit_hash_cannot_override_conflicting_archived_hash(self):
        self.write_metadata(training_data_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Conflicting expected"):
            comparison.resolve_comparison_inputs(
                self.training, self.metadata, self.checkpoint, self.expected_hash,
            )

    def test_invalid_expected_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "64 hexadecimal characters"):
            comparison.resolve_comparison_inputs(
                self.training, self.metadata, self.checkpoint, "not-a-sha256",
            )

    def test_missing_checkpoint_requires_override(self):
        self.write_metadata(training_data_sha256=self.expected_hash)
        with self.assertRaisesRegex(FileNotFoundError, "--fno-checkpoint"):
            comparison.resolve_comparison_inputs(self.training, self.metadata)


if __name__ == "__main__":
    unittest.main()
