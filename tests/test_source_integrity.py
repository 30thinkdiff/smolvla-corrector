"""Verify pinned source and result files have not been changed."""
import unittest
from pathlib import Path
from scripts.check_source_integrity import verify


class PinnedEvidenceTest(unittest.TestCase):
    def test_committed_files_match_provenance_manifest(self):
        self.assertEqual(verify(), 23)


    def test_exercise_file_not_shipped(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "implementation/corrector_online_practice.py").exists())


if __name__ == "__main__":
    unittest.main()
