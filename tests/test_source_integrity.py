"""Verify pinned source and result files have not been changed."""
import unittest
from scripts.check_source_integrity import verify


class PinnedEvidenceTest(unittest.TestCase):
    def test_files_match_original_git_blobs(self):
        self.assertEqual(verify(), 24)


if __name__ == "__main__":
    unittest.main()
