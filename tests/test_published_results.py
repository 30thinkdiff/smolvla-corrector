"""Non-GPU regression checks for the public evidence reproduction utility."""
import unittest
from scripts.reproduce_results import EvidenceError, as_bool, markdown, summarize


class PublishedResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = summarize()

    def test_object10_recomputed(self):
        z = self.result["object10"]
        self.assertEqual([z["methods"][m]["successes"] for m in ("H50", "H20", "lvm", "ogg")],
                         [44, 51, 55, 55])
        self.assertEqual(z["ogg_extra_success_over_lvm"], 0)
        self.assertEqual(z["mlp_coverage"]["lvm"], 1.0)

    def test_cross_suite_trial_ledger(self):
        z = self.result["cross_suite"]
        self.assertEqual(z["formal_trials"], 880)
        self.assertEqual(z["paired_quartets"], 220)
        self.assertEqual(z["methods"]["H20"]["successes"], 93)
        self.assertEqual(z["methods"]["mlp_ogg"]["successes"], 77)

    def test_plus_horizon(self):
        z = self.result["plus_horizon"]
        self.assertEqual(z["methods"]["H20"], {"successes": 20, "trials": 32})
        self.assertEqual(z["methods"]["H1"]["successes"], 6)

    def test_unknown_boolean_rejected(self):
        with self.assertRaises(EvidenceError):
            as_bool("maybe")

    def test_markdown_is_machine_generated(self):
        s = markdown(self.result)
        self.assertIn("55/60", s)
        self.assertIn("93/120", s)
        self.assertIn("20/32", s)


if __name__ == "__main__":
    unittest.main()
