"""Verify Chinese-primary README/report navigation and local links.

Only Python standard library is used. Does not touch experiments or the GPU.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("README.md", "README.en.md", "docs/TECHNICAL_REPORT.md",
         "docs/TECHNICAL_REPORT.en.md")


class BilingualNavigationTest(unittest.TestCase):
    def test_required_pages_exist(self):
        for page in PAGES:
            with self.subTest(page=page):
                self.assertTrue((ROOT / page).is_file())

    def test_bidirectional_language_switch(self):
        switches = {
            "README.md": "README.en.md",
            "README.en.md": "README.md",
            "docs/TECHNICAL_REPORT.md": "TECHNICAL_REPORT.en.md",
            "docs/TECHNICAL_REPORT.en.md": "TECHNICAL_REPORT.md",
        }
        for page, target in switches.items():
            with self.subTest(page=page):
                self.assertIn(f'href="{target}"', (ROOT / page).read_text(encoding="utf-8"))

    def test_every_local_link_resolves(self):
        for page in PAGES:
            path = ROOT / page
            body = path.read_text(encoding="utf-8")
            links = re.findall(r"\]\(([^)]+)\)", body)
            links += re.findall(r'href="([^"]+)"', body)
            for link in links:
                if link.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                target = path.parent / link.split("#", 1)[0]
                with self.subTest(page=page, link=link):
                    self.assertTrue(target.exists(), f"Broken internal link: {page} -> {link}")

    def test_primary_documentation_is_chinese(self):
        main = (ROOT / "README.md").read_text(encoding="utf-8")
        report = (ROOT / "docs/TECHNICAL_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("主要实验结果", main)
        self.assertIn("待作者完善", report)
        self.assertIn("英文技术报告", report)


if __name__ == "__main__":
    unittest.main()
