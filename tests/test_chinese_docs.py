"""Check the two Chinese documentation pages and their local links.

Only Python standard library is used. Does not touch experiments or the GPU.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("README.md", "docs/TECHNICAL_REPORT.md")


class ChineseDocumentationTest(unittest.TestCase):
    def test_required_pages_exist(self):
        for page in PAGES:
            with self.subTest(page=page):
                self.assertTrue((ROOT / page).is_file())
        for removed in ("README.en.md", "docs/TECHNICAL_REPORT.en.md"):
            with self.subTest(removed=removed):
                self.assertFalse((ROOT / removed).exists())

    def test_chinese_document_navigation(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        report = (ROOT / "docs/TECHNICAL_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("](docs/TECHNICAL_REPORT.md)", readme)
        self.assertIn("](../README.md)", report)
        for body in (readme, report):
            self.assertNotIn("README.en.md", body)
            self.assertNotIn("TECHNICAL_REPORT.en.md", body)

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
        self.assertIn("## 摘要", report)
        self.assertIn("## 6. 结论", report)


if __name__ == "__main__":
    unittest.main()
