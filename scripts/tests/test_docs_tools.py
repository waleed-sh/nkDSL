"""Regression checks for saved notebooks, docstring rendering, and published pages."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notebooks = load_script("docs_notebooks")
links = load_script("check_docs")
docstrings = load_script("docs_docstrings")


class DocstringRenderingTests(unittest.TestCase):
    def test_singular_example_parses_as_doctest_code(self):
        from griffe import Docstring

        for gap in ("", "\n"):
            with self.subTest(gap=gap):
                source = f"A selector.\n\nExample:\n{gap}    >>> print(1)\n    1\n"
                sections = Docstring(docstrings.markdown_docstring(source)).parse("google")
                example = sections[-1]
                self.assertEqual(example.kind.value, "examples")
                self.assertEqual(example.value[0][0].value, "examples")
                self.assertEqual(example.value[0][1], ">>> print(1)\n1")

    def test_rst_literal_is_rendered_as_code_without_visible_fences(self):
        import markdown

        for heading in ("Typical usage", "Example"):
            with self.subTest(heading=heading):
                source = f"A working example.\n\n{heading}::\n\n    first = 1\n    print(first)\n"
                value = docstrings.markdown_docstring(source)
                if heading == "Example":
                    from griffe import Docstring

                    value = Docstring(value).parse("google")[-1].value[0][1]
                html = markdown.markdown(value, extensions=["pymdownx.superfences"])
                self.assertIn("<pre>", html)
                self.assertNotIn("```", html)
                self.assertNotIn("<blockquote>", html)

    def test_roles_directives_and_non_python_literals(self):
        source = (
            "See :class:`Example` and :math:`x^2`.\n\n"
            ".. code-block:: python\n\n    print(1)\n\n"
            "Rotation::\n\n    x'[i] ← x[j]\n"
        )
        value = docstrings.markdown_docstring(source)
        self.assertIn("`Example`", value)
        self.assertIn("$x^2$", value)
        self.assertIn("```python\nprint(1)", value)
        self.assertIn("```text\nx'[i] ← x[j]", value)

    def test_prose_punctuation_preserves_code_math_and_entities(self):
        source = (
            "First clause; second clause. `a; b` and $x\\;y$ &amp;\n"
            "```python\na = 1; b = 2\n```\n"
        )
        value = docstrings.prose_without_semicolons(source)
        self.assertIn("First clause, second clause.", value)
        self.assertIn("`a; b`", value)
        self.assertIn("$x\\;y$ &amp;", value)
        self.assertIn("a = 1; b = 2", value)

    def test_doctest_output_is_preserved(self):
        source = "Examples:\n    >>> print('a; b')\n    a; b\n\nProse; more prose."
        value = docstrings.markdown_docstring(source)
        self.assertIn("    a; b", value)
        self.assertIn("Prose, more prose.", value)


class NotebookResultsTests(unittest.TestCase):
    def setUp(self):
        self.notebook = {
            "cells": [
                {"cell_type": "markdown", "source": "A description."},
                {
                    "cell_type": "code",
                    "source": ["print(2 + 2)\n"],
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": "4\n"}],
                },
            ]
        }

    def test_prose_edits_keep_results_current(self):
        revised = copy.deepcopy(self.notebook)
        revised["cells"][0]["source"] = "A clearer explanation."
        self.assertEqual(notebooks.source_digest(self.notebook), notebooks.source_digest(revised))
        self.assertEqual(notebooks.result_digest(self.notebook), notebooks.result_digest(revised))

    def test_code_changes_invalidate_results(self):
        revised = copy.deepcopy(self.notebook)
        revised["cells"][1]["source"] = "print(2 + 3)\n"
        self.assertNotEqual(
            notebooks.source_digest(self.notebook), notebooks.source_digest(revised)
        )

    def test_edited_results_are_detected(self):
        revised = copy.deepcopy(self.notebook)
        revised["cells"][1]["outputs"][0]["text"] = "5\n"
        self.assertNotEqual(
            notebooks.result_digest(self.notebook), notebooks.result_digest(revised)
        )

    def test_unexecuted_error_and_skipped_cells_fail(self):
        changes = [
            {"execution_count": None},
            {"outputs": [{"output_type": "error", "ename": "ValueError"}]},
            {"metadata": {"tags": ["skip-execution"]}},
        ]
        for change in changes:
            with self.subTest(change=change):
                revised = copy.deepcopy(self.notebook)
                revised["cells"][1].update(change)
                with self.assertRaises(ValueError):
                    notebooks.validate_results(Path("example.ipynb"), revised)

    def test_warning_locations_are_portable_and_messages_are_preserved(self):
        examples = [
            (
                "/Users/alice/project/src/nkdsl/compiler/core/pipeline.py:98: UserWarning: message\n",
                "nkdsl/compiler/core/pipeline.py:98: UserWarning: message\n",
            ),
            (
                "/home/runner/venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: message\n",
                "tqdm/auto.py:21: TqdmWarning: message\n",
            ),
            (
                r"C:\Users\alice\project\src\nkdsl\compiler\core\pipeline.py:98: UserWarning: message",
                "nkdsl/compiler/core/pipeline.py:98: UserWarning: message",
            ),
            (
                'File "/Users/alice/Work Folder/example.py", line 12',
                'File "example.py", line 12',
            ),
            (
                "/usr/lib/python3.12/importlib/__init__.py:12: UserWarning: message",
                "importlib/__init__.py:12: UserWarning: message",
            ),
        ]
        for source, expected in examples:
            with self.subTest(source=source):
                self.assertEqual(notebooks.portable_output_text(source), expected)
                self.assertEqual(notebooks.portable_output_text(expected), expected)
        url = "https://example.org/src/nkdsl/file.py:98: documentation"
        self.assertEqual(notebooks.portable_output_text(url), url)

    def test_output_sanitization_preserves_code_results_and_images(self):
        revised = copy.deepcopy(self.notebook)
        output = revised["cells"][1]["outputs"][0]
        output["text"] = ["/tmp/venv/site-packages/tqdm/auto.py:21: Warning: message\n"]
        revised["cells"][1]["outputs"].append(
            {
                "output_type": "display_data",
                "data": {
                    "text/html": "<pre>/home/alice/project/src/nkdsl/example.py:4: Warning: message</pre>",
                    "image/png": "unchanged-binary-image",
                    "application/json": {"result": [1, 2, 3]},
                },
                "metadata": {},
            }
        )
        before = copy.deepcopy(revised)
        self.assertTrue(notebooks.sanitize_outputs(revised, rewrite=False))
        self.assertEqual(before, revised)
        with self.assertRaisesRegex(ValueError, "machine-specific"):
            notebooks.validate_results(Path("example.ipynb"), revised)
        self.assertTrue(notebooks.sanitize_outputs(revised))
        self.assertFalse(notebooks.sanitize_outputs(revised))
        notebooks.validate_results(Path("example.ipynb"), revised)
        self.assertEqual(notebooks.source_digest(before), notebooks.source_digest(revised))
        self.assertEqual(revised["cells"][1]["execution_count"], 1)
        self.assertEqual(output["text"], ["tqdm/auto.py:21: Warning: message\n"])
        data = revised["cells"][1]["outputs"][1]["data"]
        self.assertEqual(data["text/html"], "<pre>nkdsl/example.py:4: Warning: message</pre>")
        self.assertEqual(data["image/png"], "unchanged-binary-image")
        self.assertEqual(data["application/json"], {"result": [1, 2, 3]})

    def test_sanitize_does_not_bless_changed_results(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            doc = root / "example.ipynb"
            doc.write_text(json.dumps(self.notebook))
            manifest = root / "results.json"
            original = json.dumps(
                {"schema": 1, "notebooks": {"example.ipynb": {"results_sha256": "invalid"}}}
            )
            manifest.write_text(original)
            with (
                patch.object(notebooks, "ROOT", root),
                patch.object(notebooks, "MANIFEST", manifest),
                patch.object(notebooks, "notebook_paths", return_value=[doc]),
                patch.object(sys, "argv", ["docs_notebooks.py", "--sanitize"]),
            ):
                self.assertEqual(notebooks.main(), 1)
            self.assertEqual(manifest.read_text(), original)


class PublishedLinksTests(unittest.TestCase):
    def test_prose_punctuation_check_excludes_code_and_math(self):
        page = links.PageLinks()
        page.feed(
            "<article><p>One clause; another clause.</p>"
            '<pre>print("a; b")</pre><code>a; b</code>'
            '<span class="arithmatex">$x\\;y$</span><p>&amp; is an entity.</p>'
            '<img src="plot.png"><p>Final clause; more text.</p></article>'
        )
        self.assertEqual(
            page.prose_semicolons,
            ["One clause; another clause.", "Final clause; more text."],
        )

    def test_rtd_version_prefix_and_fragments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts/docs_migration_inventory.json").write_text("[]")
            (root / "src/nkdsl").mkdir(parents=True)
            (root / "src/nkdsl/__init__.py").write_text("__all__ = []\n")
            (root / "docs").mkdir()
            site = root / "site"
            site.mkdir()
            (site / "index.html").write_text(
                '<link rel="canonical" href="https://nkdsl.readthedocs.io/en/stable/index.html"><a href="guide.html#result">Guide</a>'
            )
            (site / "guide.html").write_text('<h1 id="result">Result</h1>')
            (site / "404.html").write_text('<a href="/en/stable/guide.html#result">Guide</a>')
            with patch.object(links, "ROOT", root):
                self.assertEqual(links.check_site(site), [])
                (site / "guide.html").write_text('<h1 id="renamed">Result</h1>')
                failures = links.check_site(site)
            self.assertTrue(any("missing anchor" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
