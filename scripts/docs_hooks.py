"""Small build-time checks shared by local MkDocs builds and Read the Docs."""

from __future__ import annotations

import json
import os
import posixpath
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from mkdocs.exceptions import PluginError
from mkdocs.utils import get_relative_url

ROOT = Path(__file__).resolve().parents[1]


def on_config(config):
    """Use RTD's canonical URL and reject missing pages from the migration."""
    canonical = os.environ.get("READTHEDOCS_CANONICAL_URL")
    if canonical:
        config.site_url = canonical.rstrip("/") + "/"
    inventory = json.loads((ROOT / "scripts/docs_migration_inventory.json").read_text())
    missing = [
        item["current"] for item in inventory if not (ROOT / "docs" / item["current"]).is_file()
    ]
    if missing:
        raise PluginError(f"Documentation pages missing after migration: {', '.join(missing)}")
    return config


def on_page_content(html, page, config, files):
    """Resolve notebook links that nbconvert renders outside MkDocs' Markdown pipeline."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    # The site loads MathJax 3 once; nbconvert's legacy MathJax 2 loader is redundant.
    html = re.sub(
        r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->",
        "",
        html,
        flags=re.S,
    )

    def replace(match):
        quote, href = match.groups()
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path.endswith((".md", ".ipynb")):
            return match[0]
        source = posixpath.normpath(posixpath.join(posixpath.dirname(page.file.src_uri), url.path))
        target = files.get_file_from_path(source)
        if target is None:
            raise PluginError(f"Notebook {page.file.src_uri} links to missing page {href}")
        href = urlunsplit(("", "", get_relative_url(target.url, page.url), url.query, url.fragment))
        return f"href={quote}{href}{quote}"

    return re.sub(r"href=([\"'])(.*?)\1", replace, html)
