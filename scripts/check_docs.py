"""Check pages, links, assets, and prose in a generated documentation site."""

from __future__ import annotations

import argparse
import ast
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class PageLinks(HTMLParser):
    """Collect local references, anchors, and prose violations without fetching sites."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()
        self.links: list[str] = []
        self.canonical: str | None = None
        self.prose_semicolons: list[str] = []
        self._elements: list[tuple[str, bool, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        excluded = (
            tag in {"pre", "code", "script", "style"}
            or "arithmatex" in (values.get("class") or "").split()
        )
        in_article = tag == "article"
        if self._elements:
            excluded |= self._elements[-1][1]
            in_article |= self._elements[-1][2]
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self._elements.append((tag, excluded, in_article))
        if tag == "link" and values.get("rel") == "canonical":
            self.canonical = values.get("href")
        if values.get("id"):
            self.anchors.add(values["id"])
        if tag == "a" and values.get("name"):
            self.anchors.add(values["name"])
        for field in ("href", "src"):
            if values.get(field):
                self.links.append(values[field])

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._elements) - 1, -1, -1):
            if self._elements[index][0] == tag:
                del self._elements[index:]
                break

    def handle_data(self, data: str) -> None:
        if self._elements:
            _, excluded, in_article = self._elements[-1]
            if in_article and not excluded and ";" in data:
                self.prose_semicolons.append(" ".join(data.split())[:120])


def check_site(site: Path) -> list[str]:
    """Return failures for missing migration pages, API entries, or local links."""
    site = site.resolve()
    pages = {}
    for path in sorted(site.rglob("*.html")):
        parsed = PageLinks()
        parsed.feed(path.read_text())
        pages[path.resolve()] = parsed
    failures = []
    if not pages:
        return [f"No HTML pages found in {site}."]
    homepage = pages.get((site / "index.html").resolve())
    prefix = urlsplit(homepage.canonical).path if homepage and homepage.canonical else "/"
    if prefix.endswith(".html"):
        prefix = prefix.rsplit("/", 1)[0] + "/"
    for path, parsed in pages.items():
        for text in parsed.prose_semicolons:
            failures.append(f"{path.relative_to(site)}: semicolon in prose: {text}")
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            local_path = unquote(url.path)
            if local_path.startswith("/") and local_path.startswith(prefix):
                local_path = local_path[len(prefix) :]
                target = site / local_path
            else:
                target = (
                    (site / local_path.lstrip("/"))
                    if local_path.startswith("/")
                    else path.parent / local_path
                )
            if not url.path:
                target = path
            if target.is_dir():
                target = target / "index.html"
            target = target.resolve()
            if not target.is_file():
                failures.append(f"{path.relative_to(site)}: missing target {link}")
            elif (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].anchors
            ):
                failures.append(f"{path.relative_to(site)}: missing anchor {link}")
    inventory = json.loads((ROOT / "scripts/docs_migration_inventory.json").read_text())
    for item in inventory:
        expected = site / Path(item["current"]).with_suffix(".html")
        if not expected.is_file():
            failures.append(f"Missing migrated page: {expected.relative_to(site)}")
    package = ROOT / "src/nkdsl/__init__.py"
    tree = ast.parse(package.read_text())
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    for export in exports:
        expected = site / f"api/generated/nkdsl.{export}.html"
        if not expected.is_file():
            failures.append(f"Missing public API page: {export}")
    notebooks = [
        p
        for p in (ROOT / "docs").rglob("*.ipynb")
        if not any(part.startswith((".", "_")) for part in p.relative_to(ROOT / "docs").parts)
    ]
    for path in notebooks:
        relative = path.relative_to(ROOT / "docs")
        for expected in (site / relative, (site / relative).with_suffix(".html")):
            if not expected.is_file():
                failures.append(f"Missing notebook page or download: {expected.relative_to(site)}")
    print(
        f"Checked {len(pages)} HTML pages, {len(exports)} public API entries, and {len(notebooks)} notebook downloads."
    )
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Directory produced by mkdocs build.")
    args = parser.parse_args()
    failures = check_site(args.site.resolve())
    if failures:
        print("\n".join(failures))
        return 1
    print("All local documentation links and assets resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
