"""Generate API pages from source without importing NetKet, JAX, or nkDSL."""

from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PACKAGE = SOURCE / "nkdsl"


def public_exports() -> list[str]:
    """Read the public export list without importing the package."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("nkdsl.__all__ is missing")


def write_reference(identifier: str, source: Path, *, module: bool = False) -> str:
    """Keep the previous Sphinx autosummary URLs and document public members."""
    path = f"api/generated/{identifier}.md"
    options = ""
    if module:
        tree = ast.parse(source.read_text())
        members = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        options = (
            "\n    options:\n      members: "
            + ("[" + ", ".join(members) + "]" if members else "false")
            + "\n"
        )
    with mkdocs_gen_files.open(path, "w") as stream:
        stream.write(f"---\ntitle: {identifier}\n---\n\n::: {identifier}\n" + options)
    mkdocs_gen_files.set_edit_path(path, "../" + source.relative_to(ROOT).as_posix())
    return path


exports = public_exports()
object_links = []
for name in exports:
    identifier = f"nkdsl.{name}"
    path = write_reference(identifier, PACKAGE / "__init__.py")
    object_links.append(f"- [`{name}`](generated/{identifier}.md)")

module_links = []
for source in sorted(PACKAGE.rglob("*.py")):
    relative = source.relative_to(SOURCE)
    if any(part.startswith("_") and part != "__init__.py" for part in relative.parts):
        continue
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    identifier = ".".join(parts)
    write_reference(identifier, source, module=True)
    module_links.append(f"- [`{identifier}`](generated/{identifier}.md)")

with mkdocs_gen_files.open("api/catalog.md", "w") as stream:
    stream.write(
        "# API catalog\n\nEvery public top-level export and source module is listed below. "
        "The reference is generated from the current source tree.\n\n## Public exports\n\n"
    )
    stream.write("\n".join(object_links))
    stream.write("\n\n## Modules\n\n" + "\n".join(module_links) + "\n")
