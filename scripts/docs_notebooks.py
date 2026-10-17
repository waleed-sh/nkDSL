"""Validate saved notebook results or refresh them in a Jupyter kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = ROOT / "scripts/notebook-results.json"
PYTHON_LOCATION = re.compile(
    r"(?<![\w:/\\])(?:[A-Za-z]:[/\\]|/)[^\r\n<>\"'\x1b]*?\.py" r"(?=:\d+(?::|\b)|[\"'])"
)


def portable_output_text(text: str) -> str:
    """Keep diagnostic file names without exposing machine-specific directories."""

    def replace(match: re.Match) -> str:
        path = match[0].replace("\\", "/")
        for marker in ("/site-packages/", "/dist-packages/", "/src/"):
            if marker in path:
                return path.rsplit(marker, 1)[1]
        root = ROOT.as_posix() + "/"
        if path.startswith(root):
            return path[len(root) :]
        standard_library = re.search(r"/python\d+(?:\.\d+)+/(.+)$", path)
        return standard_library[1] if standard_library else path.rsplit("/", 1)[-1]

    return PYTHON_LOCATION.sub(replace, text)


def output_text_fields(notebook: dict):
    """Yield textual output fields, leaving binary images and numerical data alone."""
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for key in ("text", "traceback"):
                if key in output:
                    yield output, key
            for mime in output.get("data", {}):
                if mime.startswith("text/"):
                    yield output["data"], mime


def sanitize_outputs(notebook: dict, *, rewrite: bool = True) -> bool:
    """Normalize diagnostic locations in saved text and HTML outputs in place."""
    changed = False
    for fields, key in output_text_fields(notebook):
        value = fields[key]
        text = "".join(value) if isinstance(value, list) else value
        normalized = portable_output_text(text)
        if normalized != text:
            changed = True
            if rewrite:
                fields[key] = (
                    normalized.splitlines(keepends=True) if isinstance(value, list) else normalized
                )
    return changed


def notebook_paths() -> list[Path]:
    """Return published notebooks, excluding local build and checkpoint files."""
    return sorted(
        path
        for path in DOCS.rglob("*.ipynb")
        if not any(part.startswith((".", "_")) for part in path.relative_to(DOCS).parts)
    )


def source_digest(notebook: dict) -> str:
    """Hash executable sources and tags; prose edits do not invalidate results."""
    code = [
        {"source": "".join(cell["source"]), "tags": cell.get("metadata", {}).get("tags", [])}
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    ]
    return hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()


def result_digest(notebook: dict) -> str:
    """Hash outputs and execution counts so cleared or edited results are detected."""
    results = [
        {"count": cell.get("execution_count"), "outputs": cell.get("outputs", [])}
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    ]
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def validate_results(path: Path, notebook: dict, *, check_paths: bool = True) -> None:
    """Reject notebooks with errors, unexecuted cells, or skipped examples."""
    for number, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        if cell.get("execution_count") is None:
            raise ValueError(f"{path}: code cell {number} has not been executed.")
        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            raise ValueError(f"{path}: code cell {number} contains an execution error.")
        if "skip-execution" in cell.get("metadata", {}).get("tags", []):
            raise ValueError(f"{path}: code cell {number} is marked skip-execution.")
    if check_paths and sanitize_outputs(notebook, rewrite=False):
        raise ValueError(
            f"{path}: saved output contains a machine-specific Python location. "
            "Run python scripts/docs_notebooks.py --sanitize."
        )


def environment() -> dict[str, str]:
    """Record the interpreter and numerical packages used for a successful run."""
    packages = ("netket", "jax", "jaxlib", "numpy", "flax", "nbclient", "ipykernel")
    return {
        "python": platform.python_version(),
        **{p: importlib.metadata.version(p) for p in packages},
    }


def execute(path: Path, timeout: int) -> None:
    """Execute a notebook with this interpreter and save it only after success."""
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=timeout,
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cleanup_kc=True, cwd=str(ROOT))
    sanitize_outputs(notebook)
    nbformat.validate(notebook)
    validate_results(path, notebook)
    nbformat.write(notebook, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check", action="store_true", help="Check saved results without starting Jupyter."
    )
    action.add_argument(
        "--execute", action="store_true", help="Execute notebooks with changed code or results."
    )
    action.add_argument(
        "--sanitize", action="store_true", help="Remove local paths from verified saved outputs."
    )
    parser.add_argument("--force", action="store_true", help="Re-execute all selected notebooks.")
    parser.add_argument("--timeout", type=int, default=600, help="Maximum seconds per code cell.")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional notebook paths to refresh.")
    args = parser.parse_args()
    if args.force and not args.execute:
        parser.error("--force requires --execute")
    manifest = (
        json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"schema": 1, "notebooks": {}}
    )
    paths = notebook_paths()
    selected = {path.resolve() for path in args.paths} if args.paths else set(paths)
    unknown = selected - set(paths)
    if unknown:
        parser.error(f"Not a published notebook: {', '.join(map(str, sorted(unknown)))}")
    failures = []
    for path in paths:
        key = path.relative_to(ROOT).as_posix()
        notebook = json.loads(path.read_text())
        saved = manifest["notebooks"].get(key, {})
        fresh = saved.get("source_sha256") == source_digest(notebook) and saved.get(
            "results_sha256"
        ) == result_digest(notebook)
        if args.execute and path in selected and (args.force or not fresh):
            print(f"Executing {key}", flush=True)
            execute(path, args.timeout)
            notebook = json.loads(path.read_text())
            manifest["notebooks"][key] = {
                "source_sha256": source_digest(notebook),
                "results_sha256": result_digest(notebook),
                "environment": environment(),
                "output_sanitization": "portable-python-locations-v1",
            }
            MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            fresh = True
        if args.sanitize and path in selected:
            if not fresh:
                failures.append(f"{key}: cannot sanitize results that have not been verified.")
                continue
            try:
                validate_results(path, notebook, check_paths=False)
            except ValueError as error:
                failures.append(str(error))
                continue
            if sanitize_outputs(notebook):
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
                print(f"Sanitized {key}", flush=True)
            saved["results_sha256"] = result_digest(notebook)
            saved["output_sanitization"] = "portable-python-locations-v1"
            MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            validate_results(path, notebook)
            if not fresh:
                raise ValueError(
                    f"{key}: saved results do not match the notebook source or outputs."
                )
        except ValueError as error:
            failures.append(str(error))
        else:
            print(f"Verified {key}", flush=True)
    stale = set(manifest["notebooks"]) - {path.relative_to(ROOT).as_posix() for path in paths}
    if stale:
        failures.append(f"Notebook manifest contains removed pages: {', '.join(sorted(stale))}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        print(
            "Run python scripts/docs_notebooks.py --execute and commit the notebooks and manifest.",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(paths)} notebooks have verified saved results.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NKDSL_EXPERIMENTAL", "1")
    os.environ.setdefault("NETKET_EXPERIMENTAL", "1")
    raise SystemExit(main())
