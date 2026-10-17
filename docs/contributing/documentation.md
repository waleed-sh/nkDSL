# Contributing to the documentation

Documentation lives alongside the code so that an API change and its
explanation can be reviewed together. The site uses MkDocs and Material, with
the same configuration for local previews, CI, and Read the Docs. You can
work on prose and API pages without installing JAX or NetKet.

## Choose where the explanation belongs

Use the getting-started pages to introduce a complete task and the DSL pages
to explain how a language feature behaves. A reader should understand when to
use a feature, how it composes with the rest of a term, and what assumptions
the example makes. Keep implementation details in the developer guides unless
they affect a user's decision.

API documentation is generated from source docstrings. Update the docstring
when a signature, parameter, return value, or API contract changes. A new
concept usually also needs an authored guide. The API reference should not
have to carry the whole explanation of a feature.

For a new page, add a Markdown file under `docs/` and place it in `mkdocs.yml`.
Use relative links to source pages, including notebook paths. Existing
published `.html` paths are retained, so preserve page filenames when revising
content. The migration inventory guards against accidentally dropping a page.

## Preview your changes

Create a documentation environment from the repository root:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

The preview rebuilds as you edit. Check the page at a narrow width as well as
on a desktop, and check both color modes when changing styles. In particular,
preformatted notebook output should keep its columns and scroll horizontally
when needed. It should never be squeezed into broken lines.

The pinned tools in `docs/requirements.txt` are sufficient for a normal build.
mkdocstrings reads `src/nkdsl` statically. It does not import the package or
start a numerical runtime. The rendering extension in
`scripts/docs_docstrings.py` adapts existing RST literals and doctest sections
to Markdown, allowing source docstrings to retain their current format.

## Write examples that explain a decision

Introduce the problem before showing code, then explain the result and any
important limits. Prefer a small, valid operator over a fragment that hides
the relevant Hilbert-space assumption. Use short paragraphs to connect the
steps. Avoid semicolons in prose.

Use fenced blocks with a language name. Ordinary code blocks and notebook
inputs have copy buttons. For mathematics, use `$...$` inline and `$$...$$`
for display equations. Mermaid fences can illustrate compiler stages, but
the surrounding text must also explain the flow. Keep diagrams short enough
to read on a narrow screen.

The theme follows the supplied neuraLQX reference. nkDSL displays its name
without a logo. Shared styles belong in `docs/stylesheets/extra.css`, while
notebook-specific overrides belong in `docs/stylesheets/notebooks.css`.

## Decide whether a notebook needs execution

Notebook pages display saved results from their `.ipynb` files. Routine site
builds do not run code. This keeps previews and Read the Docs builds fast and
avoids changing numerical output during an unrelated documentation edit.

After execution, the runner makes warning locations portable before saving.
For example, a checkout path ending in `src/nkdsl/compiler/core/pipeline.py`
becomes `nkdsl/compiler/core/pipeline.py`. Installed-package paths retain the
package and filename. Warning messages, categories, line numbers, and numerical
results remain intact. This keeps personal directories out of both the rendered
page and the downloadable notebook.

To clean locations in existing results that already match the manifest, run:

```bash
python scripts/docs_notebooks.py --sanitize
```

This command does not start a kernel. It updates only output locations and
their recorded hashes, retaining the original execution environment. It refuses
to accept stale or edited results as verified. Normal builds check that saved
locations are portable as well as checking the result hashes.

Every nonempty code cell must already have completed successfully. The
manifest at `scripts/notebook-results.json` records code and result hashes,
along with the runtime versions used. Editing a Markdown cell does not make
the results stale. Changing code, execution tags, or outputs does. A runtime
change can also require a refresh even if the notebook itself is unchanged.

To refresh results, use a separate environment with Python 3.12 or 3.13 and
install `docs/requirements-notebooks.txt`. The numerical dependency pins
require Python 3.12 or later, and the CI refresh job uses Python 3.12.

```bash
python -m pip install -r docs/requirements-notebooks.txt
python scripts/docs_notebooks.py --execute
```

The runner executes notebooks whose saved results are stale. To recheck one
after a compiler change, force execution of that notebook:

```bash
python scripts/docs_notebooks.py --execute --force docs/tutorials/ising_end_to_end.ipynb
```

Omit the path to force a refresh of all notebooks. The runner stops at a failing
cell and does not save that notebook's partial execution. Review the resulting
plots, tables, and numerical comparisons before committing. Commit changed
notebooks and the updated manifest together.

You can also select **execute_notebooks** when manually starting the
**Documentation** workflow in GitHub Actions. It runs every notebook, checks
the resulting site, and uploads an `executed-notebooks` artifact. Review the
downloaded files before bringing them into a commit.

## Check the rendered site before submitting

Run the same checks used by the documentation workflow:

```bash
python scripts/docs_notebooks.py --check
python -m mkdocs build --strict
python scripts/check_docs.py site
python -m unittest discover -s scripts/tests
node scripts/tests/test_notebook_copy.mjs
```

The build treats warnings as errors. The site checker follows local links,
including fragment anchors, API pages, assets, and notebook downloads. The
Python checks cover the documentation helpers, and the Node.js check exercises
the notebook copy controls without extra JavaScript dependencies.

The rendered-site check also rejects semicolons in prose, including generated
page notices. Code and mathematical notation are excluded from that prose
rule.

If you change dependency pins, update `docs/requirements.txt` and the matching
extras in `pyproject.toml` together. Check a clean documentation-only install
so that an undeclared runtime dependency cannot pass unnoticed.

## How publication works

Read the Docs reads `.readthedocs.yaml`, installs the documentation requirements,
checks saved notebook results, and runs the strict MkDocs build. It neither
installs the numerical runtime nor executes notebooks. The site uses RTD's
canonical URL and shows a version menu when RTD Addons supplies version data.

A documentation commit therefore needs current saved results and a clean
strict build. No separate publishing command is needed for a branch that RTD
is configured to build.
