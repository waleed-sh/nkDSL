Linting
=======

The DSL linter is a compiler pass that analyzes symbolic operators during
``.compile()``. It is designed to catch likely mistakes early, explain what they
mean, and give practical next steps.

Linting runs through
:class:`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`.

.. toctree::
   :maxdepth: 2

   messages

Why linting exists
------------------

Symbolic operators are expressive, which also means they can fail in subtle ways
that are easy to miss in code review:

* free parameters left unresolved
* static index expressions outside Hilbert bounds
* terms that never emit due to constant-false predicates
* generated states that violate support, constraints, or local basis rules

Linting standardizes these findings into stable ``NKDSL-*`` codes so teams can
search, automate, and gate builds consistently.

When linting runs
-----------------

In the default pipeline, linting executes before normalization:

#. :class:`nkdsl.compiler.passes.validation.SymbolicValidationPass`
#. :class:`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`
#. :class:`nkdsl.compiler.passes.normalization.SymbolicNormalizationPass`

Post-cache passes (analysis and fusion) run afterward on cache misses.

Configuration
-------------

Lint behavior is controlled by :class:`nkdsl.SymbolicCompilerOptions`:

* ``diagnostics_enabled``: enable/disable linting
* ``diagnostics_min_severity``: minimum reported/enforced level (``info``,
  ``warning``, ``error``)
* ``fail_on_warnings``: treat warnings as compile-blocking
* ``max_diagnostics``: cap number of reported diagnostics
* ``lint_state_sample_size``: sampled source-state count for connectivity checks
* ``lint_branch_sample_cap``: cap sampled branch evaluations
* ``lint_max_exact_hilbert_states``: limit exact support-membership checks to
  manageable Hilbert sizes

Example:

.. code-block:: python

   from nkdsl import SymbolicCompilerOptions

   opts = SymbolicCompilerOptions(
       diagnostics_enabled=True,
       diagnostics_min_severity="warning",
       fail_on_warnings=True,
       lint_state_sample_size=64,
       lint_branch_sample_cap=4096,
   )

Reading lint output
-------------------

Diagnostics are printed in a structured, readable block:

.. code-block:: text

   Diagnostics summary: total=2 (errors=1, warnings=1, info=0)
   Read more: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html
   - [NKDSL-E001] ERROR @ my_op.term0
     Unresolved free symbol(s): %J.
     Suggestion: Replace with constants or bind symbols before compilation. DSL compilation requires all free symbols to be resolved.
     Docs: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html#lint-code-nkdsl-e001

Every diagnostic includes:

* a stable code (for example ``NKDSL-W302``)
* severity
* location (operator and term when available)
* message and suggested fix
* a direct documentation link for that code

Severity semantics
------------------

Lint severities are intentionally policy-oriented:

* ``error`` means compilation should usually stop until resolved
* ``warning`` means likely correctness/runtime risk that many teams gate in CI
* ``info`` means non-blocking quality signal (readability, diagnostics coverage,
  or analysis completeness)

You can enforce stricter behavior with ``diagnostics_min_severity`` and
``fail_on_warnings``.

Lint message catalog
--------------------

Use :doc:`messages` for the complete per-code reference, including:

* what each lint means
* typical causes
* concrete examples
* practical remediation guidance

Recommended workflow
--------------------

When you hit diagnostics during operator development:

#. Fix all ``NKDSL-E*`` issues first.
#. Resolve or consciously waive ``NKDSL-W*`` warnings.
#. Use ``NKDSL-I*`` signals to improve readability and diagnostic coverage.
#. Re-run ``.compile()`` after each logical change so regressions are caught
   immediately.

For production CI, a common strict profile is:

* ``diagnostics_min_severity="warning"``
* ``fail_on_warnings=True``
* fixed sampling knobs (``lint_state_sample_size`` / ``lint_branch_sample_cap``)
  so diagnostics are stable across runs.

Important notes
---------------

* Connectivity diagnostics are sample-based and intended as early warnings.
* No-warning output is strong signal, but not a formal proof of correctness.
* For strict CI workflows, combine:

  - ``diagnostics_min_severity="warning"``
  - ``fail_on_warnings=True``

  to fail compilation on warning-level issues.
