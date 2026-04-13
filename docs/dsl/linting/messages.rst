Lint Messages
=============

This page is the detailed lint message catalog for ``nkDSL`` diagnostics.
Use it when a compile-time diagnostic points you to a specific code.

Quick navigation
----------------

Errors:
   :ref:`NKDSL-E001 <lint-code-nkdsl-e001>`, :ref:`NKDSL-E002 <lint-code-nkdsl-e002>`

Warnings:
   :ref:`NKDSL-W101 <lint-code-nkdsl-w101>`, :ref:`NKDSL-W103 <lint-code-nkdsl-w103>`,
   :ref:`NKDSL-W104 <lint-code-nkdsl-w104>`, :ref:`NKDSL-W301 <lint-code-nkdsl-w301>`,
   :ref:`NKDSL-W302 <lint-code-nkdsl-w302>`, :ref:`NKDSL-W303 <lint-code-nkdsl-w303>`

Info:
   :ref:`NKDSL-I201 <lint-code-nkdsl-i201>`, :ref:`NKDSL-I301 <lint-code-nkdsl-i301>`,
   :ref:`NKDSL-I302 <lint-code-nkdsl-i302>`

Errors
------

.. _lint-code-nkdsl-e001:

NKDSL-E001: unresolved free symbols
+++++++++++++++++++++++++++++++++++

Meaning:
   The operator IR contains free symbolic names (for example ``symbol("J")``)
   that were not resolved before compilation.

Typical causes:
   A numeric value was intended but left symbolic, or a parameter-binding step
   in the application workflow was skipped.

Example:

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "hop")
       .for_each_site("i")
       .emit(identity(), matrix_element=symbol("J"))
       .build()
   )

How to resolve:
   Replace free symbols with concrete values before compile, or add an explicit
   parameter-binding layer in your workflow and ensure it runs before ``.compile()``.


.. _lint-code-nkdsl-e002:

NKDSL-E002: static index out of Hilbert bounds
++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   A static source/target index expression (for example ``source_index(k)`` or
   ``target_index(k)``) refers to a site outside ``[0, hilbert_size)``.

Typical causes:
   Hard-coded indices copied from a different model size, stale formulas after
   changing Hilbert layout, or incorrect assumptions about flattened indexing.

Example:

.. code-block:: python

   .emit(identity(), matrix_element=source_index(42))

How to resolve:
   Ensure every static index expression is valid for the current Hilbert size.
   If the index should depend on iterator labels, use symbolic selector/index
   expressions instead of fixed constants.

Warnings
--------

.. _lint-code-nkdsl-w101:

NKDSL-W101: predicate is constant false
+++++++++++++++++++++++++++++++++++++++

Meaning:
   The term predicate always evaluates to false, so the term can never emit.

Typical causes:
   Placeholder predicates left in place, contradictory conditions, or accidental
   boolean simplification to ``False``.

How to resolve:
   Remove dead terms or fix predicate logic so intended visits are reachable.


.. _lint-code-nkdsl-w103:

NKDSL-W103: duplicate emissions
+++++++++++++++++++++++++++++++

Meaning:
   Two or more emissions in the same term are structurally identical (same update
   program and amplitude expression).

Typical causes:
   Copy/paste duplication while building multi-emission terms.

How to resolve:
   Remove duplicates, or intentionally differentiate branch behavior and/or matrix
   element expressions.


.. _lint-code-nkdsl-w104:

NKDSL-W104: max connection hint below static upper bound
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   ``max_conn_size_hint`` is smaller than the term's static upper bound
   (``n_iterator_rows * n_emissions``).

Typical causes:
   The hint was set early and not updated after adding rows/emissions.

How to resolve:
   Increase the hint to a conservative valid value, or remove the explicit hint
   and let the compiler infer one.


.. _lint-code-nkdsl-w301:

NKDSL-W301: generated states outside Hilbert support
++++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   Sampled emitted states are not members of the Hilbert support set.

Typical causes:
   Update programs produce states that are not representable in the chosen
   Hilbert space.

Example scenario:
   A rewrite writes values outside what the Hilbert model can encode, or applies
   shifts that move occupations beyond support.

How to resolve:
   Constrain updates and predicates so emitted states remain in Hilbert support.
   If the warning is expected in exploratory code, reduce scope or guard branches
   until model assumptions are finalized.


.. _lint-code-nkdsl-w302:

NKDSL-W302: generated states violate Hilbert constraints
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   Sampled emitted states fail ``hilbert.constraint(state)``.

Typical causes:
   Rewrites break particle-number, magnetization, parity, or custom constrained
   subspace rules.

Example scenario:
   A term shifts occupation on one site without complementary update, violating
   a fixed-particle-number constraint.

How to resolve:
   Tighten predicates and/or adjust update programs to preserve the constrained
   manifold for all emitted branches.


.. _lint-code-nkdsl-w303:

NKDSL-W303: generated states contain illegal local basis values
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   One or more site values in sampled emitted states are outside
   ``hilbert.local_states``.

Typical causes:
   Unbounded ``shift`` operations, writes with invalid values, or assumptions
   about local basis that do not match the configured Hilbert object.

How to resolve:
   Gate shifts with predicates, use bounded rewrites (for example ``shift_mod``
   where physically appropriate), and validate write targets against local basis.

Info
----

.. _lint-code-nkdsl-i201:

NKDSL-I201: missing branch tags on multi-emission term
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   A term with multiple emissions does not assign ``tag=...`` for one or more
   branches, reducing readability of diagnostics and IR output.

How to resolve:
   Add explicit tags to each emission so logs and diagnostics stay easy to parse.


.. _lint-code-nkdsl-i301:

NKDSL-I301: skipped exact support-membership checks
+++++++++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   Exact support lookup was skipped because Hilbert cardinality exceeded
   ``lint_max_exact_hilbert_states``.

Typical causes:
   Hilbert space is too large for exact support enumeration within configured
   diagnostics limits.

How to resolve:
   Increase ``lint_max_exact_hilbert_states`` if memory/time budget allows, or
   rely on sampled diagnostics for large spaces.


.. _lint-code-nkdsl-i302:

NKDSL-I302: skipped sampled branch evaluation
+++++++++++++++++++++++++++++++++++++++++++++

Meaning:
   Some sampled branch evaluations could not run in diagnostics mode (for example
   unresolved runtime values or branch evaluation failures).

Typical causes:
   Missing runtime symbols, non-evaluable branch context during diagnostics, or
   unexpected evaluation failures in sampled paths.

How to resolve:
   Resolve runtime bindings before compile and ensure branch expressions can be
   evaluated under diagnostics sampling.
