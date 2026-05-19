Changelog
=========

Unreleased
------------

New features
~~~~~~~~~~~~

* Added :class:`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`,
  a compiler-integrated DSL linting pass that reports symbol, index, and
  connectivity diagnostics during ``.compile()``.

* Extended :func:`nkdsl.symbol` (and ``ExpressionContext.symbol``) with
  first-class symbol declarations: ``default=...``, ``doc=...``, and
  ``dtype=...``. Declared defaults are now used during compilation/evaluation
  and are no longer reported as unresolved free symbols.

* Added conditional emission chaining with ``emit_if(...)``,
  ``emit_elseif(...)``, and ``emit_else(...)``.

* Added first-class emission clause extensions via
  :class:`nkdsl.AbstractEmissionClause`, :func:`nkdsl.register_emission_clause`,
  and :func:`nkdsl.available_emission_clause_names`.

* Added fluent math helpers on :class:`nkdsl.AmplitudeExpr` so matrix-element
  expressions can be written as chained calls like
  ``(site("i").value + 1).sqrt().conj()`` (including ``neg()``, ``abs_()``,
  ``wrap_mod()``, and ``pow(...)``), while preserving existing
  ``AmplitudeExpr.<helper>(...)`` usage.

* Added ``hop(src, dst, amount=1)`` as a small occupation-transfer update helper,
  equivalent to ``shift(src, -amount).shift(dst, +amount)``.


Improvements
~~~~~~~~~~~~

* Improved operator algebra ergonomics: ``sum([op1, op2, ...])`` now works for
  symbolic and compiled operators by treating numeric zero as the additive
  identity in reverse-add dispatch (``0 + op -> op``).

* Lowered connectivity kernels now deduplicate connected states by default,
  summing matrix elements for duplicate ``x'`` targets and dropping
  zero-amplitude components (including invalidated branches). This behavior is
  controlled by ``deduplicate_connected_components`` and defaults to ``True``.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* Fixed masked-branch evaluation in the JAX lowerer so matrix-element
  expressions are evaluated lazily on active branches, avoiding inactive-path
  numerical issues (for example ``1 / site("i").value`` when masked out).

* Added lint error ``NKDSL-E003`` for potential division-by-zero on direct
  runtime state reads, including ``site("i").value``, ``emitted("i").value``,
  ``source_index(k)``, and ``target_index(k)``.

Documentation
~~~~~~~~~~~~~

* Updated DSL docs for symbolic parameters, including examples for
  ``symbol("J", default=..., doc=..., dtype=...)`` and guidance on how this
  interacts with lint diagnostics for unresolved symbols.

* Extended linting docs with the new ``NKDSL-E003`` message and guidance for
  guarding divisions over runtime state values.

* Updated emission/connectivity semantics docs to describe default connected
  component deduplication and zero-amplitude pruning.



nkDSL v0.1.1a0
-----------------------

New features
~~~~~~~~~~~~

* Added :class:`nkdsl.compiler.SymbolicOperatorLoweringRegistry` and
  :class:`nkdsl.compiler.SymbolicOperatorLoweringTarget`.

* Added :data:`nkdsl.compiler.DEFAULT_SYMBOLIC_OPERATOR_LOWERING` and
  :func:`nkdsl.compiler.default_symbolic_operator_lowering_registry`.

* Added ``operator_lowering`` selection in compiler options and convenience
  compile entry points.

* Added support for custom connection methods such as ``_get_conn_padded`` for
  computational-style operator subclasses.

* Added a clause abstraction layer for DSL extensions:
  :class:`nkdsl.AbstractIteratorClause` and :class:`nkdsl.AbstractPredicateClause`,
  together with registration APIs, so custom fluent iterator/predicate methods can
  be added without patching builder internals.

* Added :func:`nkdsl.source_index` / :func:`nkdsl.target_index` (and matching
  ``ExpressionContext`` methods) as user-facing aliases for static source/target
  flat-index reads.

Improvements
~~~~~~~~~~~~

* Preserved default runtime behavior:
  ``"netket_discrete_jax"`` -> ``DiscreteJaxOperator.get_conn_padded``.

* Included operator-lowering target identity in compilation signatures so cache
  keys remain consistent across different lowering targets.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Added :doc:`compiler_operator_lowering_registry` with rationale, API surface,
  and usage examples.

* Added a new :doc:`guides/index` section with detailed DSL extension guides,
  including custom iterator and predicate clause examples.
