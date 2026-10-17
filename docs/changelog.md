# Changelog

## Unreleased

### Improvements

- Added static connectivity assembly planning with target value numbering, bounded collision proofs, reduced output capacities, and finite-image bounds. Unresolved collisions use exact packed keys and an adaptive direct-workspace, pairwise, or single-sort algorithm. Distinct targets avoid runtime uniqueness operations.
- Added [`nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass`](api/generated/nkdsl.compiler.passes.connectivity_cost.md#nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass) and `NKDSL-W401` to report estimated deduplication assembly overhead. Added reproducible before/after benchmarks across nine operator families.

### Backwards incompatible changes

- With deduplication enabled, compiled `max_conn_size` and padded connection shapes now use the proven unique-target capacity instead of the raw branch count. Code that assumes the raw padded width should use `max_conn_size` from the compiled operator or disable deduplication explicitly.

### Tests

- Updated VMC integration checks to use the supported NetKet driver API.
- Added documentation checks for saved notebook results, API example rendering,
  local links, prose punctuation, and notebook copy controls.

### Documentation

- Migrated the site to MkDocs and Material while retaining existing page URLs
  and the complete API reference. Reorganised the guides and revised the prose.
- Added copy buttons, notebook downloads, and verified saved notebook outputs.
  Routine builds validate saved results without re-executing notebooks.
- Added strict documentation builds to CI and an optional manual notebook
  execution job. Read the Docs now uses the MkDocs configuration.
- Expanded the DSL guides with term and update chaining, iterator ordering, boundary conditions, and source/target semantics. Added user and developer compiler diagrams, an IR guide, and a detailed pass reference.
- Corrected API doctest and literal-block rendering without changing source examples. Preserved notebook output columns on narrow screens and strengthened the beta warning.
- Removed machine-specific directories from saved notebook warnings while keeping
  diagnostic messages and source locations useful. Future notebook refreshes
  apply the same cleanup without requiring execution on every RTD build.
- Removed the Operators and connectivity notebook from the site. Bounded desktop
  sidebars so expanded navigation scrolls and stops above the footer, including
  when JavaScript is unavailable.

## nkDSL v0.1.2a0

### New features

- Added [`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`](api/generated/nkdsl.compiler.passes.diagnostics.md#nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass), a compiler-integrated DSL linting pass that reports symbol, index, and connectivity diagnostics during `.compile()`.
- Extended [`nkdsl.symbol`](api/generated/nkdsl.symbol.md#nkdsl.symbol) (and `ExpressionContext.symbol`) with first-class symbol declarations: `default=...`, `doc=...`, and `dtype=...`. Declared defaults are now used during compilation/evaluation and are no longer reported as unresolved free symbols.
- Added conditional emission chaining with `emit_if(...)`, `emit_elseif(...)`, and `emit_else(...)`.
- Added `hop(src, dst, amount=1)` as a small occupation-transfer update helper, equivalent to `shift(src, -amount).shift(dst, +amount)`.
- Added first-class emission clause extensions via [`nkdsl.AbstractEmissionClause`](api/generated/nkdsl.AbstractEmissionClause.md#nkdsl.AbstractEmissionClause), [`nkdsl.register_emission_clause`](api/generated/nkdsl.register_emission_clause.md#nkdsl.register_emission_clause), and [`nkdsl.available_emission_clause_names`](api/generated/nkdsl.available_emission_clause_names.md#nkdsl.available_emission_clause_names).
- Added fluent math helpers on [`nkdsl.AmplitudeExpr`](api/generated/nkdsl.AmplitudeExpr.md#nkdsl.AmplitudeExpr) so matrix-element expressions can be written as chained calls like `(site("i").value + 1).sqrt().conj()` (including `neg()`, `abs_()`, `wrap_mod()`, and `pow(...)`), while preserving existing `AmplitudeExpr.<helper>(...)` usage.

### Improvements

- Improved operator algebra ergonomics: `sum([op1, op2, ...])` now works for symbolic and compiled operators by treating numeric zero as the additive identity in reverse-add dispatch (`0 + op -> op`).
- Lowered connectivity kernels now deduplicate connected states by default, summing matrix elements for duplicate `x'` targets and dropping zero-amplitude components (including invalidated branches). This behavior is controlled by `deduplicate_connected_components` and defaults to `True`.

### Backwards incompatible changes

- None.

### Deprecations

- None.

### Bug fixes

- Fixed masked-branch evaluation in the JAX lowerer so matrix-element expressions are evaluated lazily on active branches, avoiding inactive-path numerical issues (for example `1 / site("i").value` when masked out).
- Added lint error `NKDSL-E003` for potential division-by-zero on direct runtime state reads, including `site("i").value`, `emitted("i").value`, `source_index(k)`, and `target_index(k)`.

### Documentation

- Updated DSL docs for symbolic parameters, including examples for `symbol("J", default=..., doc=..., dtype=...)` and guidance on how this interacts with lint diagnostics for unresolved symbols.
- Extended linting docs with the new `NKDSL-E003` message and guidance for guarding divisions over runtime state values.
- Updated emission/connectivity semantics docs to describe default connected component deduplication and zero-amplitude pruning.

## nkDSL v0.1.1a0

### New features

- Added [`nkdsl.compiler.SymbolicOperatorLoweringRegistry`](api/generated/nkdsl.SymbolicOperatorLoweringRegistry.md#nkdsl.SymbolicOperatorLoweringRegistry) and [`nkdsl.compiler.SymbolicOperatorLoweringTarget`](api/generated/nkdsl.SymbolicOperatorLoweringTarget.md#nkdsl.SymbolicOperatorLoweringTarget).
- Added `nkdsl.compiler.DEFAULT_SYMBOLIC_OPERATOR_LOWERING` and [`nkdsl.compiler.default_symbolic_operator_lowering_registry`](api/generated/nkdsl.default_symbolic_operator_lowering_registry.md#nkdsl.default_symbolic_operator_lowering_registry).
- Added `operator_lowering` selection in compiler options and convenience compile entry points.
- Added support for custom connection methods such as `_get_conn_padded` for computational-style operator subclasses.
- Added a clause abstraction layer for DSL extensions: [`nkdsl.AbstractIteratorClause`](api/generated/nkdsl.AbstractIteratorClause.md#nkdsl.AbstractIteratorClause) and [`nkdsl.AbstractPredicateClause`](api/generated/nkdsl.AbstractPredicateClause.md#nkdsl.AbstractPredicateClause), together with registration APIs, so custom fluent iterator/predicate methods can be added without patching builder internals.
- Added [`nkdsl.source_index`](api/generated/nkdsl.source_index.md#nkdsl.source_index) / [`nkdsl.target_index`](api/generated/nkdsl.target_index.md#nkdsl.target_index) (and matching `ExpressionContext` methods) as user-facing aliases for static source/target flat-index reads.

### Improvements

- Preserved default runtime behavior: `"netket_discrete_jax"` -\> `DiscreteJaxOperator.get_conn_padded`.
- Included operator-lowering target identity in compilation signatures so cache keys remain consistent across different lowering targets.

### Backwards incompatible changes

- None.

### Deprecations

- None.

### Bug fixes

- None.

### Documentation

- Added [Operator lowering registry](compiler_operator_lowering_registry.md) with rationale, API surface, and usage examples.
- Added a new [Guides](guides/index.md) section with detailed DSL extension guides, including custom iterator and predicate clause examples.
