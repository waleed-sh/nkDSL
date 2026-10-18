# Diagnostic message reference

Find a compiler message by code below. Each entry explains the condition,
common causes, and how to resolve it. The
[diagnostics guide](index.md) describes severity and compiler options.

## Quick navigation

Errors:  
[NKDSL-E001](messages.md#lint-code-nkdsl-e001), [NKDSL-E002](messages.md#lint-code-nkdsl-e002), [NKDSL-E003](messages.md#lint-code-nkdsl-e003)

Warnings:  
[NKDSL-W101](messages.md#lint-code-nkdsl-w101), [NKDSL-W103](messages.md#lint-code-nkdsl-w103), [NKDSL-W104](messages.md#lint-code-nkdsl-w104), [NKDSL-W301](messages.md#lint-code-nkdsl-w301), [NKDSL-W302](messages.md#lint-code-nkdsl-w302), [NKDSL-W303](messages.md#lint-code-nkdsl-w303), [NKDSL-W401](messages.md#lint-code-nkdsl-w401)

Info:  
[NKDSL-I201](messages.md#lint-code-nkdsl-i201), [NKDSL-I301](messages.md#lint-code-nkdsl-i301), [NKDSL-I302](messages.md#lint-code-nkdsl-i302), [NKDSL-I303](messages.md#lint-code-nkdsl-i303)

## Errors

<a id="lint-code-nkdsl-e001"></a>

### NKDSL-E001: unresolved free symbols

Meaning:  
The operator IR contains free symbolic names (for example `symbol("J")`) that were not resolved before compilation. Symbols declared with `default=...` are treated as resolved and do not trigger this lint.

Typical causes:  
A numeric value was intended but left symbolic, or a parameter-binding step in the application workflow was skipped.

Example:

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "hop")
    .for_each_site("i")
    .emit(identity(), matrix_element=symbol("J"))
    .build()
)
```

How to resolve:  
Replace free symbols with concrete values before compile, or add an explicit parameter-binding layer in your workflow and ensure it runs before `.compile()`. For optional parameters, declare a fallback directly with `symbol("J", default=...)`.

<a id="lint-code-nkdsl-e002"></a>

### NKDSL-E002: static index out of Hilbert bounds

Meaning:  
A static source/target index expression (for example `source_index(k)` or `target_index(k)`) refers to a site outside `[0, hilbert_size)`.

Typical causes:  
Hard-coded indices copied from a different model size, stale formulas after changing Hilbert layout, or incorrect assumptions about flattened indexing.

Example:

```python
.emit(identity(), matrix_element=source_index(42))
```

How to resolve:  
Ensure every static index expression is valid for the current Hilbert size. If the index should depend on iterator labels, use symbolic selector/index expressions instead of fixed constants.

<a id="lint-code-nkdsl-e003"></a>

### NKDSL-E003: potential zero division on runtime state values

Meaning:  
A division denominator is a direct runtime state read that may be zero. This includes iterator selectors (for example `site("i").value`, `emitted("i").value`) and flat-index source/target reads (`source_index(k)`, `target_index(k)`).

Typical causes:  
Matrix elements use expressions like `1 / site("i").value` or `1 / source_index(0)` without a non-zero guard.

Example:

```python
.emit(identity(), matrix_element=1 / target_index(0))
```

How to resolve:  
Add an explicit non-zero guard (for example `where(site("i").value != 0)`), or rewrite with explicit conditional branching so division executes only on safe branches.

## Warnings

<a id="lint-code-nkdsl-w101"></a>

### NKDSL-W101: predicate is constant false

Meaning:  
The term predicate always evaluates to false, so the term can never emit.

Typical causes:  
Placeholder predicates left in place, contradictory conditions, or accidental boolean simplification to `False`.

How to resolve:  
Remove dead terms or fix predicate logic so intended visits are reachable.

<a id="lint-code-nkdsl-w103"></a>

### NKDSL-W103: duplicate emissions

Meaning:  
Two or more emissions in the same term are structurally identical (same update program and amplitude expression).

Typical causes:  
Copy/paste duplication while building multi-emission terms.

How to resolve:  
Remove duplicates, or intentionally differentiate branch behavior and/or matrix element expressions.

<a id="lint-code-nkdsl-w104"></a>

### NKDSL-W104: max connection hint below static upper bound

Meaning:  
`max_conn_size_hint` is smaller than the term's static upper bound (`n_iterator_rows * n_emissions`).

Typical causes:  
The hint was set early and not updated after adding rows/emissions.

How to resolve:  
Increase the hint to a conservative valid value, or remove the explicit hint and let the compiler infer one.

<a id="lint-code-nkdsl-w301"></a>

### NKDSL-W301: generated states outside Hilbert support

Meaning:  
Sampled emitted states are not members of the Hilbert support set.

Typical causes:  
Update programs produce states that are not representable in the chosen Hilbert space.

Example scenario:  
A rewrite writes values outside what the Hilbert model can encode, or applies shifts that move occupations beyond support.

How to resolve:  
Constrain updates and predicates so emitted states remain in Hilbert support. If the warning is expected in exploratory code, reduce scope or guard branches until model assumptions are finalized.

<a id="lint-code-nkdsl-w302"></a>

### NKDSL-W302: generated states violate Hilbert constraints

Meaning:  
Sampled emitted states fail `hilbert.constraint(state)`.

Typical causes:  
Rewrites break particle-number, magnetization, parity, or custom constrained subspace rules.

Example scenario:  
A term shifts occupation on one site without complementary update, violating a fixed-particle-number constraint.

How to resolve:  
Tighten predicates and/or adjust update programs to preserve the constrained manifold for all emitted branches.

<a id="lint-code-nkdsl-w303"></a>

### NKDSL-W303: generated states contain illegal local basis values

Meaning:  
One or more site values in sampled emitted states are outside `hilbert.local_states`.

Typical causes:  
Unbounded `shift` operations, writes with invalid values, or assumptions about local basis that do not match the configured Hilbert object.

How to resolve:  
Gate shifts with predicates, use bounded rewrites (for example `shift_mod` where physically appropriate), and validate write targets against local basis.

<a id="lint-code-nkdsl-w401"></a>

### NKDSL-W401: estimated deduplication assembly overhead

Meaning:  
Static cost analysis estimates more connectivity assembly work with deduplication enabled than with raw branch output. The estimate accounts for target aggregation, residual collision checks, lossless key packing, and nonzero compaction after static optimization.

Typical causes:  
State-dependent target collisions, few removable static duplicates, or compaction costs that exceed the savings from a smaller output capacity.

How to resolve:  
Benchmark `get_conn_padded` with and without `deduplicate_connected_components` on the intended device and batch size. Retain deduplication when unique targets are required or when fewer target states reduce downstream model evaluation cost. Disabling deduplication preserves repeated contributions as separate rows.

The warning is a cost-model finding, not a wall-clock measurement. It never changes compilation options automatically. It follows the diagnostics settings, including `fail_on_warnings`, and is emitted on cache misses. See [Connectivity assembly and deduplication](../../compiler_connectivity.md) for algorithms, metadata, and benchmarks.

## Info

<a id="lint-code-nkdsl-i201"></a>

### NKDSL-I201: missing branch tags on multi-emission term

Meaning:  
A term with multiple emissions does not assign `tag=...` for one or more branches, reducing readability of diagnostics and IR output.

How to resolve:  
Add explicit tags to each emission so logs and diagnostics stay easy to parse.

<a id="lint-code-nkdsl-i301"></a>

### NKDSL-I301: skipped exact support-membership checks

Meaning:  
Exact support lookup was skipped because the Hilbert space could not be indexed
or enumerated, or its cardinality exceeded `lint_max_exact_hilbert_states`.

Typical causes:  
The Hilbert space is not indexable, does not provide full state enumeration,
or is too large for exact support enumeration within the configured limits.

How to resolve:  
Sampled local-basis and constraint checks continue when source sampling is
available. Increasing `lint_max_exact_hilbert_states` can enable exact checks
for a small indexable space, but it cannot make a non-indexable space indexable.

<a id="lint-code-nkdsl-i302"></a>

### NKDSL-I302: skipped sampled branch evaluation

Meaning:  
Some sampled branch evaluations could not run in diagnostics mode (for example unresolved runtime values or branch evaluation failures).

Typical causes:  
Missing runtime symbols, non-evaluable branch context during diagnostics, or unexpected evaluation failures in sampled paths.

How to resolve:  
Resolve runtime bindings before compile and ensure branch expressions can be evaluated under diagnostics sampling.

<a id="lint-code-nkdsl-i303"></a>

### NKDSL-I303: skipped source-state sampling

Meaning:  
Dynamic connectivity checks were skipped because direct sampling and bounded
indexing could not provide source configurations. Static diagnostics still run.

Typical causes:  
The Hilbert space lacks a working `random_state` implementation and cannot be
indexed. A custom constraint may also use NetKet's generic rejection sampler,
which diagnostics skip to avoid an unbounded search for valid states.

How to resolve:  
Provide a constraint-aware `hilbert.random_state` implementation or register
an efficient sampler with NetKet. The informational message does not prevent
compilation. Check operator action on valid configurations from your own
sampler when no compiler-side source sample is available.
