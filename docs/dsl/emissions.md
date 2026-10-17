# Emissions and matrix elements

Each call to `emit(...)` adds one branch to the current term.

The update chooses the target configuration and `matrix_element` gives its
weight. An omitted update is the identity, which is useful for diagonal
energies. A tag is optional and helps identify a branch in diagnostics.
It does not affect the numerical operator.

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "two_branch")
    .for_each_site("i")
    .where((site("i") > 0) & (site("i") < nmax))
    .emit(shift("i", +1), matrix_element=+0.5, tag="raise")
    .emit(shift("i", -1), matrix_element=-0.5, tag="lower")
    .build()
)
```

An emission contains:

- an update program that maps `x` to `x'`
- a matrix element expression
- an optional diagnostic branch tag

## Multi-emission semantics

Multiple emissions share the current iterator and term predicate. Use them when
one visit can contribute several transitions.

The first example assumes a bosonic basis from zero to `nmax`. Its shared
guard allows both the raise and lower only at interior occupations. Each
emission starts from the original configuration. The lowering branch does
not act on the result of the raising branch. To allow different boundary
conditions for the two moves, use separate terms or separate conditional
chains.

## Conditional emission chains

nkDSL also supports `if / elseif / else` style branching at the emission level.
The following example assumes local values from zero to three:

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "piecewise")
    .for_each_site("i")
    .emit_if(site("i") == 0, write("i", 1), matrix_element=1.0, tag="if")
    .emit_elseif(site("i") == 1, write("i", 2), matrix_element=2.0, tag="elseif")
    .emit_else(write("i", 3), matrix_element=3.0, tag="else")
    .build()
)
```

Chaining rules:

- `emit_if(...)` starts a new conditional chain.
- `emit_elseif(...)` extends the current chain.
- `emit_else(...)` closes the current chain.
- Calling plain `emit(...)` closes any currently open chain.

The first matching alternative contributes. An `elseif` requires both its
own condition and the failure of all earlier conditions in the chain.
`else` covers what remains. The term's `where` condition still applies to
every alternative. Keep the chain consecutive, as term modifiers also close
it. Two consecutive `emit_if` calls start two independent chains rather than
forming alternatives in one chain.

Active branches contribute candidate connected states. Before final padding, duplicate `x'` targets are coalesced (matrix elements summed) and zero-amplitude components are removed.

## Connected-state dedup semantics

Connected states are deduplicated by default. If two branches emit the same `x'`, they are merged into one row and their matrix elements are summed. Entries with final matrix element `0` are dropped before padding.

The output capacity comes from static target-equivalence and finite-image analysis. Removed static duplicates are not restored as unused raw-branch padding. The connection axis retains a static upper bound across input configurations. State-dependent collisions can leave zero padding within that bound. See [Connectivity assembly and deduplication](../compiler_connectivity.md) for assembly algorithms and cost diagnostics.

For example, two branches reaching the same target with matrix elements two
and minus two cancel. They do not leave a nonzero connection behind. Padded
slot positions are therefore not stable branch identifiers. Use the target
and its matrix element when interpreting connectivity, and use tags or the
symbolic IR when investigating the original branch structure.

Set `deduplicate_connected_components=False` at compile time if you need raw per-branch connectivity output.

## Source and emitted state access

Matrix elements can depend on both the source configuration and the emitted configuration.

- `site("i")` reads from `x`
- `emitted("i")` reads from `x'`

This is especially useful when amplitudes depend on the value after a rewrite.

## Complex matrix elements and dtype promotion

Complex constants in `matrix_element=...` expressions are supported directly. When a complex constant is present, the operator dtype is automatically promoted to a compatible complex dtype during `build()`.

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "complex_branch")
    .for_each_site("i")
    .where(site("i") < nmax)
    .emit(shift("i", +1), matrix_element=1.0 + 0.5j)
    .build()
)

# op.dtype is complex-valued after promotion
```

## Extending emission clauses

To reuse a branch construction across operators, subclass
[`AbstractEmissionClause`](../api/generated/nkdsl.AbstractEmissionClause.md#nkdsl.AbstractEmissionClause)
and register it as a custom emission method.

For a full extension walkthrough, see [Extending DSL Emissions](../guides/extending_dsl/emissions.md).
