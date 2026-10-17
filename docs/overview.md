# Overview

## What nkDSL is for

Use nkDSL when the action of an operator is easier to describe as transitions
between configurations than as a collection of local matrices. It handles the
masking, batching, and padding needed by NetKet's JAX operator interface.

It is most useful when:

- an operator has many related terms and you want one readable definition instead of many hand-written kernels
- the action follows graph or geometry data that you already have as static index sets
- you want to iterate quickly on the physics without rewriting masking and padding logic every time
- you want the final object to plug into the same NetKet operator interface used by matrix-free operators

For a short sum of small local matrices, NetKet's `LocalOperator` may already
provide everything you need. A custom JAX kernel remains useful when you need
direct control over the computation.

## Describe one term at a time

A term selects an iteration domain, applies an optional condition, and emits
one or more transitions:

```python
operator = (
    SymbolicDiscreteJaxOperator(hilbert, "raise", dtype="float64")
    .for_each_site("i")
    .where(site("i") < n_max)
    .emit(shift("i", 1), matrix_element=1.0)
    .compile()
)
```

The source code mirrors that split closely.

- [`nkdsl.dsl.operator.SymbolicDiscreteJaxOperator`](api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator) owns term construction.
- iterator methods create [`nkdsl.ir.term.KBodyIteratorSpec`](api/generated/nkdsl.KBodyIteratorSpec.md#nkdsl.KBodyIteratorSpec) objects.
- predicates are stored as [`nkdsl.ir.predicates.PredicateExpr`](api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr).
- rewrite programs are stored as [`nkdsl.ir.update.UpdateProgram`](api/generated/nkdsl.UpdateProgram.md#nkdsl.UpdateProgram).
- the compiler lowers the resulting [`nkdsl.ir.program.SymbolicOperatorIR`](api/generated/nkdsl.SymbolicOperatorIR.md#nkdsl.SymbolicOperatorIR) into a concrete executable operator target (default: [`nkdsl.core.compiled.CompiledOperator`](api/generated/nkdsl.CompiledOperator.md#nkdsl.CompiledOperator)).

## What gets compiled

For one input configuration `x`, one term behaves conceptually like this:

```
for visit in iterator.index_sets:
    if predicate(x, visit):
        for emission in emissions:
            x_prime = apply(update, x, visit)
            mel = evaluate(matrix_element, x, x_prime, visit)
            branches.append((x_prime, mel))
```

The compiled implementation evaluates these operations with JAX. It also sums
matrix elements for repeated targets by default. See
[connectivity assembly](compiler_connectivity.md) for output ordering and bounds.

## Current scope

The package includes:

- the fluent builder API
- symbolic selectors and expression trees
- immutable update-program builders
- the symbolic IR
- a compiler with validation, diagnostics linting, normalization, analysis, fusion planning, caching, and JAX lowering
- compiled operators that expose `get_conn_padded`

The core workflow is available in the beta release. Advanced compiler and
extension interfaces may change before the stable release.

Continue with [how compilation works](getting_started/compilation.md) for a
short visual walkthrough, or [the DSL guide](dsl/index.md) to learn how to
compose a definition. The [developer section](developers/index.md) covers the
IR and compiler implementation.
