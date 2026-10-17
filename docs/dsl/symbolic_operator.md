# Symbolic operators

Start with [`SymbolicDiscreteJaxOperator`](../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator).
The builder collects terms and freezes them into an IR when you call `build()`.
It describes an operator through transitions from an input configuration `x`
to a target `x'`. A diagonal contribution uses an unchanged target. An off-diagonal
contribution changes one or more site values. The matrix element specifies
the weight of that contribution.

A builder accumulates **terms**. Each term has:

- an iterator
- an optional predicate
- one or more emissions

A minimal shape looks like this:

```python
from nkdsl import SymbolicDiscreteJaxOperator, shift, site

op = (
    SymbolicDiscreteJaxOperator(hi, "raise_one")
    .for_each_site("i")
    .where(site("i") < 2)
    .emit(shift("i", +1), matrix_element=1.0)
    .build()
)
```

## Important builder rules

### Iterator methods open terms

Every call to `globally` or `for_each*` seals the current term and starts a new one. That means `where` and `emit` always attach to the most recently opened term.

### `build()` returns a symbolic object

`build()` returns [`nkdsl.core.operator.SymbolicOperator`](../api/generated/nkdsl.SymbolicOperator.md#nkdsl.SymbolicOperator). This object holds typed IR terms and can later be compiled.

### `compile()` is a convenience shortcut

Calling `compile()` on the builder is equivalent to `build().compile()`.

### Connectivity semantics

By default, `compile()` enables connected-state deduplication:

- duplicate emitted targets `x'` are merged
- matrix elements are summed across merged entries
- zero-amplitude components are dropped before padding

Static target analysis reduces `max_conn_size` when equivalence or finite target images prove a smaller capacity. The kernel uses exact runtime checks for unresolved collisions. `NKDSL-W401` warns about estimated assembly overhead without changing the requested behavior. See [Connectivity assembly and deduplication](../compiler_connectivity.md) for details and benchmarks.

Pass `deduplicate_connected_components=False` to keep raw per-branch connectivity output.

## Term metadata

Two builder helpers matter when definitions become larger.

`named(name)`  
gives the current term a readable identifier for IR dumps and compiler output

`max_conn_size(hint)`  
sets a manual static upper bound for the term fanout when you know a tighter bound than the generic iterator-size times emission-count estimate. Treat a hint as a correctness contract, not a performance preference. An underestimated bound can make the padded representation insufficient.

## Hermiticity and dtype

The builder constructor accepts `dtype` and `hermitian`. These become part of both the symbolic IR and the compiled artifact metadata.

!!! warning

    Set `hermitian=True` only when the operator is Hermitian. NetKet uses this
    declaration when computing gradients. An incorrect declaration can produce
    incorrect gradients.

## Build first when inspecting a definition

Keeping the symbolic and compiled objects in separate variables makes it easy
to inspect the definition without mixing it with runtime work:

```python
symbolic = builder.build()
print(symbolic.to_ir())
compiled = symbolic.compile()
```

Term names and emission tags help relate diagnostics to the definition. The
printed IR shows iterator rows, shared conditions, updates, and amplitudes.
It is particularly useful for checking whether a new iterator started where
you intended. See [chaining terms and updates](chaining.md) for the builder's
scope and mutation rules, and [how compilation works](../getting_started/compilation.md)
for what happens after `build()`.
