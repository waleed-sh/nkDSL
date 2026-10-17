# Chaining terms and updates

nkDSL uses method chaining in two places. An operator builder collects terms,
while an update builder describes the ordered changes made by one branch.
Keeping these two roles separate makes longer definitions easier to read.

## Read an operator chain in blocks

Every iterator call starts a new term. The following `where`, `named`, and
`emit` calls belong to that term until another iterator starts.

```python
import netket as nk
from nkdsl import SymbolicDiscreteJaxOperator, affine, site

hi = nk.hilbert.Spin(s=0.5, N=3)
edges = [(0, 1), (1, 2)]
J, h = 1.0, 0.5

builder = (
    SymbolicDiscreteJaxOperator(hi, "ising", hermitian=True)
    .for_each(("i", "j"), over=edges)
    .named("interaction")
    .emit(matrix_element=-J * site("i").value * site("j").value)
    .for_each_site("i")
    .named("field")
    .emit(affine("i", scale=-1), matrix_element=-h)
)
symbolic = builder.build()
```

This operator has two terms, not one term with nested loops. The first
contributes a diagonal interaction for every edge. The second flips each
site. Reusing `"i"` is fine because each term has its own label scope. The
`"j"` label from the edge term is unavailable in the field term.

Multiple emissions within one term share the iterator and term condition.
They each start from the same input configuration. A second `emit` adds
another candidate transition, rather than continuing the previous emission's
update.

## Conditions apply to the whole term

Repeated `where` calls combine with logical AND. They describe the term as a
whole, including emissions already added to it. Put conditions before
emissions so that this scope is clear when reading the code.

```python
from nkdsl import hop

hopping = (
    SymbolicDiscreteJaxOperator(boson_hi, "hopping")
    .for_each(("i", "j"), over=bonds)
    .where(site("i").value > 0)
    .where(site("j").value < nmax)
    .emit(hop("i", "j"), matrix_element=-t)
    .build()
)
```

For conditions that differ between emissions, use
[`emit_if`, `emit_elseif`, and `emit_else`](emissions.md#conditional-emission-chains).
Keep a conditional chain consecutive. A plain emission, a term modifier, or
a new iterator closes the open chain.

## Operator builders are mutable

Fluent operator methods return the same builder after modifying it. Assigning
a builder to another variable does not create an independent operator.

```python
builder = SymbolicDiscreteJaxOperator(hi, "field")
same_builder = builder.for_each_site("i")
assert same_builder is builder
builder.emit(affine("i", scale=-1), matrix_element=1.0)
symbolic = builder.build()
```

`build()` seals the open term and returns a symbolic operator containing the
completed terms. Use a fresh builder for an independent definition. Calling
`compile()` on a builder performs `build().compile()` for you.

## Update chains are immutable and ordered

An update chain returns a new `Update` each time. You can safely reuse an
earlier update as the starting point for another program.

```python
from nkdsl import shift, write

lower = shift("i", -1)
transfer = lower.shift("j", +1)
reset_then_raise = write("i", 0).shift("i", +1)
```

`lower` still contains just one shift. `transfer` contains two. Within one
program, shifts, affine operations, swaps, and permutations act on the
working target in order. `reset_then_raise` therefore leaves site `i` at one.

Expression reads have a separate rule. `site("i").value` always refers to
the original source, even when used in a later update instruction:

```python
update = shift("i", +1).write("j", site("i").value)
```

If the source has values `(2, 0)` at `(i, j)`, this program produces `(3, 2)`.
It does not copy the newly incremented value. In a matrix element,
`emitted("i").value` can read the completed target. See
[selectors and expressions](selectors_and_expressions.md) for the distinction.
