# Custom iterator clauses

Add an iterator clause when a site selection or topology is reused across
operators. After registration, it is available as a builder method alongside
`for_each_site` and `for_each_pair`.

## Abstraction hierarchy

Iterator clauses are built on one public base class:

- [`nkdsl.AbstractIteratorClause`](../../api/generated/nkdsl.AbstractIteratorClause.md#nkdsl.AbstractIteratorClause)

Your subclass is registered with:

- [`nkdsl.register_iterator_clause`](../../api/generated/nkdsl.register_iterator_clause.md#nkdsl.register_iterator_clause)
- or the generic decorator [`nkdsl.register`](../../api/generated/nkdsl.register.md#nkdsl.register)

After registration, the clause becomes a fluent method on [`nkdsl.SymbolicDiscreteJaxOperator`](../../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator).

## What every iterator clause must provide

Minimum requirements:

1.  Subclass [`nkdsl.AbstractIteratorClause`](../../api/generated/nkdsl.AbstractIteratorClause.md#nkdsl.AbstractIteratorClause).
2.  Implement `build_iterator(self, hilbert, *args, **kwargs)`.
3.  Return a valid iterator specification:
    - either [`nkdsl.KBodyIteratorSpec`](../../api/generated/nkdsl.KBodyIteratorSpec.md#nkdsl.KBodyIteratorSpec)
    - or `(labels, index_sets)` where labels are strings and index sets are tuples of integer tuples.

Optional but recommended:

- Set `clause_name` for a stable public method name.
- Validate user inputs early and raise clear `ValueError` messages.
- Keep iterator generation deterministic (important for reproducibility and tests).

## Name resolution rules

Registering the same name again emits `nkdsl.RegistrationWarning` and keeps
the existing implementation. This makes repeated setup cells safe to run.
Pass `replace=True` to install the new definition. The examples below use
explicit replacement so edits take effect when you rerun them.

If you set `clause_name`, that name is used for the fluent method. Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

- valid Python identifier
- must not start with `_`
- must not collide with reserved builder method names (for example `build`)

## Example: Even-site iterator

The clause below iterates only over even lattice sites.

```python
import netket as nk
import nkdsl


class EvenSites(nkdsl.AbstractIteratorClause):
    clause_name = "for_each_even_site"

    def build_iterator(self, hilbert, label: str = "i"):
        n = int(hilbert.size)
        rows = tuple((k,) for k in range(n) if k % 2 == 0)
        if not rows:
            raise ValueError("No even sites available for this Hilbert space.")
        return (str(label),), rows


nkdsl.register_iterator_clause(EvenSites, replace=True)
```

Usage:

```python
hi = nk.hilbert.Fock(n_max=3, N=6)

op = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "even-diagonal")
    .for_each_even_site("i")
    .emit(nkdsl.identity(), matrix_element=nkdsl.site("i").value)
    .build()
)
```

The registered method supplies the iterator rows. Callers can add conditions
and emissions in the usual way.

## Example: Graph-edge iterator

A common use case is iterating over a fixed edge list from a graph.

```python
import netket as nk
import nkdsl


class ForEachEdge(nkdsl.AbstractIteratorClause):
    clause_name = "for_each_edge"

    def build_iterator(self, hilbert, label_a: str = "i", label_b: str = "j", *, edges):
        rows = tuple((int(i), int(j)) for i, j in edges)
        if not rows:
            raise ValueError("edges must contain at least one pair.")
        return (str(label_a), str(label_b)), rows


nkdsl.register_iterator_clause(ForEachEdge, replace=True)
```

Usage:

```python
edges = [(0, 1), (1, 2), (2, 3)]
hi = nk.hilbert.Fock(n_max=2, N=4)

hop = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "edge-hop")
    .for_each_edge("i", "j", edges=edges)
    .where(nkdsl.site("i") > 0)
    .emit(nkdsl.shift("i", -1).shift("j", +1), matrix_element=1.0)
    .build()
)
```

## Check the iterator

Check that there is at least one row, each row matches the label arity, and all
indices are in bounds. Document the method name and test registration,
successful construction, and invalid inputs.

## Discoverability

You can inspect currently available iterator clause names at runtime:

```python
names = nkdsl.available_iterator_clause_names()
print(names)
```
