# Static K-body iterators

For graph-driven or geometry-driven operators, the most important iterator is the fully general `for_each(labels, over=...)`.

## `for_each(labels, over=...)`

This binds an arbitrary sequence of labels to an explicit static list of index rows.

```python
edges = [(0, 1), (1, 2), (2, 3)]

op = (
    SymbolicDiscreteJaxOperator(hi, "edge_term")
    .for_each(("src", "dst"), over=edges)
    .emit(identity(), matrix_element=site("src").value * site("dst").value)
    .build()
)
```

This is usually the best choice whenever the operator follows a graph, plaquette list, triangle list, or other precomputed structure.

## Convenience wrappers

The builder also provides wrappers for common fixed arities.

`for_each_triplet(label_a, label_b, label_c, over=...)`  
static list of three-tuples

`for_each_plaquette(label_a, label_b, label_c, label_d, over=...)`  
static list of four-tuples

These wrappers delegate to the same underlying machinery and mainly improve readability.

## Rows are part of the definition

The row list is consumed while constructing the operator and becomes part of
its static iteration domain. Each row must have one integer site index per
label. For the edge example, the first row binds `src` to zero and `dst` to
one. Values at those positions are read later from each input configuration.
Changing the graph means constructing a new symbolic definition.

The iterator preserves repeated rows. If `(0, 1)` appears twice, its
contribution appears twice in the operator. Likewise, `(0, 1)` and `(1, 0)`
are separate visits. This is useful for directed moves but needs care for
symmetric interactions. Decide whether your geometry supplies directed or
undirected edges before assigning coefficients.

With `M` rows and `E` emissions, the raw domain contributes up to `M * E`
candidate branches. Conditions can deactivate branches, and deduplication
can reduce the unique-target capacity. Neither operation changes the meaning
of the row list.

Use descriptive labels for larger tuples, such as `("a", "b", "c", "d")`
around a plaquette. The labels establish scope within the term. They do not
impose an orientation or a geometric relation beyond the supplied rows.
