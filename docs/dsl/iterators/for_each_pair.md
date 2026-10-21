# `for_each_pair(label_a, label_b)`

`for_each_pair` iterates over all ordered pairs `(i, j)` in `[0, N) x [0, N)`.

This includes diagonal pairs `(i, i)`.

Use it when:

- you need a dense two-body sweep
- diagonal and off-diagonal pairs are both meaningful
- you prefer to remove illegal cases with `where(...)` instead of choosing a more restrictive iterator

Example:

```python
from nkdsl import ExpressionContext

ctx = ExpressionContext()
hop = (
    SymbolicDiscreteJaxOperator(hi, "hop")
    .for_each_pair("i", "j")
    .where(ctx.ne(site("i").index, site("j").index))
    .where(site("i") > 0)
    .where(site("j") < nmax)
    .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
    .build()
)
```

The example assumes bosonic occupations from zero to `nmax`. Both bounds
matter: a hop needs a particle at the source and room at the destination.

## Pair order matters

There are `N * N` visits. For two sites the rows are `(0, 0)`, `(0, 1)`,
`(1, 0)`, and `(1, 1)`. Both orientations occur, so an update that transfers
occupation from `i` to `j` includes both transfer directions after diagonal
pairs are removed.

A symmetric diagonal expression is different. Summing
`site("i").value * site("j").value` over ordered unequal pairs counts each
unordered interaction twice. If the intended sum is over `i < j`, add that
index condition or supply a list with each edge once. Deduplication adds
repeated contributions. It does not correct a factor of two in the model.

For a sparse graph, an explicit [edge iterator](for_each_kbody.md) is normally
clearer and smaller than this dense domain. If only equal indices are unwanted,
[`for_each_distinct_pair`](for_each_distinct_pair.md) encodes that restriction
without a separate predicate.
