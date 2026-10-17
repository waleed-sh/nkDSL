# `for_each_distinct_pair(label_a, label_b)`

`for_each_distinct_pair` is the dense pair iterator without diagonal visits.

It iterates over all ordered pairs with `i != j`.

Use it when the diagonal is never meaningful and you want that fact encoded in the iterator itself rather than in an extra predicate.

Example:

```python
hop = (
    SymbolicDiscreteJaxOperator(hi, "hop")
    .for_each_distinct_pair("i", "j")
    .where(site("i") > 0)
    .where(site("j") < nmax)
    .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
    .build()
)
```

Here the local occupation range is zero to `nmax`. The two value guards ensure
that lowering the source and raising the destination stay in that range.
The iterator itself guarantees only different site indices, not different
occupations. Two distinct sites may have equal values in the input state.

## Distinct does not mean unordered

For three sites, the domain contains six rows: `(0, 1)`, `(0, 2)`, `(1, 0)`,
`(1, 2)`, `(2, 0)`, and `(2, 1)`. It has `N * (N - 1)` visits in general.
Both directions of an occupation transfer are therefore present.

For a symmetric interaction that should appear once per pair, use an index
condition such as `site("i").index < site("j").index`, or provide the desired
unordered edges through [`for_each`](for_each_kbody.md). Target deduplication
preserves the sum of contributions and will not remove accidental double
counting of a physical coupling.

Choose this iterator for dense interactions between different sites. Use a
static row list for graph edges or a restricted set of pairs. With several
emissions per visit, each emission shares this same domain and the term's
conditions.
