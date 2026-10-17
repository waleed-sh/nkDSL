# `for_each_site(label)`

`for_each_site` iterates over all site indices `0` through `hilbert.size - 1`.

The bound label gives access to:

- `site(label).value`
- `site(label).index`

Typical use cases:

- single-site raising or lowering operators
- on-site diagonal energies
- local flips or local constraints

Example:

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "raise")
    .for_each_site("i")
    .where(site("i") < 2)
    .emit(shift("i", +1), matrix_element=1.0)
    .build()
)
```

Here `hi` must have local values from zero to two for the guard to describe a
valid raising move. At a site already equal to two, the condition is false and
the branch has zero matrix element. The iterator still has `hi.size` visits.
A condition changes which visits are active, not the static iteration domain.

## Values and positions have different roles

For a source configuration `[2, 0, 1]`, the visit at position one binds
`site("i").index` to `1` and `site("i").value` to `0`. Use the index for
site-dependent coefficients or geometry, and the value for occupation or spin
dependence. The label is a symbolic name for the visit, not a fixed position.

If the update is omitted, all visits are diagonal contributions. For example,
`.emit(matrix_element=site("i").value)` defines the sum of occupations over
sites. Deduplication combines these contributions into one target with the
summed matrix element. With one emission, the raw branch count is `N`, even
when the compiled unique-target capacity is one.

To visit only selected sites, use
[`for_each(("i",), over=[(0,), (2,)])`](for_each_kbody.md) instead of changing
the all-sites iterator.
