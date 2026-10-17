# Comparison predicates

The predicate system supports the standard comparison family.

## Available comparisons

- `==`
- `!=`
- `<`
- `<=`
- `>`
- `>=`

These comparisons operate on symbolic amplitude expressions.

## Examples

```python
site("i") > 0
site("i").index != site("j").index
site("i").value + site("j").value <= 2
```

Internally, these become [`nkdsl.ir.PredicateExpr`](../../api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr) nodes such as `gt`, `ne`, or `le`.

## Compare the quantity you mean

`site("i").value != site("j").value` asks whether the two occupations or
spin values differ in the current configuration. In contrast,
`site("i").index != site("j").index` asks whether the iterator bound
different positions. Distinct positions can hold equal values, so the two
conditions are not interchangeable.

Comparisons can include arithmetic on either side. A constraint on the total
occupation of a pair might be written as
`site("i").value + site("j").value <= 2`. The expression is recorded when the
operator is built and evaluated on each input at runtime.

Write an interval as two comparisons joined by `&`:

```python
within_bounds = (site("i").value >= 0) & (site("i").value < nmax)
```

Python's chained form `0 <= value < nmax` asks Python to convert an intermediate
comparison to a truth value. It does not construct the intended symbolic
predicate. For the same reason, do not put a symbolic comparison directly in
a Python `if`. Use [`where`](where.md) or a conditional emission instead.
