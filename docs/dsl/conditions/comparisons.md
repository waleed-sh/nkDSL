# Comparison predicates

The predicate system supports the standard comparison family.

## Available comparisons

- `==`
- `!=`
- `<`
- `<=`
- `>`
- `>=`

These comparisons build predicates that the compiler evaluates for each
configuration. Equality and inequality against numbers work on selectors,
their `.value` and `.index` properties, and arithmetic expressions.

## Examples

```python
site("i") > 0
(site("i").index == 3) & (site("i").value == 1)
site("i").value != 0
site("i").value + site("j").value <= 2
```

Internally, these become [`nkdsl.ir.PredicateExpr`](../../api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr) nodes such as `gt`, `ne`, or `le`.

## Select an index and a value

An index condition selects a position in the configuration. A value condition
selects the occupation, spin, or charge at that position. Combine them to
construct a diagonal projector onto a particular charge on one edge:

```python
projector = (
    SymbolicDiscreteJaxOperator(hi, "charge-projector", hermitian=True)
    .for_each_site("e")
    .where((site("e").index == edge_index) & (site("e").value == charge))
    .emit(identity(), matrix_element=1.0)
)
```

Only the selected edge can contribute. Its matrix element is one when its
charge matches and zero otherwise. The identity update leaves the
configuration unchanged.

## Compare two expressions

For backwards compatibility, `==` and `!=` between two `AmplitudeExpr` objects
compare the structure of their expression trees. This behavior keeps IR
comparisons, dictionary keys, and compiler caches unchanged. Equality against
a number or a symbol-name string constructs a runtime predicate.

Use `ExpressionContext.eq` or `ExpressionContext.ne` to compare the evaluated
values of two expressions:

```python
from nkdsl import ExpressionContext, site

ctx = ExpressionContext()
different_values = ctx.ne(site("i").value, site("j").value)
different_indices = ctx.ne(site("i").index, site("j").index)
```

`different_values` tests the occupations or spin values in the current
configuration. `different_indices` tests whether the iterator bound different
positions. Distinct positions can hold equal values, so these conditions are
not interchangeable. The existing `PredicateExpr.eq` and `PredicateExpr.ne`
helpers remain available as well.

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
