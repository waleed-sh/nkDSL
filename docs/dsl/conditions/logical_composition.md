# Logical composition

Predicates can be combined with boolean operators.

## Supported forms

- `a & b` for logical AND
- `a | b` for logical OR
- `~a` for logical NOT

## Examples

```python
pred = (site("i") > 0) & (site("j") < 2)
pred = (site("i").value == 1) | (site("j").value == 0)
pred = ~(site("i") < 0)
```

For callback-style construction, [`nkdsl.dsl.context.ExpressionContext`](../../api/generated/nkdsl.ExpressionContext.md#nkdsl.ExpressionContext) also exposes `all_of`, `any_of`, and `not_` helpers.

## Keep Python and symbolic logic separate

Use the symbolic operators above rather than Python's `and`, `or`, and `not`.
The Python keywords request an immediate truth value while the builder is
running. nkDSL needs an expression that can be evaluated later for every
configuration.

Parentheses around each comparison are essential because bitwise operators
have different precedence from comparisons. For example, the occupation
transfer guard is:

```python
allowed = (site("i").value > 0) & (site("j").value < nmax)
```

For a longer rule, name intermediate predicates. This keeps the physical
requirements visible without changing how the expression is compiled:

```python
from nkdsl import ExpressionContext

ctx = ExpressionContext()
different_sites = ctx.ne(site("i").index, site("j").index)
source_occupied = site("i").value > 0
destination_available = site("j").value < nmax
allowed = different_sites & source_occupied & destination_available
```

Several consecutive `where` calls express the same conjunction on a term.
Use whichever form makes the rule easier to read. For mutually exclusive
alternatives with different outputs, use an emission chain rather than trying
to encode the whole choice in one boolean expression.
