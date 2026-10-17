# Custom predicate clauses

Add a predicate clause when several operators share a selection rule. The
registered builder method constructs the condition and combines it with the
term's existing predicates.

## Abstraction hierarchy

Predicate clauses are built on one public base class:

- [`nkdsl.AbstractPredicateClause`](../../api/generated/nkdsl.AbstractPredicateClause.md#nkdsl.AbstractPredicateClause)

Your subclass is registered with:

- [`nkdsl.register_predicate_clause`](../../api/generated/nkdsl.register_predicate_clause.md#nkdsl.register_predicate_clause)
- or the generic decorator [`nkdsl.register`](../../api/generated/nkdsl.register.md#nkdsl.register)

After registration, the clause becomes a fluent method on [`nkdsl.SymbolicDiscreteJaxOperator`](../../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator).

## What every predicate clause must provide

Minimum requirements:

1.  Subclass [`nkdsl.AbstractPredicateClause`](../../api/generated/nkdsl.AbstractPredicateClause.md#nkdsl.AbstractPredicateClause).
2.  Implement `build_predicate(self, ctx, *args, **kwargs)`.
3.  Return a value coercible to [`nkdsl.PredicateExpr`](../../api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr).

About `ctx`:

- `ctx` is an [`nkdsl.ExpressionContext`](../../api/generated/nkdsl.ExpressionContext.md#nkdsl.ExpressionContext).
- Use it to build selector-based expressions in a consistent way.
- Example: `ctx.site("i").value >= 1`.

Optional but recommended:

- Set `clause_name` for a stable public method name.
- Validate user arguments and fail fast with clear errors.
- Keep logic focused: one clause should represent one reusable predicate idea.

## Name resolution rules

If you set `clause_name`, that name is used for the fluent method. Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

- valid Python identifier
- must not start with `_`
- must not collide with reserved builder method names

## Example: Occupancy threshold predicate

This clause checks whether one site value is above a threshold.

```python
import netket as nk
import nkdsl


class AtLeast(nkdsl.AbstractPredicateClause):
    clause_name = "at_least"

    def build_predicate(self, ctx, label: str = "i", cutoff: int = 0):
        return ctx.site(label).value >= int(cutoff)


nkdsl.register_predicate_clause(AtLeast, replace=True)
```

Usage:

```python
hi = nk.hilbert.Fock(n_max=3, N=4)

op = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "filtered")
    .for_each_site("i")
    .at_least("i", cutoff=1)
    .emit(nkdsl.identity(), matrix_element=1.0)
    .build()
)
```

## Example: Band predicate (inclusive range)

This clause keeps visits with values in `[lower, upper]`.

```python
import nkdsl


class ValueBand(nkdsl.AbstractPredicateClause):
    clause_name = "value_in_band"

    def build_predicate(self, ctx, label: str = "i", *, lower: int, upper: int):
        if int(lower) > int(upper):
            raise ValueError("lower must be <= upper.")
        site_value = ctx.site(label).value
        return (site_value >= int(lower)) & (site_value <= int(upper))


nkdsl.register_predicate_clause(ValueBand, replace=True)
```

Usage:

```python
op = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "banded")
    .for_each_site("i")
    .value_in_band("i", lower=1, upper=2)
    .emit(nkdsl.identity(), matrix_element=1.0)
    .build()
)
```

## Composition semantics with `where`

Predicate clauses compose with existing conditions using logical AND. That means this:

```python
.at_least("i", cutoff=1).where(nkdsl.site("i") < 3)
```

is equivalent to:

```python
.where((nkdsl.site("i") >= 1) & (nkdsl.site("i") < 3))
```

## Check the predicate

Validate arguments, return a value coercible to `PredicateExpr`, and document
the method name. Test composition with `where(...)` and rejection of bad
arguments or invalid clause names.

## Discoverability

You can inspect currently available predicate clause names at runtime:

```python
names = nkdsl.available_predicate_clause_names()
print(names)
```
