# Custom emission clauses

An emission clause packages a recurring branch condition, update, amplitude,
and tag into a builder method. Use it when that combination appears in several
operator definitions.

## Abstraction hierarchy

Emission clauses are built on one public base class:

- [`nkdsl.AbstractEmissionClause`](../../api/generated/nkdsl.AbstractEmissionClause.md#nkdsl.AbstractEmissionClause)

Your subclass is registered with:

- [`nkdsl.register_emission_clause`](../../api/generated/nkdsl.register_emission_clause.md#nkdsl.register_emission_clause)
- or the generic decorator [`nkdsl.register`](../../api/generated/nkdsl.register.md#nkdsl.register)

After registration, the clause becomes a fluent method on [`nkdsl.SymbolicDiscreteJaxOperator`](../../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator).

## What every emission clause must provide

Minimum requirements:

1.  Subclass [`nkdsl.AbstractEmissionClause`](../../api/generated/nkdsl.AbstractEmissionClause.md#nkdsl.AbstractEmissionClause).
2.  Implement `build_emission(self, ctx, *args, **kwargs)`.
3.  Return [`nkdsl.dsl.emissions.types.EmissionClauseSpec`](../../api/generated/nkdsl.EmissionClauseSpec.md#nkdsl.EmissionClauseSpec).

`EmissionClauseSpec` supports these modes:

- `"emit"`
- `"emit_if"`
- `"emit_elseif"`
- `"emit_else"`

Optional but recommended:

- Set `clause_name` for a stable public method name.
- Validate arguments and fail fast with clear `ValueError` messages.
- Add branch tags for diagnostics clarity.

## Name resolution rules

Registering the same name again emits `nkdsl.RegistrationWarning` and keeps
the existing implementation. This makes repeated setup cells safe to run.
Pass `replace=True` to install the new definition. The examples below use
explicit replacement so edits take effect when you rerun them.

If you set `clause_name`, that name is used for the fluent method. Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

- valid Python identifier
- must not start with `_`
- must not collide with reserved builder method names

## Example: conditional emission convenience clause

The clause below emits a branch only when one site value is above a cutoff.

```python
import netket as nk
import nkdsl
from nkdsl.dsl.emissions.types import EmissionClauseSpec


class EmitWhenAtLeast(nkdsl.AbstractEmissionClause):
    clause_name = "emit_when_at_least"

    def build_emission(self, ctx, label: str = "i", cutoff: int = 1):
        predicate = ctx.site(label).value >= int(cutoff)
        return EmissionClauseSpec(
            mode="emit_if",
            predicate=predicate,
            update=nkdsl.identity(),
            matrix_element=ctx.site(label).value,
            tag="emit-when-at-least",
        )


nkdsl.register_emission_clause(EmitWhenAtLeast, replace=True)
```

Usage:

```python
hi = nk.hilbert.Fock(n_max=3, N=4)

op = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "custom-emission")
    .for_each_site("i")
    .emit_when_at_least("i", cutoff=2)
    .emit_else(nkdsl.identity(), matrix_element=0.0, tag="fallback")
    .build()
)
```

## Check the emission

Check that the result is a valid `EmissionClauseSpec` and that conditional
modes follow `if -> elseif* -> else?`. Use tags to identify branches in
diagnostics, and test both valid calls and invalid chaining.

## Discoverability

You can inspect currently available emission clause names at runtime:

```python
names = nkdsl.available_emission_clause_names()
print(names)
```
