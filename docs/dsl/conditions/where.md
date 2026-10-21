# `where(predicate)`

`where` attaches a predicate to the current term.

If the predicate is false for a given iterator visit, that visit produces no valid emitted branch.

Example:

```python
constrained = (
    SymbolicDiscreteJaxOperator(hi, "constrained_raise")
    .for_each_site("i")
    .where(site("i") < 2)
    .emit(shift("i", +1), matrix_element=1.0)
    .build()
)
```

## Chaining `where` calls

Multiple `where` calls on the same term are combined with logical AND.

```python
from nkdsl import ExpressionContext

ctx = ExpressionContext()
op = (
    SymbolicDiscreteJaxOperator(hi, "hop")
    .for_each_pair("i", "j")
    .where(ctx.ne(site("i").index, site("j").index))
    .where(site("i") > 0)
    .where(site("j") < nmax)
    .emit(shift("i", -1).shift("j", +1))
    .build()
)
```

The pair example assumes a local basis from zero to `nmax`. It checks
different source and destination positions, a nonempty source, and available
capacity at the destination. Each guard contributes to the same term-wide
predicate.

## Scope and numerical safety

The predicate reads the source configuration. It does not see the result of
the emission's update. To express a condition on a proposed move, derive that
condition from the source values. For example, a raise by one is allowed when
`site("i").value < nmax`.

Adding a `where` later in a term changes the shared predicate for all of its
emissions, including ones already added. Place guards before emissions to
make that behavior clear. Starting a new iterator starts a fresh term with
its own predicate.

A guard can also exclude a singular amplitude, such as a division by a zero
occupation. Write the guard explicitly rather than relying on a zero
coefficient to cancel an undefined value. Compiler diagnostics check common
cases, but sampled checks cannot establish numerical safety for every large
Hilbert space. See [diagnostic messages](../linting/messages.md) for
`NKDSL-E003`.
