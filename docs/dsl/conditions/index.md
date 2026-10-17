# Conditions

Conditions decide which iterator visits produce output branches.

The main entry point is `where(...)` and the underlying boolean IR type is [`nkdsl.ir.PredicateExpr`](../../api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr).

A condition is evaluated for each input configuration and each row of the
term's iterator. It can compare occupations, site indices, or more involved
symbolic expressions. The result gates all emissions on that term. A rejected
visit contributes zero matrix element and does not become a physical
connection.

Use conditions to state the allowed moves explicitly. For a bosonic hop on
a truncated basis, check that the source is occupied and the destination is
below its maximum. A shift update performs arithmetic. It does not add those
physical constraints for you.

Conditions describe runtime choices inside the compiled operator. They are
not Python `if` statements evaluated while building it. Combine comparisons
with `&`, `|`, and `~`, and put each comparison in parentheses. The pages below
explain the available forms and their scope.

For alternatives with different updates or matrix elements, keep a common
term condition in `where` and use an
[emission-level conditional chain](../emissions.md#conditional-emission-chains)
for the alternatives. This distinguishes a shared validity constraint from
the choice of which branch to emit.

- [``where(predicate)``](where.md)
- [Comparison predicates](comparisons.md)
- [Logical composition](logical_composition.md)
