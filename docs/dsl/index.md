# The DSL

The DSL describes which configurations an operator connects and the matrix
element for each transition. Start with a builder, choose an iteration domain,
and add conditions and emissions to each term.

A useful way to read a definition is to ask four questions: which site tuples
does the term visit, which source configurations allow the move, what target
does the update produce, and what matrix element does it carry? These questions
correspond to iterators, conditions, updates, and emissions.

The [quick start](../quickstart.md) puts them together in an Ising Hamiltonian.
The [chaining guide](chaining.md) explains how fluent calls form terms and how
an update chain differs from an operator chain. Read
[how compilation works](../getting_started/compilation.md) to follow the
definition from those terms to an executable JAX operator.

The core pieces are:

- [`SymbolicDiscreteJaxOperator`](../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator) creates the operator definition
- selectors such as [`nkdsl.site`](../api/generated/nkdsl.site.md#nkdsl.site), [`nkdsl.emitted`](../api/generated/nkdsl.emitted.md#nkdsl.emitted), and [`nkdsl.symbol`](../api/generated/nkdsl.symbol.md#nkdsl.symbol) (including `default/doc/dtype` declarations)
- predicates built with [`nkdsl.ir.PredicateExpr`](../api/generated/nkdsl.PredicateExpr.md#nkdsl.PredicateExpr)
- update programs built with [`nkdsl.Update`](../api/generated/nkdsl.Update.md#nkdsl.Update) and helper functions such as [`nkdsl.shift`](../api/generated/nkdsl.shift.md#nkdsl.shift) or [`nkdsl.write`](../api/generated/nkdsl.write.md#nkdsl.write)
- emissions that attach a rewrite rule and matrix element to each active visit
- compiler-integrated linting that reports actionable diagnostics at compile time

For custom fluent extensions, see [Extending the DSL](../guides/extending_dsl/index.md).

## Linting

The DSL includes a compiler-integrated linting pass that runs during `.compile()`. It checks unresolved symbols, index-safety issues, and sampled connectivity validity (support, constraints, and local basis values).

Start with [Linting](linting/index.md) for the overall workflow and policy options. Use [Lint Messages](linting/messages.md) for the full diagnostic message catalog with examples and resolutions.

- [The Symbolic Operator](symbolic_operator.md)
- [Chaining terms and updates](chaining.md)
- [Selectors and expressions](selectors_and_expressions.md)
- [Emissions and matrix elements](emissions.md)
- [Linting](linting/index.md)
- [Currently Supported](currently_supported.md)
- [Iterators](iterators/index.md)
- [Conditions](conditions/index.md)
- [Updates](updates/index.md)
