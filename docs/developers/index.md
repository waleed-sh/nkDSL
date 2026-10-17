# Developing nkDSL

The DSL, symbolic representation, and executable operator are separate parts
of nkDSL. Most user-facing extensions only need the DSL clause interfaces.
Compiler changes require understanding what information survives into the IR
and which analyses the lowerer relies on.

Start with the [user's compilation walkthrough](../getting_started/compilation.md)
if you are new to the project. Then read the
[IR guide](ir.md) and [compiler pipeline](compiler.md) together. The IR guide
defines the objects passed between stages. The pipeline guide explains how
those objects become an executable connectivity function.

## Find the relevant layer

| Location under `src/nkdsl/` | Responsibility |
| --- | --- |
| `dsl/` | Fluent builders, selectors, expression context, and clause registration |
| `ir/` | Immutable expression, predicate, update, term, and operator records |
| `core/` | Symbolic operators and executable operator wrappers |
| `compiler/passes/` | Validation, diagnostics, connectivity analysis, and planning |
| `compiler/diagnostics/` | Diagnostic rules, state sampling, and message formatting |
| `compiler/lowering/` | JAX generation, connectivity assembly, and target registries |
| `compiler/core/` and `compiler/cache/` | Pipeline context, options, artifacts, signatures, and stores |

For a reusable domain-specific iterator, condition, or emission, begin with
the [extension guides](../guides/extending_dsl/index.md). A clause should
translate user intent into the existing IR wherever possible. This lets the
same validation and connectivity analyses apply to the new syntax.

For integration with another operator class, use the
[operator lowering registry](../compiler_operator_lowering_registry.md).
Changing the wrapper class does not require a new backend. A new execution
backend belongs in the lowerer registry instead.

## Validate a change at its boundary

A DSL change should be checked at the resulting IR as well as at runtime.
A compiler transformation must preserve the sum of matrix elements for each
target, including cancellation and inactive branches. On small Hilbert spaces,
compare dense matrices or aggregate raw connectivity independently. Include
the padded capacity in checks when the transformation changes output shapes.

Performance work should distinguish symbolic analysis time, JAX compilation
time, and synchronised steady-state execution. The connectivity benchmark
procedure is described in [the deduplication guide](../compiler_connectivity.md).
Keep correctness checks independent of the algorithm being benchmarked.

Documentation changes have a lighter environment than runtime development.
The [documentation contribution guide](../contributing/documentation.md)
explains local previews, strict builds, API rendering, and when notebook
results need to be refreshed.
