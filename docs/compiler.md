# Compiler and execution

The compiler turns a symbolic operator into a callable connectivity kernel.
For a first introduction, read [how compilation works](getting_started/compilation.md).
For the implementation, the [developer pipeline guide](developers/compiler.md)
explains the context, each pass, cache lookup, and lowering. The
[IR guide](developers/ir.md) describes the records those stages exchange.

This page collects the compilation interfaces and runtime behavior. More
specialised guides cover [operator targets](compiler_operator_lowering_registry.md)
and [connectivity assembly](compiler_connectivity.md).

## From DSL to executable operator

The DSL builder does not execute operators directly. It first produces a [`nkdsl.core.operator.SymbolicOperator`](api/generated/nkdsl.SymbolicOperator.md#nkdsl.SymbolicOperator), which can then be lowered by the compiler.

The default flow is:

1.  build symbolic terms into [`nkdsl.ir.program.SymbolicOperatorIR`](api/generated/nkdsl.SymbolicOperatorIR.md#nkdsl.SymbolicOperatorIR)
2.  validate symbol scope and update-program consistency
3.  run DSL linting diagnostics (symbol/index/connectivity quality checks)
4.  normalize and fingerprint the IR
5.  look up a cached compiled artifact
6.  on a cache miss, plan output capacity, deduplication, estimated assembly cost, and fusion groups
7.  lower to a concrete executable operator target (default: [`nkdsl.core.compiled.CompiledOperator`](api/generated/nkdsl.CompiledOperator.md#nkdsl.CompiledOperator))

This structure is visible directly in the source tree under `nkdsl.compiler`.

## Default passes

The default pipeline created by [`nkdsl.compiler.defaults.default_symbolic_pass_pipeline`](api/generated/nkdsl.default_symbolic_pass_pipeline.md#nkdsl.default_symbolic_pass_pipeline) contains seven passes.

### Pre-cache passes

- [`nkdsl.compiler.passes.validation.SymbolicValidationPass`](api/generated/nkdsl.compiler.passes.validation.md#nkdsl.compiler.passes.validation.SymbolicValidationPass)
- [`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`](api/generated/nkdsl.compiler.passes.diagnostics.md#nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass)
- [`nkdsl.compiler.passes.normalization.SymbolicNormalizationPass`](api/generated/nkdsl.compiler.passes.normalization.md#nkdsl.compiler.passes.normalization.SymbolicNormalizationPass)

### Post-cache passes

- [`nkdsl.compiler.passes.analysis.SymbolicMaxConnSizeAnalysisPass`](api/generated/nkdsl.compiler.passes.analysis.md#nkdsl.compiler.passes.analysis.SymbolicMaxConnSizeAnalysisPass)
- [`nkdsl.compiler.passes.deduplication.SymbolicDeduplicationPass`](api/generated/nkdsl.compiler.passes.deduplication.md#nkdsl.compiler.passes.deduplication.SymbolicDeduplicationPass)
- [`nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass`](api/generated/nkdsl.compiler.passes.connectivity_cost.md#nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass)
- [`nkdsl.compiler.passes.fusion.SymbolicFusionPass`](api/generated/nkdsl.compiler.passes.fusion.md#nkdsl.compiler.passes.fusion.SymbolicFusionPass)

## Diagnostics policy and strictness

Diagnostics are configurable through [`nkdsl.SymbolicCompilerOptions`](api/generated/nkdsl.SymbolicCompilerOptions.md#nkdsl.SymbolicCompilerOptions):

- `diagnostics_enabled`
- `diagnostics_min_severity`
- `fail_on_warnings`
- `max_diagnostics`
- `lint_state_sample_size`
- `lint_branch_sample_cap`
- `lint_max_exact_hilbert_states`

For full details, see [Linting](dsl/linting/index.md). For the per-code lint catalog with examples and fixes, see [Lint Messages](dsl/linting/messages.md).

## Caching

The compiler can cache compiled artifacts in an in-memory store. The relevant public pieces are:

- [`nkdsl.compiler.SymbolicCompiler`](api/generated/nkdsl.SymbolicCompiler.md#nkdsl.SymbolicCompiler)
- [`nkdsl.compiler.SymbolicCompilerOptions`](api/generated/nkdsl.SymbolicCompilerOptions.md#nkdsl.SymbolicCompilerOptions)
- [`nkdsl.compiler.SymbolicCompilationSignature`](api/generated/nkdsl.SymbolicCompilationSignature.md#nkdsl.SymbolicCompilationSignature)
- [`nkdsl.compiler.SymbolicCacheKey`](api/generated/nkdsl.SymbolicCacheKey.md#nkdsl.SymbolicCacheKey)
- [`nkdsl.compiler.defaults.default_symbolic_artifact_store`](api/generated/nkdsl.default_symbolic_artifact_store.md#nkdsl.default_symbolic_artifact_store)

## Direct compiler usage

```python
from nkdsl import SymbolicCompiler

compiler = SymbolicCompiler()
artifact = compiler.compile(symbolic_operator)
compiled = artifact.compiled_operator
```

Convenience compilation from the builder or symbolic operator is also supported:

```python
compiled = SymbolicDiscreteJaxOperator(hi, "hop").for_each_site("i").emit(...).compile()
```

## Compiled operators

Compiled objects are normal executable operators. The default target exposes `get_conn_padded` and is represented by [`nkdsl.core.compiled.CompiledOperator`](api/generated/nkdsl.CompiledOperator.md#nkdsl.CompiledOperator). Custom lowering targets can bind the generated kernel to a different method name (for example `_get_conn_padded`).

By default, compiled connectivity kernels deduplicate connected states with the same `x'` target, sum their matrix elements, and drop zero-amplitude entries before final padding. This is controlled by `deduplicate_connected_components` (default: `True`).

The compiler groups identical target expressions, proves disjoint targets where possible, and limits runtime checks to unresolved collision components. `max_conn_size` uses the resulting unique-target bound, which can be smaller than the raw emission count. The `NKDSL-W401` diagnostic reports an estimated assembly cost penalty when deduplication remains more expensive than raw output. See [Connectivity assembly and deduplication](compiler_connectivity.md) for algorithms, output-capacity semantics, metadata, and reproducible benchmarks.

```python
xp, mels = compiled.get_conn_padded(x_batch)
```

## How to read printed IR

Every symbolic operator can be printed in a readable textual IR form:

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "heisenberg_sym", hermitian=True)
    .for_each(("i", "j"), over=edges)
    .emit(identity(), matrix_element=site("i").value * site("j").value)
    .for_each(("i", "j"), over=edges)
    .where(site("i").value * site("j").value < 0)
    .emit(swap("i", "j"), matrix_element=2.0)
    .build()
)

print(op.to_ir())
```

Example output:

```
symbolic.operator @"heisenberg_sym" [dtype=float64, hermitian=true, hilbert_size=8] {
  term #0 "0" [kbody, n_iter=8, max_conn_size=8] {
    iterate: for (i, j) in [(0, 1), (1, 2), (2, 3), ... +5 more]
    where:   true
    emit #0:
      update:    identity
      amplitude: (x[i] * x[j])
  }

  term #1 "1" [kbody, n_iter=8, max_conn_size=8] {
    iterate: for (i, j) in [(0, 1), (1, 2), (2, 3), ... +5 more]
    where:   ((x[i] * x[j]) < 0)
    emit #0:
      update:    x'[i], x'[j] = x[j], x[i]
      amplitude: 2
  }

}
```

Interpretation guide:

- `symbolic.operator ...`: global header with operator name, dtype, hermiticity, and Hilbert size.
- `term #k`: one independent contribution to the operator action.
- `iterate: ...`: static iteration domain (the tuples the term runs over).
- `where: ...`: predicate gate for each iterator tuple.
- `emit #m`: one emitted branch for that term.
- `update: ...`: state rewrite rule from `x` to `x'`.
- `amplitude: ...`: matrix element assigned to that emitted branch.
