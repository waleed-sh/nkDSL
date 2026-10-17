# Linting

Compilation checks the symbolic definition for unresolved parameters, unsafe
indices, and likely connectivity mistakes. Each finding has an `NKDSL-*` code,
a location, and a suggested action.

Linting runs through [`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`](../../api/generated/nkdsl.compiler.passes.diagnostics.md#nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass).

- [Lint Messages](messages.md)

## What is checked

The checks cover:

- free parameters left unresolved
- static index expressions outside Hilbert bounds
- divisions by runtime state values that may be zero
- terms that never emit due to constant-false predicates
- generated states that violate support, constraints, or local basis rules

The stable codes can be used to look up a message or enforce a diagnostics
policy in automated checks.

## When linting runs

In the default pipeline, linting executes before normalization:

1.  [`nkdsl.compiler.passes.validation.SymbolicValidationPass`](../../api/generated/nkdsl.compiler.passes.validation.md#nkdsl.compiler.passes.validation.SymbolicValidationPass)
2.  [`nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass`](../../api/generated/nkdsl.compiler.passes.diagnostics.md#nkdsl.compiler.passes.diagnostics.SymbolicDiagnosticsPass)
3.  [`nkdsl.compiler.passes.normalization.SymbolicNormalizationPass`](../../api/generated/nkdsl.compiler.passes.normalization.md#nkdsl.compiler.passes.normalization.SymbolicNormalizationPass)

Post-cache passes (analysis and fusion) run afterward on cache misses. The post-cache [`nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass`](../../api/generated/nkdsl.compiler.passes.connectivity_cost.md#nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass) also reports `NKDSL-W401` when its assembly cost model predicts a deduplication penalty. It follows the same diagnostics policy. See [Connectivity assembly and deduplication](../../compiler_connectivity.md) for the model and its scope.

## Configuration

Lint behavior is controlled by [`nkdsl.SymbolicCompilerOptions`](../../api/generated/nkdsl.SymbolicCompilerOptions.md#nkdsl.SymbolicCompilerOptions):

- `diagnostics_enabled`: enable/disable linting
- `diagnostics_min_severity`: minimum reported/enforced level (`info`, `warning`, `error`)
- `fail_on_warnings`: treat warnings as compile-blocking
- `max_diagnostics`: cap number of reported diagnostics
- `lint_state_sample_size`: sampled source-state count for connectivity checks
- `lint_branch_sample_cap`: cap sampled branch evaluations
- `lint_max_exact_hilbert_states`: limit exact support-membership checks to manageable Hilbert sizes

Example:

```python
from nkdsl import SymbolicCompilerOptions

opts = SymbolicCompilerOptions(
    diagnostics_enabled=True,
    diagnostics_min_severity="warning",
    fail_on_warnings=True,
    lint_state_sample_size=64,
    lint_branch_sample_cap=4096,
)
```

## Reading lint output

Diagnostics are printed in a structured, readable block:

```
Diagnostics summary: total=2 (errors=1, warnings=1, info=0)
Read more: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html
- [NKDSL-E001] ERROR @ my_op.term0
  Unresolved free symbol(s): %J.
  Suggestion: Replace with constants or bind symbols before compilation. DSL compilation requires all free symbols to be resolved.
  Docs: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html#lint-code-nkdsl-e001
```

Every diagnostic includes:

- a stable code (for example `NKDSL-W302`)
- severity
- location (operator and term when available)
- message and suggested fix
- a direct documentation link for that code

## Severity semantics

Severity determines how a finding is reported and whether compilation stops:

- `error` means compilation should usually stop until resolved
- `warning` means likely correctness/runtime risk that many teams gate in CI
- `info` means non-blocking quality signal (readability, diagnostics coverage, or analysis completeness)

For symbol diagnostics specifically, remember that only unresolved symbols trigger `NKDSL-E001`. Symbols declared with `default=...` are considered resolved.

You can enforce stricter behavior with `diagnostics_min_severity` and `fail_on_warnings`.

## Lint message catalog

Use [Lint Messages](messages.md) for the complete per-code reference, including:

- what each lint means
- typical causes
- concrete examples
- practical remediation guidance

## Recommended workflow

When you hit diagnostics during operator development:

1.  Fix all `NKDSL-E*` issues first.
2.  Resolve or consciously waive `NKDSL-W*` warnings.
3.  Use `NKDSL-I*` signals to improve readability and diagnostic coverage.
4.  Re-run `.compile()` after each logical change so regressions are caught immediately.

For production CI, a common strict profile is:

- `diagnostics_min_severity="warning"`
- `fail_on_warnings=True`
- fixed sampling knobs (`lint_state_sample_size` / `lint_branch_sample_cap`) so diagnostics are stable across runs.

## Important notes

- Connectivity diagnostics are sample-based and intended as early warnings.

- Passing sampled checks does not prove that every generated configuration is valid.

- For strict CI workflows, combine:

  - `diagnostics_min_severity="warning"`
  - `fail_on_warnings=True`

  to fail compilation on warning-level issues.
