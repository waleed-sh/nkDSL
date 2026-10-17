# API reference

Use the builder, selectors, and update helpers to describe an operator. Compile it
when you are ready to evaluate its connections or use it in NetKet. The pages
below document the public interfaces, including their arguments, return values,
and source code.

## Building operators

| Interface | Purpose |
| --- | --- |
| [`SymbolicDiscreteJaxOperator`](generated/nkdsl.SymbolicDiscreteJaxOperator.md) | Build terms with iterators, conditions, and emissions. |
| [`SymbolicOperator`](generated/nkdsl.SymbolicOperator.md) | Store a finished symbolic definition and compile it. |
| [`CompiledOperator`](generated/nkdsl.CompiledOperator.md) | Evaluate padded connections through NetKet's JAX operator interface. |
| [`SymbolicOperatorSum`](generated/nkdsl.SymbolicOperatorSum.md) | Compose symbolic contributions. |

## Selectors and expressions

Read the source with [`site`](generated/nkdsl.site.md) or
[`source_index`](generated/nkdsl.source_index.md), and the updated state with
[`emitted`](generated/nkdsl.emitted.md) or
[`target_index`](generated/nkdsl.target_index.md).
[`symbol`](generated/nkdsl.symbol.md) declares named parameters and defaults.
[`SiteSelector`](generated/nkdsl.SiteSelector.md),
[`AmplitudeExpr`](generated/nkdsl.AmplitudeExpr.md), and
[`PredicateExpr`](generated/nkdsl.PredicateExpr.md) describe the underlying
expression types. Callback-based definitions receive an
[`ExpressionContext`](generated/nkdsl.ExpressionContext.md).

## Updates

[`Update`](generated/nkdsl.Update.md) builds a sequence of state rewrites. Start
with one of the helper functions and chain further updates as needed:

- [`identity`](generated/nkdsl.identity.md): leave the configuration unchanged.
- [`write`](generated/nkdsl.write.md): replace a site value.
- [`shift`](generated/nkdsl.shift.md), [`hop`](generated/nkdsl.hop.md), and
  [`shift_mod`](generated/nkdsl.shift_mod.md): change occupations.
- [`swap`](generated/nkdsl.swap.md) and [`permute`](generated/nkdsl.permute.md):
  exchange site values.
- [`affine`](generated/nkdsl.affine.md) and [`scatter`](generated/nkdsl.scatter.md):
  apply affine maps or several static writes.

## Compiler and IR

[`SymbolicCompiler`](generated/nkdsl.SymbolicCompiler.md) accepts a symbolic
operator and returns a [`SymbolicCompiledArtifact`](generated/nkdsl.SymbolicCompiledArtifact.md).
Configure it through [`SymbolicCompilerOptions`](generated/nkdsl.SymbolicCompilerOptions.md).
The [compiler guide](../compiler.md) explains passes, caching, and lowering.

The IR consists of [`SymbolicOperatorIR`](generated/nkdsl.SymbolicOperatorIR.md),
[`SymbolicIRTerm`](generated/nkdsl.SymbolicIRTerm.md),
[`KBodyIteratorSpec`](generated/nkdsl.KBodyIteratorSpec.md),
[`EmissionSpec`](generated/nkdsl.EmissionSpec.md),
[`UpdateProgram`](generated/nkdsl.UpdateProgram.md), and
[`UpdateOp`](generated/nkdsl.UpdateOp.md).

## Extension and configuration interfaces

Custom clauses inherit from
[`AbstractIteratorClause`](generated/nkdsl.AbstractIteratorClause.md),
[`AbstractPredicateClause`](generated/nkdsl.AbstractPredicateClause.md), or
[`AbstractEmissionClause`](generated/nkdsl.AbstractEmissionClause.md). See the
[extension guides](../guides/extending_dsl/index.md) for registration and examples.

[`cfg`](generated/nkdsl.cfg.md) exposes package configuration. The
[`nkdsl.configs` module](generated/nkdsl.configs.md) documents configuration
sources, mutability, overrides, and runtime locking.

## Complete reference

The [API catalog](catalog.md) lists every public top-level export and source
module, including compiler passes, diagnostics, lowering registries, and errors.
