# Selectors and expressions

## Selectors

Selectors are how you refer to site-local data symbolically.

`site(label)`  
reads from the source configuration `x`

`emitted(label)`  
reads from the emitted configuration `x'` after the update program has been applied for the current branch

`symbol(name)`  
creates a free named symbol that is not tied to an iterator label. You can also declare `default`, `doc`, and `dtype` metadata.

The selector object is [`nkdsl.dsl.selectors.SiteSelector`](../api/generated/nkdsl.SiteSelector.md#nkdsl.SiteSelector).

## Useful selector properties

`.value`  
the quantum number at the current site label

`.index`  
the integer site index bound by the iterator

Example:

```python
from nkdsl import emitted, site, symbol

amplitude = symbol("g") * site("i").value * emitted("j").value
predicate = (site("i").index != site("j").index) & (site("i") > 0)
```

## Symbol declarations

Free symbols can be declared with optional metadata:

```python
from nkdsl import symbol

J = symbol(
    "J",
    default=1.0,
    doc="Nearest-neighbor exchange coupling",
    dtype="float32",
)

amplitude = J * site("i").value * emitted("j").value
```

Declaration fields:

`default`  
fallback value used when no runtime binding for the symbol is provided

`doc`  
human-readable description for tooling/readability

`dtype`  
optional NumPy-compatible dtype declaration. If omitted and `default` is provided, dtype is inferred from the default value.

Practical behavior:

- `symbol("J")` is unresolved and can trigger `NKDSL-E001` at compile time.
- `symbol("J", default=1.0)` is considered resolved and does not trigger unresolved-symbol linting.
- if `dtype` is declared, runtime values are coerced to that dtype in the lowered evaluator.

## Amplitude expressions

Numeric expressions are represented by [`nkdsl.ir.AmplitudeExpr`](../api/generated/nkdsl.AmplitudeExpr.md#nkdsl.AmplitudeExpr). In user code they are usually built through normal Python operators.

```python
expr = 0.5 * site("i").value + 2 * site("j").value
```

Common helpers exposed through the IR and expression context include:

- `sqrt`
- `conj`
- `abs_`
- `pow`
- `wrap_mod`
- `source_index` and `target_index` for flat-index source/target access
- `static_index` and `static_emitted_index` remain available on `AmplitudeExpr`

## ExpressionContext

For callback-based construction, `nkDSL` exposes [`nkdsl.dsl.context.ExpressionContext`](../api/generated/nkdsl.ExpressionContext.md#nkdsl.ExpressionContext).

```python
def amplitude(ctx):
    i = ctx.site("i")
    return ctx.sqrt(i.value + 1)
```

The context API mirrors symbol declarations:

```python
def amplitude(ctx):
    J = ctx.symbol("J", default=1.0, dtype="float32")
    i = ctx.site("i")
    return J * i.value
```

This is mainly useful when a symbolic expression is easier to write as a small function than as one inline statement.

## Source values and target values

For a hopping branch, the source occupation and the occupation after the move
are different quantities. Suppose `hop("i", "j")` acts on occupations `(2, 0)`.
In its matrix element, `site("i").value` is two and
`emitted("i").value` is one. Both selectors use the same iterator-bound site
index. Only the state being read changes.

Use source selectors in validity conditions and update expressions. Emitted
selectors become available for the matrix element after the branch's update
has completed. They do not refer to a partially updated intermediate value.
Each emission starts from the original source, even when it follows another
emission in the same term.

## Compose expressions before evaluation

Expressions are symbolic trees rather than arrays of occupations. Python
arithmetic constructs those trees, and the compiler evaluates them for each
visit. Fluent helpers can make a matrix element easier to follow:

```python
ni, nj = site("i").value, site("j").value
boson_factor = (ni * (nj + 1)).sqrt()
complex_weight = (symbol("g", default=1.0) * boson_factor).conj()
```

The equivalent class helpers, such as `AmplitudeExpr.sqrt(expr)`, remain
available. Ordinary Python math functions are not substitutes for these
symbolic operations. For example, use `.sqrt()` rather than `math.sqrt` on a
selector expression.

For a term without bound labels, `source_index(0)` reads position zero directly
and `target_index(0)` reads that position in the completed target. These flat
indices are static parts of the definition. See
[global terms](iterators/globally.md) for a complete use case.
