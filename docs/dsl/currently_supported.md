# Supported features

nkDSL compiles discrete operator definitions into NetKet JAX operators. This
page lists the supported Hilbert spaces, builder clauses, and expressions,
along with the conditions under which they can be used.

## NetKet integration

`build()` returns a `SymbolicOperator` containing the uncompiled definition.
`compile()` returns a `CompiledOperator`, a subclass of
[`netket.operator.DiscreteJaxOperator`](https://netket.readthedocs.io/en/latest/api/_generated/operator/netket.operator.DiscreteJaxOperator.html).
The compiled object exposes `get_conn_padded` and works with NetKet APIs that
accept discrete operators. The builder and symbolic definition do not execute
connections themselves.

## Supported Hilbert spaces

The builder expects a [`netket.hilbert.DiscreteHilbert`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.DiscreteHilbert.html).

Configurations must be flat arrays of local quantum numbers. This includes:

- [`netket.hilbert.Qubit`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.Qubit.html)
- [`netket.hilbert.Spin`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.Spin.html)
- [`netket.hilbert.Fock`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.Fock.html)
- [`netket.hilbert.SpinOrbitalFermions`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.SpinOrbitalFermions.html)
- [`netket.hilbert.TensorHilbert`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.TensorHilbert.html)
- [`netket.hilbert.DoubledHilbert`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.DoubledHilbert.html)
- constrained discrete spaces derived from [`netket.hilbert.DiscreteHilbert`](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.DiscreteHilbert.html)

The main requirement is that your rewrite rules produce valid configurations for that Hilbert space.

### Not supported

- continuous Hilbert spaces
- symbolic operators before compilation

### Indexability caveat

Compilation and `get_conn_padded` work with discrete Hilbert spaces in general. However, conversions such as dense or sparse matrix materialisation are subject to NetKet's usual `is_indexable` requirement.

### `shift_mod` caveat

The modular update helper `shift_mod(...)` and the expression helper `wrap_mod(...)` need more structure than a generic discrete space. At the moment they require:

- finite `hilbert.local_states`
- a one-dimensional local basis
- contiguous, unit-spaced integer local states

Typical examples that fit this contract are local bases such as `[0, 1]` or `[-1, 0, 1]`.

## Current DSL dictionary

### Builder clauses

These clauses are methods on
[`SymbolicDiscreteJaxOperator`](../api/generated/nkdsl.SymbolicDiscreteJaxOperator.md#nkdsl.SymbolicDiscreteJaxOperator).

`globally()`  
create one global term with no iterator labels

`for_each_site(label)`  
iterate over all sites

`for_each_pair(label_a, label_b)`  
iterate over all ordered pairs, including diagonal pairs

`for_each_distinct_pair(label_a, label_b)`  
iterate over all ordered pairs with `i != j`

`for_each_triplet(..., over=...)`  
iterate over a static list of ordered triplets

`for_each_plaquette(..., over=...)`  
iterate over a static list of ordered 4-tuples

`for_each(labels, over=...)`  
the general static K-body iterator over a user-provided list of index tuples

`where(predicate)`  
attach a branch predicate to the current term

`emit(update, matrix_element=..., tag=...)`  
add one emitted branch to the current term

`emit_if(predicate, update, matrix_element=..., tag=...)`  
start a conditional emission chain (if branch)

`emit_elseif(predicate, update, matrix_element=..., tag=...)`  
extend the current conditional chain with an elseif branch

`emit_else(update, matrix_element=..., tag=...)`  
close the current conditional chain with an else branch

`named(name)`  
give the current term a readable diagnostic name

`max_conn_size(hint)`  
provide a manual static upper bound for the current term's fanout

`fanout(hint)`  
backward-compatible alias for `max_conn_size`

`build()`  
freeze the fluent definition into a symbolic operator

`compile()`  
lower the operator to an executable JAX-backed NetKet operator

### Iterator model

All current iterators are static. Internally, they compile to fixed tuples of site-index tuples. There is no dynamic runtime iterator generation in the current implementation.

### Selectors

The current selector helpers are:

`site(label)`  
select a source-site binding from the current iterator environment

`emitted(label)`  
select the same site position after the branch update has been applied

`symbol(name, default=..., doc=..., dtype=...)`  
create a free symbolic parameter not bound to an iterator label, optionally with a fallback default value and dtype declaration

On `site(...)` and `emitted(...)`, the currently documented fields are:

`.value`  
the quantum number at that site

`.index`  
the integer site index bound by the iterator

`.abs()`  
a convenience constructor for `|value|`

### Predicates and conditions

Current predicate support includes:

- comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- boolean composition: `&`, `|`, `~`
- repeated `where(...)` calls, which compose by logical AND

### Matrix-element expressions

Current amplitude-expression support includes:

- numeric constants
- free symbols via `symbol(name)`
- declared symbols with defaults and metadata via `symbol(name, default=..., doc=..., dtype=...)`
- source-site selectors such as `site("i").value`
- emitted-state selectors such as `emitted("i").value`
- arithmetic: `+`, `-`, `*`, `/`, `**`
- unary negation
- `sqrt(...)`
- `conj(...)`
- absolute value via `abs_` or selector `.abs()`
- static source reads and emitted-state reads through the expression context
- Hilbert-aware modular wrapping via `wrap_mod(...)`

### Updates and rewrites

The currently implemented update primitives are:

`identity()`  
no-op update, useful for diagonal terms

`shift(site, delta)`  
set `x'[i] = x[i] + delta`

`hop(src, dst, amount=1)`  
transfer occupation from `src` to `dst`, equivalent to `shift(src, -amount).shift(dst, +amount)`

`shift_mod(site, delta)`  
Hilbert-aware wrapped shift using the local basis metadata

`write(site, value)`  
set `x'[i] = value`

`swap(site_a, site_b)`  
exchange two sites

`permute(site_a, site_b, ...)`  
cyclic permutation over two or more sites

`affine(site, scale=..., bias=...)`  
set `x'[i] = scale * x[i] + bias`

`scatter(flat_indices, values)`  
bulk writes to static flat indices

`Update.cond(predicate, if_true=..., if_false=...)`  
conditional update program lowered through `jax.lax.cond`

`Update.invalidate(reason=...)`  
mark a branch as invalid so it contributes zero amplitude

### Emissions

One term may contain more than one emission. This is the current multi-emission model:

- one iterator visit may produce several emitted branches
- each `emit(...)` call adds one branch
- conditional chains are supported through `emit_if/emit_elseif/emit_else`
- by default, duplicate emitted configurations are merged by target `x'` and their matrix elements are summed
- zero-amplitude components (including invalidated branches) are dropped before final padding
- this behavior is controlled by `deduplicate_connected_components` (default: `True`)

## Useful current limits to remember

- labels used in `site(...)` or `emitted(...)` must be bound by the current iterator
- global terms do not bind site labels
- iterators are static, not data-dependent
- `shift_mod` and `wrap_mod` have stronger Hilbert requirements than the rest of the DSL
