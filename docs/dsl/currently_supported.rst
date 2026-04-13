Currently Supported
===================

This page summarises the current support surface of ``nkdsl``.
It has two goals:

* clarify what kind of NetKet operator ``nkdsl`` produces
* give a compact dictionary of the DSL features that are implemented today

NetKet integration
------------------

``nkdsl`` produces two layers of objects.

``SymbolicOperator``
   the symbolic, uncompiled object returned by ``build()``

``CompiledOperator``
   the executable object returned by ``compile()``

The executable form is the important one for NetKet integration. It is a
subclass of :class:`netket.operator.DiscreteJaxOperator`. In practice, this is
what makes an ``nkdsl`` operator behave like a NetKet matrix-free discrete JAX
operator.

That means the intended integration target is:

* :class:`netket.operator.DiscreteJaxOperator`
* and, by inheritance, APIs that work with discrete NetKet operators more generally

Practical note
~~~~~~~~~~~~~~

The symbolic builder itself is not executable. Only the compiled form exposes
``get_conn_padded`` and plugs directly into NetKet execution paths.

Supported Hilbert spaces
------------------------

The builder expects a :class:`netket.hilbert.DiscreteHilbert`.

In practice, ``nkdsl`` is currently intended for any discrete NetKet Hilbert
space whose configurations are represented as a flat array of local quantum
numbers. Typical examples include:

* :class:`netket.hilbert.Qubit`
* :class:`netket.hilbert.Spin`
* :class:`netket.hilbert.Fock`
* :class:`netket.hilbert.SpinOrbitalFermions`
* :class:`netket.hilbert.TensorHilbert`
* :class:`netket.hilbert.DoubledHilbert`
* constrained discrete spaces derived from :class:`netket.hilbert.DiscreteHilbert`

The main requirement is that your rewrite rules produce valid configurations for
that Hilbert space.

Not supported
~~~~~~~~~~~~~

* continuous Hilbert spaces
* symbolic operators before compilation

Indexability caveat
~~~~~~~~~~~~~~~~~~~

Compilation and ``get_conn_padded`` work with discrete Hilbert spaces in
general. However, conversions such as dense or sparse matrix materialisation are
subject to NetKet's usual ``is_indexable`` requirement.

``shift_mod`` caveat
~~~~~~~~~~~~~~~~~~~~

The modular update helper ``shift_mod(...)`` and the expression helper
``wrap_mod(...)`` need more structure than a generic discrete space. At the
moment they require:

* finite ``hilbert.local_states``
* a one-dimensional local basis
* contiguous, unit-spaced integer local states

Typical examples that fit this contract are local bases such as ``[0, 1]`` or
``[-1, 0, 1]``.

Current DSL dictionary
----------------------

Builder clauses
~~~~~~~~~~~~~~~

These are the main fluent clauses on :class:`nkdsl.DOperator` or
:class:`nkdsl.SymbolicDiscreteJaxOperator`.

``globally()``
   create one global term with no iterator labels

``for_each_site(label)``
   iterate over all sites

``for_each_pair(label_a, label_b)``
   iterate over all ordered pairs, including diagonal pairs

``for_each_distinct_pair(label_a, label_b)``
   iterate over all ordered pairs with ``i != j``

``for_each_triplet(..., over=...)``
   iterate over a static list of ordered triplets

``for_each_plaquette(..., over=...)``
   iterate over a static list of ordered 4-tuples

``for_each(labels, over=...)``
   the general static K-body iterator over a user-provided list of index tuples

``where(predicate)``
   attach a branch predicate to the current term

``emit(update, matrix_element=..., tag=...)``
   add one emitted branch to the current term

``named(name)``
   give the current term a readable diagnostic name

``max_conn_size(hint)``
   provide a manual static upper bound for the current term's fanout

``fanout(hint)``
   backward-compatible alias for ``max_conn_size``

``build()``
   freeze the fluent definition into a symbolic operator

``compile()``
   lower the operator to an executable JAX-backed NetKet operator

Iterator model
~~~~~~~~~~~~~~

All current iterators are static. Internally, they compile to fixed tuples of
site-index tuples. There is no dynamic runtime iterator generation in the
current implementation.

Selectors
~~~~~~~~~

The current selector helpers are:

``site(label)``
   select a source-site binding from the current iterator environment

``emitted(label)``
   select the same site position after the branch update has been applied

``symbol(name, default=..., doc=..., dtype=...)``
   create a free symbolic parameter not bound to an iterator label, optionally
   with a fallback default value and dtype declaration

On ``site(...)`` and ``emitted(...)``, the currently documented fields are:

``.value``
   the quantum number at that site

``.index``
   the integer site index bound by the iterator

``.abs()``
   a convenience constructor for ``|value|``

Predicates and conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

Current predicate support includes:

* comparisons: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
* boolean composition: ``&``, ``|``, ``~``
* repeated ``where(...)`` calls, which compose by logical AND

Matrix-element expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Current amplitude-expression support includes:

* numeric constants
* free symbols via ``symbol(name)``
* declared symbols with defaults and metadata via
  ``symbol(name, default=..., doc=..., dtype=...)``
* source-site selectors such as ``site("i").value``
* emitted-state selectors such as ``emitted("i").value``
* arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
* unary negation
* ``sqrt(...)``
* ``conj(...)``
* absolute value via ``abs_`` or selector ``.abs()``
* static source reads and emitted-state reads through the expression context
* Hilbert-aware modular wrapping via ``wrap_mod(...)``

Updates and rewrites
~~~~~~~~~~~~~~~~~~~~

The currently implemented update primitives are:

``identity()``
   no-op update, useful for diagonal terms

``shift(site, delta)``
   set ``x'[i] = x[i] + delta``

``shift_mod(site, delta)``
   Hilbert-aware wrapped shift using the local basis metadata

``write(site, value)``
   set ``x'[i] = value``

``swap(site_a, site_b)``
   exchange two sites

``permute(site_a, site_b, ...)``
   cyclic permutation over two or more sites

``affine(site, scale=..., bias=...)``
   set ``x'[i] = scale * x[i] + bias``

``scatter(flat_indices, values)``
   bulk writes to static flat indices

``Update.cond(predicate, if_true=..., if_false=...)``
   conditional update program lowered through ``jax.lax.cond``

``Update.invalidate(reason=...)``
   mark a branch as invalid so it contributes zero amplitude

Emissions
~~~~~~~~~

One term may contain more than one emission. This is the current multi-emission
model:

* one iterator visit may produce several emitted branches
* each ``emit(...)`` call adds one branch
* duplicate emitted configurations are not automatically merged

In other words, the connected output is currently a branch multiset, not a
canonical deduplicated adjacency row.

Useful current limits to remember
---------------------------------

* labels used in ``site(...)`` or ``emitted(...)`` must be bound by the current iterator
* global terms do not bind site labels
* iterators are static, not data-dependent
* ``shift_mod`` and ``wrap_mod`` have stronger Hilbert requirements than the rest of the DSL
