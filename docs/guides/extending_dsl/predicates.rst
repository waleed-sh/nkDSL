Extending DSL Predicates
========================

This guide shows how to add custom predicate methods to the DSL so users can write
domain-specific conditions as fluent builder calls.

Abstraction hierarchy
---------------------

Predicate clauses are built on one public base class:

* :class:`nkdsl.AbstractPredicateClause`

Your subclass is registered with:

* :func:`nkdsl.register_predicate_clause`
* or the generic decorator :func:`nkdsl.register`

After registration, the clause becomes a fluent method on
:class:`nkdsl.SymbolicDiscreteJaxOperator`.

What every predicate clause must provide
----------------------------------------

Minimum requirements:

1. Subclass :class:`nkdsl.AbstractPredicateClause`.
2. Implement ``build_predicate(self, ctx, *args, **kwargs)``.
3. Return a value coercible to :class:`nkdsl.PredicateExpr`.

About ``ctx``:

* ``ctx`` is an :class:`nkdsl.ExpressionContext`.
* Use it to build selector-based expressions in a consistent way.
* Example: ``ctx.site("i").value >= 1``.

Optional but recommended:

* Set ``clause_name`` for a stable public method name.
* Validate user arguments and fail fast with clear errors.
* Keep logic focused: one clause should represent one reusable predicate idea.

Name resolution rules
---------------------

If you set ``clause_name``, that name is used for the fluent method.
Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

* valid Python identifier
* must not start with ``_``
* must not collide with reserved builder method names

Example: Occupancy threshold predicate
--------------------------------------

This clause checks whether one site value is above a threshold.

.. code-block:: python

   import netket as nk
   import nkdsl


   class AtLeast(nkdsl.AbstractPredicateClause):
       clause_name = "at_least"

       def build_predicate(self, ctx, label: str = "i", cutoff: int = 0):
           return ctx.site(label).value >= int(cutoff)


   nkdsl.register_predicate_clause(AtLeast, replace=True)

Usage:

.. code-block:: python

   hi = nk.hilbert.Fock(n_max=3, N=4)

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "filtered")
       .for_each_site("i")
       .at_least("i", cutoff=1)
       .emit(nkdsl.identity(), matrix_element=1.0)
       .build()
   )

Example: Band predicate (inclusive range)
-----------------------------------------

This clause keeps visits with values in ``[lower, upper]``.

.. code-block:: python

   import nkdsl


   class ValueBand(nkdsl.AbstractPredicateClause):
       clause_name = "value_in_band"

       def build_predicate(self, ctx, label: str = "i", *, lower: int, upper: int):
           if int(lower) > int(upper):
               raise ValueError("lower must be <= upper.")
           site_value = ctx.site(label).value
           return (site_value >= int(lower)) & (site_value <= int(upper))


   nkdsl.register_predicate_clause(ValueBand, replace=True)

Usage:

.. code-block:: python

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "banded")
       .for_each_site("i")
       .value_in_band("i", lower=1, upper=2)
       .emit(nkdsl.identity(), matrix_element=1.0)
       .build()
   )

Composition semantics with ``where``
------------------------------------

Predicate clauses compose with existing conditions using logical AND.
That means this:

.. code-block:: python

   .at_least("i", cutoff=1).where(nkdsl.site("i") < 3)

is equivalent to:

.. code-block:: python

   .where((nkdsl.site("i") >= 1) & (nkdsl.site("i") < 3))

Practical checklist before shipping a custom predicate
------------------------------------------------------

* Are arguments validated with clear error messages?
* Is the return value always coercible to ``PredicateExpr``?
* Is the method name stable and documented?
* Did you test successful composition with ``where(...)``?
* Did you test failure paths (bad arguments and invalid clause names)?

Discoverability
---------------

You can inspect currently available predicate clause names at runtime:

.. code-block:: python

   names = nkdsl.available_predicate_clause_names()
   print(names)
