Extending DSL Emissions
=======================

This guide shows how to add custom emission methods to the DSL so you can
encapsulate reusable branch-emission behavior as fluent builder calls.

Abstraction hierarchy
---------------------

Emission clauses are built on one public base class:

* :class:`nkdsl.AbstractEmissionClause`

Your subclass is registered with:

* :func:`nkdsl.register_emission_clause`
* or the generic decorator :func:`nkdsl.register`

After registration, the clause becomes a fluent method on
:class:`nkdsl.SymbolicDiscreteJaxOperator`.

What every emission clause must provide
---------------------------------------

Minimum requirements:

1. Subclass :class:`nkdsl.AbstractEmissionClause`.
2. Implement ``build_emission(self, ctx, *args, **kwargs)``.
3. Return :class:`nkdsl.dsl.emissions.types.EmissionClauseSpec`.

``EmissionClauseSpec`` supports these modes:

* ``"emit"``
* ``"emit_if"``
* ``"emit_elseif"``
* ``"emit_else"``

Optional but recommended:

* Set ``clause_name`` for a stable public method name.
* Validate arguments and fail fast with clear ``ValueError`` messages.
* Add branch tags for diagnostics clarity.

Name resolution rules
---------------------

If you set ``clause_name``, that name is used for the fluent method.
Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

* valid Python identifier
* must not start with ``_``
* must not collide with reserved builder method names

Example: conditional emission convenience clause
------------------------------------------------

The clause below emits a branch only when one site value is above a cutoff.

.. code-block:: python

   import netket as nk
   import nkdsl
   from nkdsl.dsl.emissions.types import EmissionClauseSpec


   class EmitWhenAtLeast(nkdsl.AbstractEmissionClause):
       clause_name = "emit_when_at_least"

       def build_emission(self, ctx, label: str = "i", cutoff: int = 1):
           predicate = ctx.site(label).value >= int(cutoff)
           return EmissionClauseSpec(
               mode="emit_if",
               predicate=predicate,
               update=nkdsl.identity(),
               matrix_element=ctx.site(label).value,
               tag="emit-when-at-least",
           )


   nkdsl.register_emission_clause(EmitWhenAtLeast, replace=True)

Usage:

.. code-block:: python

   hi = nk.hilbert.Fock(n_max=3, N=4)

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "custom-emission")
       .for_each_site("i")
       .emit_when_at_least("i", cutoff=2)
       .emit_else(nkdsl.identity(), matrix_element=0.0, tag="fallback")
       .build()
   )

Practical checklist before shipping a custom emission clause
------------------------------------------------------------

* Does the clause return a valid ``EmissionClauseSpec``?
* Are conditional modes used in legal sequence (``if -> elseif* -> else?``)?
* Are branch tags set when diagnostics readability matters?
* Did you add tests for valid and invalid usage paths?

Discoverability
---------------

You can inspect currently available emission clause names at runtime:

.. code-block:: python

   names = nkdsl.available_emission_clause_names()
   print(names)
