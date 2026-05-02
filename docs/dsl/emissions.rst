Emissions and matrix elements
=============================

Each call to ``emit(...)`` adds one branch to the current term.

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "two_branch")
       .for_each_site("i")
       .where(site("i").abs() < 2)
       .emit(shift("i", +1), matrix_element=+0.5, tag="raise")
       .emit(shift("i", -1), matrix_element=-0.5, tag="lower")
       .build()
   )

An emission contains:

* an update program that maps ``x`` to ``x'``
* a matrix element expression
* an optional diagnostic branch tag

Multi-emission semantics
------------------------

Multiple emissions share the same iterator visit and predicate evaluation. This
is often cleaner and cheaper than repeating the iterator block.

Conditional emission chains
---------------------------

nkDSL also supports ``if / elseif / else`` style branching at the emission level.

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "piecewise")
       .for_each_site("i")
       .emit_if(site("i") == 0, write("i", 1), matrix_element=1.0, tag="if")
       .emit_elseif(site("i") == 1, write("i", 2), matrix_element=2.0, tag="elseif")
       .emit_else(write("i", 3), matrix_element=3.0, tag="else")
       .build()
   )

Chaining rules:

* ``emit_if(...)`` starts a new conditional chain.
* ``emit_elseif(...)`` extends the current chain.
* ``emit_else(...)`` closes the current chain.
* Calling plain ``emit(...)`` closes any currently open chain.

Active branches contribute candidate connected states. Before final padding,
duplicate ``x'`` targets are coalesced (matrix elements summed) and
zero-amplitude components are removed.

Connected-state dedup semantics
-------------------------------

Connected states are deduplicated by default. If two branches emit the same
``x'``, they are merged into one row and their matrix elements are summed.
Entries with final matrix element ``0`` are dropped before padding.

Set ``deduplicate_connected_components=False`` at compile time if you need raw
per-branch connectivity output.

Source and emitted state access
-------------------------------

Matrix elements can depend on both the source configuration and the emitted
configuration.

* ``site("i")`` reads from ``x``
* ``emitted("i")`` reads from ``x'``

This is especially useful when amplitudes depend on the value after a rewrite.

Complex matrix elements and dtype promotion
-------------------------------------------

Complex constants in ``matrix_element=...`` expressions are supported directly.
When a complex constant is present, the operator dtype is automatically promoted
to a compatible complex dtype during ``build()``.

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "complex_branch")
       .for_each_site("i")
       .emit(shift("i", +1), matrix_element=1.0 + 0.5j)
       .build()
   )

   # op.dtype is complex-valued after promotion

Extending emission clauses
--------------------------

Emission methods are now first-class clause extension points, similar to
iterators and predicates. Advanced users can register custom emission clauses
by subclassing :class:`nkdsl.AbstractEmissionClause`.

For a full extension walkthrough, see :doc:`../guides/extending_dsl/emissions`.
