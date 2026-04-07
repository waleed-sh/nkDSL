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

Branch multiset semantics
-------------------------

Connected states are not automatically deduplicated. If two branches emit the
same ``x'``, both rows remain present in the padded output. Any later reduction
that wants merged matrix elements must do so explicitly.

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
