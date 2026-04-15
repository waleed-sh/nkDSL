Extending the DSL
=================

nkDSL now exposes a first-class clause abstraction layer for extending the fluent API.
In practice, this means you can add your own iterator and predicate methods and call
them exactly like built-ins:

.. code-block:: python

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "my-op")
       .my_custom_iterator(...)
       .my_custom_predicate(...)
       .emit(...)
       .build()
   )

The extension points are separated by concern:

* **Iterator clauses** decide where the term iterates.
* **Predicate clauses** decide when a visit is active.
* **Emission clauses** decide how branches are emitted (including conditional chains).

Use the pages below as implementation guides:

.. toctree::
   :maxdepth: 2

   iterators
   predicates
   emissions
