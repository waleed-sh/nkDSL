Extending DSL Iterators
=======================

This guide shows how to add custom iterator methods to the DSL in a way that
feels identical to built-in methods such as ``for_each_site`` or ``for_each_pair``.

Abstraction hierarchy
---------------------

Iterator clauses are built on one public base class:

* :class:`nkdsl.AbstractIteratorClause`

Your subclass is registered with:

* :func:`nkdsl.register_iterator_clause`
* or the generic decorator :func:`nkdsl.register`

After registration, the clause becomes a fluent method on
:class:`nkdsl.SymbolicDiscreteJaxOperator`.

What every iterator clause must provide
---------------------------------------

Minimum requirements:

1. Subclass :class:`nkdsl.AbstractIteratorClause`.
2. Implement ``build_iterator(self, hilbert, *args, **kwargs)``.
3. Return a valid iterator specification:

   * either :class:`nkdsl.KBodyIteratorSpec`
   * or ``(labels, index_sets)`` where labels are strings and index sets are tuples of integer tuples.

Optional but recommended:

* Set ``clause_name`` for a stable public method name.
* Validate user inputs early and raise clear ``ValueError`` messages.
* Keep iterator generation deterministic (important for reproducibility and tests).

Name resolution rules
---------------------

If you set ``clause_name``, that name is used for the fluent method.
Otherwise nkDSL derives a name from the class name.

Names must satisfy all of the following:

* valid Python identifier
* must not start with ``_``
* must not collide with reserved builder method names (for example ``build``)

Example: Even-site iterator
---------------------------

The clause below iterates only over even lattice sites.

.. code-block:: python

   import netket as nk
   import nkdsl


   class EvenSites(nkdsl.AbstractIteratorClause):
       clause_name = "for_each_even_site"

       def build_iterator(self, hilbert, label: str = "i"):
           n = int(hilbert.size)
           rows = tuple((k,) for k in range(n) if k % 2 == 0)
           if not rows:
               raise ValueError("No even sites available for this Hilbert space.")
           return (str(label),), rows


   nkdsl.register_iterator_clause(EvenSites, replace=True)

Usage:

.. code-block:: python

   hi = nk.hilbert.Fock(n_max=3, N=6)

   op = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "even-diagonal")
       .for_each_even_site("i")
       .emit(nkdsl.identity(), matrix_element=nkdsl.site("i").value)
       .build()
   )

This is the entire user-facing API surface. Once registered, users just call
``.for_each_even_site(...)`` like any built-in iterator.

Example: Graph-edge iterator
----------------------------

A common use case is iterating over a fixed edge list from a graph.

.. code-block:: python

   import netket as nk
   import nkdsl


   class ForEachEdge(nkdsl.AbstractIteratorClause):
       clause_name = "for_each_edge"

       def build_iterator(self, hilbert, label_a: str = "i", label_b: str = "j", *, edges):
           rows = tuple((int(i), int(j)) for i, j in edges)
           if not rows:
               raise ValueError("edges must contain at least one pair.")
           return (str(label_a), str(label_b)), rows


   nkdsl.register_iterator_clause(ForEachEdge, replace=True)

Usage:

.. code-block:: python

   edges = [(0, 1), (1, 2), (2, 3)]
   hi = nk.hilbert.Fock(n_max=2, N=4)

   hop = (
       nkdsl.SymbolicDiscreteJaxOperator(hi, "edge-hop")
       .for_each_edge("i", "j", edges=edges)
       .where(nkdsl.site("i") > 0)
       .emit(nkdsl.shift("i", -1).shift("j", +1), matrix_element=1.0)
       .build()
   )

Practical checklist before shipping a custom iterator
-----------------------------------------------------

* Does the clause return at least one index tuple?
* Do all index rows match label arity?
* Are indices in bounds for your Hilbert size?
* Is the method name stable and documented for your users?
* Did you add tests for registration, successful use, and invalid input paths?

Discoverability
---------------

You can inspect currently available iterator clause names at runtime:

.. code-block:: python

   names = nkdsl.available_iterator_clause_names()
   print(names)
