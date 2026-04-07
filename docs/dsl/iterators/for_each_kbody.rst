Static K-body iterators
=======================

For graph-driven or geometry-driven operators, the most important iterator is the
fully general ``for_each(labels, over=...)``.

``for_each(labels, over=...)``
------------------------------

This binds an arbitrary sequence of labels to an explicit static list of index
rows.

.. code-block:: python

   edges = [(0, 1), (1, 2), (2, 3)]

   op = (
       SymbolicDiscreteJaxOperator(hi, "edge_term")
       .for_each(("src", "dst"), over=edges)
       .emit(identity(), matrix_element=site("src").value * site("dst").value)
       .build()
   )

This is usually the best choice whenever the operator follows a graph, plaquette
list, triangle list, or other precomputed structure.

Convenience wrappers
--------------------

The builder also provides wrappers for common fixed arities.

``for_each_triplet(label_a, label_b, label_c, over=...)``
   static list of three-tuples

``for_each_plaquette(label_a, label_b, label_c, label_d, over=...)``
   static list of four-tuples

These wrappers delegate to the same underlying machinery and mainly improve
readability.
