``for_each_distinct_pair(label_a, label_b)``
============================================

``for_each_distinct_pair`` is the dense pair iterator without diagonal visits.

It iterates over all ordered pairs with ``i != j``.

Use it when the diagonal is never meaningful and you want that fact encoded in
the iterator itself rather than in an extra predicate.

Example:

.. code-block:: python

   hop = (
       SymbolicDiscreteJaxOperator(hi, "hop")
       .for_each_distinct_pair("i", "j")
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
       .build()
   )
