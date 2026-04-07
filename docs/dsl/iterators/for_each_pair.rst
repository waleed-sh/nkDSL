``for_each_pair(label_a, label_b)``
===================================

``for_each_pair`` iterates over all ordered pairs ``(i, j)`` in ``[0, N) x [0, N)``.

This includes diagonal pairs ``(i, i)``.

Use it when:

* you need a dense two-body sweep
* diagonal and off-diagonal pairs are both meaningful
* you prefer to remove illegal cases with ``where(...)`` instead of choosing a
  more restrictive iterator

Example:

.. code-block:: python

   hop = (
       SymbolicDiscreteJaxOperator(hi, "hop")
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
       .build()
   )
