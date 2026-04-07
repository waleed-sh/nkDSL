``identity()``
==============

``identity()`` returns a no-op update.

Use it for diagonal operators where the connected configuration should remain the
same and only the matrix element changes.

Example:

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "diag")
       .for_each_site("i")
       .emit(identity(), matrix_element=site("i").value)
       .build()
   )
