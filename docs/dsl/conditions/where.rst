``where(predicate)``
====================

``where`` attaches a predicate to the current term.

If the predicate is false for a given iterator visit, that visit produces no
valid emitted branch.

Example:

.. code-block:: python

   constrained = (
       SymbolicDiscreteJaxOperator(hi, "constrained_raise")
       .for_each_site("i")
       .where(site("i") < 2)
       .emit(shift("i", +1), matrix_element=1.0)
       .build()
   )

Chaining ``where`` calls
------------------------

Multiple ``where`` calls on the same term are combined with logical AND.

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "hop")
       .for_each_pair("i", "j")
       .where(site("i").index != site("j").index)
       .where(site("i") > 0)
       .emit(shift("i", -1).shift("j", +1))
       .build()
   )
