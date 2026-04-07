``globally()``
==============

``globally()`` creates exactly one visit per input configuration.

Use it for:

* diagonal operators that only inspect the current configuration
* global quantities where sites are accessed through static indices or helper
  expressions

Example:

.. code-block:: python

   from nkdsl import SymbolicDiscreteJaxOperator, identity, site

   number = (
       SymbolicDiscreteJaxOperator(hi, "number")
       .globally()
       .emit(identity(), matrix_element=-1.2)
       .build()
   )

In practice, global terms are most useful together with static index expressions
or callback-based expression construction.
