``for_each_site(label)``
========================

``for_each_site`` iterates over all site indices ``0`` through ``hilbert.size - 1``.

The bound label gives access to:

* ``site(label).value``
* ``site(label).index``

Typical use cases:

* single-site raising or lowering operators
* on-site diagonal energies
* local flips or local constraints

Example:

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "raise")
       .for_each_site("i")
       .where(site("i") < 2)
       .emit(shift("i", +1), matrix_element=1.0)
       .build()
   )
