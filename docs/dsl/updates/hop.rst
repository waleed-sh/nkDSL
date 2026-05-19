``hop(src, dst, amount=1)``
=============================

``hop`` transfers occupation from one site to another.

It is shorthand for a paired shift:

.. code-block:: python

   hop("i", "j", amount=a)
   # equivalent to:
   shift("i", -a).shift("j", +a)

Use it for off-diagonal hopping terms where one quantum number is removed from
the source site and added to the destination site.

Example:

.. code-block:: python

   hopping = (
       SymbolicDiscreteJaxOperator(hi, "hopping")
       .for_each(("i", "j"), over=graph.edges())
       .where(site("i") > 0)
       .emit(hop("i", "j"), matrix_element=-t)
       .build()
   )

For larger transfer steps, pass ``amount``:

.. code-block:: python

   pair_hop = hop("i", "j", amount=2)
