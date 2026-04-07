``shift(site_ref, delta)``
==========================

``shift`` adds ``delta`` to the selected site value.

It is the most common primitive for bosonic occupancies and any model where the
local basis is encoded by additive integer updates.

Example:

.. code-block:: python

   hop = shift("i", -1).shift("j", +1)

Because updates are immutable and chainable, this is the natural way to express a
hopping move.
