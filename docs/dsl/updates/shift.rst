``shift(site_ref, delta)``
==========================

``shift`` adds ``delta`` to the selected site value.

It is the most common primitive for bosonic occupancies and any model where the
local basis is encoded by additive integer updates.

Example:

.. code-block:: python

   hopping_update = shift("i", -1).shift("j", +1)

Because updates are immutable and chainable, paired shifts are a natural way to
express a hopping move. For the common occupation-transfer case, prefer the
equivalent :doc:`hop` helper.
