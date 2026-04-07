``swap(...)`` and ``permute(...)``
===================================

``swap(site_a, site_b)`` exchanges two site values.

``permute(site_0, site_1, ..., site_k)`` performs a cyclic permutation over the
listed sites.

Examples
--------

.. code-block:: python

   swap("i", "j")
   permute("i", "j", "k")

These helpers are useful when the update is naturally expressed as a rearrangement
rather than as several independent writes.
