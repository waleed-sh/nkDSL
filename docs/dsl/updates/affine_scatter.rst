``affine(...)`` and ``scatter(...)``
=====================================

``affine(site_ref, scale=..., bias=...)`` applies an affine transform to one site.

A common use is spin flipping on a ``\{-1, +1\}`` basis.

.. code-block:: python

   flip = affine("i", scale=-1, bias=0)

``scatter(indices, values)`` writes several flat positions in one update program.

Use it when a branch naturally modifies a fixed collection of sites at once.
