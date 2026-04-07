``shift_mod(site_ref, delta)``
==============================

``shift_mod`` applies a Hilbert-aware wrapped shift.

This helper is intended for finite discrete local bases with contiguous,
unit-spaced integer ``local_states``. The builder records metadata so the lowerer
can reproduce the same wrap semantics at runtime.

Use it when a degree of freedom should wrap around instead of saturating or
leaving the basis.

Example:

.. code-block:: python

   plaquette = shift_mod("e0", +1).shift_mod("e1", -1)

Because wrapped semantics depend on the Hilbert local basis, ``shift_mod`` is more
specialized than plain ``shift``.
