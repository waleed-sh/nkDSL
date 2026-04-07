Quick start
===========

Installation
------------

``nkDSL`` follows the same simple installation style commonly used in the NetKet
ecosystem. It is available to install through ``pip`` as

.. code-block:: bash

   pip install nkDSL

For editable development installs:

.. code-block:: bash

   git clone https://www.github.com/waleed-sh/nkDSL
   cd nkDSL
   pip install -e .

Example
--------

Below is a compact but complete example showing how to build a transverse-field
Ising Hamiltonian with ``nkDSL``.

The model is

.. math::

   H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i

In the DSL:

* the diagonal ``\sigma^z_i \sigma^z_j`` term is an identity update with a
  bond-dependent matrix element
* the transverse field ``\sigma^x_i`` term flips one spin at a time

.. code-block:: python

   import netket as nk
   from nkdsl import SymbolicDiscreteJaxOperator, affine, identity, site

   L = 6
   J = 1.0
   h = 0.7

   hi = nk.hilbert.Spin(s=1 / 2, N=L)
   graph = nk.graph.Chain(length=L)

   zz = (
       SymbolicDiscreteJaxOperator(hi, "ising_zz", dtype="float64", hermitian=True)
       .for_each(("i", "j"), over=graph.edges())
       .emit(
           identity(),  # can also just omit it in case of identity
           matrix_element=-J * site("i").value * site("j").value,
       )
       .build()
   )

   x_field = (
       SymbolicDiscreteJaxOperator(hi, "ising_x", dtype="float64", hermitian=True)
       .for_each_site("i")
       .emit(
           affine("i", scale=-1, bias=0),
           matrix_element=-h,
       )
       .build()
   )

   # Operator algebra stays compatible with the NetKet operator interface.
   H = zz + x_field

What this example shows
------------------------

The Ising example is a good first template because it exercises the three main
pieces of the DSL without becoming noisy.

* ``for_each(("i", "j"), over=graph.edges())`` ties the diagonal term directly to the
  bond list instead of sweeping over all site pairs.
* ``identity()`` makes the ``zz`` contribution diagonal.
* ``affine("i", scale=-1, bias=0)`` implements a spin flip on a ``\{-1, +1\}``
  local basis.
* the matrix element is just a normal symbolic expression built from
  ``site("i").value`` and Python scalars.

Typical next step
------------------

Once an operator is built, the usual next step is compilation:

.. code-block:: python

   compiled_zz = zz.compile()
   xp, mels = compiled_zz.get_conn_padded(x_batch)

For larger Hamiltonians you will usually define several symbolic pieces and then
compose them with operator algebra.

