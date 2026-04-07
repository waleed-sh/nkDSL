The Symbolic Operator
=====================

The main entry point is :class:`nkdsl.SymbolicDiscreteJaxOperator`, also exported
as :class:`nkdsl.SymbolicDiscreteJaxOperator`.

A builder accumulates **terms**. Each term has:

* an iterator
* an optional predicate
* one or more emissions

A minimal shape looks like this:

.. code-block:: python

   from nkdsl import SymbolicDiscreteJaxOperator, shift, site

   op = (
       SymbolicDiscreteJaxOperator(hi, "raise_one")
       .for_each_site("i")
       .where(site("i") < 2)
       .emit(shift("i", +1), matrix_element=1.0)
       .build()
   )

Important builder rules
-----------------------

Iterator methods open terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every call to ``globally`` or ``for_each*`` seals the current term and starts a
new one. That means ``where`` and ``emit`` always attach to the most recently
opened term.

``build()`` returns a symbolic object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build()`` returns :class:`nkdsl.core.operator.SymbolicOperator`. This object
holds typed IR terms and can later be compiled.

``compile()`` is a convenience shortcut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling ``compile()`` on the builder is equivalent to ``build().compile()``.

Term metadata
-------------

Two builder helpers matter when definitions become larger.

``named(name)``
   gives the current term a readable identifier for IR dumps and compiler output

``max_conn_size(hint)``
   sets a manual static upper bound for the term fanout when you know a tighter
   bound than the generic iterator-size times emission-count estimate

Hermiticity and dtype
---------------------

The builder constructor accepts ``dtype`` and ``hermitian``. These become part of
both the symbolic IR and the compiled artifact metadata.

.. warning::

    You are responsible to ensure that the hermiticity of the operator is correct!
    If you set ``hermitian=True`` for a non-Hermitian operator, gradients will be
    incorrectly computed.
