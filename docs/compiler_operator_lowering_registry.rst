Operator lowering registry
==========================

Why this exists
---------------

Different operator families expose different connectivity entry points.
For example:

* NetKet ``DiscreteJaxOperator`` uses ``get_conn_padded``.
* Other operator styles, such as neuraLQX's ``ComputationalJaxOperator`` may expose ``_get_conn_padded``.

The operator-lowering registry lets the compiler target either style without
hard-coding one operator base class or one method name in the lowering code.

Concept
-------

A lowering target is a small mapping:

* ``name``: user-facing key (for example ``"netket_discrete_jax"``)
* ``operator_type``: class to instantiate for compiled operators
* ``connection_method``: method that should execute the generated kernel

The default target is:

* ``"netket_discrete_jax"`` -> ``DiscreteJaxOperator.get_conn_padded``

Public API
----------

The main entry points are:

* :class:`nkdsl.compiler.SymbolicOperatorLoweringRegistry`
* :class:`nkdsl.compiler.SymbolicOperatorLoweringTarget`
* :data:`nkdsl.compiler.DEFAULT_SYMBOLIC_OPERATOR_LOWERING`
* :func:`nkdsl.compiler.default_symbolic_operator_lowering_registry`

Register a custom target
------------------------

.. code-block:: python

   import nkdsl

   class ComputationalLikeOperator:
       def __init__(self, hilbert):
           self.hilbert = hilbert

       def get_conn_padded(self, x):
           return self._get_conn_padded(x)

   registry = nkdsl.default_symbolic_operator_lowering_registry()
   registry.register(
       name="computational_like",
       operator_type=ComputationalLikeOperator,
       connection_method="_get_conn_padded",
   )

Select a target during compilation
----------------------------------

You can choose a target from any compile entry point.

From a symbolic operator:

.. code-block:: python

   compiled = symbolic_op.compile(
       operator_lowering="computational_like",
       cache=False,
   )

From the DSL builder directly:

.. code-block:: python

   compiled = (
       SymbolicDiscreteJaxOperator(hi, "hop")
       .for_each_site("i")
       .emit(...)
       .compile(operator_lowering="computational_like")
   )

From explicit compiler options:

.. code-block:: python

   import nkdsl

   compiler = nkdsl.SymbolicCompiler(
       options=nkdsl.SymbolicCompilerOptions(
           backend_preference="jax",
           operator_lowering="computational_like",
           cache_enabled=True,
       )
   )
   compiled = compiler.compile_operator(symbolic_op)

Notes
-----

* ``connection_method`` must be a valid Python identifier.
* Target names are included in compiler signatures, so cache keys stay
  consistent across different lowering targets.
