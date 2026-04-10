Changelog
=========

This page follows a release-notes style inspired by JAX and PyTorch:
one top-level section per release, with consistent change categories.

Unreleased
-----------------------

New features
~~~~~~~~~~~~

* Added :class:`nkdsl.compiler.SymbolicOperatorLoweringRegistry` and
  :class:`nkdsl.compiler.SymbolicOperatorLoweringTarget`.

* Added :data:`nkdsl.compiler.DEFAULT_SYMBOLIC_OPERATOR_LOWERING` and
  :func:`nkdsl.compiler.default_symbolic_operator_lowering_registry`.

* Added ``operator_lowering`` selection in compiler options and convenience
  compile entry points.

* Added support for custom connection methods such as ``_get_conn_padded`` for
  computational-style operator subclasses.

Improvements
~~~~~~~~~~~~

* Preserved default runtime behavior:
  ``"netket_discrete_jax"`` -> ``DiscreteJaxOperator.get_conn_padded``.

* Included operator-lowering target identity in compilation signatures so cache
  keys remain consistent across different lowering targets.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Added :doc:`compiler_operator_lowering_registry` with rationale, API surface,
  and usage examples.
