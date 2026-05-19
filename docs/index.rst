nkDSL
=====

``nkDSL`` is a small standalone package that extracts the symbolic operator DSL from
``neuraLQX`` and makes it usable on its own. The core workflow is to describe a discrete
operator in a compact, readable [D]eclarative [S]ymbolic [L]anguage (DSL) and compile it into a NetKet-compatible JAX operator.

.. note::

   ``nkDSL`` is currently in beta. The core DSL and compiler workflow are ready
   for broader testing, while some advanced APIs may still evolve before a fully
   stable release.

The package is built around a simple four-step mental model:

#. choose **where** to act with an iterator
#. choose **when** to act with a predicate
#. choose **how** to update the state
#. choose the **matrix element** attached to each emitted branch

That description is lowered into a static-shape JAX connectivity kernel
(default target: ``get_conn_padded``).

Compilation also runs a linting pass that reports symbol, index, and connectivity
diagnostics early. See :doc:`dsl/linting/index` for details.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   overview
   quickstart
   compiler

.. toctree::
   :maxdepth: 3
   :caption: The DSL

   dsl/index

.. toctree::
   :maxdepth: 3
   :caption: Guides

   guides/index


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   advanced_tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   changelog
