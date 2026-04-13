The DSL
=======

The DSL is the user-facing layer of ``nkdsl``. It is designed to keep operator
construction declarative and readable.

The core pieces are:

* :class:`nkdsl.SymbolicDiscreteJaxOperator` or the alias :class:`nkdsl.DOperator`
* selectors such as :func:`nkdsl.site`, :func:`nkdsl.emitted`, and
  :func:`nkdsl.symbol` (including ``default/doc/dtype`` declarations)
* predicates built with :class:`nkdsl.ir.PredicateExpr`
* update programs built with :class:`nkdsl.Update` and helper functions such as
  :func:`nkdsl.shift` or :func:`nkdsl.write`
* emissions that attach a rewrite rule and matrix element to each active visit
* compiler-integrated linting that reports actionable diagnostics at compile time

For custom fluent extensions, see :doc:`../guides/extending_dsl/index`.

Linting
-------

The DSL includes a compiler-integrated linting pass that runs during
``.compile()``. It checks unresolved symbols, index-safety issues, and sampled
connectivity validity (support, constraints, and local basis values).

Start with :doc:`linting/index` for the overall workflow and policy options.
Use :doc:`linting/messages` for the full diagnostic message catalog with
examples and resolutions.

.. toctree::
   :maxdepth: 2

   symbolic_operator
   selectors_and_expressions
   emissions
   linting/index
   currently_supported
   iterators/index
   conditions/index
   updates/index
