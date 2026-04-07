The DSL
=======

The DSL is the user-facing layer of ``nkdsl``. It is designed to keep operator
construction declarative and readable.

The core pieces are:

* :class:`nkdsl.SymbolicDiscreteJaxOperator` or the alias :class:`nkdsl.DOperator`
* selectors such as :func:`nkdsl.site`, :func:`nkdsl.emitted`, and
  :func:`nkdsl.symbol`
* predicates built with :class:`nkdsl.ir.PredicateExpr`
* update programs built with :class:`nkdsl.Update` and helper functions such as
  :func:`nkdsl.shift` or :func:`nkdsl.write`
* emissions that attach a rewrite rule and matrix element to each active visit

.. toctree::
   :maxdepth: 2

   symbolic_operator
   selectors_and_expressions
   emissions
   currently_supported
   iterators/index
   conditions/index
   updates/index
