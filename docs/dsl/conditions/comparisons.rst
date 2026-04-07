Comparison predicates
=====================

The predicate system supports the standard comparison family.

Available comparisons
---------------------

* ``==``
* ``!=``
* ``<``
* ``<=``
* ``>``
* ``>=``

These comparisons operate on symbolic amplitude expressions.

Examples
--------

.. code-block:: python

   site("i") > 0
   site("i").index != site("j").index
   site("i").value + site("j").value <= 2

Internally, these become :class:`nkdsl.ir.PredicateExpr` nodes such as
``gt``, ``ne``, or ``le``.
