Logical composition
===================

Predicates can be combined with boolean operators.

Supported forms
---------------

* ``a & b`` for logical AND
* ``a | b`` for logical OR
* ``~a`` for logical NOT

Examples
--------

.. code-block:: python

   pred = (site("i") > 0) & (site("j") < 2)
   pred = (site("i") == site("j")) | (site("i") == 0)
   pred = ~(site("i") < 0)

For callback-style construction, :class:`nkdsl.dsl.context.ExpressionContext`
also exposes ``all_of``, ``any_of``, and ``not_`` helpers.
