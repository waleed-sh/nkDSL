Selectors and expressions
=========================

Selectors
---------

Selectors are how you refer to site-local data symbolically.

``site(label)``
   reads from the source configuration ``x``

``emitted(label)``
   reads from the emitted configuration ``x'`` after the update program has been
   applied for the current branch

``symbol(name)``
   creates a free named symbol that is not tied to an iterator label

The selector object is :class:`nkdsl.dsl.selectors.SiteSelector`.

Useful selector properties
--------------------------

``.value``
   the quantum number at the current site label

``.index``
   the integer site index bound by the iterator

Example:

.. code-block:: python

   from nkdsl import emitted, site, symbol

   amplitude = symbol("g") * site("i").value * emitted("j").value
   predicate = (site("i").index != site("j").index) & (site("i") > 0)

Amplitude expressions
---------------------

Numeric expressions are represented by :class:`nkdsl.ir.AmplitudeExpr`. In user
code they are usually built through normal Python operators.

.. code-block:: python

   expr = 0.5 * site("i").value + 2 * site("j").value

Common helpers exposed through the IR and expression context include:

* ``sqrt``
* ``conj``
* ``abs_``
* ``pow``
* ``wrap_mod``
* ``static_index`` and ``static_emitted_index`` for flat-index access

ExpressionContext
-----------------

For callback-based construction, ``nkDSL`` exposes
:class:`nkdsl.dsl.context.ExpressionContext`.

.. code-block:: python

   def amplitude(ctx):
       i = ctx.site("i")
       return ctx.sqrt(i.value + 1)

This is mainly useful when a symbolic expression is easier to write as a small
function than as one inline statement.
