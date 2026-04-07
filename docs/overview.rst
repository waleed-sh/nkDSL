Overview
========

What nkDSL is for
-----------------

``nkDSL`` exists for the cases where writing a local matrix for a NetKet ``LocalOperator`` is too limiting and
writing a full custom JAX kernel for a custom ``DiscreteOperator`` is too low level.

It is most useful when:

* an operator has many related terms and you want one readable definition instead
  of many hand-written kernels
* the action follows graph or geometry data that you already have as static index
  sets
* you want to iterate quickly on the physics without rewriting masking and padding
  logic every time
* you want the final object to plug into the same NetKet operator interface used by
  matrix-free operators

**Not all operators should be written using symbolic operators.**

The mental model
----------------

A symbolic operator in ``nkDSL`` reads like a sentence:

.. code-block:: python

   SymbolicDiscreteJaxOperator(hilbert, "name", dtype="float64", hermitian=True)
     .for_each_*(...)
     .where(...)
     .emit(...)
     .compile()

The source code mirrors that split closely.

* :class:`nkdsl.dsl.operator.SymbolicDiscreteJaxOperator` owns term construction.
* iterator methods create :class:`nkdsl.ir.term.KBodyIteratorSpec` objects.
* predicates are stored as :class:`nkdsl.ir.predicates.PredicateExpr`.
* rewrite programs are stored as :class:`nkdsl.ir.update.UpdateProgram`.
* the compiler lowers the resulting :class:`nkdsl.ir.program.SymbolicOperatorIR`
  into a concrete :class:`nkdsl.core.compiled.CompiledOperator`.

What gets compiled
------------------

For one input configuration ``x``, one term behaves conceptually like this:

.. code-block:: text

   for visit in iterator.index_sets:
       if predicate(x, visit):
           for emission in emissions:
               x_prime = apply(update, x, visit)
               mel = evaluate(matrix_element, x, x_prime, visit)
               branches.append((x_prime, mel))

The real implementation is vectorised and lowered to JAX, but this is the right
way to think about the semantics.

Current scope
-------------

The extracted package already includes:

* the fluent builder API
* symbolic selectors and expression trees
* immutable update-program builders
* the symbolic IR
* a compiler with validation, normalization, analysis, fusion planning, caching,
  and JAX lowering
* compiled operators that expose ``get_conn_padded``

It is still an early package. That means the public surface is usable, while some
internals and naming may continue to evolve.
