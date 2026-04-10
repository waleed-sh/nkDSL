Compiler and execution
======================

.. toctree::
   :maxdepth: 1

   compiler_operator_lowering_registry

From DSL to executable operator
-------------------------------

The DSL builder does not execute operators directly. It first produces a
:class:`nkdsl.core.operator.SymbolicOperator`, which can then be lowered by the
compiler.

The default flow is:

#. build symbolic terms into :class:`nkdsl.ir.program.SymbolicOperatorIR`
#. validate symbol scope and update-program consistency
#. normalize and fingerprint the IR
#. look up a cached compiled artifact
#. on a cache miss, run analysis and fusion planning
#. lower to a concrete executable operator target (default:
   :class:`nkdsl.core.compiled.CompiledOperator`)

This structure is visible directly in the source tree under ``nkdsl.compiler``.

Default passes
--------------

The default pipeline created by
:func:`nkdsl.compiler.defaults.default_symbolic_pass_pipeline` contains four
passes.

Pre-cache passes
~~~~~~~~~~~~~~~~

* :class:`nkdsl.compiler.passes.validation.SymbolicValidationPass`
* :class:`nkdsl.compiler.passes.normalization.SymbolicNormalizationPass`

Post-cache passes
~~~~~~~~~~~~~~~~~

* :class:`nkdsl.compiler.passes.analysis.SymbolicMaxConnSizeAnalysisPass`
* :class:`nkdsl.compiler.passes.fusion.SymbolicFusionPass`

Caching
-------

The compiler can cache compiled artifacts in an in-memory store. The relevant
public pieces are:

* :class:`nkdsl.compiler.SymbolicCompiler`
* :class:`nkdsl.compiler.SymbolicCompilerOptions`
* :class:`nkdsl.compiler.SymbolicCompilationSignature`
* :class:`nkdsl.compiler.SymbolicCacheKey`
* :func:`nkdsl.compiler.defaults.default_symbolic_artifact_store`

Direct compiler usage
---------------------

.. code-block:: python

   from nkdsl import SymbolicCompiler

   compiler = SymbolicCompiler()
   artifact = compiler.compile(symbolic_operator)
   compiled = artifact.operator

Convenience compilation from the builder or symbolic operator is also supported:

.. code-block:: python

   compiled = SymbolicDiscreteJaxOperator(hi, "hop").for_each_site("i").emit(...).compile()

Compiled operators
------------------

Compiled objects are normal executable operators. The default target exposes
``get_conn_padded`` and is represented by
:class:`nkdsl.core.compiled.CompiledOperator`. Custom lowering targets can
bind the generated kernel to a different method name (for example
``_get_conn_padded``).

.. code-block:: python

   xp, mels = compiled.get_conn_padded(x_batch)

How to read printed IR
----------------------

Every symbolic operator can be printed in a readable textual IR form:

.. code-block:: python

   op = (
       SymbolicDiscreteJaxOperator(hi, "heisenberg_sym", hermitian=True)
       .for_each(("i", "j"), over=edges)
       .emit(identity(), matrix_element=site("i").value * site("j").value)
       .for_each(("i", "j"), over=edges)
       .where(site("i").value * site("j").value < 0)
       .emit(swap("i", "j"), matrix_element=2.0)
       .build()
   )

   print(op.to_ir())

Example output:

.. code-block:: text

   symbolic.operator @"heisenberg_sym" [dtype=float64, hermitian=true, hilbert_size=8] {
     ; 2 term(s)

     term #0 "0" [kbody, n_iter=8, max_conn_size=8] {
       iterate: for (i, j) in [(0, 1), (1, 2), (2, 3), ... +5 more]
       where:   true
       emit #0:
         update:    identity
         amplitude: (x[i] * x[j])
     }

     term #1 "1" [kbody, n_iter=8, max_conn_size=8] {
       iterate: for (i, j) in [(0, 1), (1, 2), (2, 3), ... +5 more]
       where:   ((x[i] * x[j]) < 0)
       emit #0:
         update:    x'[i], x'[j] = x[j], x[i]
         amplitude: 2
     }

   }

Interpretation guide:

* ``symbolic.operator ...``: global header with operator name, dtype, hermiticity, and Hilbert size.
* ``term #k``: one independent contribution to the operator action.
* ``iterate: ...``: static iteration domain (the tuples the term runs over).
* ``where: ...``: predicate gate for each iterator tuple.
* ``emit #m``: one emitted branch for that term.
* ``update: ...``: state rewrite rule from ``x`` to ``x'``.
* ``amplitude: ...``: matrix element assigned to that emitted branch.
