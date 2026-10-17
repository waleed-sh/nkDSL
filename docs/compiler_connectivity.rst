Connectivity assembly and deduplication
=======================================

By default, ``compile()`` combines branches with the same target configuration,
sums their matrix elements, and removes entries whose final matrix element is
zero. ``deduplicate_connected_components=False`` retains the raw branches.

Deduplication is planned from the operator IR. The compiler uses target
structure to remove work before generating a JAX kernel, and generates exact
runtime checks for collisions that depend on the input configuration.

Static assembly planning
------------------------

:class:`nkdsl.compiler.passes.deduplication.SymbolicDeduplicationPass` runs after
the raw max-connection analysis. It performs the following steps:

#. Substitute the static site-index tuples into each update program.
#. Symbolically execute updates, retaining their evaluation order and casts.
#. Group structurally identical target expressions across terms, emissions,
   and iterator labels. Remove constant-zero, invalidated, and provably inactive
   branches. Two unconditional opposite constant contributions to an identical
   target are removed together.
#. Use affine interval analysis and bounded reasoning over the local basis to
   prove target classes disjoint. Simple displacements do not require
   enumeration of a large local basis.
   Predicates can establish that a collision is impossible on active branches.
   Exact changed-site sets partition the analysis before pairwise checks.
#. Build connected components of the remaining possible collisions, project each
   component onto the coordinates needed to distinguish its targets, and derive
   finite coordinate ranges where possible.
#. Derive the output capacity from the target classes and finite target images.

This applies to all update kinds, including writes, shifts, modular shifts,
affine updates, swaps, permutations, scatter updates, and conditionals. Programs
with state-dependent update indices retain an opaque structural key and use
general runtime collision checks. Declared symbol defaults participate in static
analysis.

The proofs assume input configurations belong to the Hilbert space's local
basis. They do not require an exhaustive enumeration of the Hilbert space or
infer correctness from sampled configurations. Analysis has explicit work
budgets, exhausting a budget leaves a runtime obligation. Integer proofs also
record intermediate ranges and operand precision. If the runtime configuration
dtype cannot represent those intermediates exactly, lowering selects general
exact collision checks while retaining the proven structural output bound.

Output capacity
---------------

``max_conn_size`` and the connection axis of ``get_conn_padded`` use the reduced
bound. The original raw branch count is not restored as padding. For example,
five diagonal contributions and eight distinct hopping targets require nine
slots rather than thirteen. A diagonal sum over any number of sites needs one
slot. Several branches copying binary site values into one coordinate need at
most two target slots, even when their target expressions differ.

JAX requires a static output shape. The capacity therefore remains a proven
upper bound over configurations, not the number of active connections in a
particular sample. State-dependent collisions or cancellations can leave unused
slots. Nonzero entries are packed in deterministic first-occurrence order and
unused slots contain zero states and zero matrix elements. An identically zero
operator retains one zero padding slot for compatibility with the operator API.

Conservative analysis can retain an upper bound larger than the true maximum
when it cannot prove a tighter one. It never truncates connections to an
unproved smaller size. Explicit max-connection hints do not override the bound
proved by deduplication analysis.

Runtime assembly
----------------

Identical-target contributions are summed with static gather/reduction schedules.
Targets proved distinct require no runtime uniqueness operation. When their
amplitudes are also known nonzero, they need no compaction.

For unresolved collision components, lowering chooses between three algorithms:

* Small proven key domains use a direct-address workspace. Amplitudes and the
  first occurrence of each key are accumulated directly, without comparisons
  or sorting. Workspace size is bounded by 4,096 slots and four times the
  component size. Components below eight entries use the smaller pairwise path.
* Small components use exact pairwise key equality and amplitude accumulation.
  The temporary footprint is bounded by the strategy selector.
* Larger components use one stable lexicographic sort, an adjacent-run scan,
  and accumulation into the first original occurrence. Auxiliary storage grows
  linearly with component size. There is no inverse unique mapping or second
  sort to recover emission order.

Proven integer coordinate ranges are encoded into 32-bit words using all bits
needed by each coordinate. This encoding is lossless, it does not use hashes
or assume that fingerprint collisions are unlikely. Unknown coordinate ranges
retain full exact keys. Nonzero compaction uses a prefix sum and a destination
map. Single-slot outputs use a direct zero mask.

Cost diagnostics
----------------

:class:`nkdsl.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass`
compares estimated raw and deduplicated assembly work after planning. It emits
:ref:`NKDSL-W401 <lint-code-nkdsl-w401>` when aggregation, residual collision
checks, key packing, and compaction are estimated to outweigh output savings.

The model counts scalar work in assembly. It is a deterministic heuristic, not
a prediction of microseconds. Actual latency depends on the device, batch size,
configuration dtype, and input states. Matrix-element evaluation is shared by
both variants and is excluded from the comparison. Downstream model evaluation
is also excluded: fewer target configurations can improve total simulation
performance even when connectivity assembly becomes slower.

The warning preserves the requested deduplication behavior. It follows
``diagnostics_enabled``, ``diagnostics_min_severity``, ``max_diagnostics``, and
``fail_on_warnings``. This pass runs on cache misses, a cache hit reuses the
existing assembly plan and artifact diagnostics.

Inspecting a plan
-----------------

.. code-block:: python

   compiler = nkdsl.SymbolicCompiler()
   artifact = compiler.compile(symbolic_operator)
   compiled = artifact.compiled_operator

   metadata = artifact.metadata_map()
   plan = metadata["connectivity_deduplication"]
   cost = metadata["connectivity_cost"]

   print(plan["raw_size"], plan["output_size"])
   print(plan["runtime_component_sizes"])
   print(plan["packed_key_widths"])
   print(cost["estimated_assembly_ratio"])

``plan`` is ``None`` when deduplication is disabled. These metadata entries are
intended for inspecting lowering decisions and performance diagnostics.

