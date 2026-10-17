# nkDSL

nkDSL lets you describe a discrete quantum operator by its action on a
configuration, then compile that description into a NetKet-compatible JAX
operator. It began as the symbolic operator language in
[neuraLQX](https://github.com/waleed-sh/neuraLQX) and is available as a standalone
package.

!!! warning "Beta software"

    nkDSL is in beta and is not ready for production use. APIs and compiler
    behavior may change. Check your operator against an independent reference
    before relying on numerical results, and report unexpected behavior.

!!! note "About this documentation"

    Most of this documentation was generated using OpenAI's GPT-6 model in
    Codex. If you spot errors, unclear explanations, or incomplete details,
    please [open an issue on GitHub](https://github.com/waleed-sh/nkDSL/issues/new).

An operator definition follows the way you would describe a transition:

1. Choose the sites or tuples to visit with an **iterator**.
2. State any **condition** that makes a transition allowed.
3. **Update** the configuration at those sites.
4. Attach the transition's **matrix element**.

The compiler turns these terms into a static-shape `get_conn_padded` kernel.
It checks symbols and indices, reports connectivity diagnostics, and plans how
to combine branches that reach the same state.

## Start with an operator

The [quick start](quickstart.md) builds and evaluates an Ising Hamiltonian.
Read the [overview](overview.md) first if you are deciding whether a symbolic
definition suits your model. The [supported features](dsl/currently_supported.md)
page lists the available Hilbert spaces, operations, and current limits.

For a complete optimisation, work through the [Ising](tutorials/ising_end_to_end.ipynb),
[Heisenberg](tutorials/heisenberg_end_to_end.ipynb), or
[J1–J2](tutorials/j1j2_end_to_end.ipynb) notebook. Each includes saved results and
can be downloaded and run in Jupyter.

## Find the details you need

- [The DSL](dsl/index.md) explains selectors, conditions, updates, and emissions.
- [Diagnostics](dsl/linting/index.md) explains compiler messages and how to act on them.
- [How compilation works](getting_started/compilation.md) follows a definition through the compiler.
- [Compiler and execution](compiler.md) covers compilation interfaces and connectivity assembly.
- [Developers](developers/index.md) explains the IR, compiler passes, and extension architecture.
- [Extension guides](guides/extending_dsl/index.md) and
  [worked notebooks](advanced_tutorials/index.md) show how to add domain-specific clauses.
- [API reference](api/index.md) documents the public interfaces and source modules.

See the [changelog](changelog.md) for release changes and the
[documentation guide](contributing/documentation.md) to contribute or build the site.
