<div align="center">

[![PyPI](https://img.shields.io/pypi/v/nkDSL.svg)](https://pypi.org/project/nkDSL/)
[![License](https://img.shields.io/github/license/waleed-sh/nkDSL.svg)](https://github.com/waleed-sh/nkDSL/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/nkdsl/badge/?version=latest)](https://nkdsl.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/waleed-sh/nkDSL/graph/badge.svg?token=MPNEEEISNI)](https://codecov.io/github/waleed-sh/nkDSL)

</div>

# nkDSL

`nkDSL` is a standalone extraction of the symbolic operator DSL from [neuraLQX](https://www.github.com/waleed-sh/neuraLQX).
It provides a compact, declarative way to define discrete operators in [NetKet](https://github.com/netket/netket) and compile them into executable JAX-backed operator kernels.

> [!NOTE]
> `nkDSL` is currently in **beta**.
> The core DSL and compiler workflow are ready for broader testing, while some
> advanced APIs may still evolve before a fully stable release.

## What it does

With `nkDSL`, you describe an operator in terms of

- **where** to iterate, such as sites, pairs, or user-defined index sets,
- **when** a branch is active, through symbolic predicates,
- **how** to rewrite a configuration, through update programs,
- **what** matrix element to assign to each emitted branch.

The result is a symbolic operator that can be compiled into a NetKet-compatible executable operator.

## Example: 1D transverse-field Ising model

The example below builds $` H = -J \sum_{\langle i,j \rangle} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x `$
for a periodic chain. The diagonal Ising interaction is emitted with an identity update, while the transverse-field term flips one spin at a time.

```python
import netket as nk
from nkdsl import SymbolicDiscreteJaxOperator, site, write

L = 8
J = 1.0
h = 0.5

hilbert = nk.hilbert.Spin(s=0.5, N=L)
graph = nk.graph.Chain(length=L, pbc=True)

ising = (
    SymbolicDiscreteJaxOperator(hilbert, "ising_tfim", hermitian=True)
    .for_each(("i", "j"), over=graph.edges())
    .emit(
        matrix_element=1.2 * site("i").value * site("j").value,
    )
    .for_each_site("i")
    .emit(
        write("i", -site("i").value),
        matrix_element=-0.7,
    )
    .compile()
)

# now, `ising` is a NetKet DiscreteJaxOperator subclass, and can be used in NetKet
```

## Installation

Install directly from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install nkdsl
```

## Status

`nkdsl` is a small, focused beta package for the symbolic operator work developed in neuraLQX. The goal is a clear, usable DSL with a stable core workflow and room for advanced APIs to continue maturing.

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) for the full text.

## Documentation

The [documentation](https://nkdsl.readthedocs.io/en/latest/) uses MkDocs and
Material. To preview it locally:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

Notebook pages display verified, saved outputs and do not execute during site
builds. See the [documentation guide](docs/contributing/documentation.md) for
refreshing notebooks and running the checks used by CI and Read the Docs.
