<p align="center">
  <img src="docs/_static/nkDSL.png" alt="nkDSL logo" width="350">
</p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/nkDSL.svg)](https://pypi.org/project/nkDSL/)
[![License](https://img.shields.io/github/license/waleed-sh/nkDSL.svg)](https://github.com/waleed-sh/nkDSL/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/nkdsl/badge/?version=latest)](https://nkdsl.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/waleed-sh/nkDSL/graph/badge.svg?token=MPNEEEISNI)](https://codecov.io/github/waleed-sh/nkDSL)

</div>

# nkDSL

`nkDSL` is a standalone extraction of the symbolic operator DSL from [neuraLQX](https://www.github.com/waleed-sh/neuraLQX).
It provides a compact, declarative way to define discrete operators in [NetKet](https://github.com/netket/netket) and compile them into executable JAX-backed operator kernels.

> [!WARNING]
> `nkDSL` is currently **experimental (alpha)** and **not a production release**.
> We are still testing and refining core behavior, interfaces, and performance.
> Expect API changes between releases while the package matures.

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

`nkdsl` is currently a small, focused package for the recent DSL operator work developed in neuraLQX. Right now the goal is clarity and usability rather than a broad, frozen API surface.

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) for the full text.
