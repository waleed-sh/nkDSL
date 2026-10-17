# Quick start

## Installation

nkDSL requires Python 3.11–3.13. Install it with pip. NetKet and its numerical
dependencies are installed with it:

```bash
python -m pip install nkDSL
```

For editable development installs:

```bash
git clone https://github.com/waleed-sh/nkDSL.git
cd nkDSL
python -m pip install -e .
```

## Build an Ising Hamiltonian

Consider a transverse-field Ising chain:

$$
H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i
$$

The interaction term leaves the configuration unchanged and contributes a
bond-dependent matrix element. The field term flips one spin. Define them
separately so they can be inspected or reused:

```python
import netket as nk
from nkdsl import SymbolicDiscreteJaxOperator, affine, identity, site

L = 6
J = 1.0
h = 0.7

hi = nk.hilbert.Spin(s=1 / 2, N=L)
graph = nk.graph.Chain(length=L)

zz = (
    SymbolicDiscreteJaxOperator(hi, "ising_zz", dtype="float64", hermitian=True)
    .for_each(("i", "j"), over=graph.edges())
    .emit(
        identity(),  # Omitting the update also leaves the state unchanged.
        matrix_element=-J * site("i").value * site("j").value,
    )
    .build()
)

x_field = (
    SymbolicDiscreteJaxOperator(hi, "ising_x", dtype="float64", hermitian=True)
    .for_each_site("i")
    .emit(
        affine("i", scale=-1, bias=0),
        matrix_element=-h,
    )
    .build()
)

# Compile the pieces, then combine them with NetKet's operator algebra.
H = zz.compile() + x_field.compile()
```

## Read the definition

`for_each(("i", "j"), over=graph.edges())` visits the graph's bonds. At each bond,
the identity update keeps the source configuration and the matrix element reads
the two spin values. The site iterator in `x_field` visits each spin once.

On the local basis $\{-1,+1\}$, multiplying a value by $-1$ flips the spin.
Both terms use ordinary Python arithmetic to construct symbolic matrix-element
expressions.

## Compile and evaluate connections

`build()` freezes a definition and `compile()` makes it executable. The sum
`H` above is ready to use in NetKet. Evaluate a batch of configurations:

```python
import jax

x_batch = hi.random_state(jax.random.PRNGKey(0), size=4)
xp, mels = H.get_conn_padded(x_batch)
print(xp.shape, mels.shape)
```

`xp` contains connected configurations and `mels` their matrix elements. The
connection axis is padded to the compiled operator's `max_conn_size`. Unused
entries have zero matrix element. The compiler combines repeated targets by
default.

To follow the stages behind `.compile()`, read
[how compilation works](getting_started/compilation.md). For a single builder
that collects several terms, see [chaining terms and updates](dsl/chaining.md).

Continue with the [Ising notebook](tutorials/ising_end_to_end.ipynb) to optimise a
variational state and compare the result with NetKet's native operator.
