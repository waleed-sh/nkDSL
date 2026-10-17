# `hop(src, dst, amount=1)`

`hop` transfers occupation from one site to another.

It is shorthand for a paired shift:

```python
hop("i", "j", amount=a)
# equivalent to:
shift("i", -a).shift("j", +a)
```

Use it for off-diagonal hopping terms where one quantum number is removed from the source site and added to the destination site.

Example:

```python
hopping = (
    SymbolicDiscreteJaxOperator(hi, "hopping")
    .for_each(("i", "j"), over=graph.edges())
    .where((site("i") > 0) & (site("j") < nmax))
    .emit(hop("i", "j"), matrix_element=-t)
    .build()
)
```

For larger transfer steps, pass `amount`:

```python
pair_hop = hop("i", "j", amount=2)
```

## Supply the physical amplitude and bounds

`hop` only transfers the site values. A bosonic hopping matrix element also
contains occupation factors. For a basis from zero to `nmax`, one direction
of the usual hopping term is:

```python
ni, nj = site("i").value, site("j").value
builder.where((ni > 0) & (nj < nmax)).emit(
    hop("i", "j"), matrix_element=-t * (ni * (nj + 1)).sqrt()
)
```

For a transfer of `a` particles, the validity bounds become `ni >= a` and
`nj + a <= nmax`. The correct amplitude depends on the operator being
represented and is not supplied by the helper.

The update preserves the sum of the two occupations when the sites differ.
It does not add the reverse move automatically. If a bond list contains each
undirected edge once, add a reverse emission or term to obtain both directions.
If the iterator already contains both orientations, take care not to count
them twice. Set `hermitian=True` only after checking the complete operator.
