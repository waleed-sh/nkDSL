# `shift(site_ref, delta)`

`shift` adds `delta` to the selected site value.

It is the most common primitive for bosonic occupancies and any model where the local basis is encoded by additive integer updates.

Example:

```python
hopping_update = shift("i", -1).shift("j", +1)
```

Because updates are immutable and chainable, paired shifts are a natural way to express a hopping move. For the common occupation-transfer case, prefer the equivalent [``hop(src, dst, amount=1)``](hop.md) helper.

## Arithmetic, not automatic boundary handling

For a current site value two, `shift("i", +1)` produces three. It does not
clip the result or check a particle-number constraint. On a Fock basis with
maximum occupation `nmax`, guard a raising emission with
`site("i").value < nmax`. Guard a lowering emission with a positive source
occupation. Use [`shift_mod`](shift_mod.md) only when wrapping is part of the
intended model.

The shift amount may itself be a symbolic expression. Such expressions read
the original source configuration. The addition is applied to the current
working target, so repeated shifts accumulate:

```python
twice = shift("i", +1).shift("i", +1)
```

This adds two, whereas `write("i", site("i").value + 1)` repeated twice
still writes the original occupation plus one. Use the primitive whose
semantics match the intended transition.

For a spin basis encoded as `{-1, +1}`, an additive shift is usually less
convenient than [`affine("i", scale=-1)`](affine_scatter.md), which flips
either sign without a separate case for the source value.
