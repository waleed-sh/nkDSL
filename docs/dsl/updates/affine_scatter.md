# `affine(...)` and `scatter(...)`

`affine(site_ref, scale=..., bias=...)` applies an affine transform to one site.

A common use is spin flipping on a `\{-1, +1\}` basis.

```python
flip = affine("i", scale=-1, bias=0)
```

`scatter(indices, values)` writes several flat positions in one update program.

Use it when a branch naturally modifies a fixed collection of sites at once.

## Affine transformations act on the working value

The operation computes `scale * current + bias` at the selected position.
For a spin-half configuration represented by values minus one and plus one,
the example above changes either sign. The scale and bias can be symbolic
expressions, which read the source state, while `current` is the value after
any preceding operations in the update chain.

An affine transform is ordinary arithmetic. It does not round, wrap, or infer
which values are allowed. Choose a transform that preserves the local basis,
or restrict the branch with a condition.

## Scatter uses fixed positions

`scatter` accepts equal-length sequences of integer flat indices and values:

```python
from nkdsl import scatter, source_index

copy_pair = scatter((0, 2), (source_index(2), source_index(0)))
```

This copies the original value at position two to zero and the original value
at zero to two. Indices are fixed when building the program, so this is useful
for a known set of coordinates in a global term. Use label-based writes when
positions should follow an iterator.

Values are evaluated from the source environment and writes are applied in
sequence. Repeating an index therefore keeps its last assigned value. For
a rotation of values from the current working target, prefer `permute` so its
atomic capture semantics are explicit.
