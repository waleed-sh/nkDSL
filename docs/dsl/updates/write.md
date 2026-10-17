# `write(site_ref, value)`

`write` overwrites one site with a new symbolic or numeric value.

Example:

```python
op = write("i", 0)
```

This is the most direct update when you know the target value explicitly.

## Write a constant or a source expression

The value can be numeric or symbolic. For example,
`write("j", site("i").value)` copies the source occupation at `i` into `j`.
The rest of the configuration is unchanged. This differs from a swap, which
also replaces the value at `i`.

Expressions read the original source throughout an update program. Thus:

```python
update = write("i", 0).write("j", site("i").value)
```

sets `i` to zero but copies its original value to `j`. To transform the
working value at one site, use a shift or affine operation. For simultaneous
rearrangement of several working values, use a swap or permutation.

If several writes target the same position, the last write wins. Values are
stored in the configuration's array dtype, so do not use writes to introduce
fractional or complex quantum numbers into an integer-valued basis. Matrix
elements have their own dtype and are a separate part of the emission.

Like other arithmetic updates, a write does not establish that the target
belongs to the Hilbert space. Choose valid constants and guard source-dependent
writes when needed, especially in a constrained basis.
