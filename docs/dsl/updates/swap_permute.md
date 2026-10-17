# `swap(...)` and `permute(...)`

`swap(site_a, site_b)` exchanges two site values.

`permute(site_0, site_1, ..., site_k)` performs a cyclic permutation over the listed sites.

## Examples

```python
swap("i", "j")
permute("i", "j", "k")
```

These helpers are useful when the update is naturally expressed as a rearrangement rather than as several independent writes.

## Rotation direction

For `permute("i", "j", "k")`, the new value at `i` comes from `j`, the new
value at `j` comes from `k`, and the new value at `k` comes from `i`. A source
triple `(1, 2, 3)` therefore becomes `(2, 3, 1)`. At least two site references
are required.

The operation captures all participating values before writing any of them.
This avoids the overwrite problem that occurs when a rotation is expressed
as sequential copies from a changing target. Within a larger update chain,
the captured values come from the working target at that point, including
earlier updates.

Swapping two equal occupations leaves the configuration unchanged. That branch
can then coincide with a diagonal contribution or another move. Default
deduplication combines their matrix elements. If the physical exchange term
should act only on unequal values, state that explicitly with a condition.

A permutation preserves the multiset of values at distinct participating
positions. It need not preserve a constraint that depends on where each value
is placed. Use distinct iterator labels and an appropriate domain for a
geometric cycle, and check any spatial constraints of the model.
