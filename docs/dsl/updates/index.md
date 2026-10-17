# Updates

Update programs describe how one active branch rewrites `x` into `x'`.

The user-facing builder is [`nkdsl.Update`](../../api/generated/nkdsl.Update.md#nkdsl.Update), but most code starts from the module-level helpers.

An update describes one branch. Calling `.shift(...)` or `.write(...)` on an
update returns a new program with that operation appended. The original is
unchanged, so small update programs can be reused. At execution, operations
run in order on a working copy of the source configuration.

Choose the primitive that expresses the move directly. `shift` changes an
occupation by an amount, `write` sets an explicit value, and `swap` rearranges
existing values. `identity` leaves the target unchanged for a diagonal
contribution. The guides below show the remaining choices.

## State reads and validity

Expressions such as `site("i").value` refer to the source, even inside a
later operation in the chain. Shifts and affine operations act on the current
working target. These rules differ when the same site is updated more than
once. The [chaining guide](../chaining.md#update-chains-are-immutable-and-ordered)
works through an example.

Updates do not infer the physical constraints of your model. Guard a bosonic
raise at the truncation boundary and a lower at zero. A move that preserves
each local basis can still violate a global constraint on the Hilbert space.
For a new operator, check its action on a small constrained system as well as
on individual local values.

- [``identity()``](identity.md)
- [``shift(site_ref, delta)``](shift.md)
- [``hop(src, dst, amount=1)``](hop.md)
- [``shift_mod(site_ref, delta)``](shift_mod.md)
- [``write(site_ref, value)``](write.md)
- [``swap(...)`` and ``permute(...)``](swap_permute.md)
- [``affine(...)`` and ``scatter(...)``](affine_scatter.md)
