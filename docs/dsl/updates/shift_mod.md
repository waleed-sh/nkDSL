# `shift_mod(site_ref, delta)`

`shift_mod` applies a Hilbert-aware wrapped shift.

This helper is intended for finite discrete local bases with contiguous, unit-spaced integer `local_states`. The builder records metadata so the lowerer can reproduce the same wrap semantics at runtime.

Use it when a degree of freedom should wrap around instead of saturating or leaving the basis.

Example:

```python
plaquette = shift_mod("e0", +1).shift_mod("e1", -1)
```

Because wrapped semantics depend on the Hilbert local basis, `shift_mod` is more specialized than plain `shift`.

## Wrap within the declared local basis

If the local states run from `state_min` to `state_max`, the span is
`state_max - state_min + 1`. A wrapped shift computes

```text
target = ((current + delta - state_min) % span) + state_min
```

On the basis `[-1, 0, 1]`, shifting one by `+1` yields minus one. This is a
cyclic degree of freedom, not an ordinary boson creation operation. A truncated
bosonic raise should normally be rejected at its upper boundary rather than
wrapped to zero.

The builder infers and records the wrap specification from the enclosing
Hilbert space when the operator is built. A noncontiguous basis or a basis
with non-unit spacing is outside this helper's supported semantics. Use
explicit writes or a suitable extension for those cases.

Wrapped shifts preserve membership in the supported local basis. They do
not, by themselves, preserve constraints across sites such as a fixed total
charge. Compose the required shifts and guards according to the model. In an
update chain, each modular shift acts on the current working target.
