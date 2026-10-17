# `identity()`

`identity()` returns a no-op update.

Use it for diagonal operators where the connected configuration should remain the same and only the matrix element changes.

Example:

```python
op = (
    SymbolicDiscreteJaxOperator(hi, "diag")
    .for_each_site("i")
    .emit(identity(), matrix_element=site("i").value)
    .build()
)
```

For a source `[2, 0, 1]`, every visit emits that same source. The matrix
elements are two, zero, and one, so default deduplication produces one
diagonal entry with matrix element three. This defines the total number
operator on an occupation basis.

## An unchanged target still needs a coefficient

The helper specifies only the state update. It does not assign the matrix
element or mean that the whole operator is the identity operator. With
`matrix_element=1.0` under an all-sites iterator, the operator is `N` times
the identity. A single global emission with coefficient one defines the
identity itself.

You can omit the update argument to `emit` for the same diagonal behavior:

```python
.emit(matrix_element=site("i").value)
```

An identity update also provides an empty starting program for a longer update
chain. Once an operation is appended, that branch can be off diagonal. A
matrix element that evaluates to zero contributes no active connection after
deduplication, even if the update itself is the identity.
