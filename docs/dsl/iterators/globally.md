# `globally()`

`globally()` creates exactly one visit per input configuration.

Use it for:

- diagonal operators that only inspect the current configuration
- global quantities where sites are accessed through static indices or helper expressions

Example:

```python
from nkdsl import SymbolicDiscreteJaxOperator, identity, source_index

offset = (
    SymbolicDiscreteJaxOperator(hi, "offset")
    .globally()
    .emit(identity(), matrix_element=-1.2)
    .build()
)
```

In practice, global terms are most useful together with static index expressions or callback-based expression construction.

## Read fixed positions explicitly

A global term binds no site labels. An expression such as `site("i").value`
therefore has no meaning in its scope. Use `source_index(k)` to read a known
position of the input, or `target_index(k)` in a matrix element to read the
completed target.

```python
number = (
    SymbolicDiscreteJaxOperator(hi, "number")
    .globally()
    .emit(matrix_element=sum(source_index(k) for k in range(hi.size)))
    .build()
)
```

The Python `sum` here constructs one symbolic expression over static indices.
At runtime the term contributes one diagonal branch whose matrix element is
the total occupation. An all-sites diagonal term can express the same operator,
but starts from one contribution per site before aggregation.

The constant-offset example above also contributes one diagonal branch. Its
coefficient is applied once per configuration, not once per site. This
distinction matters for energy offsets. A global term may contain several
emissions if a fixed collection of whole-configuration moves is needed.
