# Extending the DSL

Register custom clauses when several operators share a domain-specific
iteration pattern, condition, or transition. Registered clauses become fluent
builder methods and compose with the built-in clauses:

```python
op = (
    nkdsl.SymbolicDiscreteJaxOperator(hi, "my-op")
    .my_custom_iterator(...)
    .my_custom_predicate(...)
    .emit(...)
    .build()
)
```

The extension points are separated by concern:

- **Iterator clauses** decide where the term iterates.
- **Predicate clauses** decide when a visit is active.
- **Emission clauses** decide how branches are emitted (including conditional chains).

Use the pages below as implementation guides:

- [Extending DSL Iterators](iterators.md)
- [Extending DSL Predicates](predicates.md)
- [Extending DSL Emissions](emissions.md)
