# Choosing the compiled operator type

The lowering target determines the base class of the executable operator.
Choose it once near the start of a script, then use `.compile()` normally.
NetKet is the default.

## Built-in targets

| Name | Compiled operator base | Kernel entry point |
| --- | --- | --- |
| `"netket"` | `netket.operator.DiscreteJaxOperator` | `get_conn_padded` |
| `"neuralqx"` | `neuralqx.operators.types.ComputationalJaxOperator` | `_get_conn_padded` |

The older name `"netket_discrete_jax"` remains an alias for `"netket"`.
Both names select the same target and share cached artifacts.

For NetKet, no extra setup is needed:

```python
import nkdsl

nkdsl.config.LOWERING = "netket"
```

For neuraLQX, first install the optional dependency:

```bash
python -m pip install "nkDSL[neuralqx]"
```

The extra selects NetKet `>=3.20.0,<3.22.0`, the range supported by neuraLQX.
This avoids an import failure in neuraLQX 1.1.2 with newer NetKet releases.
Base nkDSL installations keep their existing NetKet version range.

Then select its built-in target:

```python
import nkdsl

nkdsl.config.LOWERING = "neuralqx"
```

You do not need to register either built-in target. neuraLQX is imported only
when its target is used for compilation. A NetKet script does not require it.

The selected default applies to builders, built symbolic operators, and the
compiler convenience function:

```python
compiled = builder.compile()
compiled = symbolic_op.compile()
compiled = nkdsl.compile_symbolic_operator(symbolic_op)
```

`nkdsl.config` and the existing `nkdsl.cfg` are the same configuration object.
For batch jobs, `NKDSL_LOWERING=neuralqx` also selects the initial default.

## Register a custom operator family

Register a custom base class through
[`nkdsl.LOWERING_REGISTRY`](api/generated/nkdsl.LOWERING_REGISTRY.md).
Registration selects the new target for subsequent compilations:

```python
import nkdsl


class MyComputationalOperator:
    def __init__(self, hilbert):
        self.hilbert = hilbert

    def get_conn_padded(self, x):
        return self._get_conn_padded(x)


nkdsl.LOWERING_REGISTRY.register(
    name="my_operator",
    operator_type=MyComputationalOperator,
    connection_method="_get_conn_padded",
)

compiled = builder.compile()
```

The class above illustrates the constructor and connectivity contract. Use
your framework's operator base class when the result needs its sampling,
expectation, or operator-algebra APIs. The compiler subclasses that base and
implements the named connectivity method. The method name must be a valid
Python identifier.

To make a target available without selecting it, pass `set_default=False`.
You can select it later with either form:

```python
nkdsl.config.LOWERING = "my_operator"
# Equivalent selection through the registry:
nkdsl.LOWERING_REGISTRY.set_default("my_operator")
```

[`default_symbolic_operator_lowering_registry()`](api/generated/nkdsl.default_symbolic_operator_lowering_registry.md#nkdsl.default_symbolic_operator_lowering_registry)
returns the same shared object. Existing registration code can continue using
that function. New registrations in the shared registry now select their
target unless `set_default=False` is supplied.

## Change the default or override one compilation

You can change the default at any point in a script:

```python
nkdsl.config.LOWERING = "neuralqx"
first = builder.compile()

nkdsl.config.LOWERING = "netket"
second = builder.compile()
```

Assigning the default again emits
[`LoweringWarning`](api/generated/nkdsl.LoweringWarning.md). The new setting
takes effect, including when it repeats the previous value. Already compiled
operators retain their original type and remain usable.

For a single exception to the default, keep using `operator_lowering`:

```python
special = builder.compile(operator_lowering="my_operator")
usual = builder.compile()
```

The override applies only to `special`. It does not change the configured
default. Compilations with different target classes or kernel entry points
use separate cache entries. Replacing a registered class also prevents reuse
of an artifact compiled for the previous implementation.

## Explicit compiler instances

A compiler with an omitted target follows the current default each time it
compiles, even if the instance was created before the default changed:

```python
compiler = nkdsl.SymbolicCompiler()
nkdsl.config.LOWERING = "my_operator"
compiled = compiler.compile_operator(symbolic_op)
```

Set `operator_lowering` in the compiler constructor or
[`SymbolicCompilerOptions`](api/generated/nkdsl.SymbolicCompilerOptions.md)
to pin that compiler to a target. Options passed to an individual compile
call can override it. `symbolic_op.compile(compiler=compiler,
operator_lowering="netket")` also overrides the target for that invocation.

The selection order is:

1. An explicit target for the current compilation.
2. An explicit target in the compiler's default options.
3. The current registry default.

An independently constructed
[`SymbolicOperatorLoweringRegistry`](api/generated/nkdsl.SymbolicOperatorLoweringRegistry.md)
maintains its own default. Pass `set_default=True` when registering a target
to select it there. It does not change the process-wide configuration.

## Repeated registrations

Registering a name that already exists emits
[`RegistrationWarning`](api/generated/nkdsl.RegistrationWarning.md) and keeps
the existing implementation and current default. This also applies to a
built-in target whose optional dependency has not been imported yet.

Use `replace=True` when you intend to replace an implementation:

```python
nkdsl.LOWERING_REGISTRY.register(
    name="my_operator",
    operator_type=MyNewComputationalOperator,
    connection_method="_get_conn_padded",
    replace=True,
)
```

Duplicate iterator, predicate, and emission registrations follow the same
rule. A repeated notebook cell or module import keeps the running definition
unless replacement is explicit. Invalid clause names and unknown lowering
targets still report errors, as does selecting neuraLQX without installing it.
