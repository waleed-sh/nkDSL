# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Compiled executable operators produced by symbolic lowering."""

from __future__ import annotations

import inspect
import numbers
import re
from typing import Any, Callable

import numpy as np
from jax.tree_util import register_pytree_node_class

from netket.operator import AbstractOperator, DiscreteJaxOperator

from nkdsl.debug import event as debug_event

_DEFAULT_CONNECTION_METHOD: str = "get_conn_padded"
_DYNAMIC_CLASS_CACHE: dict[tuple[type[Any], str], type[Any]] = {}


def _is_additive_identity(value: Any) -> bool:
    """Returns True when ``value`` is a numeric zero usable as additive identity."""
    return isinstance(value, numbers.Number) and not isinstance(value, bool) and value == 0


class _CompiledOperatorMixin:
    """Shared runtime behavior for all compiled operator wrappers."""

    _CONNECTION_METHOD_NAME = _DEFAULT_CONNECTION_METHOD

    __slots__ = (
        "_connection_method_name",
        "_dtype_val",
        "_fn",
        "_hermitian",
        "_max_conn_size",
        "_name",
    )

    def __init__(
        self,
        hilbert: Any,
        *,
        name: str,
        fn: Callable,
        is_hermitian: bool,
        dtype: Any,
        max_conn_size: int,
    ) -> None:
        super().__init__(hilbert)
        self._name: str = str(name)
        self._fn: Callable = fn
        self._hermitian: bool = bool(is_hermitian)
        self._dtype_val: np.dtype = np.dtype(dtype)
        self._max_conn_size: int = int(max_conn_size)
        self._connection_method_name = str(self._CONNECTION_METHOD_NAME)

    @property
    def name(self) -> str:
        """Returns the readable operator name."""
        return self._name

    @property
    def is_hermitian(self) -> bool:
        return self._hermitian

    @property
    def dtype(self) -> np.dtype:
        return self._dtype_val

    @property
    def max_conn_size(self) -> int:
        return self._max_conn_size

    def _execute_connection(self, x: Any) -> tuple[Any, Any]:
        debug_event(
            "executing compiled connectivity kernel",
            scope="runtime",
            tag="RUNTIME",
            operator_name=self._name,
            connection_method=self._connection_method_name,
            input_shape=getattr(x, "shape", None),
            max_conn_size=self._max_conn_size,
        )
        return self._fn(x)

    def __add__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import SumOperator

            return SumOperator(self, other)
        return super().__add__(other)

    def __radd__(self, other):
        if _is_additive_identity(other):
            return self
        if isinstance(other, AbstractOperator):
            from netket.operator import SumOperator

            return SumOperator(other, self)
        return super().__radd__(other)

    def __matmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import ProductOperator

            return ProductOperator(self, other)
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import ProductOperator

            return ProductOperator(other, self)
        return super().__rmatmul__(other)

    def tree_flatten(self):
        return [], (
            self._name,
            self._fn,
            self._hermitian,
            self._dtype_val,
            self._max_conn_size,
            self.hilbert,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        name, fn, is_hermitian, dtype, max_conn_size, hilbert = aux_data
        return cls(
            hilbert,
            name=name,
            fn=fn,
            is_hermitian=is_hermitian,
            dtype=dtype,
            max_conn_size=max_conn_size,
        )


@register_pytree_node_class
class CompiledOperator(_CompiledOperatorMixin, DiscreteJaxOperator):
    """
    An executable operator produced by lowering a
    :class:`~nkdsl.core.operator.SymbolicOperator`.

    ``CompiledOperator`` is the concrete result of ``symbolic_op.compile()``
    or ``SymbolicDiscreteJaxOperator(...).compile()``. Its ``get_conn_padded`` kernel is a
    pure JAX function that can be JIT-compiled, vmapped, and differentiated.

    The class name is fixed and stable, it does not encode the operator name
    or structure. The readable operator name is available via the
    :attr:`name` property.

    Attributes:
        name: Operator name (from the DSL definition).
        is_hermitian: Whether this operator is declared Hermitian.
        dtype: Matrix-element NumPy dtype.
    """

    _CONNECTION_METHOD_NAME = _DEFAULT_CONNECTION_METHOD

    def get_conn_padded(self, x: Any) -> tuple[Any, Any]:
        return self._execute_connection(x)

    def __repr__(self) -> str:
        return (
            f"CompiledOperator("
            f"name={self._name!r}, "
            f"dtype={self._dtype_val}, "
            f"hermitian={self._hermitian}, "
            f"max_conn_size={self._max_conn_size})"
        )


def _method_dispatch(connection_method: str) -> Callable[[Any, Any], tuple[Any, Any]]:
    def _dispatch(self, x: Any) -> tuple[Any, Any]:
        return self._execute_connection(x)

    _dispatch.__name__ = connection_method
    _dispatch.__qualname__ = connection_method
    return _dispatch


def _dynamic_class_name(operator_type: type[Any], connection_method: str) -> str:
    base = re.sub(r"[^0-9A-Za-z_]+", "_", operator_type.__name__).strip("_")
    method = re.sub(r"[^0-9A-Za-z_]+", "_", connection_method).strip("_")
    return f"Compiled_{base}_{method}"


def _resolve_dynamic_class(
    operator_type: type[Any],
    connection_method: str,
) -> type[Any]:
    cache_key = (operator_type, connection_method)
    cached = _DYNAMIC_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    attrs: dict[str, Any] = {
        "__module__": __name__,
        "_CONNECTION_METHOD_NAME": connection_method,
        connection_method: _method_dispatch(connection_method),
    }
    class_name = _dynamic_class_name(operator_type, connection_method)
    dynamic_cls = type(class_name, (_CompiledOperatorMixin, operator_type), attrs)
    dynamic_cls = register_pytree_node_class(dynamic_cls)

    if inspect.isabstract(dynamic_cls):
        missing = sorted(getattr(dynamic_cls, "__abstractmethods__", ()))
        raise TypeError(
            f"Cannot build compiled operator class for {operator_type!r} with "
            f"connection method {connection_method!r}; unresolved abstract methods: {missing!r}."
        )

    _DYNAMIC_CLASS_CACHE[cache_key] = dynamic_cls
    return dynamic_cls


def create_compiled_operator(
    hilbert: Any,
    *,
    name: str,
    fn: Callable,
    is_hermitian: bool,
    dtype: Any,
    max_conn_size: int,
    operator_type: type[Any] = DiscreteJaxOperator,
    connection_method: str = _DEFAULT_CONNECTION_METHOD,
) -> Any:
    """
    Creates a compiled operator instance for the requested operator target.

    The default target returns a concrete :class:`CompiledOperator`. For all
    other targets a dynamic subclass is generated that injects the compiled
    kernel into ``connection_method``.
    """
    normalized_method = str(connection_method).strip()
    if not normalized_method:
        raise ValueError("connection_method must be a non-empty string.")
    if not normalized_method.isidentifier():
        raise ValueError(
            f"connection_method {normalized_method!r} is not a valid Python identifier."
        )
    if not isinstance(operator_type, type):
        raise TypeError(f"operator_type must be a class, got {type(operator_type)!r}.")

    if operator_type is DiscreteJaxOperator and normalized_method == _DEFAULT_CONNECTION_METHOD:
        return CompiledOperator(
            hilbert,
            name=name,
            fn=fn,
            is_hermitian=is_hermitian,
            dtype=dtype,
            max_conn_size=max_conn_size,
        )

    dynamic_cls = _resolve_dynamic_class(operator_type, normalized_method)
    try:
        return dynamic_cls(
            hilbert,
            name=name,
            fn=fn,
            is_hermitian=is_hermitian,
            dtype=dtype,
            max_conn_size=max_conn_size,
        )
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate compiled operator target {operator_type!r} "
            f"with connection method {normalized_method!r}: {exc}"
        ) from exc


__all__ = ["CompiledOperator", "create_compiled_operator"]
