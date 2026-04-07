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


"""Compiled executable operator produced by SymbolicOperator.compile()."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from jax.tree_util import register_pytree_node_class

from netket.operator import AbstractOperator, DiscreteJaxOperator

from nkdsl.debug import event as debug_event


@register_pytree_node_class
class CompiledOperator(DiscreteJaxOperator):
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

    def get_conn_padded(self, x: Any) -> tuple:
        debug_event(
            "executing compiled get_conn_padded",
            scope="runtime",
            tag="RUNTIME",
            operator_name=self._name,
            input_shape=getattr(x, "shape", None),
            max_conn_size=self._max_conn_size,
        )
        return self._fn(x)

    def __add__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator._sum import SumOperator

            return SumOperator(self, other)
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator._sum import SumOperator

            return SumOperator(other, self)
        return super().__radd__(other)

    def __matmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator._prod import ProductOperator

            return ProductOperator(self, other)
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator._prod import ProductOperator

            return ProductOperator(other, self)
        return super().__rmatmul__(other)

    def __repr__(self) -> str:
        return (
            f"CompiledOperator("
            f"name={self._name!r}, "
            f"dtype={self._dtype_val}, "
            f"hermitian={self._hermitian}, "
            f"max_conn_size={self._max_conn_size})"
        )

    #
    #
    #   JAX pytree registration

    # CompiledOperator carries no JAX arrays, its _fn is a plain Python
    # callable and all other fields are Python scalars/strings. We register
    # it as an empty pytree (no leaves) so that JAX does not try to trace it
    # as an array when it is passed as a non-static argument to a jit-compiled
    # function (e.g. as local_value_args in _expect). The full object is
    # reconstructed from aux_data, which JAX treats as a static hashable key.

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


__all__ = ["CompiledOperator"]
