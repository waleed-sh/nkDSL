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


"""Abstract base class for all symbolic operator implementations."""

from __future__ import annotations

import abc
import numbers
from typing import Any

from netket.hilbert import DiscreteHilbert
from netket.operator import AbstractOperator, DiscreteJaxOperator

from nkdsl.errors import SymbolicOperatorExecutionError
from nkdsl.ir.program import SymbolicOperatorIR


def _is_additive_identity(value: Any) -> bool:
    """Returns True when ``value`` is a numeric zero usable as additive identity."""
    return isinstance(value, numbers.Number) and not isinstance(value, bool) and value == 0


class AbstractSymbolicOperator(DiscreteJaxOperator):
    """
    Abstract base class for all symbolic (DSL-defined) operators.

    Symbolic operators extend :class:`DiscreteJaxOperator` and declare
    their action through a typed IR rather than a hand-written JAX kernel.
    They **cannot** execute until the compiler has lowered them to a concrete
    JAX kernel via :meth:`nkdsl.compiler.SymbolicCompiler.compile`.

    Attempting to call :meth:`get_conn_padded` before compilation raises
    :class:`~nkdsl.errors.SymbolicOperatorExecutionError`.

    Args:
        hilbert: Discrete Hilbert space this operator is defined on.
        name: User-facing operator name.
        dtype_str: String label for the matrix-element dtype.
        is_hermitian: Whether this operator is declared Hermitian.
        metadata: Optional extra metadata dictionary.
    """

    __slots__ = ("_dtype_val", "_is_hermitian_val", "_metadata_dict", "_name_val")

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        *,
        name: str,
        dtype_str: str = "complex64",
        is_hermitian: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(hilbert)
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Operator name must be a non-empty string.")
        self._name_val: str = normalized
        self._dtype_val: str = str(dtype_str)
        self._is_hermitian_val: bool = bool(is_hermitian)
        self._metadata_dict: dict[str, Any] = dict(metadata) if metadata else {}

    @property
    def operator_name(self) -> str:
        """Returns user-facing operator name."""
        return self._name_val

    @property
    def name(self) -> str:
        """
        Returns user-facing operator name.

        Returns:
            str: Stable operator name.
        """
        return self._name_val

    @property
    def dtype(self):
        """Returns matrix-element dtype."""
        import numpy as np

        return np.dtype(self._dtype_val)

    @property
    def dtype_str(self) -> str:
        """
        Returns matrix-element dtype as a normalized string.

        Returns:
            str: Dtype name string.
        """
        return self._dtype_val

    @property
    def is_hermitian(self) -> bool:
        """Returns whether this operator is declared Hermitian."""
        return self._is_hermitian_val

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns operator metadata dictionary."""
        return self._metadata_dict

    @property
    def max_conn_size(self) -> int:
        """
        Returns a static upper bound on the number of connected states.

        For symbolic operators this is estimated from IR max-connection metadata and is
        mainly used for buffer sizing before lowering.
        """
        if hasattr(self, "estimate_max_conn_size"):
            return int(self.estimate_max_conn_size())
        return 1

    @abc.abstractmethod
    def to_ir(self) -> SymbolicOperatorIR:
        """Builds the symbolic action IR for this operator."""

    def get_conn_padded(self, x: Any) -> tuple[Any, Any]:
        """Raises until this operator has been compiled."""
        raise SymbolicOperatorExecutionError(
            f"Symbolic operator {self._name_val!r} cannot execute before "
            "compilation. Lower it through SymbolicCompiler.compile() first."
        )

    def __add__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import SumOperator

            lhs = self.compile()
            rhs = other.compile() if isinstance(other, AbstractSymbolicOperator) else other
            return SumOperator(lhs, rhs)
        return super().__add__(other)

    def __radd__(self, other):
        if _is_additive_identity(other):
            return self
        if isinstance(other, AbstractOperator):
            from netket.operator import SumOperator

            lhs = other.compile() if isinstance(other, AbstractSymbolicOperator) else other
            rhs = self.compile()
            return SumOperator(lhs, rhs)
        return super().__radd__(other)

    def __matmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import ProductOperator

            lhs = self.compile()
            rhs = other.compile() if isinstance(other, AbstractSymbolicOperator) else other
            return ProductOperator(lhs, rhs)
        return self.compile().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, AbstractOperator):
            from netket.operator import ProductOperator

            lhs = other.compile() if isinstance(other, AbstractSymbolicOperator) else other
            rhs = self.compile()
            return ProductOperator(lhs, rhs)
        return self.compile().__rmatmul__(other)

    def _apply_scalar(self, scalar: "int | float | complex") -> "AbstractSymbolicOperator":
        """
        Returns a new operator whose matrix elements are all multiplied by *scalar*.

        Subclasses override this to perform the actual term-level amplitude scaling.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _apply_scalar.")

    def apply_scalar(self, scalar: "int | float | complex") -> "AbstractSymbolicOperator":
        """
        Returns a scaled copy of this operator.

        This public facade allows external modules to trigger scaling without
        relying on the protected :meth:`_apply_scalar` hook.

        Args:
            scalar: Real or complex scale factor.

        Returns:
            AbstractSymbolicOperator: Scaled operator.
        """
        return self._apply_scalar(scalar)

    def __mul__(self, scalar: Any) -> "AbstractSymbolicOperator":
        """
        Scales all matrix elements by *scalar*.

        Args:
            scalar: Real or complex numeric factor.

        Returns:
            New operator of the same concrete type with scaled amplitudes.
        """
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        return self._apply_scalar(scalar)

    def __rmul__(self, scalar: Any) -> "AbstractSymbolicOperator":
        return self.__mul__(scalar)

    def __neg__(self) -> "AbstractSymbolicOperator":
        return self.__mul__(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self._name_val!r}, "
            f"dtype={self._dtype_val!r}, "
            f"is_hermitian={self._is_hermitian_val}, "
            f"hilbert={self.hilbert!r})"
        )


__all__ = ["AbstractSymbolicOperator"]
