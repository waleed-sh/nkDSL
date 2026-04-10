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


"""Registry for compiled operator targets used by lowering backends."""

from __future__ import annotations

import dataclasses
from typing import Any

from netket.operator import DiscreteJaxOperator

from nkdsl.debug import event as debug_event

DEFAULT_SYMBOLIC_OPERATOR_LOWERING: str = "netket_discrete_jax"


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicOperatorLoweringTarget:
    """
    Describes one compiled-operator target selectable during lowering.

    Args:
        name: Unique target key used in compiler options.
        operator_type: Operator class instantiated by the lowerer.
        connection_method: Method name that should execute the compiled kernel
            (for example ``"get_conn_padded"`` or ``"_get_conn_padded"``).
    """

    name: str
    operator_type: type[Any]
    connection_method: str

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("Lowering target name must be a non-empty string.")
        if not isinstance(self.operator_type, type):
            raise TypeError(f"operator_type must be a class, got {type(self.operator_type)!r}.")
        normalized_method = str(self.connection_method).strip()
        if not normalized_method:
            raise ValueError("connection_method must be a non-empty string.")
        if not normalized_method.isidentifier():
            raise ValueError(
                f"connection_method {normalized_method!r} is not a valid Python identifier."
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "connection_method", normalized_method)

    @property
    def operator_type_qualname(self) -> str:
        """Fully qualified import path for the operator class."""
        return f"{self.operator_type.__module__}.{self.operator_type.__qualname__}"

    def __repr__(self) -> str:
        return (
            f"SymbolicOperatorLoweringTarget("
            f"name={self.name!r}, "
            f"operator_type={self.operator_type_qualname!r}, "
            f"connection_method={self.connection_method!r})"
        )


class SymbolicOperatorLoweringRegistry:
    """
    Registry for selectable compiled-operator targets.

    A target maps a public lowering name (for example
    ``"netket_discrete_jax"``) to an operator class and the method that
    should host the generated connectivity kernel.
    """

    def __init__(
        self,
        targets: list[SymbolicOperatorLoweringTarget] | None = None,
        *,
        default_target: str | None = None,
    ) -> None:
        self._targets: dict[str, SymbolicOperatorLoweringTarget] = {}
        self._default_target: str | None = None

        for target in targets or ():
            self.register_target(target)

        if default_target is not None:
            self.set_default(default_target)
        elif targets:
            self._default_target = targets[0].name

    def register_target(
        self,
        target: SymbolicOperatorLoweringTarget,
        *,
        replace: bool = False,
        set_default: bool = False,
    ) -> None:
        """
        Registers one lowering target.

        Args:
            target: Target descriptor.
            replace: Whether to replace an existing target with the same name.
            set_default: Whether this target becomes the default target.
        """
        if not isinstance(target, SymbolicOperatorLoweringTarget):
            raise TypeError(
                "target must be a SymbolicOperatorLoweringTarget, " f"got {type(target)!r}."
            )
        if target.name in self._targets and not replace:
            raise ValueError(
                f"Lowering target {target.name!r} is already registered. "
                "Use replace=True to overwrite."
            )
        self._targets[target.name] = target
        if self._default_target is None or set_default:
            self._default_target = target.name
        debug_event(
            "registered operator lowering target",
            scope="lowering",
            tag="LOWERING",
            target_name=target.name,
            operator_type=target.operator_type_qualname,
            connection_method=target.connection_method,
            set_default=bool(set_default),
        )

    def register(
        self,
        *,
        name: str,
        operator_type: type[Any],
        connection_method: str,
        replace: bool = False,
        set_default: bool = False,
    ) -> None:
        """Convenience wrapper around :meth:`register_target`."""
        self.register_target(
            SymbolicOperatorLoweringTarget(
                name=name,
                operator_type=operator_type,
                connection_method=connection_method,
            ),
            replace=replace,
            set_default=set_default,
        )

    def set_default(self, name: str) -> None:
        """Sets the default target name used by :meth:`resolve`."""
        normalized = str(name).strip()
        if normalized not in self._targets:
            raise KeyError(
                f"Unknown lowering target {normalized!r}. "
                f"Available targets: {sorted(self._targets)!r}."
            )
        self._default_target = normalized
        debug_event(
            "set default operator lowering target",
            scope="lowering",
            tag="LOWERING",
            target_name=normalized,
        )

    def resolve(self, name: str | None = None) -> SymbolicOperatorLoweringTarget:
        """
        Resolves one target by name.

        Args:
            name: Target name, or ``None`` to use the registry default.

        Returns:
            The resolved target.

        Raises:
            RuntimeError: If no targets are registered.
            KeyError: If *name* does not exist in the registry.
        """
        if not self._targets:
            raise RuntimeError("No symbolic operator lowering targets are registered.")

        target_name = self._default_target if name is None else str(name).strip()
        if not target_name:
            raise ValueError("Lowering target name must be a non-empty string.")
        target = self._targets.get(target_name)
        if target is None:
            raise KeyError(
                f"Unknown lowering target {target_name!r}. "
                f"Available targets: {sorted(self._targets)!r}."
            )
        return target

    @property
    def default_target(self) -> str | None:
        """Returns the configured default target name."""
        return self._default_target

    @property
    def target_names(self) -> tuple[str, ...]:
        """Returns target names sorted lexicographically."""
        return tuple(sorted(self._targets))

    def __len__(self) -> int:
        return len(self._targets)

    def __repr__(self) -> str:
        return (
            f"SymbolicOperatorLoweringRegistry("
            f"default_target={self._default_target!r}, "
            f"targets={self.target_names!r})"
        )


def build_default_symbolic_operator_lowering_registry() -> SymbolicOperatorLoweringRegistry:
    """
    Builds a registry with the default NetKet discrete JAX target.

    The default lowering target name is :data:`DEFAULT_SYMBOLIC_OPERATOR_LOWERING`.
    """
    registry = SymbolicOperatorLoweringRegistry()
    registry.register(
        name=DEFAULT_SYMBOLIC_OPERATOR_LOWERING,
        operator_type=DiscreteJaxOperator,
        connection_method="get_conn_padded",
        set_default=True,
    )
    return registry


__all__ = [
    "DEFAULT_SYMBOLIC_OPERATOR_LOWERING",
    "SymbolicOperatorLoweringTarget",
    "SymbolicOperatorLoweringRegistry",
    "build_default_symbolic_operator_lowering_registry",
]
