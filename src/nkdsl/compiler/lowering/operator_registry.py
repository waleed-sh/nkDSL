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
import threading
import warnings
from collections.abc import Callable
from typing import Any

from netket.operator import DiscreteJaxOperator

from nkdsl.debug import event as debug_event
from nkdsl.configs import cfg
from nkdsl.warnings import RegistrationWarning

DEFAULT_SYMBOLIC_OPERATOR_LOWERING: str = "netket_discrete_jax"


def _normalize_target_name(name: str) -> str:
    """Normalizes built-in aliases while preserving custom target names."""
    normalized = str(name).strip()
    if normalized.lower() in {"netket", DEFAULT_SYMBOLIC_OPERATOR_LOWERING}:
        return DEFAULT_SYMBOLIC_OPERATOR_LOWERING
    if normalized.lower() == "neuralqx":
        return "neuralqx"
    return normalized


def _neuralqx_target() -> SymbolicOperatorLoweringTarget:
    """Loads the optional neuraLQX target only when it is selected."""
    try:
        from neuralqx.operators.types import ComputationalJaxOperator
    except ModuleNotFoundError as exc:
        if exc.name != "neuralqx":
            raise
        raise ImportError(
            "The 'neuralqx' lowering requires neuraLQX. "
            'Install it with pip install "nkdsl[neuralqx]".'
        ) from exc
    return SymbolicOperatorLoweringTarget(
        name="neuralqx",
        operator_type=ComputationalJaxOperator,
        connection_method="_get_conn_padded",
    )


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
        normalized_name = _normalize_target_name(self.name)
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

    @property
    def cache_signature(self) -> tuple:
        """Identifies the target class and entry point for in-process caching."""
        return (
            self.name,
            self.operator_type_qualname,
            self.connection_method,
            id(self.operator_type),
        )

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

    Args:
        targets: Initial target descriptors.
        default_target: Initial default target name.
        default_from_config: Whether defaults follow ``nkdsl.config.LOWERING``.
            The shared :data:`nkdsl.LOWERING_REGISTRY` enables this behavior and
            selects newly registered targets by default. Independent registries
            retain their own default unless ``set_default=True`` is supplied.
    """

    def __init__(
        self,
        targets: list[SymbolicOperatorLoweringTarget] | None = None,
        *,
        default_target: str | None = None,
        default_from_config: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._targets: dict[str, SymbolicOperatorLoweringTarget] = {}
        self._lazy_targets: dict[str, Callable[[], SymbolicOperatorLoweringTarget]] = {}
        self._default_target: str | None = None
        self._default_from_config = bool(default_from_config)

        for target in targets or ():
            self.register_target(target, set_default=False)

        if default_target is not None:
            self.set_default(default_target)
        elif targets:
            self._default_target = targets[0].name

    def register_target(
        self,
        target: SymbolicOperatorLoweringTarget,
        *,
        replace: bool = False,
        set_default: bool | None = None,
    ) -> None:
        """
        Registers one lowering target.

        Args:
            target: Target descriptor.
            replace: Whether to replace an existing target with the same name.
                Otherwise, a duplicate emits :class:`nkdsl.RegistrationWarning`
                and keeps both the existing target and the current default.
            set_default: Whether this target becomes the default. ``None``
                selects new targets in the shared registry and leaves the
                default unchanged in independent registries.
        """
        if not isinstance(target, SymbolicOperatorLoweringTarget):
            raise TypeError(
                "target must be a SymbolicOperatorLoweringTarget, " f"got {type(target)!r}."
            )
        with self._lock:
            if target.name in self.target_names and not replace:
                warnings.warn(
                    f"Lowering target {target.name!r} is already registered. "
                    "Keeping the existing target and default. "
                    "Use replace=True to use the new implementation.",
                    RegistrationWarning,
                    stacklevel=3,
                )
                return
            self._targets[target.name] = target
            self._lazy_targets.pop(target.name, None)
            if self._default_target is None:
                self._default_target = target.name
            select_default = self._default_from_config if set_default is None else set_default
            if select_default:
                self.set_default(target.name)
        debug_event(
            "registered operator lowering target",
            scope="lowering",
            tag="LOWERING",
            target_name=target.name,
            operator_type=target.operator_type_qualname,
            connection_method=target.connection_method,
            set_default=bool(select_default),
        )

    def register(
        self,
        *,
        name: str,
        operator_type: type[Any],
        connection_method: str,
        replace: bool = False,
        set_default: bool | None = None,
    ) -> None:
        """Registers a target and selects it by default in the shared registry.

        Args:
            name: Unique target name.
            operator_type: Operator base class used by the generated wrapper.
            connection_method: Method receiving the compiled connectivity kernel.
            replace: Whether an existing implementation should be replaced.
                Otherwise, duplicates warn and keep the existing registration.
            set_default: Whether to select this target for later compilations.
                Defaults to ``True`` for :data:`nkdsl.LOWERING_REGISTRY` and
                ``False`` for independent registries.
        """
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
        """Selects a registered target, synchronizing the shared configuration.

        Selecting a default again is allowed. The shared registry emits
        :class:`nkdsl.LoweringWarning` when a script assigns its default again.
        Optional dependencies are loaded when the target is resolved.
        """
        normalized = _normalize_target_name(name)
        with self._lock:
            if normalized not in self.target_names:
                raise KeyError(
                    f"Unknown lowering target {normalized!r}. "
                    f"Available targets: {list(self.target_names)!r}."
                )
            if self._default_from_config:
                cfg.LOWERING = (
                    "netket" if normalized == DEFAULT_SYMBOLIC_OPERATOR_LOWERING else normalized
                )
            else:
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
        with self._lock:
            if not self.target_names:
                raise RuntimeError("No symbolic operator lowering targets are registered.")
            target_name = self.default_target if name is None else _normalize_target_name(name)
            if not target_name:
                raise ValueError("Lowering target name must be a non-empty string.")
            target = self._targets.get(target_name)
            if target is None and target_name in self._lazy_targets:
                target = self._lazy_targets[target_name]()
                self._targets[target_name] = target
            if target is None:
                raise KeyError(
                    f"Unknown lowering target {target_name!r}. "
                    f"Available targets: {list(self.target_names)!r}."
                )
            return target

    @property
    def default_target(self) -> str | None:
        """Returns the configured default target name."""
        if self._default_from_config:
            return _normalize_target_name(cfg.LOWERING)
        return self._default_target

    @property
    def target_names(self) -> tuple[str, ...]:
        """Returns target names sorted lexicographically."""
        with self._lock:
            return tuple(sorted(self._targets.keys() | self._lazy_targets.keys()))

    def __len__(self) -> int:
        return len(self.target_names)

    def __repr__(self) -> str:
        return (
            f"SymbolicOperatorLoweringRegistry("
            f"default_target={self.default_target!r}, "
            f"targets={self.target_names!r})"
        )


def build_default_symbolic_operator_lowering_registry(
    *, default_from_config: bool = False
) -> SymbolicOperatorLoweringRegistry:
    """
    Builds a registry with NetKet and optional neuraLQX JAX targets.

    NetKet accepts both ``"netket"`` and the legacy name
    :data:`DEFAULT_SYMBOLIC_OPERATOR_LOWERING`. The ``"neuralqx"`` target
    imports neuraLQX only when selected for compilation.

    Args:
        default_from_config: Whether the registry follows ``config.LOWERING``
            and selects newly registered custom targets automatically.
    """
    registry = SymbolicOperatorLoweringRegistry(default_from_config=default_from_config)
    registry.register(
        name=DEFAULT_SYMBOLIC_OPERATOR_LOWERING,
        operator_type=DiscreteJaxOperator,
        connection_method="get_conn_padded",
        set_default=False,
    )
    registry._lazy_targets["neuralqx"] = _neuralqx_target
    return registry


__all__ = [
    "DEFAULT_SYMBOLIC_OPERATOR_LOWERING",
    "SymbolicOperatorLoweringTarget",
    "SymbolicOperatorLoweringRegistry",
    "build_default_symbolic_operator_lowering_registry",
]
