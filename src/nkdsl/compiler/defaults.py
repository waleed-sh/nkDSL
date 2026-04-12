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


"""Default pipeline and registry factories for the symbolic compiler."""

from __future__ import annotations

from nkdsl.compiler.cache.store import (
    InMemorySymbolicArtifactStore,
)
from nkdsl.compiler.core.pipeline import (
    SymbolicPassPipeline,
)
from nkdsl.compiler.lowering.jax_lowerer import (
    JAXSymbolicLowerer,
)
from nkdsl.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringRegistry,
    build_default_symbolic_operator_lowering_registry,
)
from nkdsl.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from nkdsl.compiler.passes.analysis import (
    SymbolicMaxConnSizeAnalysisPass,
)
from nkdsl.compiler.passes.fusion import (
    SymbolicFusionPass,
)
from nkdsl.compiler.passes.normalization import (
    SymbolicNormalizationPass,
)
from nkdsl.compiler.passes.validation import (
    SymbolicValidationPass,
)

# Module-level shared in-memory store (one per process)
_DEFAULT_STORE: InMemorySymbolicArtifactStore | None = None
_DEFAULT_OPERATOR_LOWERING_REGISTRY: SymbolicOperatorLoweringRegistry | None = None


def default_symbolic_pass_pipeline() -> SymbolicPassPipeline:
    """
    Builds the default two-stage symbolic compiler pass pipeline.

    **Pre-cache passes** (run on every :meth:`compile` call):
        1. :class:`~nkdsl.compiler.passes.validation.SymbolicValidationPass`
           - validates IR symbol scopes and update-op parameters.
        2. :class:`~nkdsl.compiler.passes.normalization.SymbolicNormalizationPass`
           - computes the IR fingerprint and resolves the target backend.

    **Post-cache passes** (run only on cache misses):
        1. :class:`~nkdsl.compiler.passes.analysis.SymbolicMaxConnSizeAnalysisPass`
           - derives per-term max-connection-size bounds and the total padded output size.
        2. :class:`~nkdsl.compiler.passes.fusion.SymbolicFusionPass`
           - groups terms into fusion-compatible clusters for the lowerer.

    Returns:
        Configured :class:`~nkdsl.compiler.core.pipeline.SymbolicPassPipeline`.
    """
    return SymbolicPassPipeline(
        pre_cache_passes=[
            SymbolicValidationPass(),
            SymbolicNormalizationPass(),
        ],
        post_cache_passes=[
            SymbolicMaxConnSizeAnalysisPass(),
            SymbolicFusionPass(),
        ],
    )


def default_symbolic_lowerer_registry(
    *,
    operator_lowering_registry: SymbolicOperatorLoweringRegistry | None = None,
) -> SymbolicLowererRegistry:
    """
    Builds the default symbolic lowerer registry.

    Currently registers only the JAX backend lowerer
    (:class:`~nkdsl.compiler.lowering.jax_lowerer.JAXSymbolicLowerer`).
    The lowerer is configured with *operator_lowering_registry*.

    Returns:
        Configured :class:`~nkdsl.compiler.lowering.registry.SymbolicLowererRegistry`.
    """
    resolved_targets = operator_lowering_registry or default_symbolic_operator_lowering_registry()
    registry = SymbolicLowererRegistry()
    registry.register(
        JAXSymbolicLowerer(
            operator_lowering_registry=resolved_targets,
        )
    )
    return registry


def default_symbolic_operator_lowering_registry() -> SymbolicOperatorLoweringRegistry:
    """
    Returns the module-level shared operator-lowering target registry.

    The default registry contains the NetKet discrete JAX target
    (``"netket_discrete_jax"`` -> ``DiscreteJaxOperator.get_conn_padded``).
    """
    global _DEFAULT_OPERATOR_LOWERING_REGISTRY  # noqa: PLW0603
    if _DEFAULT_OPERATOR_LOWERING_REGISTRY is None:
        _DEFAULT_OPERATOR_LOWERING_REGISTRY = build_default_symbolic_operator_lowering_registry()
    return _DEFAULT_OPERATOR_LOWERING_REGISTRY


def default_symbolic_artifact_store() -> InMemorySymbolicArtifactStore:
    """
    Returns the module-level shared in-memory artifact store.

    The store is lazily created and reused across compiler instances in the
    same process. Call :meth:`InMemorySymbolicArtifactStore.clear` to
    evict all compiled artifacts if needed.

    Returns:
        Shared :class:`~nkdsl.compiler.cache.store.InMemorySymbolicArtifactStore`.
    """
    global _DEFAULT_STORE  # noqa: PLW0603
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = InMemorySymbolicArtifactStore()
    return _DEFAULT_STORE


def reset_default_symbolic_singletons() -> None:
    """
    Resets module-level default singleton instances.

    This helper primarily exists for tests that need deterministic singleton
    lifecycle behavior across cases.
    """
    global _DEFAULT_STORE  # noqa: PLW0603
    global _DEFAULT_OPERATOR_LOWERING_REGISTRY  # noqa: PLW0603
    _DEFAULT_STORE = None
    _DEFAULT_OPERATOR_LOWERING_REGISTRY = None


__all__ = [
    "default_symbolic_pass_pipeline",
    "default_symbolic_lowerer_registry",
    "default_symbolic_operator_lowering_registry",
    "default_symbolic_artifact_store",
    "reset_default_symbolic_singletons",
]
