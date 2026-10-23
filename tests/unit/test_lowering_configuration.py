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


"""Exercise lowering selection through public compilation and registration APIs."""

from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl
from nkdsl.compiler.cache.store import InMemorySymbolicArtifactStore
from nkdsl.compiler.core.pipeline import SymbolicPassPipeline
from nkdsl.compiler.defaults import reset_default_symbolic_singletons
from nkdsl.compiler.lowering.operator_registry import (
    build_default_symbolic_operator_lowering_registry,
)
from nkdsl.configs import ConfigValidationError
from nkdsl.compiler.passes.base import AbstractSymbolicPass

pytestmark = pytest.mark.unit


class ComputationalLike:
    """Minimal target with the same connection entry point as neuraLQX."""

    def __init__(self, hilbert):
        self.hilbert = hilbert

    def get_conn_padded(self, x):
        return self._get_conn_padded(x)


class OtherComputationalLike(ComputationalLike):
    """A distinct target used to check changes and replacement."""


@pytest.fixture(autouse=True)
def isolated_lowering(monkeypatch):
    """Keep tests independent of process-global configuration and registrations."""
    registry = nkdsl.LOWERING_REGISTRY
    monkeypatch.setattr(registry, "_targets", registry._targets.copy())
    monkeypatch.setattr(registry, "_lazy_targets", registry._lazy_targets.copy())
    monkeypatch.setattr(registry, "_default_target", registry._default_target)
    with nkdsl.config.patch(LOWERING="netket"):
        yield


def builder():
    """A physical diagonal operator with a known matrix on all basis states."""
    hi = nk.hilbert.Fock(n_max=2, N=2)
    return (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "number", hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.identity(), matrix_element=nkdsl.site("i").value)
    )


def register_target(name="custom", operator_type=ComputationalLike, **kwargs):
    nkdsl.LOWERING_REGISTRY.register(
        name=name,
        operator_type=operator_type,
        connection_method="_get_conn_padded",
        **kwargs,
    )


def check_kernel(operator):
    """Check generated connectivity and pytree behavior, not just its class."""
    states = jnp.asarray([[0, 0], [1, 2], [2, 1]], dtype=jnp.int32)
    leaves, structure = jax.tree_util.tree_flatten(operator)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    xp, mels = jax.jit(lambda op, x: op.get_conn_padded(x))(restored, states)
    np.testing.assert_array_equal(xp, np.asarray(states)[:, None, :])
    np.testing.assert_allclose(mels, [[0], [3], [3]])


@pytest.mark.parametrize("name", ["netket", "netket_discrete_jax", " NetKet "])
def test_builtin_netket_aliases_share_a_target_and_cached_artifact(name):
    assert nkdsl.config is nkdsl.cfg
    assert nkdsl.LOWERING_REGISTRY is nkdsl.default_symbolic_operator_lowering_registry()
    registry = nkdsl.LOWERING_REGISTRY
    assert "neuralqx" in registry.target_names
    assert registry.resolve(name) is registry.resolve("netket")
    compiler = nkdsl.SymbolicCompiler(artifact_store=InMemorySymbolicArtifactStore())
    symbolic = builder().build()
    canonical = compiler.compile(
        symbolic, options=nkdsl.SymbolicCompilerOptions(operator_lowering="netket")
    )
    alias = compiler.compile(
        symbolic, options=nkdsl.SymbolicCompilerOptions(operator_lowering=name)
    )
    assert canonical is alias
    assert isinstance(alias.compiled_operator, nk.operator.DiscreteJaxOperator)
    check_kernel(alias.compiled_operator)


@pytest.mark.parametrize("entry", ["builder", "operator", "compiler", "convenience"])
def test_custom_registration_selects_default_for_every_compile_entry(entry):
    symbolic_builder = builder()
    compiler = nkdsl.SymbolicCompiler()
    with pytest.warns(nkdsl.LoweringWarning):
        register_target()
    assert nkdsl.config.LOWERING == "custom"
    if entry == "builder":
        compiled = symbolic_builder.compile()
    elif entry == "operator":
        compiled = symbolic_builder.build().compile()
    elif entry == "compiler":
        compiled = compiler.compile_operator(symbolic_builder.build())
    else:
        compiled = nkdsl.compile_symbolic_operator(symbolic_builder.build())
    assert isinstance(compiled, ComputationalLike)
    check_kernel(compiled)


def test_existing_compiler_and_options_follow_changes_without_stale_cache():
    register_target(set_default=False)
    options = nkdsl.SymbolicCompilerOptions.from_mapping()
    compiler = nkdsl.SymbolicCompiler(
        options=options, artifact_store=InMemorySymbolicArtifactStore()
    )
    symbolic = builder().build()
    original = compiler.compile(symbolic)
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.config.LOWERING = "custom"
    custom = compiler.compile(symbolic)
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.config.LOWERING = "netket"
    assert compiler.compile(symbolic) is original
    assert compiler.compile(symbolic, options=options) is original
    assert isinstance(custom.compiled_operator, ComputationalLike)
    assert original.cache_key != custom.cache_key
    assert options.operator_lowering is None
    check_kernel(original.compiled_operator)
    check_kernel(custom.compiled_operator)


def test_explicit_overrides_do_not_change_global_or_compiler_default():
    register_target(set_default=False)
    symbolic = builder().build()
    compiler = nkdsl.SymbolicCompiler(operator_lowering="custom")
    assert isinstance(symbolic.compile(compiler=compiler), ComputationalLike)
    # Unrelated invocation options must not unpin the compiler's target.
    invocation = nkdsl.SymbolicCompilerOptions(cache_enabled=False)
    assert isinstance(compiler.compile_operator(symbolic, options=invocation), ComputationalLike)
    invocation = nkdsl.SymbolicCompilerOptions(operator_lowering="netket")
    assert not isinstance(
        compiler.compile_operator(symbolic, options=invocation), ComputationalLike
    )
    overridden = symbolic.compile(compiler=compiler, operator_lowering="netket")
    assert isinstance(overridden, nk.operator.DiscreteJaxOperator)
    assert not isinstance(overridden, ComputationalLike)
    assert compiler.options.operator_lowering == "custom"
    assert nkdsl.config.LOWERING == "netket"
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.config.LOWERING = "custom"
    check_kernel(builder().compile(operator_lowering="netket"))
    assert nkdsl.config.LOWERING == "custom"
    assert isinstance(symbolic.compile(compiler=compiler), ComputationalLike)


def test_registering_again_keeps_existing_target_and_current_default_until_replaced():
    register_target(set_default=False)
    with pytest.warns(nkdsl.RegistrationWarning, match="replace=True"):
        register_target(operator_type=OtherComputationalLike)
    assert nkdsl.LOWERING_REGISTRY.resolve("custom").operator_type is ComputationalLike
    assert nkdsl.config.LOWERING == "netket"
    with pytest.warns(nkdsl.LoweringWarning):
        register_target(operator_type=OtherComputationalLike, replace=True)
    assert nkdsl.config.LOWERING == "custom"
    compiled = builder().compile()
    assert isinstance(compiled, OtherComputationalLike)
    check_kernel(compiled)


def test_replacement_changes_cache_even_when_class_qualnames_are_identical():
    first = type("RepeatedTarget", (ComputationalLike,), {})
    second = type("RepeatedTarget", (ComputationalLike,), {})
    assert first.__qualname__ == second.__qualname__
    assert first.__module__ == second.__module__
    register_target(operator_type=first, set_default=False)
    compiler = nkdsl.SymbolicCompiler(
        operator_lowering="custom", artifact_store=InMemorySymbolicArtifactStore()
    )
    symbolic = builder().build()
    original = compiler.compile(symbolic)
    register_target(operator_type=second, replace=True, set_default=False)
    replacement = compiler.compile(symbolic)
    assert isinstance(original.compiled_operator, first)
    assert isinstance(replacement.compiled_operator, second)
    assert original.cache_key != replacement.cache_key
    assert compiler.compile(symbolic) is replacement
    check_kernel(original.compiled_operator)
    check_kernel(replacement.compiled_operator)


def test_cache_separates_connection_methods_and_independent_registries():
    store = InMemorySymbolicArtifactStore()
    symbolic = builder().build()
    artifacts = []
    for cls, method in [
        (ComputationalLike, "_get_conn_padded"),
        (OtherComputationalLike, "_get_conn_padded"),
        (ComputationalLike, "get_conn_padded"),
    ]:
        registry = nkdsl.SymbolicOperatorLoweringRegistry()
        registry.register(name="same", operator_type=cls, connection_method=method)
        artifacts.append(
            nkdsl.SymbolicCompiler(
                operator_lowering_registry=registry, artifact_store=store
            ).compile(symbolic)
        )
    assert len({artifact.cache_key for artifact in artifacts}) == 3
    assert len(store) == 3
    for artifact in artifacts:
        check_kernel(artifact.compiled_operator)


def test_target_is_bound_before_passes_even_if_configuration_and_registration_change():
    register_target(set_default=False)
    changed = False

    class ChangeTarget(AbstractSymbolicPass):
        @property
        def name(self):
            return "change_target"

        def run(self, context):
            nonlocal changed
            if not changed:
                with pytest.warns(nkdsl.LoweringWarning):
                    nkdsl.config.LOWERING = "custom"
                register_target(
                    operator_type=OtherComputationalLike, replace=True, set_default=False
                )
                changed = True

    defaults = nkdsl.default_symbolic_pass_pipeline()
    pipeline = SymbolicPassPipeline(
        pre_cache_passes=[ChangeTarget(), *defaults.pre_cache_passes],
        post_cache_passes=defaults.post_cache_passes,
    )
    compiler = nkdsl.SymbolicCompiler(
        operator_lowering="custom",
        artifact_store=InMemorySymbolicArtifactStore(),
        pipeline=pipeline,
    )
    symbolic = builder().build()
    original = compiler.compile(symbolic)
    assert isinstance(original.compiled_operator, ComputationalLike)
    assert not isinstance(original.compiled_operator, OtherComputationalLike)
    replacement = compiler.compile(symbolic)
    assert isinstance(replacement.compiled_operator, OtherComputationalLike)
    assert original.cache_key != replacement.cache_key
    check_kernel(original.compiled_operator)
    check_kernel(replacement.compiled_operator)


def test_registry_selection_and_config_are_one_default():
    register_target(set_default=False)
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.LOWERING_REGISTRY.set_default("custom")
    assert nkdsl.config.LOWERING == "custom"
    assert isinstance(builder().compile(), ComputationalLike)
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.config.LOWERING = "NetKet"
    assert nkdsl.config.LOWERING == "netket"
    assert nkdsl.LOWERING_REGISTRY.default_target == nkdsl.DEFAULT_SYMBOLIC_OPERATOR_LOWERING
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.LOWERING_REGISTRY.set_default("netket")
    check_kernel(builder().compile())


def test_first_assignment_is_quiet_and_subsequent_assignments_warn():
    nkdsl.config.clear_override("LOWERING")
    with pytest.warns() as recorded:
        nkdsl.config.LOWERING = "netket"
        nkdsl.config.LOWERING = "netket"
    assert len(recorded) == 1
    assert recorded[0].category is nkdsl.LoweringWarning
    with pytest.warns(nkdsl.LoweringWarning, match="subsequent compilations"):
        nkdsl.config.LOWERING = "neuralqx"
    assert nkdsl.config.LOWERING == "neuralqx"


@pytest.mark.parametrize("value", [None, "", "  ", 42, False])
def test_invalid_config_values_do_not_change_the_selected_target(value):
    with pytest.raises(ConfigValidationError, match="non-empty string"):
        nkdsl.config.LOWERING = value
    assert nkdsl.config.LOWERING == "netket"


def test_unknown_target_has_an_actionable_error_and_does_not_poison_future_compiles():
    with pytest.warns(nkdsl.LoweringWarning):
        nkdsl.config.LOWERING = "missing_target"
    with pytest.raises(
        nkdsl.SymbolicCompilerError, match="Unknown lowering target.*missing_target"
    ):
        builder().compile()
    check_kernel(builder().compile(operator_lowering="netket"))
    assert nkdsl.config.LOWERING == "missing_target"


def test_optional_neuralqx_is_loaded_only_when_resolved_and_can_be_retried(monkeypatch):
    registry = build_default_symbolic_operator_lowering_registry()
    original_import = builtins.__import__
    attempts = []

    def missing_neuralqx(name, *args, **kwargs):
        if name.startswith("neuralqx"):
            attempts.append(name)
            raise ModuleNotFoundError("No module named 'neuralqx'", name="neuralqx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_neuralqx)
    assert "neuralqx" in registry.target_names
    assert len(registry) == 2
    assert "neuralqx" in repr(registry)
    registry.set_default("neuralqx")
    registry.resolve("netket")
    assert attempts == []
    for _ in range(2):
        with pytest.raises(ImportError, match=r"nkdsl\[neuralqx\]"):
            registry.resolve()
    assert len(attempts) == 2
    with pytest.warns(nkdsl.RegistrationWarning):
        registry.register(
            name="neuralqx", operator_type=ComputationalLike, connection_method="_get_conn_padded"
        )
    registry.register(
        name="neuralqx",
        operator_type=ComputationalLike,
        connection_method="_get_conn_padded",
        replace=True,
    )
    assert registry.resolve().operator_type is ComputationalLike


def test_missing_neuralqx_dependency_is_not_misreported_as_missing_neuralqx(monkeypatch):
    original_import = builtins.__import__

    def broken_dependency(name, *args, **kwargs):
        if name.startswith("neuralqx"):
            raise ModuleNotFoundError("No module named 'dependency'", name="dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_dependency)
    registry = build_default_symbolic_operator_lowering_registry()
    with pytest.raises(ModuleNotFoundError, match="dependency") as failure:
        registry.resolve("neuralqx")
    assert failure.value.name == "dependency"


def test_global_default_is_runtime_mutable_and_thread_local_overrides_are_isolated():
    register_target(set_default=False)
    nkdsl.config.lock_runtime()
    try:
        with pytest.warns(nkdsl.LoweringWarning):
            nkdsl.config.LOWERING = "custom"
        registry = nkdsl.LOWERING_REGISTRY
        with nkdsl.config.thread_local_override("LOWERING", "netket"):
            assert registry.resolve().operator_type is nk.operator.DiscreteJaxOperator
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert (
                    pool.submit(lambda: registry.resolve().operator_type).result()
                    is ComputationalLike
                )
        assert registry.resolve().operator_type is ComputationalLike
    finally:
        nkdsl.config.unlock_runtime_for_testing()


def test_independent_registry_default_does_not_modify_global_configuration():
    registry = nkdsl.SymbolicOperatorLoweringRegistry()
    registry.register(
        name="a", operator_type=ComputationalLike, connection_method="_get_conn_padded"
    )
    registry.register(
        name="b", operator_type=OtherComputationalLike, connection_method="_get_conn_padded"
    )
    assert registry.default_target == "a"
    registry.set_default("b")
    assert registry.resolve().operator_type is OtherComputationalLike
    assert nkdsl.config.LOWERING == "netket"
    check_kernel(
        nkdsl.SymbolicCompiler(operator_lowering_registry=registry).compile_operator(
            builder().build()
        )
    )


@pytest.mark.parametrize(
    "default,expected", [(None, ComputationalLike), ("b", OtherComputationalLike)]
)
def test_initial_target_descriptors_preserve_independent_registry_defaults(default, expected):
    targets = [
        nkdsl.SymbolicOperatorLoweringTarget("a", ComputationalLike, "_get_conn_padded"),
        nkdsl.SymbolicOperatorLoweringTarget("b", OtherComputationalLike, "_get_conn_padded"),
    ]
    registry = nkdsl.SymbolicOperatorLoweringRegistry(targets, default_target=default)
    assert registry.resolve().operator_type is expected
    assert nkdsl.config.LOWERING == "netket"
    check_kernel(
        nkdsl.SymbolicCompiler(operator_lowering_registry=registry).compile_operator(
            builder().build()
        )
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("name", " ", ValueError),
        ("operator_type", object(), TypeError),
        ("connection_method", " ", ValueError),
        ("connection_method", "invalid-method", ValueError),
    ],
)
def test_invalid_replacement_leaves_the_existing_target_and_default_usable(field, value, error):
    register_target(set_default=False)
    arguments = dict(
        name="custom",
        operator_type=OtherComputationalLike,
        connection_method="_get_conn_padded",
        replace=True,
    )
    arguments[field] = value
    with pytest.raises(error):
        nkdsl.LOWERING_REGISTRY.register(**arguments)
    assert nkdsl.config.LOWERING == "netket"
    assert nkdsl.LOWERING_REGISTRY.resolve("custom").operator_type is ComputationalLike
    check_kernel(builder().compile(operator_lowering="custom"))


def test_invalid_registry_calls_preserve_the_current_selection():
    registry = nkdsl.LOWERING_REGISTRY
    with pytest.raises(TypeError, match="SymbolicOperatorLoweringTarget"):
        registry.register_target(object())
    with pytest.raises(KeyError, match="Unknown lowering target"):
        registry.set_default("not_registered")
    with pytest.raises(ValueError, match="non-empty string"):
        registry.resolve(" ")
    assert nkdsl.config.LOWERING == "netket"
    check_kernel(builder().compile())


def test_empty_explicit_registry_is_not_silently_replaced_by_the_global_registry():
    compiler = nkdsl.SymbolicCompiler(
        operator_lowering_registry=nkdsl.SymbolicOperatorLoweringRegistry()
    )
    with pytest.raises(nkdsl.SymbolicCompilerError, match="No symbolic operator lowering targets"):
        compiler.compile(builder().build())


def test_reset_keeps_public_registry_identity_and_restores_builtins():
    registry = nkdsl.LOWERING_REGISTRY
    register_target(set_default=False)
    reset_default_symbolic_singletons()
    assert nkdsl.LOWERING_REGISTRY is registry
    assert nkdsl.default_symbolic_operator_lowering_registry() is registry
    assert registry.target_names == ("netket_discrete_jax", "neuralqx")
    assert registry.resolve().operator_type is nk.operator.DiscreteJaxOperator


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("NKDSL_LOWERING", " neuralqx ", "neuralqx"),
        ("nkdsl_lowering", "NetKet", "netket"),
        ("NKDSL_LOWERING", "CustomTarget", "CustomTarget"),
    ],
)
def test_environment_selects_the_default_before_registration(monkeypatch, name, value, expected):
    from nkdsl.configs import ConfigManager

    monkeypatch.delenv("NKDSL_LOWERING", raising=False)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(ConfigManager, "_instance", None)
    manager = ConfigManager()
    assert manager.LOWERING == expected
    assert manager.is_runtime_mutable("LOWERING")


@pytest.mark.parametrize(
    "factory", [nkdsl.SymbolicCompilerOptions, nkdsl.SymbolicCompilerOptions.from_mapping]
)
@pytest.mark.parametrize("value", ["", "  ", 1, False])
def test_invalid_explicit_lowering_is_rejected(factory, value):
    with pytest.raises(ValueError, match="non-empty string"):
        factory(operator_lowering=value)
