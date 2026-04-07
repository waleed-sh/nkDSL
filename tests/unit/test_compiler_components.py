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


from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import netket as nk
import pytest

import nkdsl
from nkdsl.compiler.cache.store import InMemorySymbolicArtifactStore
from nkdsl.compiler.compiler import SymbolicCompiler
from nkdsl.compiler.compiler import compile_symbolic_operator
from nkdsl.compiler.core.artifact import SymbolicCompiledArtifact
from nkdsl.compiler.core.context import SymbolicCompilationContext
from nkdsl.compiler.core.options import SymbolicCompilerOptions
from nkdsl.compiler.core.pass_report import SymbolicPassReport
from nkdsl.compiler.core.pipeline import SymbolicPassPipeline
from nkdsl.compiler.core.signature import SymbolicCompilationSignature
from nkdsl.compiler.lowering.base import AbstractSymbolicLowerer
from nkdsl.compiler.lowering.registry import SymbolicLowererRegistry
from nkdsl.compiler.passes.base import AbstractSymbolicPass
from nkdsl.errors import SymbolicCompilerError

pytestmark = pytest.mark.unit


def _make_symbolic_operator(name: str = "cmp"):
    hi = nk.hilbert.Fock(n_max=2, N=2)
    return (
        nkdsl.SymbolicDiscreteJaxOperator(hi, name, hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )


def _make_context() -> SymbolicCompilationContext:
    op = _make_symbolic_operator("ctx")
    return SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(),
        metadata={"source": "test"},
    )


def test_options_signature_and_validation():
    opts = SymbolicCompilerOptions.from_mapping(
        backend_preference="jax",
        enable_fusion=False,
        strict_validation=True,
        cache_enabled=True,
        cache_namespace="ns",
        debug_flags={"dump_ir": True},
    )
    sig = opts.static_signature()
    assert ("backend_preference", "jax") in sig
    assert opts.debug_flag_map()["dump_ir"] is True

    with pytest.raises(ValueError, match="Unsupported backend_preference"):
        SymbolicCompilerOptions(backend_preference="torch")

    with pytest.raises(ValueError, match="non-empty string"):
        SymbolicCompilerOptions(cache_namespace="  ")


def test_signature_and_cache_key_are_deterministic():
    ctx = _make_context()
    sig = SymbolicCompilationSignature.from_context(ctx)
    key_a = sig.build_cache_key(namespace="A")
    key_b = sig.build_cache_key(namespace="A")
    key_c = sig.build_cache_key(namespace="A", extension_context={"x": 1})

    assert key_a == key_b
    assert key_a != key_c
    assert str(key_a) == key_a.token


def test_artifact_store_get_put_invalidate_clear_and_eviction():
    store = InMemorySymbolicArtifactStore(max_entries=1)
    ctx = _make_context()
    sig = SymbolicCompilationSignature.from_context(ctx)
    key1 = sig.build_cache_key(namespace="T")
    key2 = sig.build_cache_key(namespace="T", extension_context={"other": 1})

    fake = object()
    art1 = SymbolicCompiledArtifact.create(
        operator_name="a",
        backend="jax",
        lowerer_name="L",
        compiled_operator=fake,
    )
    art2 = SymbolicCompiledArtifact.create(
        operator_name="b",
        backend="jax",
        lowerer_name="L",
        compiled_operator=fake,
    )

    assert store.get(key1) is None
    store.put(key1, art1)
    assert store.get(key1) is art1
    store.put(key2, art2)
    assert store.get(key1) is None
    assert store.get(key2) is art2
    assert store.invalidate(key2) is True
    assert store.invalidate(key2) is False
    store.clear()
    assert len(store) == 0


def test_pass_report_and_artifact_metadata_helpers():
    report = SymbolicPassReport.create(pass_name="p", duration_ms=1.23, metadata={"ok": True})
    assert report.metadata_map()["ok"] is True

    art = SymbolicCompiledArtifact.create(
        operator_name="x",
        backend="jax",
        lowerer_name="L",
        compiled_operator=object(),
        metadata={"a": 1},
    )
    assert art.metadata_map()["a"] == 1
    assert art.cache_token() is None


class _PassOk(AbstractSymbolicPass):
    @property
    def name(self) -> str:
        return "ok"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        context.set_analysis("ok", True)
        return {"written": True}


class _PassFail(AbstractSymbolicPass):
    @property
    def name(self) -> str:
        return "fail"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        raise RuntimeError("boom")


class _NoSupportLowerer(AbstractSymbolicLowerer):
    @property
    def name(self) -> str:
        return "none"

    @property
    def backend(self) -> str:
        return "none"

    def supports(self, context: SymbolicCompilationContext) -> bool:
        return False

    def lower(self, context: SymbolicCompilationContext) -> SymbolicCompiledArtifact:
        raise AssertionError("unreachable")


class _SupportLowerer(AbstractSymbolicLowerer):
    @property
    def name(self) -> str:
        return "support"

    @property
    def backend(self) -> str:
        return "jax"

    def supports(self, context: SymbolicCompilationContext) -> bool:
        return True

    def lower(self, context: SymbolicCompilationContext) -> SymbolicCompiledArtifact:
        op = context.operator.compile(cache=False)
        context.set_selected_lowerer(self.name)
        return SymbolicCompiledArtifact.create(
            operator_name=context.ir.operator_name,
            backend="jax",
            lowerer_name=self.name,
            compiled_operator=op,
            pass_reports=context.pass_reports,
            metadata={"lowered": True},
        )


def test_pipeline_and_registry_success_and_failure_paths():
    ctx = _make_context()
    pipeline = SymbolicPassPipeline(pre_cache_passes=[_PassOk()], post_cache_passes=[])
    pipeline.run_pre_cache(ctx)
    assert ctx.analysis("ok") is True
    assert pipeline.pass_names() == ("ok",)

    with pytest.raises(ValueError, match="at least one pre-cache pass"):
        SymbolicPassPipeline(pre_cache_passes=[], post_cache_passes=[])

    failing = _PassFail()
    with pytest.raises(RuntimeError, match="boom"):
        failing.execute(ctx)
    assert ctx.pass_reports[-1].pass_name == "fail"

    registry = SymbolicLowererRegistry()
    registry.register(_NoSupportLowerer())
    with pytest.raises(RuntimeError, match="No registered symbolic lowerer"):
        registry.resolve(ctx)

    registry.register_first(_SupportLowerer())
    resolved = registry.resolve(ctx)
    assert resolved.name == "support"
    assert "support" in registry.lowerer_names


def test_compiler_success_cache_and_errors():
    op = _make_symbolic_operator("compiler")

    compiler = SymbolicCompiler(cache_enabled=True)
    c1 = compiler.compile_operator(op)
    c2 = compiler.compile_operator(op)
    assert c1 is c2
    assert compiler.cache_size == 1

    artifact = compiler.compile(op)
    assert artifact.operator_name == "compiler"
    assert artifact.backend == "jax"

    class _BadOperator:
        pass

    with pytest.raises(SymbolicCompilerError, match="Failed to extract IR"):
        compiler.compile_operator(_BadOperator())

    direct = compile_symbolic_operator(op)
    assert hasattr(direct, "get_conn_padded")
