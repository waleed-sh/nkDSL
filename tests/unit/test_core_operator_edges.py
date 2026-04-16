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

import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest
from netket.operator._prod import ProductOperator
from netket.operator._sum import SumOperator

import nkdsl
from nkdsl.core.base import AbstractSymbolicOperator
from nkdsl.core.compiled import CompiledOperator
from nkdsl.core.operator import SymbolicOperator
from nkdsl.core.operator import _merge_metadata
from nkdsl.core.operator import _promote_dtype_for_scalar
from nkdsl.core.operator import _resolve_dtype
from nkdsl.core.operator import _scalar_str
from nkdsl.core.sum import SymbolicOperatorSum
from nkdsl.ir.program import SymbolicOperatorIR
from nkdsl.ir.term import KBodyIteratorSpec
from nkdsl.ir.term import SymbolicIRTerm

pytestmark = pytest.mark.unit


def _compiled_identity(hi, name: str = "compiled") -> CompiledOperator:
    def fn(x):
        if x.ndim == 1:
            return jnp.expand_dims(x, 0), jnp.ones((1,), dtype=jnp.float32)
        batch = x.shape[0]
        return jnp.expand_dims(x, 1), jnp.ones((batch, 1), dtype=jnp.float32)

    return CompiledOperator(
        hi,
        name=name,
        fn=fn,
        is_hermitian=True,
        dtype=np.float32,
        max_conn_size=1,
    )


def _single_term_ir(hi, name: str) -> SymbolicOperatorIR:
    term = SymbolicIRTerm.create(
        name="t0",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=nkdsl.identity().to_program(),
        amplitude=1.0,
    )
    return SymbolicOperatorIR.from_terms(
        operator_name=name,
        hilbert_size=int(hi.size),
        dtype_str="float32",
        is_hermitian=True,
        terms=(term,),
    )


class _NoEstimateSymbolic(AbstractSymbolicOperator):
    def to_ir(self) -> SymbolicOperatorIR:
        return _single_term_ir(self.hilbert, self.operator_name)

    def compile(self, *_, **__):
        return _compiled_identity(self.hilbert, name=f"c-{self.operator_name}")

    def _apply_scalar(self, scalar: int | float | complex):
        return self


class _NoScalarImpl(AbstractSymbolicOperator):
    def to_ir(self) -> SymbolicOperatorIR:
        return _single_term_ir(self.hilbert, self.operator_name)

    def compile(self, *_, **__):
        return _compiled_identity(self.hilbert, name=f"c-{self.operator_name}")


class _FakeNonSymbolicChild:
    def __init__(self, hilbert):
        self.hilbert = hilbert
        self._name_val = "fake_child"
        self._dtype_val = "float32"
        self.is_hermitian = False

    def to_ir(self) -> SymbolicOperatorIR:
        return SymbolicOperatorIR(
            operator_name="fake_child",
            mode="jax_kernel",
            hilbert_size=int(self.hilbert.size),
            dtype_str="float32",
            is_hermitian=False,
            terms=(),
            metadata=(),
        )


def _build_symbolic(name: str, *, dtype: str = "float64") -> SymbolicOperator:
    hi = nk.hilbert.Fock(n_max=2, N=2)
    return (
        nkdsl.SymbolicDiscreteJaxOperator(hi, name, dtype=dtype, hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )


def test_abstract_symbolic_operator_runtime_and_arithmetic_paths():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = _NoEstimateSymbolic(
        hi,
        name="dummy",
        dtype_str="float32",
        is_hermitian=True,
        metadata={"source": "unit"},
    )

    assert op.operator_name == "dummy"
    assert op.dtype == np.dtype(np.float32)
    assert op.is_hermitian is True
    assert op.metadata["source"] == "unit"
    assert op.max_conn_size == 1
    assert "dummy" in repr(op)

    with pytest.raises(nkdsl.SymbolicOperatorExecutionError, match="cannot execute"):
        op.get_conn_padded(jnp.asarray([0, 1], dtype=jnp.int32))

    compiled = _compiled_identity(hi, name="other")
    assert isinstance(op + compiled, SumOperator)
    assert isinstance(compiled + op, SumOperator)
    assert isinstance(op @ compiled, ProductOperator)
    assert isinstance(compiled @ op, ProductOperator)
    assert (0 + op) is op
    assert isinstance(sum([op, op]), SumOperator)

    with pytest.raises(TypeError):
        _ = op + 1
    with pytest.raises(TypeError):
        _ = 1 + op
    with pytest.raises(TypeError):
        _ = op @ 1
    with pytest.raises(TypeError):
        _ = 1 @ op

    assert op.__mul__("bad") is NotImplemented
    assert op.__rmul__("bad") is NotImplemented
    assert (-op) is op


def test_base_apply_scalar_notimplemented_and_compiled_fallback_dunders():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = _NoScalarImpl(hi, name="noscalar")
    with pytest.raises(NotImplementedError, match="does not implement _apply_scalar"):
        _ = op * 2

    compiled = _compiled_identity(hi, name="compiled-fallback")
    assert "CompiledOperator(" in repr(compiled)
    assert (0 + compiled) is compiled
    assert isinstance(sum([compiled, compiled]), SumOperator)

    with pytest.raises(TypeError):
        _ = compiled + 1
    with pytest.raises(TypeError):
        _ = 1 + compiled
    with pytest.raises(TypeError):
        _ = compiled @ 1
    with pytest.raises(TypeError):
        _ = 1 @ compiled


def test_symbolic_operator_helper_functions_and_custom_compile_path():
    assert _resolve_dtype("float32", "complex64") == "complex64"
    assert _resolve_dtype("float64", "complex64") == "complex128"
    assert _resolve_dtype("float32", "float64") == "float64"
    assert _scalar_str(2.0) == "2"
    assert "1+2j" in _scalar_str(1 + 2j)
    assert _promote_dtype_for_scalar("float32", np.complex64(1j)) == "complex64"
    assert _promote_dtype_for_scalar("float32", 1j) == "complex128"
    assert _promote_dtype_for_scalar("float64", 1j) == "complex128"
    assert _promote_dtype_for_scalar("complex64", 2) == "complex64"
    assert _merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    with pytest.raises(ValueError, match="Cannot merge symbolic operator metadata key"):
        _merge_metadata({"a": 1}, {"a": 2})

    op = _build_symbolic("custom-compile")
    assert "SymbolicOperator(" in repr(op)

    class _StubCompiler:
        def __init__(self):
            self.called = False

        def compile_operator(self, operator):
            self.called = True
            assert operator is op
            return "stubbed"

    stub = _StubCompiler()
    out = op.compile(compiler=stub)
    assert out == "stubbed"
    assert stub.called is True


def test_symbolic_operator_estimate_errors_and_symbolic_sum_edges():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    bad_term = SymbolicIRTerm.create(
        name="bad",
        iterator=object(),
        predicate=True,
        update_program=nkdsl.identity().to_program(),
        amplitude=1.0,
    )
    bad_op = SymbolicOperator(hi, "bad-op", (bad_term,), dtype_str="float64")
    with pytest.raises(TypeError, match="Unsupported iterator type"):
        bad_op.estimate_max_conn_size()

    op_a = _build_symbolic("a", dtype="float32")
    op_b = _build_symbolic("b", dtype="complex64")
    inner = SymbolicOperatorSum(hi, [op_a], name="inner")
    outer = SymbolicOperatorSum(hi, [inner, op_b], name="outer")

    assert len(outer) == 2
    assert list(iter(outer)) == list(outer.terms)
    assert outer.is_hermitian is True
    assert outer.dtype == np.dtype("complex64")
    assert "SymbolicOperatorSum(" in repr(outer)

    scaled = outer * (1 + 0j)
    assert isinstance(scaled, SymbolicOperatorSum)
    assert scaled.is_hermitian is False
    assert scaled.dtype == np.dtype("complex128")

    fake = _FakeNonSymbolicChild(hi)
    mixed = SymbolicOperatorSum(hi, [op_a, fake], name="mixed")
    assert mixed.free_symbols == op_a.free_symbols
    assert mixed.estimate_max_conn_size() == (op_a.estimate_max_conn_size() + hi.size)

    with pytest.raises(ValueError, match="mixed modes"):
        mixed.to_ir()
