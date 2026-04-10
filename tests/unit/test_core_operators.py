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
from nkdsl.core.compiled import CompiledOperator
from nkdsl.core.compiled import create_compiled_operator
from nkdsl.errors import SymbolicOperatorExecutionError

pytestmark = pytest.mark.unit


def _build_simple_symbolic(name: str = "op"):
    hi = nk.hilbert.Spin(s=0.5, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, name, hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.write("i", -nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    return hi, op


def test_uncompiled_symbolic_execution_raises_helpful_error():
    _, op = _build_simple_symbolic("raw")
    with pytest.raises(SymbolicOperatorExecutionError, match="cannot execute"):
        op.get_conn_padded(jnp.asarray([0, 1], dtype=jnp.int32))


def test_symbolic_operator_scalar_algebra_and_dense_equivalence():
    _, op = _build_simple_symbolic("a")
    c = op.compile(cache=False)

    scaled = (2 * op).compile(cache=False)
    negated = (-op).compile(cache=False)

    dense = np.asarray(c.to_dense())
    np.testing.assert_allclose(np.asarray(scaled.to_dense()), 2.0 * dense)
    np.testing.assert_allclose(np.asarray(negated.to_dense()), -dense)


def test_symbolic_operator_add_and_matmul_use_netket_algebra():
    _, op1 = _build_simple_symbolic("op1")
    _, op2 = _build_simple_symbolic("op2")

    assert isinstance(op1 + op2, SumOperator)
    assert isinstance(op1 @ op2, ProductOperator)


def test_symbolic_operator_sum_to_ir_and_compile():
    hi, op1 = _build_simple_symbolic("s1")
    _, op2 = _build_simple_symbolic("s2")

    summed = nkdsl.SymbolicOperatorSum(hi, [op1, op2], name="ham")
    assert len(summed) == 2
    assert summed.estimate_max_conn_size() == (
        op1.estimate_max_conn_size() + op2.estimate_max_conn_size()
    )

    ir = summed.to_ir()
    assert ir.term_count == op1.term_count + op2.term_count

    compiled = nkdsl.SymbolicCompiler(cache_enabled=False).compile_operator(summed)
    dense_sum = np.asarray(compiled.to_dense())
    dense_ref = np.asarray((op1.compile(cache=False) + op2.compile(cache=False)).to_dense())
    np.testing.assert_allclose(dense_sum, dense_ref)


def test_symbolic_operator_sum_free_symbols_and_validation_errors():
    hi = nk.hilbert.Fock(n_max=2, N=1)
    op_with_symbol = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "sym", hermitian=True)
        .globally()
        .emit(nkdsl.identity(), matrix_element=nkdsl.symbol("kappa"))
        .build()
    )
    op_plain = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "plain", hermitian=True)
        .globally()
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
    )

    summed = nkdsl.SymbolicOperatorSum(hi, [op_with_symbol, op_plain])
    assert "kappa" in summed.free_symbols

    hi_other = nk.hilbert.Fock(n_max=2, N=2)
    other = (
        nkdsl.SymbolicDiscreteJaxOperator(hi_other, "other")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )
    with pytest.raises(ValueError, match="same Hilbert space"):
        nkdsl.SymbolicOperatorSum(hi, [op_plain, other])

    with pytest.raises(ValueError, match="at least one term"):
        nkdsl.SymbolicOperatorSum(hi, [])


def test_compiled_operator_properties_and_pytree_roundtrip():
    hi = nk.hilbert.Fock(n_max=2, N=2)

    def fn(x):
        if x.ndim == 1:
            return jnp.expand_dims(x, 0), jnp.ones((1,), dtype=jnp.float32)
        batch = x.shape[0]
        return jnp.expand_dims(x, 1), jnp.ones((batch, 1), dtype=jnp.float32)

    op = CompiledOperator(
        hi,
        name="manual",
        fn=fn,
        is_hermitian=True,
        dtype=np.float32,
        max_conn_size=1,
    )

    assert op.name == "manual"
    assert op.is_hermitian is True
    assert op.max_conn_size == 1
    assert op.dtype == np.dtype(np.float32)

    leaves, aux = op.tree_flatten()
    assert leaves == []
    restored = CompiledOperator.tree_unflatten(aux, leaves)
    x = jnp.asarray([0, 1], dtype=jnp.int32)
    xp, m = restored.get_conn_padded(x)
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[0, 1]]))
    np.testing.assert_allclose(np.asarray(m), np.asarray([1.0], dtype=np.float32))

    other = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "other")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
        .compile(cache=False)
    )
    assert isinstance(op + other, SumOperator)
    assert isinstance(op @ other, ProductOperator)


def test_create_compiled_operator_validates_inputs_and_supports_custom_targets():
    hi = nk.hilbert.Fock(n_max=2, N=2)

    def fn(x):
        return jnp.expand_dims(x, 0), jnp.ones((1,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="non-empty string"):
        create_compiled_operator(
            hi,
            name="bad",
            fn=fn,
            is_hermitian=False,
            dtype=np.float32,
            max_conn_size=1,
            connection_method=" ",
        )

    with pytest.raises(TypeError, match="operator_type must be a class"):
        create_compiled_operator(
            hi,
            name="bad",
            fn=fn,
            is_hermitian=False,
            dtype=np.float32,
            max_conn_size=1,
            operator_type="not-a-class",
        )

    class _ComputationalLikeOperator:
        def __init__(self, hilbert):
            self.hilbert = hilbert

        def get_conn_padded(self, x):
            return self._get_conn_padded(x)

    op = create_compiled_operator(
        hi,
        name="custom",
        fn=fn,
        is_hermitian=False,
        dtype=np.float32,
        max_conn_size=1,
        operator_type=_ComputationalLikeOperator,
        connection_method="_get_conn_padded",
    )
    xp, mel = op.get_conn_padded(jnp.asarray([0, 1], dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[0, 1]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(mel), np.asarray([1.0], dtype=np.float32))


def test_complex_matrix_elements_auto_promote_dtype():
    hi = nk.hilbert.Spin(s=0.5, N=2)

    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "complex_auto", hermitian=False)
        .for_each_site("i")
        .emit(
            nkdsl.write("i", -nkdsl.site("i").value),
            matrix_element=1.0 + 2.0j,
        )
        .build()
    )
    assert symbolic.dtype == np.dtype(np.complex128)

    compiled = symbolic.compile(cache=False)
    assert compiled.dtype == np.dtype(np.complex128)

    x = jnp.asarray([1, -1], dtype=jnp.int32)
    xp, mel = compiled.get_conn_padded(x)
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[-1, -1], [1, 1]]))
    np.testing.assert_allclose(
        np.asarray(mel),
        np.asarray([1.0 + 2.0j, 1.0 + 2.0j], dtype=np.complex128),
    )

    symbolic_f32 = (
        nkdsl.SymbolicDiscreteJaxOperator(
            hi,
            "complex_auto_f32",
            dtype="float32",
            hermitian=False,
        )
        .for_each_site("i")
        .emit(
            nkdsl.write("i", -nkdsl.site("i").value),
            matrix_element=np.complex64(0.25 + 0.5j),
        )
        .build()
    )
    assert symbolic_f32.dtype == np.dtype(np.complex64)


def test_complex_real_operator_addition_and_dense_parity():
    hi = nk.hilbert.Spin(s=0.5, N=2)

    real_op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "real_part", hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.write("i", -nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    complex_op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "complex_part", hermitian=False)
        .for_each_site("i")
        .emit(
            nkdsl.write("i", -nkdsl.site("i").value),
            matrix_element=0.0 + 1.0j,
        )
        .build()
    )

    summed_symbolic = nkdsl.SymbolicOperatorSum(hi, [real_op, complex_op], name="mix")
    assert summed_symbolic.dtype == np.dtype(np.complex128)

    dense_sum = np.asarray(
        nkdsl.SymbolicCompiler(cache_enabled=False).compile_operator(summed_symbolic).to_dense()
    )
    dense_ref = np.asarray(real_op.compile(cache=False).to_dense()) + np.asarray(
        complex_op.compile(cache=False).to_dense()
    )
    np.testing.assert_allclose(dense_sum, dense_ref)

    netket_sum = real_op + complex_op
    assert isinstance(netket_sum, SumOperator)
    np.testing.assert_allclose(np.asarray(netket_sum.to_dense()), dense_ref)


def test_operator_algebra_against_native_netket_dense():
    hi = nk.hilbert.Spin(s=0.5, N=2)
    sym = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "sym_native", hermitian=True)
        .for_each_site("i")
        .emit(
            nkdsl.write("i", -nkdsl.site("i").value),
            matrix_element=0.7,
        )
        .build()
    )
    sym_c = sym.compile(cache=False)

    native = nk.operator.spin.sigmax(hi, 0)

    sum_op = sym + native
    assert isinstance(sum_op, SumOperator)
    np.testing.assert_allclose(
        np.asarray(sum_op.to_dense()),
        np.asarray(sym_c.to_dense()) + np.asarray(native.to_dense()),
    )

    sum_op_rev = native + sym
    assert isinstance(sum_op_rev, SumOperator)
    np.testing.assert_allclose(
        np.asarray(sum_op_rev.to_dense()),
        np.asarray(native.to_dense()) + np.asarray(sym_c.to_dense()),
    )

    prod_op = sym @ native
    assert isinstance(prod_op, ProductOperator)
    np.testing.assert_allclose(
        np.asarray(prod_op.to_dense()),
        np.asarray(sym_c.to_dense()) @ np.asarray(native.to_dense()),
    )

    prod_op_rev = native @ sym
    assert isinstance(prod_op_rev, ProductOperator)
    np.testing.assert_allclose(
        np.asarray(prod_op_rev.to_dense()),
        np.asarray(native.to_dense()) @ np.asarray(sym_c.to_dense()),
    )
