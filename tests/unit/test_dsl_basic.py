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

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl

pytestmark = pytest.mark.unit


def test_global_identity_diagonal_matrix_element():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "diag")
        .globally()
        .emit(
            nkdsl.identity(),
            matrix_element=nkdsl.source_index(0) + nkdsl.source_index(1),
        )
        .build()
        .compile()
    )

    x = jnp.asarray([1, 2], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[1, 2]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([3.0]))
    assert op.max_conn_size == 1


def test_global_target_index_reads_emitted_configuration():
    hi = nk.hilbert.Fock(n_max=3, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "target-index")
        .globally()
        .emit(
            nkdsl.write(0, 2),
            matrix_element=nkdsl.target_index(0) + nkdsl.source_index(1),
        )
        .build()
        .compile()
    )

    x = jnp.asarray([1, 3], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[2, 3]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([5.0]))
    assert op.max_conn_size == 1


def test_for_each_site_where_shift():
    hi = nk.hilbert.Fock(n_max=3, N=3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "lower")
        .for_each_site("i")
        .where(nkdsl.site("i").value > 0)
        .emit(nkdsl.shift("i", -1), matrix_element=nkdsl.site("i").value)
        .build()
        .compile()
    )

    x = jnp.asarray([1, 0, 2], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    expected_xp = np.asarray(
        [
            [0, 0, 2],
            [1, -1, 2],
            [1, 0, 1],
        ],
        dtype=np.int32,
    )
    expected_mels = np.asarray([1.0, 0.0, 2.0])

    np.testing.assert_array_equal(np.asarray(xp), expected_xp)
    np.testing.assert_allclose(np.asarray(mels), expected_mels)
    assert op.max_conn_size == 3


def test_multi_emission_order_and_values():
    hi = nk.hilbert.Fock(n_max=3, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "two_branch")
        .for_each_site("i")
        .where(nkdsl.site("i").value < 2)
        .emit(nkdsl.shift("i", +1), matrix_element=0.5)
        .emit(nkdsl.shift("i", -1), matrix_element=-0.5)
        .build()
        .compile()
    )

    x = jnp.asarray([1, 0], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    expected_xp = np.asarray(
        [
            [2, 0],
            [0, 0],
            [1, 1],
            [1, -1],
        ],
        dtype=np.int32,
    )
    expected_mels = np.asarray([0.5, -0.5, 0.5, -0.5])

    np.testing.assert_array_equal(np.asarray(xp), expected_xp)
    np.testing.assert_allclose(np.asarray(mels), expected_mels)
    assert op.max_conn_size == 4


def test_emitted_selector_in_matrix_element():
    hi = nk.hilbert.Fock(n_max=3, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "emit_amp")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=nkdsl.emitted("i").value)
        .build()
        .compile()
    )

    x = jnp.asarray([0, 1], dtype=jnp.int32)
    _, mels = op.get_conn_padded(x)

    np.testing.assert_allclose(np.asarray(mels), np.asarray([1.0, 2.0]))


def test_shift_mod_and_wrap_mod_expression():
    hi = nk.hilbert.Fock(n_max=2, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "wrap")
        .for_each_site("i")
        .emit(
            nkdsl.shift_mod("i", +1),
            matrix_element=nkdsl.AmplitudeExpr.wrap_mod(nkdsl.site("i").value + 2),
        )
        .build()
        .compile()
    )

    x = jnp.asarray([2], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[0]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([1.0]))


def test_affine_permute_and_scatter_updates_execute():
    hi_affine = nk.hilbert.Fock(n_max=5, N=2)
    op_affine = (
        nkdsl.SymbolicDiscreteJaxOperator(hi_affine, "aff")
        .for_each_site("i")
        .emit(nkdsl.affine("i", scale=2, bias=1), matrix_element=1.0)
        .build()
        .compile()
    )
    x_aff = jnp.asarray([0, 1], dtype=jnp.int32)
    xp_aff, m_aff = op_affine.get_conn_padded(x_aff)
    np.testing.assert_array_equal(np.asarray(xp_aff), np.asarray([[1, 1], [0, 3]]))
    np.testing.assert_allclose(np.asarray(m_aff), np.asarray([1.0, 1.0]))

    hi_perm = nk.hilbert.Fock(n_max=9, N=3)
    op_perm = (
        nkdsl.SymbolicDiscreteJaxOperator(hi_perm, "perm")
        .globally()
        .emit(nkdsl.permute(0, 1, 2), matrix_element=1.0)
        .build()
        .compile()
    )
    x_perm = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    xp_perm, m_perm = op_perm.get_conn_padded(x_perm)
    np.testing.assert_array_equal(np.asarray(xp_perm), np.asarray([[2, 3, 1]]))
    np.testing.assert_allclose(np.asarray(m_perm), np.asarray([1.0]))

    hi_scatter = nk.hilbert.Fock(n_max=9, N=3)
    op_scatter = (
        nkdsl.SymbolicDiscreteJaxOperator(hi_scatter, "scatter")
        .globally()
        .emit(nkdsl.scatter([0, 2], [9, 8]), matrix_element=1.0)
        .build()
        .compile()
    )
    x_sc = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    xp_sc, m_sc = op_scatter.get_conn_padded(x_sc)
    np.testing.assert_array_equal(np.asarray(xp_sc), np.asarray([[9, 2, 8]]))
    np.testing.assert_allclose(np.asarray(m_sc), np.asarray([1.0]))


def test_conditional_and_invalidate_updates():
    hi = nk.hilbert.Fock(n_max=3, N=2)

    cond_update = nkdsl.Update.cond(
        nkdsl.site("i") > 0,
        if_true=nkdsl.write("i", 0),
        if_false=nkdsl.write("i", 2),
    )
    op_cond = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "cond")
        .for_each_site("i")
        .emit(cond_update, matrix_element=1.0)
        .build()
        .compile()
    )

    x = jnp.asarray([0, 1], dtype=jnp.int32)
    xp, mels = op_cond.get_conn_padded(x)
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[2, 1], [0, 0]]))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([1.0, 1.0]))

    op_invalid = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "invalid")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1).invalidate(reason="test"), matrix_element=5.0)
        .build()
        .compile()
    )
    x2 = jnp.asarray([1, 1], dtype=jnp.int32)
    _xp2, m2 = op_invalid.get_conn_padded(x2)
    np.testing.assert_allclose(np.asarray(m2), np.asarray([0.0, 0.0]))


def test_compiled_operator_is_jittable_with_operator_argument():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "jit")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
        .compile()
    )

    x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)

    @jax.jit
    def total_mels(operator, states):
        _, mels = operator.get_conn_padded(states)
        return mels.sum()

    out = total_mels(op, x)
    np.testing.assert_allclose(np.asarray(out), np.asarray(4.0))


def test_builder_validation_errors_and_aliases():
    hi = nk.hilbert.Fock(n_max=2, N=2)

    with pytest.raises(ValueError, match="non-empty"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "  ")

    with pytest.raises(ValueError, match="before any iterator"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").emit(nkdsl.identity())

    with pytest.raises(ValueError, match="zero terms"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").build()

    with pytest.raises(ValueError, match="has no emissions"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").for_each_site("i").build()

    with pytest.raises(ValueError, match="over= must not be empty"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").for_each(("i",), over=())

    with pytest.raises(ValueError, match="must have length"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").for_each(("i", "j"), over=((0,),))

    with pytest.raises(ValueError, match="positive integer"):
        (nkdsl.SymbolicDiscreteJaxOperator(hi, "e").for_each_site("i").max_conn_size(0))

    with pytest.raises(ValueError, match="non-empty"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "e").for_each_site("i").named(" ")

    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "alias")
        .for_each_site("i")
        .fanout(3)
        .emit(nkdsl.shift("i", +1), amplitude=1.0)
        .build()
    )
    assert op.estimate_max_conn_size() == 3


def test_builder_repr_contains_progress_information():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    builder = nkdsl.SymbolicDiscreteJaxOperator(hi, "repr").for_each_site("i")
    rep = repr(builder)
    assert "terms_sealed" in rep
    assert "term_open=True" in rep


def test_extended_iterators_named_terms_and_direct_compile():
    hi = nk.hilbert.Fock(n_max=3, N=3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "iterators")
        .for_each_pair("a", "b")
        .named("pair")
        .where(nkdsl.site("a") >= 0)
        .where(nkdsl.site("b") >= 0)
        .emit(nkdsl.identity(), matrix_element=nkdsl.site("a").value + nkdsl.site("b").value)
        .for_each_distinct_pair("u", "v")
        .named("distinct")
        .emit(nkdsl.identity(), matrix_element=1.0)
        .for_each_triplet("i", "j", "k", over=((0, 1, 2),))
        .named("trip")
        .emit(nkdsl.identity(), matrix_element=2.0)
        .for_each_plaquette("p", "q", "r", "s", over=((0, 1, 2, 0),))
        .named("plaq")
        .max_conn_size(1)
        .emit(matrix_element=3.0)
        .compile(cache=False)
    )

    x = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)
    assert xp.shape[0] == mels.shape[0]
    assert xp.shape[1] == hi.size
    assert op.max_conn_size >= 1
