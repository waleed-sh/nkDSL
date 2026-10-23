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


"""Validate the built-in neuraLQX target against the actual optional package."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl

nqx = pytest.importorskip("neuralqx")

pytestmark = pytest.mark.integration


def make_operator():
    """A Hermitian operator containing diagonal and off-diagonal branches."""
    hi = nk.hilbert.Spin(s=0.5, N=3)
    builder = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "field", dtype="float64", hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.identity(), matrix_element=0.7 * nkdsl.site("i").value)
        .emit(nkdsl.affine("i", scale=-1), matrix_element=-0.3)
    )
    reference = sum(
        0.7 * nk.operator.spin.sigmaz(hi, i) - 0.3 * nk.operator.spin.sigmax(hi, i)
        for i in range(hi.size)
    )
    return hi, builder, reference


@pytest.mark.parametrize("deduplicate", [False, True])
def test_real_neuralqx_target_matches_netket_and_dense_reference(deduplicate):
    hi, builder, reference = make_operator()
    with nkdsl.config.patch(LOWERING="neuralqx"):
        compiled = builder.compile(deduplicate_connected_components=deduplicate)
        netket = builder.compile(
            operator_lowering="netket", deduplicate_connected_components=deduplicate
        )
    assert isinstance(compiled, nqx.operators.types.ComputationalJaxOperator)
    assert not isinstance(netket, nqx.operators.types.ComputationalJaxOperator)
    leaves, structure = jax.tree_util.tree_flatten(compiled)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    states = jnp.asarray(hi.all_states())
    for x in (states[0], states, states.reshape(2, 4, hi.size)):
        actual = jax.jit(lambda operator, row: operator.get_conn_padded(row))(restored, x)
        expected = netket.get_conn_padded(x)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, atol=1e-12)
    np.testing.assert_allclose(compiled.to_dense(), reference.to_dense(), atol=1e-12)


def test_neuralqx_operator_products_and_scaling_use_real_framework_wrappers():
    _, builder, reference = make_operator()
    with nkdsl.config.patch(LOWERING="neuralqx"):
        compiled = builder.compile()
    dense = reference.to_dense()
    product = compiled * compiled
    assert isinstance(product, nqx.operators.types.ComputationalJaxOperator)
    np.testing.assert_allclose(product.to_dense(), dense @ dense, atol=1e-12)
    np.testing.assert_allclose((2 * compiled).to_dense(), 2 * dense, atol=1e-12)


def test_neuralqx_vstate_expectation_and_gradient_match_on_identical_samples():
    hi, builder, _ = make_operator()
    with nkdsl.config.patch(LOWERING="neuralqx"):
        compiled = builder.compile()
        reference = builder.compile(operator_lowering="netket")
    state = nqx.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains_per_rank=4),
        nk.models.RBM(alpha=1, param_dtype=float),
        n_samples=32,
        n_discard_per_chain=2,
        seed=12,
        sampler_seed=13,
    )
    # Both calls use the state's cached samples and identical model parameters.
    stats, gradient = state.expect_and_grad(compiled)
    expected_stats, expected_gradient = state.expect_and_grad(reference)
    np.testing.assert_allclose(stats.mean, expected_stats.mean, atol=1e-10)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(expected_gradient)
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
