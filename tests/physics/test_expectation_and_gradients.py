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
import netket as nk
import numpy as np
import pytest

from tests.helpers.physics_builders import max_abs_tree_diff
from tests.helpers.physics_builders import symbolic_heisenberg
from tests.helpers.physics_builders import symbolic_ising

pytestmark = pytest.mark.physics


@pytest.mark.parametrize(
    ("name", "builder", "reference_factory"),
    (
        (
            "ising",
            lambda hi, g: symbolic_ising(hi, g, J=1.2, h=0.6).compile(cache=False),
            lambda hi, g: nk.operator.Ising(hi, g, h=0.6, J=1.2),
        ),
        (
            "heisenberg",
            lambda hi, g: symbolic_heisenberg(hi, g, J=1.0).compile(cache=False),
            lambda hi, g: nk.operator.Heisenberg(hi, g, J=1.0, sign_rule=False),
        ),
    ),
)
def test_expectation_and_gradient_match_netket(name, builder, reference_factory):
    hi = nk.hilbert.Spin(s=0.5, N=4)
    g = nk.graph.Chain(length=4, pbc=True)

    h_ref = reference_factory(hi, g)
    h_sym = builder(hi, g)

    model = nk.models.RBM(alpha=1, param_dtype=float)
    state = nk.vqs.FullSumState(hi, model, seed=123)

    stats_ref = state.expect(h_ref)
    stats_sym = state.expect(h_sym)
    np.testing.assert_allclose(float(stats_sym.mean), float(stats_ref.mean), atol=1e-10, rtol=1e-10)

    grad_stats_ref, grad_ref = state.expect_and_grad(h_ref)
    grad_stats_sym, grad_sym = state.expect_and_grad(h_sym)

    np.testing.assert_allclose(
        float(grad_stats_sym.mean), float(grad_stats_ref.mean), atol=1e-10, rtol=1e-10
    )
    assert max_abs_tree_diff(grad_ref, grad_sym) < 1e-10

    # Also ensure gradients can pass through JAX tree utilities without shape mismatches.
    leaves_ref = jax.tree_util.tree_leaves(grad_ref)
    leaves_sym = jax.tree_util.tree_leaves(grad_sym)
    assert len(leaves_ref) == len(leaves_sym)
