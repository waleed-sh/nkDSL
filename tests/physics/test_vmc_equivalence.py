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

import netket as nk
import numpy as np
import pytest

from tests.helpers.physics_builders import fullsum_vmc_energy_trace
from tests.helpers.physics_builders import max_abs_tree_diff
from tests.helpers.physics_builders import symbolic_heisenberg
from tests.helpers.physics_builders import symbolic_ising

pytestmark = [pytest.mark.physics, pytest.mark.slow]


@pytest.mark.parametrize(
    ("name", "symbolic_builder", "reference_builder"),
    (
        (
            "ising",
            lambda hi, g: symbolic_ising(hi, g, J=1.2, h=0.7).compile(cache=False),
            lambda hi, g: nk.operator.Ising(hi, g, h=0.7, J=1.2),
        ),
        (
            "heisenberg",
            lambda hi, g: symbolic_heisenberg(hi, g, J=1.0).compile(cache=False),
            lambda hi, g: nk.operator.Heisenberg(hi, g, J=1.0, sign_rule=False),
        ),
    ),
)
def test_full_vmc_matches_netket_reference(name, symbolic_builder, reference_builder):
    hi = nk.hilbert.Spin(s=0.5, N=6)
    g = nk.graph.Chain(length=6, pbc=True)

    h_ref = reference_builder(hi, g)
    h_sym = symbolic_builder(hi, g)

    e_ref, state_ref = fullsum_vmc_energy_trace(
        operator=h_ref,
        hilbert=hi,
        n_iter=12,
        learning_rate=0.05,
        seed=101,
    )
    e_sym, state_sym = fullsum_vmc_energy_trace(
        operator=h_sym,
        hilbert=hi,
        n_iter=12,
        learning_rate=0.05,
        seed=101,
    )

    np.testing.assert_allclose(np.asarray(e_sym), np.asarray(e_ref), atol=1e-10, rtol=1e-10)
    assert abs(e_sym[-1] - e_ref[-1]) < 1e-10
    assert max_abs_tree_diff(state_ref.parameters, state_sym.parameters) < 1e-10
