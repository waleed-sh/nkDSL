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

from tests.helpers.physics_builders import chain_lengths
from tests.helpers.physics_builders import symbolic_heisenberg
from tests.helpers.physics_builders import symbolic_ising

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("length", chain_lengths())
def test_symbolic_ising_dense_matches_netket(length: int):
    hi = nk.hilbert.Spin(s=0.5, N=length)
    graph = nk.graph.Chain(length=length, pbc=False)

    J = 1.3
    h = 0.7

    ref = nk.operator.Ising(hi, graph, h=h, J=J)
    sym = symbolic_ising(hi, graph, J=J, h=h).compile(cache=False)

    dense_ref = np.asarray(ref.to_dense())
    dense_sym = np.asarray(sym.to_dense())

    np.testing.assert_allclose(dense_sym, dense_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("length", chain_lengths())
def test_symbolic_heisenberg_dense_matches_netket_sign_rule_false(length: int):
    hi = nk.hilbert.Spin(s=0.5, N=length)
    graph = nk.graph.Chain(length=length, pbc=False)

    J = 1.1

    ref = nk.operator.Heisenberg(hi, graph, J=J, sign_rule=False)
    sym = symbolic_heisenberg(hi, graph, J=J).compile(cache=False)

    dense_ref = np.asarray(ref.to_dense())
    dense_sym = np.asarray(sym.to_dense())

    np.testing.assert_allclose(dense_sym, dense_ref, atol=1e-10, rtol=1e-10)
