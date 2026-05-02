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

from tests.helpers.physics_builders import local_operator_heisenberg
from tests.helpers.physics_builders import local_operator_ising
from tests.helpers.physics_builders import symbolic_heisenberg
from tests.helpers.physics_builders import symbolic_ising

pytestmark = pytest.mark.physics


def _canonicalize_get_conn(x_primes, mels):
    x_arr = np.asarray(x_primes)
    m_arr = np.asarray(mels)
    if x_arr.size == 0:
        if x_arr.ndim == 2:
            return x_arr, m_arr
        return x_arr.reshape(0, 0), m_arr
    order = np.lexsort(x_arr.T[::-1])
    return x_arr[order], m_arr[order]


@pytest.mark.parametrize(
    ("name", "symbolic_builder", "local_builder"),
    (
        (
            "ising",
            lambda hi, g: symbolic_ising(hi, g, J=1.2, h=0.6).compile(cache=False),
            lambda hi, g: local_operator_ising(hi, g, J=1.2, h=0.6),
        ),
        (
            "heisenberg",
            lambda hi, g: symbolic_heisenberg(hi, g, J=1.0).compile(cache=False),
            lambda hi, g: local_operator_heisenberg(hi, g, J=1.0),
        ),
    ),
)
def test_get_conn_matches_local_operator_reference(name, symbolic_builder, local_builder):
    hi = nk.hilbert.Spin(s=0.5, N=4)
    g = nk.graph.Chain(length=4, pbc=True)

    h_sym = symbolic_builder(hi, g)
    h_ref = local_builder(hi, g)

    for x in np.asarray(hi.all_states(), dtype=np.int32):
        xp_sym, mel_sym = h_sym.get_conn(x)
        xp_ref, mel_ref = h_ref.get_conn(x)

        xp_sym, mel_sym = _canonicalize_get_conn(xp_sym, mel_sym)
        xp_ref, mel_ref = _canonicalize_get_conn(xp_ref, mel_ref)

        np.testing.assert_array_equal(xp_sym, xp_ref, err_msg=f"state={tuple(x)} model={name}")
        np.testing.assert_allclose(
            mel_sym,
            mel_ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"state={tuple(x)} model={name}",
        )
