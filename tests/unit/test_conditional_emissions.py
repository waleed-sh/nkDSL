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

import nkdsl

pytestmark = pytest.mark.unit


def test_conditional_emission_if_elseif_else_chain() -> None:
    hi = nk.hilbert.Fock(n_max=4, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "if-elseif-else")
        .for_each_site("i")
        .emit_if(nkdsl.site("i") == 0, nkdsl.write("i", 1), matrix_element=10.0)
        .emit_elseif(nkdsl.site("i") == 1, nkdsl.write("i", 2), matrix_element=20.0)
        .emit_else(nkdsl.write("i", 3), matrix_element=30.0)
        .build()
        .compile()
    )

    xp0, m0 = op.get_conn_padded(jnp.asarray([0], dtype=jnp.int32))
    xp1, m1 = op.get_conn_padded(jnp.asarray([1], dtype=jnp.int32))
    xp2, m2 = op.get_conn_padded(jnp.asarray([2], dtype=jnp.int32))

    np.testing.assert_array_equal(np.asarray(xp0).reshape(-1), np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(np.asarray(xp1).reshape(-1), np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(np.asarray(xp2).reshape(-1), np.asarray([1, 2, 3]))

    np.testing.assert_allclose(np.asarray(m0), np.asarray([10.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.asarray(m1), np.asarray([0.0, 20.0, 0.0]))
    np.testing.assert_allclose(np.asarray(m2), np.asarray([0.0, 0.0, 30.0]))


def test_conditional_emission_supports_multiple_elseif_branches() -> None:
    hi = nk.hilbert.Fock(n_max=5, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "multi-elseif")
        .for_each_site("i")
        .emit_if(nkdsl.site("i") == 0, nkdsl.write("i", 1), matrix_element=1.0)
        .emit_elseif(nkdsl.site("i") == 1, nkdsl.write("i", 2), matrix_element=2.0)
        .emit_elseif(nkdsl.site("i") == 2, nkdsl.write("i", 3), matrix_element=3.0)
        .emit_else(nkdsl.write("i", 4), matrix_element=4.0)
        .build()
        .compile()
    )

    _xp, m = op.get_conn_padded(jnp.asarray([2], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(m), np.asarray([0.0, 0.0, 3.0, 0.0]))


def test_emit_elseif_and_emit_else_require_open_conditional_chain() -> None:
    hi = nk.hilbert.Fock(n_max=3, N=1)

    with pytest.raises(ValueError, match="must follow .emit_if"):
        (
            nkdsl.SymbolicDiscreteJaxOperator(hi, "elseif-error")
            .for_each_site("i")
            .emit_elseif(nkdsl.site("i").value > 0, nkdsl.identity(), matrix_element=1.0)
        )

    with pytest.raises(ValueError, match="must follow .emit_if"):
        nkdsl.SymbolicDiscreteJaxOperator(hi, "else-error").for_each_site("i").emit_else(
            nkdsl.identity(),
            matrix_element=1.0,
        )


def test_unconditional_emit_closes_conditional_chain() -> None:
    hi = nk.hilbert.Fock(n_max=3, N=1)
    builder = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "chain-close")
        .for_each_site("i")
        .emit_if(nkdsl.site("i").value > 0, nkdsl.write("i", 0), matrix_element=1.0)
        .emit(nkdsl.identity(), matrix_element=2.0)
    )

    with pytest.raises(ValueError, match="must follow .emit_if"):
        builder.emit_elseif(nkdsl.site("i").value == 0, nkdsl.write("i", 1), matrix_element=3.0)
