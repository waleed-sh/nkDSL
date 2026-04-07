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
import nkdsl.debug as nkdebug

pytestmark = pytest.mark.integration


def test_compiler_cache_reuses_artifact_and_executes():
    hi = nk.hilbert.Fock(n_max=2, N=1)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "cache", hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )

    compiler = nkdsl.SymbolicCompiler(cache_enabled=True)
    c1 = compiler.compile_operator(symbolic)
    c2 = compiler.compile_operator(symbolic)

    assert c1 is c2
    assert compiler.cache_size == 1

    x = jnp.asarray([0], dtype=jnp.int32)
    xp, mels = c1.get_conn_padded(x)
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[1]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([1.0]))


def test_debug_pass_filter_logs_only_selected_passes(tmp_path):
    with nkdsl.cfg.patch(
        DEBUG=True,
        DEBUG_SCOPES="passes",
        DEBUG_PASSES="symbolic_validation",
        DEBUG_LOG_TO_FILE=True,
        DEBUG_LOG_DIR=str(tmp_path),
    ):
        nkdebug.refresh_settings(reinit=True)
        logfile = nkdebug.get_logfile()
        assert logfile is not None

        hi = nk.hilbert.Fock(n_max=2, N=1)
        op = (
            nkdsl.SymbolicDiscreteJaxOperator(hi, "dbg", hermitian=True)
            .for_each_site("i")
            .emit(nkdsl.shift("i", +1), matrix_element=1.0)
            .build()
            .compile()
        )

        x = jnp.asarray([0], dtype=jnp.int32)
        _xp, _mels = op.get_conn_padded(x)

        content = logfile.read_text(encoding="utf-8")
        assert "symbolic_validation" in content
        assert "symbolic_normalization" not in content
        assert "symbolic_max_conn_size_analysis" not in content

    nkdebug.refresh_settings(reinit=True)


def test_compiler_metadata_and_artifact_roundtrip():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "meta", hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )

    compiler = nkdsl.SymbolicCompiler(cache_enabled=False)
    artifact = compiler.compile(symbolic, metadata={"suite": "integration"})

    assert artifact.operator_name == "meta"
    assert artifact.backend == "jax"
    assert artifact.lowerer_name == "jax_symbolic_v1"
    assert artifact.metadata_map()["term_count"] == 1
