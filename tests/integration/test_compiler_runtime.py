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
import nkdsl.debug as nkdebug

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deduplicate", (False, True))
def test_large_plaquette_operator_compiles_without_hilbert_indexing(deduplicate):
    length = 5
    edges = [
        (i, j, i + di, j + dj)
        for i in range(length)
        for j in range(length)
        for di, dj in ((1, 0), (0, 1))
        if i + di < length and j + dj < length
    ]
    edge_index = {edge: index for index, edge in enumerate(edges)}
    plaquettes = [
        (
            edge_index[i, j, i + 1, j],
            edge_index[i + 1, j, i + 1, j + 1],
            edge_index[i, j + 1, i + 1, j + 1],
            edge_index[i, j, i, j + 1],
        )
        for i in range(length - 1)
        for j in range(length - 1)
    ]
    hi = nk.hilbert.HomogeneousHilbert(
        local_states=nk.utils.StaticRange(start=-2, step=1, length=5, dtype=np.int64),
        N=len(edges),
    )
    assert not hi.is_indexable
    forward = nkdsl.shift_mod("p", 1).shift_mod("q", 1).shift_mod("r", -1).shift_mod("s", -1)
    reverse = nkdsl.shift_mod("p", -1).shift_mod("q", -1).shift_mod("r", 1).shift_mod("s", 1)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "large-plaquettes", hermitian=True)
        .for_each_site("l")
        .named("electric")
        .emit(matrix_element=0.5 * nkdsl.site("l").value.pow(2))
        .for_each_plaquette("p", "q", "r", "s", over=plaquettes)
        .named("magnetic")
        .emit(matrix_element=1.0, tag="constant")
        .emit(forward, matrix_element=-0.5, tag="forward")
        .emit(reverse, matrix_element=-0.5, tag="reverse")
        .build()
    )
    operator = symbolic.compile(deduplicate_connected_components=deduplicate, cache=False)
    sources = np.stack((np.zeros(hi.size), np.full(hi.size, 2), np.full(hi.size, -2)))
    targets, weights = jax.block_until_ready(jax.jit(operator.get_conn_padded)(sources))
    assert targets.shape == (3, operator.max_conn_size, hi.size)
    assert operator.max_conn_size == (33 if deduplicate else 88)
    for source, row_targets, row_weights in zip(sources, np.asarray(targets), np.asarray(weights)):
        expected = {tuple(source): 0.5 * np.sum(source**2) + len(plaquettes)}
        for links in plaquettes:
            for direction in (-1, 1):
                target = source.copy()
                target[list(links)] = (
                    target[list(links)] + direction * np.array([1, 1, -1, -1]) + 2
                ) % 5 - 2
                expected[tuple(target)] = -0.5
        actual = {}
        for target, weight in zip(row_targets, row_weights):
            if weight != 0:
                key = tuple(target)
                actual[key] = actual.get(key, 0.0) + weight
        assert actual == expected


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
