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

import copy
from collections.abc import Sequence

import netket as nk

import nkdsl


def graph_edges(graph: nk.graph.AbstractGraph) -> tuple[tuple[int, int], ...]:
    """Returns graph edges as a stable tuple of integer pairs."""
    return tuple((int(i), int(j)) for i, j in graph.edges())


def symbolic_ising(
    hilbert: nk.hilbert.Spin,
    graph: nk.graph.AbstractGraph,
    *,
    J: float,
    h: float,
    name: str = "symbolic_ising",
):
    """Builds an nkdsl symbolic Ising operator matching ``nk.operator.Ising``."""
    edges = graph_edges(graph)
    return (
        nkdsl.SymbolicDiscreteJaxOperator(hilbert, name, hermitian=True)
        .for_each(("i", "j"), over=edges)
        .emit(
            nkdsl.identity(),
            matrix_element=J * nkdsl.site("i").value * nkdsl.site("j").value,
        )
        .for_each_site("i")
        .emit(
            nkdsl.write("i", -nkdsl.site("i").value),
            matrix_element=-h,
        )
        .build()
    )


def symbolic_heisenberg(
    hilbert: nk.hilbert.Spin,
    graph: nk.graph.AbstractGraph,
    *,
    J: float,
    name: str = "symbolic_heisenberg",
):
    """Builds an nkdsl symbolic Heisenberg operator matching NetKet ``sign_rule=False``."""
    edges = graph_edges(graph)
    return (
        nkdsl.SymbolicDiscreteJaxOperator(hilbert, name, hermitian=True)
        .for_each(("i", "j"), over=edges)
        .emit(
            nkdsl.identity(),
            matrix_element=J * nkdsl.site("i").value * nkdsl.site("j").value,
        )
        .for_each(("i", "j"), over=edges)
        .where(nkdsl.site("i").value * nkdsl.site("j").value < 0)
        .emit(nkdsl.swap("i", "j"), matrix_element=2.0 * J)
        .build()
    )


def fullsum_vmc_energy_trace(
    *,
    operator,
    hilbert,
    n_iter: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[float], object]:
    """Runs a deterministic FullSumState VMC loop and returns per-iteration energies."""
    model = nk.models.RBM(alpha=1, param_dtype=float)
    state = nk.vqs.FullSumState(hilbert, model, seed=seed)
    opt = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC(operator, opt, variational_state=state)

    energies: list[float] = []
    for _ in range(n_iter):
        driver.advance(1)
        energies.append(float(driver.energy.mean))

    return energies, state


def clone_parameters(params):
    """Returns a deep copy of parameter pytrees for independent state initialization."""
    return copy.deepcopy(params)


def max_abs_tree_diff(tree_a, tree_b) -> float:
    """Computes max absolute leaf-wise difference between two pytrees."""
    import jax
    import numpy as np

    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    if len(leaves_a) != len(leaves_b):
        raise AssertionError(f"Pytree leaf mismatch: {len(leaves_a)} != {len(leaves_b)}.")

    diffs = [
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(leaves_a, leaves_b, strict=True)
    ]
    return max(diffs) if diffs else 0.0


def chain_lengths() -> Sequence[int]:
    """Default chain lengths used across dense-equivalence physics checks."""
    return (2, 3, 4)
