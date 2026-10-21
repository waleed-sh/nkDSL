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

import operator
from functools import lru_cache

import jax
import netket as nk
import numpy as np
import pytest

import nkdsl

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("field", ("index", "value"))
@pytest.mark.parametrize("scalar", (2, -1.0, np.int64(2), np.float64(-1.0), "charge"))
@pytest.mark.parametrize("comparison", (operator.eq, operator.ne))
def test_scalar_equality_builds_predicates(field, scalar, comparison):
    expr = getattr(nkdsl.site("e"), field)
    predicate = comparison(expr, scalar)

    assert isinstance(predicate, nkdsl.PredicateExpr)
    assert predicate.op == comparison.__name__
    assert predicate.args == (expr, nkdsl.coerce_amplitude_expr(scalar))


@pytest.mark.parametrize("scalar", (2, -1.0, "charge"))
@pytest.mark.parametrize("comparison", (operator.eq, operator.ne))
def test_scalar_equality_supports_reflected_comparisons(scalar, comparison):
    expr = nkdsl.site("e").value
    predicate = comparison(scalar, expr)

    assert isinstance(predicate, nkdsl.PredicateExpr)
    assert predicate.op == comparison.__name__
    assert predicate.args == (expr, nkdsl.coerce_amplitude_expr(scalar))


@pytest.mark.parametrize(
    "expr",
    (
        nkdsl.site("e").value + 1,
        nkdsl.emitted("e").value,
        nkdsl.source_index(0),
        nkdsl.target_index(0),
        nkdsl.symbol("g", default=1.0),
    ),
)
def test_scalar_equality_applies_to_composed_and_static_expressions(expr):
    assert isinstance(expr == 1, nkdsl.PredicateExpr)
    assert isinstance(expr != 1, nkdsl.PredicateExpr)


def test_expression_structural_equality_hashing_and_containers_are_preserved():
    def expression():
        return (nkdsl.site("i").value + 1) * (nkdsl.site("j").value - 2)

    first = expression()
    same = expression()
    different = expression() + 1

    assert (first == same) is True
    assert (first != same) is False
    assert (first == different) is False
    assert (first != different) is True
    assert hash(first) == hash(same) == hash((first.op, first.args))
    assert {first: "cached"}[same] == "cached"
    assert len({first, same, different}) == 2
    assert same in [first]
    assert different not in [first]
    assert (first,) == (same,)
    assert (first,) != (different,)

    @lru_cache
    def cached(expr):
        return object()

    assert cached(first) is cached(same)
    assert cached(first) is not cached(different)
    assert cached.cache_info().hits == 2


@pytest.mark.parametrize("other", (None, True, False, np.bool_(True), [], {}, object()))
def test_equality_with_unsupported_values_keeps_python_comparison_behavior(other):
    expr = nkdsl.site("e").index
    assert not (expr == other)
    assert expr != other


def test_expression_subclasses_keep_dataclass_structural_comparison_rules():
    class DerivedExpr(nkdsl.AmplitudeExpr):
        pass

    expr = nkdsl.AmplitudeExpr.symbol("g")
    derived = DerivedExpr.symbol("g")
    assert (expr == derived) is False
    assert (expr != derived) is True
    assert (derived == DerivedExpr.symbol("g")) is True
    assert hash(derived) == hash(expr)


@pytest.mark.parametrize("deduplicate", (False, True))
@pytest.mark.parametrize("edge_index, charge", ((0, 0), (1, 1), (2, 2), (1, 3)))
def test_charge_projector_natural_equality_under_jit(edge_index, charge, deduplicate):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "charge-projector", hermitian=True)
        .for_each_site("e")
        .where((nkdsl.site("e").value == charge) & (nkdsl.site("e").index == edge_index))
        .emit(nkdsl.identity(), matrix_element=1.0)
        .compile(deduplicate_connected_components=deduplicate, cache=False)
    )
    states = np.asarray(hi.all_states()).reshape(3, 9, 3)
    targets, weights = jax.block_until_ready(jax.jit(op.get_conn_padded)(states))
    targets, weights = np.asarray(targets), np.asarray(weights)

    np.testing.assert_array_equal(weights.sum(axis=-1), states[..., edge_index] == charge)
    source_states = np.broadcast_to(states[..., None, :], targets.shape)
    np.testing.assert_array_equal(targets[weights != 0], source_states[weights != 0])
    assert np.all((weights == 0) | (weights == 1))


@pytest.mark.parametrize("deduplicate", (False, True))
def test_natural_inequality_filters_indices_and_values(deduplicate):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "charge-exclusion", hermitian=True)
        .for_each_site("e")
        .where((nkdsl.site("e").index != 1) & (nkdsl.site("e").value != 2))
        .emit(nkdsl.identity(), matrix_element=1.0)
        .compile(deduplicate_connected_components=deduplicate, cache=False)
    )
    states = np.asarray(hi.all_states())
    _, weights = jax.block_until_ready(jax.jit(op.get_conn_padded)(states))
    expected = np.sum(states[:, (0, 2)] != 2, axis=-1)
    np.testing.assert_array_equal(np.asarray(weights).sum(axis=-1), expected)


def test_legacy_predicates_ir_and_compiler_cache_remain_compatible():
    hi = nk.hilbert.Fock(n_max=2, N=3)
    ctx = nkdsl.ExpressionContext()
    predicates = (
        (nkdsl.site("e").value == 1) & (nkdsl.site("e").index == 1),
        (nkdsl.site("e") == 1) & nkdsl.PredicateExpr.eq(nkdsl.site("e").index, 1),
        ctx.all_of(ctx.eq(ctx.site("e").value, 1), ctx.eq(ctx.site("e").index, 1)),
    )
    operators = [
        (
            nkdsl.SymbolicDiscreteJaxOperator(hi, "compatible-projector", hermitian=True)
            .for_each_site("e")
            .where(predicate)
            .emit(nkdsl.identity(), matrix_element=1.0)
            .build()
        )
        for predicate in predicates
    ]
    first_ir = operators[0].to_ir()
    compiler = nkdsl.SymbolicCompiler(cache_enabled=True)
    compiled = compiler.compile_operator(operators[0])
    for symbolic in operators[1:]:
        assert (symbolic.to_ir() == first_ir) is True
        assert symbolic.to_ir().static_fingerprint() == first_ir.static_fingerprint()
        assert compiler.compile_operator(symbolic) is compiled
    assert compiler.cache_size == 1


def test_predicates_and_nested_update_programs_keep_structural_equality():
    def program(value):
        return nkdsl.Update.cond(
            nkdsl.site("i").value == value,
            if_true=nkdsl.write("i", nkdsl.site("i").value + 1),
            if_false=nkdsl.shift("i", -1),
        ).to_program()

    first, same, different = program(0), program(0), program(1)
    assert (first == same) is True
    assert (first != different) is True
    assert hash(first) == hash(same)
    assert {first: "cached"}[same] == "cached"
    assert len({first, same, different}) == 2
