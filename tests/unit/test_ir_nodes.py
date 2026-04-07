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

import pytest

import nkdsl
from nkdsl.ir.expressions import _collect_free_symbols
from nkdsl.ir.predicates import _collect_free_symbols_pred
from nkdsl.ir.term import EmissionSpec
from nkdsl.ir.term import KBodyIteratorSpec
from nkdsl.ir.term import SymbolicIRTerm
from nkdsl.ir.update import UpdateOp
from nkdsl.ir.update import UpdateProgram
from nkdsl.ir.validate import validate_symbolic_ir

pytestmark = pytest.mark.unit


def test_amplitude_expr_constructors_render_and_coercion():
    x_i = nkdsl.AmplitudeExpr.symbol("site:i:value")
    x_j = nkdsl.AmplitudeExpr.symbol("site:j:value")
    free = nkdsl.AmplitudeExpr.symbol("kappa")

    expr = nkdsl.AmplitudeExpr.wrap_mod(
        nkdsl.AmplitudeExpr.abs_(nkdsl.AmplitudeExpr.pow((x_i + 1) * (x_j - 2), 2) / 3)
    )

    rendered = str(expr)
    assert "wrap(" in rendered
    assert "x[i]" in rendered
    assert "x[j]" in rendered

    assert str(nkdsl.AmplitudeExpr.static_index(2)) == "x[2]"
    assert str(nkdsl.AmplitudeExpr.static_emitted_index(1)) == "x'[1]"

    with pytest.raises(TypeError, match="Boolean values"):
        nkdsl.AmplitudeExpr.constant(True)

    with pytest.raises(ValueError, match="non-negative"):
        nkdsl.AmplitudeExpr.static_index(-1)

    with pytest.raises(ValueError, match="Unsupported amplitude-expression op"):
        nkdsl.AmplitudeExpr(op="unknown", args=())

    assert nkdsl.coerce_amplitude_expr(1.2).op == "const"
    assert nkdsl.coerce_amplitude_expr("alpha").op == "symbol"

    with pytest.raises(TypeError, match="Cannot coerce"):
        nkdsl.coerce_amplitude_expr([1, 2, 3])

    symbols: set[str] = set()
    _collect_free_symbols(free + x_i, symbols)
    assert symbols == {"kappa"}


def test_predicate_expr_logic_and_free_symbols():
    x_i = nkdsl.site("i").value
    pred = nkdsl.PredicateExpr.and_(
        nkdsl.PredicateExpr.gt(x_i, 0),
        nkdsl.PredicateExpr.or_(
            nkdsl.PredicateExpr.lt("theta", 1),
            nkdsl.PredicateExpr.not_(False),
        ),
    )

    rendered = str(pred)
    assert "&&" in rendered
    assert "||" in rendered
    assert "(x[i] > 0)" in rendered

    assert nkdsl.coerce_predicate_expr(True).op == "const"
    with pytest.raises(TypeError, match="Cannot use an AmplitudeExpr"):
        nkdsl.coerce_predicate_expr(x_i)
    with pytest.raises(TypeError, match="Cannot coerce"):
        nkdsl.coerce_predicate_expr(1)

    syms: set[str] = set()
    _collect_free_symbols_pred(pred, syms)
    assert "theta" in syms


def test_update_ops_and_program_rendering():
    update = (
        nkdsl.shift("i", +1)
        .shift_mod("i", -1)
        .write("j", nkdsl.site("i").value)
        .swap("i", "j")
        .affine("i", scale=2, bias=-1)
        .permute("i", "j", 0)
        .scatter([0, 1], [3, nkdsl.site("j").value])
        .invalidate(reason="test")
    )

    program = update.to_program()
    assert isinstance(program, UpdateProgram)
    assert program.op_count == 8
    assert program.has_invalidate()

    text = str(program)
    assert "wrap" in text
    assert "invalidate" in text
    assert "x'[" in text

    cond = nkdsl.Update.cond(
        nkdsl.site("i") > 0,
        if_true=nkdsl.write("i", 1),
        if_false=nkdsl.write("i", 2),
    )
    cond_program = cond.to_program()
    assert cond_program.op_count == 1
    assert cond_program.ops[0].kind == "cond_branch"

    with pytest.raises(ValueError, match="same length"):
        nkdsl.scatter([0], [1, 2])

    op = UpdateOp.from_mapping(kind="write_site", params={"site": x_i(), "value": 1})
    assert op.get("site") is not None
    assert op.get("missing", default=123) == 123


def x_i():
    return nkdsl.site("i").index


def test_symbolic_term_ir_and_validation():
    iterator = KBodyIteratorSpec(labels=("i",), index_sets=((0,), (1,)))
    em_a = EmissionSpec(
        update_program=nkdsl.shift("i", +1).to_program(),
        amplitude=nkdsl.site("i").value + nkdsl.symbol("alpha"),
        branch_tag="raise",
    )
    em_b = EmissionSpec(
        update_program=nkdsl.shift("i", -1).to_program(),
        amplitude=nkdsl.site("i").value - nkdsl.symbol("alpha"),
        branch_tag="lower",
    )

    term = SymbolicIRTerm.create(
        name="t0",
        iterator=iterator,
        predicate=nkdsl.site("i") >= 0,
        update_program=em_a.update_program,
        amplitude=em_a.amplitude,
        max_conn_size_hint=4,
        emissions=(em_a, em_b),
    )

    assert len(term.effective_emissions) == 2
    assert term.max_conn_size_hint == 4
    assert "alpha" in term.free_symbols
    assert "max_conn_size" in str(term)

    ir = nkdsl.SymbolicOperatorIR.from_terms(
        operator_name="t",
        hilbert_size=2,
        dtype_str="float64",
        is_hermitian=False,
        terms=(term,),
    )
    summary = validate_symbolic_ir(ir)
    assert summary["mode"] == "symbolic"
    assert summary["term_count"] == 1

    bad_term = SymbolicIRTerm.create(
        name="bad",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=nkdsl.site("k") > 0,
        update_program=nkdsl.identity().to_program(),
        amplitude=1.0,
    )
    bad_ir = nkdsl.SymbolicOperatorIR.from_terms(
        operator_name="bad",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(bad_term,),
    )
    with pytest.raises(ValueError, match="not bound by its iterator"):
        validate_symbolic_ir(bad_ir)

    with pytest.raises(ValueError, match="requires at least one term"):
        nkdsl.SymbolicOperatorIR.from_terms(
            operator_name="empty",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=False,
            terms=(),
        )

    with pytest.raises(ValueError, match="positive integer"):
        SymbolicIRTerm.create(
            name="x",
            iterator=iterator,
            predicate=True,
            update_program=nkdsl.identity().to_program(),
            amplitude=1.0,
            max_conn_size_hint=0,
        )
