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
from nkdsl.dsl.context import ExpressionContext

pytestmark = pytest.mark.unit


def test_expression_context_methods_cover_main_api_surface():
    ctx = ExpressionContext()

    c = ctx.const(2)
    s = ctx.symbol("theta")
    i = ctx.site("i")
    e = ctx.emitted("i")

    assert str(c) == "2"
    assert str(s) == "%theta"
    assert i.label == "i"
    assert e.namespace == "emit"

    expr = ctx.sqrt(ctx.abs_(ctx.pow(ctx.neg(i.value + 1), 2)))
    assert "sqrt(" in str(expr)

    pred = ctx.all_of(ctx.gt(i.value, 0), ctx.any_of(ctx.lt(i.value, 2), ctx.not_(False)))
    assert "&&" in str(pred)

    assert ctx.eq(i.value, 1).op == "eq"
    assert ctx.ne(i.value, 1).op == "ne"
    assert ctx.lt(i.value, 1).op == "lt"
    assert ctx.le(i.value, 1).op == "le"
    assert ctx.gt(i.value, 1).op == "gt"
    assert ctx.ge(i.value, 1).op == "ge"

    assert ctx.coerce_amplitude(1).op == "const"
    assert ctx.coerce_predicate(True).op == "const"

    sq = ctx.sq_norm(i.value, e.value, 3)
    n = ctx.norm2(i.value, e.value)
    assert "*" in str(sq)
    assert "sqrt(" in str(n)

    edge = ctx.edge_value(edge_idx=2, gauge_copy=1, n_edges_per_copy=10)
    edge_em = ctx.emitted_edge_value(edge_idx=2, gauge_copy=1, n_edges_per_copy=10)
    comps = ctx.edge_components(edge_idx=2, n_edges_per_copy=10, gauge_dim=3)
    assert str(edge) == "x[12]"
    assert str(edge_em) == "x'[12]"
    assert len(comps) == 3
    assert "sqrt(" in str(ctx.edge_norm(edge_idx=1, n_edges_per_copy=4, gauge_dim=2))
    assert "*" in str(ctx.edge_sq_norm(edge_idx=1, n_edges_per_copy=4, gauge_dim=2))

    with pytest.raises(ValueError, match="at least one component"):
        ctx.sq_norm()


def test_site_selector_attr_and_comparisons():
    sel = nkdsl.site("i")
    emit_sel = nkdsl.emitted("j")

    assert str(sel.value) == "x[i]"
    assert str(sel.index) == "i"
    assert str(emit_sel.value) == "x'[j]"

    assert sel.as_site_ref().op == "symbol"
    assert sel.attr("foo").op == "symbol"
    assert sel.foo.op == "symbol"
    assert sel.abs().op == "abs_"
    assert "SiteSelector(" in repr(sel)

    assert (sel < 1).op == "lt"
    assert (sel <= 1).op == "le"
    assert (sel > 1).op == "gt"
    assert (sel >= 1).op == "ge"
    assert (sel == 1).op == "eq"  # type: ignore[comparison-overlap]
    assert (sel != 1).op == "ne"  # type: ignore[comparison-overlap]

    with pytest.raises(ValueError, match="non-empty string"):
        nkdsl.dsl.selectors.SiteSelector(" ")
    with pytest.raises(ValueError, match="non-empty string"):
        nkdsl.dsl.selectors.SiteSelector("i", namespace=" ")
    with pytest.raises(ValueError, match="non-empty"):
        sel.attr(" ")
