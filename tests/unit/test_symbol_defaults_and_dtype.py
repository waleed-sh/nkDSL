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
from nkdsl.compiler.lowering import jax_lowerer as jl
from nkdsl.dsl.context import ExpressionContext
from nkdsl.ir.expressions import _collect_free_symbols
from nkdsl.ir.expressions import parse_symbol_declaration_args

pytestmark = pytest.mark.unit


def test_symbol_declaration_infers_dtype_and_keeps_metadata():
    expr = nkdsl.symbol("J", default=1.0, doc="Coupling strength")
    name, declaration = parse_symbol_declaration_args(expr.args)

    assert name == "J"
    assert declaration["default"] == 1.0
    assert declaration["doc"] == "Coupling strength"
    assert declaration["dtype"] == np.asarray(1.0).dtype.name
    assert str(expr) == "%J"


def test_symbol_declaration_accepts_user_dtype_and_coerces_default():
    expr = nkdsl.symbol("J", default=1.0, dtype=np.float32)
    _name, declaration = parse_symbol_declaration_args(expr.args)

    assert declaration["dtype"] == "float32"
    out = jl.eval_amplitude(expr, {})
    assert np.asarray(out).dtype == np.dtype(np.float32)
    assert np.isclose(float(np.asarray(out)), 1.0)


def test_symbol_declaration_dtype_is_applied_to_runtime_bindings():
    expr = nkdsl.symbol("J", default=1.0, dtype=np.float32)
    out = jl.eval_amplitude(expr, {"J": np.float64(2.75)})

    assert np.asarray(out).dtype == np.dtype(np.float32)
    assert np.isclose(float(np.asarray(out)), 2.75)


def test_symbol_defaults_are_not_reported_as_free_symbols():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    op_with_default = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "with-default", hermitian=True)
        .globally()
        .emit(nkdsl.identity(), matrix_element=nkdsl.symbol("J", default=1.0))
        .build()
    )
    op_unresolved = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "unresolved", hermitian=True)
        .globally()
        .emit(nkdsl.identity(), matrix_element=nkdsl.symbol("K"))
        .build()
    )

    assert op_with_default.free_symbols == frozenset()
    assert op_with_default.to_ir().free_symbols == frozenset()
    assert op_unresolved.free_symbols == frozenset({"K"})

    compiled = op_with_default.compile(cache=False)
    _x_prime, mel = compiled.get_conn_padded(jnp.asarray([0], dtype=jnp.int32))
    assert np.isclose(np.asarray(mel)[0], 1.0)


def test_symbol_declaration_parse_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="at least one argument"):
        parse_symbol_declaration_args(())

    with pytest.raises(ValueError, match="either one item"):
        parse_symbol_declaration_args(("J", (), ()))

    with pytest.raises(TypeError, match="tuple of \\(key, value\\) pairs"):
        parse_symbol_declaration_args(("J", ["bad"]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported symbol declaration key"):
        parse_symbol_declaration_args(("J", (("unknown", 1),)))

    with pytest.raises(ValueError, match="Duplicate symbol declaration key"):
        parse_symbol_declaration_args(("J", (("doc", "a"), ("doc", "b"))))


def test_symbol_declaration_validates_dtype_inputs_and_default_conversion():
    with pytest.raises(ValueError, match="Unsupported symbol dtype declaration"):
        nkdsl.symbol("J", dtype="not-a-valid-dtype")

    with pytest.raises(TypeError, match="cannot be converted"):
        nkdsl.symbol("J", default="abc", dtype=np.float32)


def test_symbol_declaration_skips_empty_doc_and_none_dtype_entries():
    name, declaration = parse_symbol_declaration_args(
        ("J", (("doc", "  "), ("dtype", None), ("default", 3.0)))
    )

    assert name == "J"
    assert declaration == {"default": 3.0}


def test_expression_context_symbol_supports_default_doc_and_dtype():
    ctx = ExpressionContext()
    expr = ctx.symbol("g", default=2.0, doc="  Gauge coupling  ", dtype=np.float32)
    _name, declaration = parse_symbol_declaration_args(expr.args)

    assert declaration["default"] == 2.0
    assert declaration["doc"] == "Gauge coupling"
    assert declaration["dtype"] == "float32"

    free_symbols: set[str] = set()
    _collect_free_symbols(expr + nkdsl.symbol("unresolved"), free_symbols)
    assert free_symbols == {"unresolved"}
