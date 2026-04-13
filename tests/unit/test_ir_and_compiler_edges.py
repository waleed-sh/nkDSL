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

from typing import Any

import netket as nk
import pytest

import nkdsl
from nkdsl.compiler.compiler import SymbolicCompiler
from nkdsl.compiler.compiler import compile_symbolic_operator
from nkdsl.compiler.compiler import reset_default_symbolic_compiler
from nkdsl.compiler.core.context import SymbolicCompilationContext
from nkdsl.compiler.core.options import SymbolicCompilerOptions
from nkdsl.compiler.core.pass_report import SymbolicPassReport
from nkdsl.compiler.defaults import default_symbolic_artifact_store
from nkdsl.compiler.defaults import default_symbolic_lowerer_registry
from nkdsl.compiler.defaults import default_symbolic_operator_lowering_registry
from nkdsl.compiler.defaults import default_symbolic_pass_pipeline
from nkdsl.compiler.defaults import reset_default_symbolic_singletons
from nkdsl.compiler.lowering.operator_registry import DEFAULT_SYMBOLIC_OPERATOR_LOWERING
from nkdsl.compiler.passes.validation import SymbolicValidationPass
from nkdsl.errors import SymbolicCompilerError
from nkdsl.ir.program import SymbolicOperatorIR
from nkdsl.ir.term import KBodyIteratorSpec
from nkdsl.ir.term import SymbolicIRTerm
from nkdsl.ir.update import UpdateOp
from nkdsl.ir.update import UpdateProgram
from nkdsl.ir.validate import validate_symbolic_ir

pytestmark = pytest.mark.unit


def _simple_operator(name: str = "edge-op"):
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, name, hermitian=True)
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )
    return hi, op


def _context_for_ir(ir: SymbolicOperatorIR, operator: Any, *, strict: bool = True):
    return SymbolicCompilationContext(
        operator=operator,
        ir=ir,
        options=SymbolicCompilerOptions(strict_validation=strict),
        metadata={"source": "unit"},
    )


def _invalid_symbol_scope_ir() -> SymbolicOperatorIR:
    term = SymbolicIRTerm.create(
        name="bad_scope",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=nkdsl.AmplitudeExpr.symbol("bogus:i:value") > 0,
        update_program=nkdsl.identity().to_program(),
        amplitude=1.0,
    )
    return SymbolicOperatorIR.from_terms(
        operator_name="bad-scope",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(term,),
    )


def test_program_rendering_fingerprint_and_ir_constructor_guards():
    iterator = KBodyIteratorSpec(labels=("i",), index_sets=((0,), (1,)))
    term = SymbolicIRTerm.create(
        name="diag",
        iterator=iterator,
        predicate=nkdsl.site("i") >= 0,
        update_program=nkdsl.identity().to_program(),
        amplitude=nkdsl.site("i").value + nkdsl.symbol("alpha"),
        metadata={"kind": "diag"},
    )
    ir = SymbolicOperatorIR.from_terms(
        operator_name="render",
        hilbert_size=2,
        dtype_str="float64",
        is_hermitian=True,
        terms=(term,),
        metadata={"block": "A"},
    )

    payload = ir.as_dict()
    assert payload["operator_name"] == "render"
    assert payload["metadata"] == [("block", "A")]
    assert payload["terms"][0]["name"] == "diag"

    text = str(ir)
    assert 'symbolic.operator @"render"' in text
    assert "free symbols: [%alpha]" in text
    assert "term #0" in text
    assert "SymbolicOperatorIR(" in repr(ir)

    fp_1 = ir.static_fingerprint()
    fp_2 = ir.static_fingerprint()
    assert fp_1 == fp_2
    assert ir.metadata_dict()["block"] == "A"
    assert ir.free_symbols == {"alpha"}

    with pytest.raises(ValueError, match="non-empty string"):
        SymbolicOperatorIR(
            operator_name=" ",
            mode="symbolic",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=False,
        )
    with pytest.raises(ValueError, match="Unsupported IR mode"):
        SymbolicOperatorIR(
            operator_name="x",
            mode="unsupported",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=False,
        )
    with pytest.raises(ValueError, match="positive integer"):
        SymbolicOperatorIR(
            operator_name="x",
            mode="symbolic",
            hilbert_size=0,
            dtype_str="float64",
            is_hermitian=False,
        )


def test_validation_non_symbolic_empty_and_update_param_errors():
    non_symbolic = SymbolicOperatorIR(
        operator_name="jax-only",
        mode="jax_kernel",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=True,
        terms=(),
        metadata=(),
    )
    summary = validate_symbolic_ir(non_symbolic)
    assert summary["mode"] == "jax_kernel"
    assert summary["term_count"] == 0

    empty_symbolic = SymbolicOperatorIR(
        operator_name="empty",
        mode="symbolic",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(),
        metadata=(),
    )
    with pytest.raises(ValueError, match="has no terms"):
        validate_symbolic_ir(empty_symbolic)

    base_iter = KBodyIteratorSpec(labels=("i",), index_sets=((0,),))
    cases = [
        (
            UpdateOp.from_mapping(
                kind="write_site", params={"value": nkdsl.AmplitudeExpr.constant(1)}
            ),
            "requires a 'site' parameter",
        ),
        (
            UpdateOp.from_mapping(kind="write_site", params={"site": nkdsl.site("i").index}),
            "requires a 'value' parameter",
        ),
        (
            UpdateOp.from_mapping(
                kind="shift_site", params={"delta": nkdsl.AmplitudeExpr.constant(1)}
            ),
            "shift_site requires a 'site' parameter",
        ),
        (
            UpdateOp.from_mapping(kind="shift_site", params={"site": nkdsl.site("i").index}),
            "shift_site requires a 'delta' parameter",
        ),
        (
            UpdateOp.from_mapping(
                kind="shift_mod_site", params={"delta": nkdsl.AmplitudeExpr.constant(1)}
            ),
            "shift_mod_site requires a 'site' parameter",
        ),
        (
            UpdateOp.from_mapping(kind="shift_mod_site", params={"site": nkdsl.site("i").index}),
            "shift_mod_site requires a 'delta' parameter",
        ),
        (
            UpdateOp.from_mapping(kind="swap_sites", params={"site_a": nkdsl.site("i").index}),
            "swap_sites requires 'site_a' and 'site_b'",
        ),
    ]

    for op, message in cases:
        term = SymbolicIRTerm.create(
            name="bad",
            iterator=base_iter,
            predicate=True,
            update_program=UpdateProgram(ops=(op,)),
            amplitude=1.0,
        )
        bad_ir = SymbolicOperatorIR.from_terms(
            operator_name="bad-op",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=False,
            terms=(term,),
        )
        with pytest.raises(ValueError, match=message):
            validate_symbolic_ir(bad_ir)

    class _LegacyIterator:
        kind = "legacy"
        label_a = "i"
        label_b = "j"

    legacy_term = SymbolicIRTerm.create(
        name="legacy",
        iterator=_LegacyIterator(),
        predicate=True,
        update_program=nkdsl.identity().to_program(),
        amplitude=nkdsl.AmplitudeExpr.symbol("site:i:value"),
    )
    legacy_ir = SymbolicOperatorIR.from_terms(
        operator_name="legacy",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(legacy_term,),
    )
    assert validate_symbolic_ir(legacy_ir)["term_count"] == 1


def test_validation_pass_non_strict_and_strict_modes():
    _, good_operator = _simple_operator("good")
    bad_ir = _invalid_symbol_scope_ir()

    vpass = SymbolicValidationPass()
    non_strict_ctx = _context_for_ir(bad_ir, good_operator, strict=False)
    result = vpass.run(non_strict_ctx)
    assert result is not None and result["valid"] is False
    assert non_strict_ctx.analysis("validation_summary")["valid"] is False

    strict_ctx = _context_for_ir(bad_ir, good_operator, strict=True)
    with pytest.raises(ValueError, match="unknown namespace"):
        vpass.run(strict_ctx)


def test_update_program_cond_render_and_updateop_guards():
    cond = nkdsl.Update.cond(
        nkdsl.site("i") > 0,
        if_true=nkdsl.write("i", 1),
        if_false=None,
    ).to_program()
    assert "cond(" in str(cond)
    assert "identity" in str(cond)

    program = UpdateProgram().append(
        UpdateOp.from_mapping(
            kind="write_site",
            params={"site": nkdsl.site("i").index, "value": nkdsl.AmplitudeExpr.constant(1)},
        )
    )
    program = program.extend(UpdateProgram())
    assert program.op_count == 1
    assert "UpdateProgram(" in repr(program)

    op = UpdateOp.from_mapping(kind="invalidate_branch", params={"reason": "x"})
    assert "invalidate" in str(op)
    assert "UpdateOp(" in repr(op)

    up_a = nkdsl.shift("i", +1)
    up_b = nkdsl.shift("i", +1)
    up_c = nkdsl.shift("i", -1)
    assert up_a == up_b
    assert up_a != up_c
    assert hash(up_a) == hash(up_b)
    assert "Update([" in repr(up_a)
    assert (up_a == 1) is False

    with pytest.raises(ValueError, match="Unsupported update operation kind"):
        UpdateOp(kind="unknown", params=())


def test_context_defaults_and_compiler_failure_paths():
    hi, op = _simple_operator("ctx")
    ctx = SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(),
        metadata={"origin": "test"},
    )
    assert ctx.operator is op
    assert ctx.ir.operator_name == "ctx"
    assert ctx.options.backend_preference == "auto"
    assert ctx.metadata["origin"] == "test"
    assert ctx.analyses == {}

    ctx.set_metadata("new_key", 5)
    assert ctx.metadata["new_key"] == 5
    ctx.set_analysis("alpha", {"x": 1})
    assert ctx.analysis("alpha") == {"x": 1}
    assert ctx.analysis("missing", default=7) == 7
    assert ctx.require_analysis("alpha") == {"x": 1}
    with pytest.raises(ValueError, match="is missing"):
        ctx.require_analysis("missing")

    ctx.add_pass_report(SymbolicPassReport.create(pass_name="p", duration_ms=0.1))
    ctx.set_selected_backend("jax")
    ctx.set_selected_lowerer("jax_symbolic_v1")
    summary = ctx.summary()
    assert summary["selected_backend"] == "jax"
    assert summary["selected_lowerer"] == "jax_symbolic_v1"
    assert summary["pass_count"] == 1

    # Defaults wiring
    reset_default_symbolic_singletons()
    store_a = default_symbolic_artifact_store()
    store_b = default_symbolic_artifact_store()
    assert store_a is store_b
    target_a = default_symbolic_operator_lowering_registry()
    target_b = default_symbolic_operator_lowering_registry()
    assert target_a is target_b
    assert target_a.resolve().name == DEFAULT_SYMBOLIC_OPERATOR_LOWERING
    assert default_symbolic_pass_pipeline().pass_names() == (
        "symbolic_validation",
        "symbolic_diagnostics",
        "symbolic_normalization",
        "symbolic_max_conn_size_analysis",
        "symbolic_fusion_planning",
    )
    assert "jax_symbolic_v1" in default_symbolic_lowerer_registry().lowerer_names

    class _PreFailPipeline:
        def run_pre_cache(self, _context):
            raise RuntimeError("preboom")

        def run_post_cache(self, _context):
            return None

        def pass_names(self):
            return ("prefail",)

    class _PostFailPipeline:
        def run_pre_cache(self, _context):
            return None

        def run_post_cache(self, _context):
            raise RuntimeError("postboom")

        def pass_names(self):
            return ("postfail",)

    class _LoweringFailLowerer:
        name = "failing"
        backend = "jax"

        def supports(self, _context):
            return True

        def lower(self, _context):
            raise RuntimeError("lowerboom")

    class _LoweringFailRegistry:
        lowerer_names = ("failing",)

        def resolve(self, _context):
            return _LoweringFailLowerer()

    with pytest.raises(SymbolicCompilerError, match="Pre-cache pass failed"):
        SymbolicCompiler(
            pipeline=_PreFailPipeline(),
            lowerer_registry=default_symbolic_lowerer_registry(),
            cache_enabled=False,
        ).compile(op)

    with pytest.raises(SymbolicCompilerError, match="Post-cache pass failed"):
        SymbolicCompiler(
            pipeline=_PostFailPipeline(),
            lowerer_registry=default_symbolic_lowerer_registry(),
            cache_enabled=False,
        ).compile(op)

    with pytest.raises(SymbolicCompilerError, match="Lowering failed"):
        SymbolicCompiler(
            pipeline=default_symbolic_pass_pipeline(),
            lowerer_registry=_LoweringFailRegistry(),
            cache_enabled=False,
        ).compile(op)

    compiler = SymbolicCompiler(cache_enabled=True)
    _ = compiler.compile_operator(op)
    assert compiler.cache_size >= 1
    assert "SymbolicCompiler(" in repr(compiler)
    assert "symbolic_validation" in compiler.pass_names
    assert "jax_symbolic_v1" in compiler.lowerer_names
    compiler.clear_cache()
    assert compiler.cache_size == 0

    reset_default_symbolic_compiler()
    c1 = compile_symbolic_operator(op)
    c2 = compile_symbolic_operator(op)
    assert hasattr(c1, "get_conn_padded")
    assert hasattr(c2, "get_conn_padded")
