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
import pytest

import nkdsl
from nkdsl.compiler.compiler import SymbolicCompiler
from nkdsl.compiler.core.context import SymbolicCompilationContext
from nkdsl.compiler.core.options import SymbolicCompilerOptions
from nkdsl.compiler.passes.diagnostics import SymbolicDiagnosticsPass
from nkdsl.errors import SymbolicCompilerError
from nkdsl.errors import SymbolicDiagnosticsError

pytestmark = pytest.mark.unit


def _context_for_operator(
    operator,
    *,
    strict_validation: bool = True,
    diagnostics_enabled: bool = True,
    diagnostics_min_severity: str = "warning",
    fail_on_warnings: bool = False,
) -> SymbolicCompilationContext:
    return SymbolicCompilationContext(
        operator=operator,
        ir=operator.to_ir(),
        options=SymbolicCompilerOptions(
            strict_validation=strict_validation,
            diagnostics_enabled=diagnostics_enabled,
            diagnostics_min_severity=diagnostics_min_severity,
            fail_on_warnings=fail_on_warnings,
            cache_enabled=False,
        ),
    )


def test_diagnostics_pass_can_be_disabled():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "disabled")
        .for_each_site("i")
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
    )
    ctx = _context_for_operator(op, diagnostics_enabled=False)
    payload = SymbolicDiagnosticsPass().run(ctx)
    assert payload is not None
    assert payload["enabled"] is False
    assert ctx.analysis("dsl_diagnostics_summary")["enabled"] is False


def test_diagnostics_pass_reports_unresolved_symbols_as_errors():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "free-symbol")
        .for_each_site("i")
        .emit(nkdsl.identity(), matrix_element=nkdsl.symbol("J"))
        .build()
    )
    strict_ctx = _context_for_operator(op, strict_validation=True, diagnostics_min_severity="error")
    with pytest.raises(SymbolicDiagnosticsError, match="NKDSL-E001"):
        SymbolicDiagnosticsPass().run(strict_ctx)

    non_strict_ctx = _context_for_operator(
        op,
        strict_validation=False,
        diagnostics_min_severity="error",
    )
    with pytest.warns(UserWarning, match="NKDSL-E001"):
        SymbolicDiagnosticsPass().run(non_strict_ctx)
    assert non_strict_ctx.analysis("dsl_diagnostics_summary")["visible_error"] == 1


def test_diagnostics_pass_reports_static_index_bounds():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "static-oob")
        .globally()
        .emit(nkdsl.identity(), matrix_element=nkdsl.source_index(7))
        .build()
    )
    ctx = _context_for_operator(op, diagnostics_min_severity="error")
    with pytest.raises(SymbolicDiagnosticsError, match="NKDSL-E002"):
        SymbolicDiagnosticsPass().run(ctx)


def test_diagnostics_pass_can_fail_on_warnings():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "warning")
        .for_each_site("i")
        .where(False)
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
    )
    ctx = _context_for_operator(
        op,
        strict_validation=False,
        diagnostics_min_severity="warning",
        fail_on_warnings=True,
    )
    with pytest.raises(SymbolicDiagnosticsError, match="NKDSL-W101"):
        SymbolicDiagnosticsPass().run(ctx)


def test_diagnostics_pass_structural_rules_and_info_visibility():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "structure")
        .for_each_site("i")
        .max_conn_size(1)
        .emit(nkdsl.identity(), matrix_element=1.0)
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
    )
    ctx = _context_for_operator(
        op,
        strict_validation=False,
        diagnostics_min_severity="info",
    )
    with pytest.warns(UserWarning):
        SymbolicDiagnosticsPass().run(ctx)
    diagnostics = ctx.analysis("dsl_diagnostics")
    codes = {entry["code"] for entry in diagnostics}
    assert "NKDSL-W103" in codes
    assert "NKDSL-W104" in codes
    assert "NKDSL-I201" in codes


def test_diagnostics_connectivity_rules_for_illegal_states_constraints_and_support():
    hi = nk.hilbert.Fock(n_max=1, N=2, n_particles=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "connectivity")
        .for_each_site("i")
        .emit(nkdsl.shift("i", +1), matrix_element=1.0)
        .build()
    )
    ctx = SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(
            strict_validation=False,
            diagnostics_min_severity="warning",
            cache_enabled=False,
            lint_state_sample_size=8,
            lint_branch_sample_cap=256,
        ),
    )
    with pytest.warns(UserWarning):
        SymbolicDiagnosticsPass().run(ctx)
    diagnostics = ctx.analysis("dsl_diagnostics")
    codes = {entry["code"] for entry in diagnostics}
    assert "NKDSL-W301" in codes
    assert "NKDSL-W302" in codes
    assert "NKDSL-W303" in codes


def test_compiler_surfaces_diagnostics_failures_during_compile():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "compile-visible")
        .for_each_site("i")
        .emit(nkdsl.identity(), matrix_element=nkdsl.symbol("alpha"))
        .build()
    )
    compiler = SymbolicCompiler(
        options=SymbolicCompilerOptions(
            cache_enabled=False,
            diagnostics_min_severity="error",
        )
    )
    with pytest.raises(SymbolicCompilerError, match="NKDSL-E001"):
        compiler.compile_operator(op)
