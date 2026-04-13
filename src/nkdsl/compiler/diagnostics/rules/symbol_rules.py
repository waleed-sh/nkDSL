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


"""Symbol and index-bound diagnostics rules."""

from __future__ import annotations

from nkdsl.compiler.diagnostics.models import DSLDiagnostic
from nkdsl.compiler.diagnostics.rules.base import AbstractDiagnosticRule
from nkdsl.compiler.diagnostics.rules.base import DiagnosticRuleContext
from nkdsl.compiler.diagnostics.traversals import iter_term_static_index_nodes


class UnresolvedFreeSymbolsRule(AbstractDiagnosticRule):
    """Reports unresolved free symbols present in the operator IR."""

    @property
    def name(self) -> str:
        return "unresolved_free_symbols"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        free_symbols = tuple(sorted(context.ir.free_symbols))
        if not free_symbols:
            return ()
        shown = ", ".join(f"%{name}" for name in free_symbols[:8])
        suffix = "" if len(free_symbols) <= 8 else f", ... +{len(free_symbols) - 8} more"
        return (
            DSLDiagnostic.create(
                code="NKDSL-E001",
                severity="error",
                message=f"Unresolved free symbol(s): {shown}{suffix}.",
                operator_name=context.ir.operator_name,
                suggestion=(
                    "Replace with constants or bind symbols before compilation. "
                    "DSL compilation requires all free symbols to be resolved."
                ),
                context={"free_symbols": free_symbols},
            ),
        )


class StaticIndexBoundsRule(AbstractDiagnosticRule):
    """Reports static source/target index reads outside Hilbert bounds."""

    @property
    def name(self) -> str:
        return "static_index_bounds"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        diagnostics: list[DSLDiagnostic] = []
        hilbert_size = int(context.ir.hilbert_size)
        for term in context.ir.terms:
            for op_name, flat_index in iter_term_static_index_nodes(term):
                if 0 <= flat_index < hilbert_size:
                    continue
                diagnostics.append(
                    DSLDiagnostic.create(
                        code="NKDSL-E002",
                        severity="error",
                        message=(
                            f"{op_name} uses flat index {flat_index}, but hilbert_size is "
                            f"{hilbert_size}."
                        ),
                        operator_name=context.ir.operator_name,
                        term_name=term.name,
                        suggestion="Use a static index in [0, hilbert_size).",
                        context={
                            "flat_index": flat_index,
                            "hilbert_size": hilbert_size,
                            "op_name": op_name,
                        },
                    )
                )
        return tuple(diagnostics)


__all__ = ["UnresolvedFreeSymbolsRule", "StaticIndexBoundsRule"]
