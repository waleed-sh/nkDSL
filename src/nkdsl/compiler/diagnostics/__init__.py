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


"""Compiler diagnostics models, rule runners, and formatting helpers."""

from nkdsl.compiler.diagnostics.collector import (
    filter_diagnostics_by_minimum_severity,
)
from nkdsl.compiler.diagnostics.collector import (
    run_default_diagnostics,
)
from nkdsl.compiler.diagnostics.formatting import (
    format_diagnostic,
)
from nkdsl.compiler.diagnostics.formatting import (
    format_diagnostics_block,
)
from nkdsl.compiler.diagnostics.models import (
    DSLDiagnostic,
)
from nkdsl.compiler.diagnostics.models import (
    DiagnosticSeverity,
)
from nkdsl.compiler.diagnostics.models import (
    count_diagnostics_by_severity,
)
from nkdsl.compiler.diagnostics.models import (
    diagnostic_severity_at_least,
)
from nkdsl.compiler.diagnostics.models import (
    diagnostic_severity_rank,
)
from nkdsl.compiler.diagnostics.models import (
    normalize_diagnostic_severity,
)

__all__ = [
    "DSLDiagnostic",
    "DiagnosticSeverity",
    "normalize_diagnostic_severity",
    "diagnostic_severity_rank",
    "diagnostic_severity_at_least",
    "count_diagnostics_by_severity",
    "run_default_diagnostics",
    "filter_diagnostics_by_minimum_severity",
    "format_diagnostic",
    "format_diagnostics_block",
]
