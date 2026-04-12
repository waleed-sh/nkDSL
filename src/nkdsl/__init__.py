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


"""
Symbolic operator subsystem for NetKet.

Typical workflow::

    from nkdsl import SymbolicDiscreteJaxOperator
    from nkdsl.dsl import site, shift, swap

    hop = (
        SymbolicDiscreteJaxOperator(hi, "hopping")
        .for_each_pair("i", "j")
        .where(site("i") > 0)
        .emit(shift("i", -1).shift("j", +1), matrix_element=1.0)
        .build()
    )
    compiled = hop.compile()
    xp, mels = compiled.get_conn_padded(x_batch)
"""

#
#
#   Primary user-facing entry points

from nkdsl.dsl import SymbolicDiscreteJaxOperator
from nkdsl.dsl import Update
from nkdsl.dsl import AbstractIteratorClause
from nkdsl.dsl import AbstractPredicateClause
from nkdsl.dsl import ExpressionContext
from nkdsl.dsl import SiteSelector
from nkdsl.dsl import available_iterator_clause_names
from nkdsl.dsl import available_predicate_clause_names
from nkdsl.dsl import affine
from nkdsl.dsl import identity
from nkdsl.dsl import permute
from nkdsl.dsl import register
from nkdsl.dsl import register_iterator_clause
from nkdsl.dsl import register_predicate_clause
from nkdsl.dsl import scatter
from nkdsl.dsl import shift
from nkdsl.dsl import shift_mod
from nkdsl.dsl import site
from nkdsl.dsl import emitted
from nkdsl.dsl import source_index
from nkdsl.dsl import swap
from nkdsl.dsl import symbol
from nkdsl.dsl import target_index
from nkdsl.dsl import write

#
#
#   Operator types

from nkdsl.core import AbstractSymbolicOperator
from nkdsl.core import CompiledOperator
from nkdsl.core import SymbolicOperator
from nkdsl.core import SymbolicOperatorSum

#
#
#   Compiler

from nkdsl.compiler import SymbolicCacheKey
from nkdsl.compiler import SymbolicCompiledArtifact
from nkdsl.compiler import SymbolicCompilationContext
from nkdsl.compiler import (
    DEFAULT_SYMBOLIC_OPERATOR_LOWERING,
)
from nkdsl.compiler import (
    SymbolicCompilationSignature,
)
from nkdsl.compiler import SymbolicCompiler
from nkdsl.compiler import SymbolicCompilerOptions
from nkdsl.compiler import SymbolicOperatorLoweringRegistry
from nkdsl.compiler import SymbolicOperatorLoweringTarget
from nkdsl.compiler import compile_symbolic_operator
from nkdsl.compiler import (
    default_symbolic_artifact_store,
)
from nkdsl.compiler import (
    default_symbolic_lowerer_registry,
)
from nkdsl.compiler import (
    default_symbolic_operator_lowering_registry,
)
from nkdsl.compiler import (
    default_symbolic_pass_pipeline,
)

#
#
#   IR

from nkdsl.ir import AmplitudeExpr
from nkdsl.ir import EmissionSpec
from nkdsl.ir import KBodyIteratorSpec
from nkdsl.ir import PredicateExpr
from nkdsl.ir import SymbolicIRTerm
from nkdsl.ir import SymbolicOperatorIR
from nkdsl.ir import UpdateOp
from nkdsl.ir import UpdateProgram
from nkdsl.ir import coerce_amplitude_expr
from nkdsl.ir import coerce_predicate_expr

#
#
#   Errors

from nkdsl.errors import NKDSLError
from nkdsl.errors import SymbolicCompilationError
from nkdsl.errors import SymbolicCompilerError
from nkdsl.errors import SymbolicError
from nkdsl.errors import SymbolicOperatorError
from nkdsl.errors import SymbolicOperatorExecutionError

#
#
#   Config
from nkdsl.configs import cfg

__all__ = [
    # Primary API
    "SymbolicDiscreteJaxOperator",
    "Update",
    "shift",
    "shift_mod",
    "write",
    "swap",
    "permute",
    "affine",
    "scatter",
    "identity",
    "site",
    "emitted",
    "symbol",
    "source_index",
    "target_index",
    "SiteSelector",
    "AbstractIteratorClause",
    "AbstractPredicateClause",
    "register",
    "register_iterator_clause",
    "register_predicate_clause",
    "available_iterator_clause_names",
    "available_predicate_clause_names",
    # Operator types
    "SymbolicOperator",
    "CompiledOperator",
    "AbstractSymbolicOperator",
    "SymbolicOperatorSum",
    "ExpressionContext",
    # Compiler
    "SymbolicCompiler",
    "compile_symbolic_operator",
    "SymbolicCompilerOptions",
    "SymbolicCompiledArtifact",
    "SymbolicCompilationContext",
    "SymbolicCacheKey",
    "SymbolicCompilationSignature",
    "default_symbolic_pass_pipeline",
    "default_symbolic_lowerer_registry",
    "default_symbolic_operator_lowering_registry",
    "default_symbolic_artifact_store",
    "DEFAULT_SYMBOLIC_OPERATOR_LOWERING",
    "SymbolicOperatorLoweringRegistry",
    "SymbolicOperatorLoweringTarget",
    # IR
    "SymbolicOperatorIR",
    "SymbolicIRTerm",
    "EmissionSpec",
    "KBodyIteratorSpec",
    "AmplitudeExpr",
    "PredicateExpr",
    "UpdateProgram",
    "UpdateOp",
    "coerce_amplitude_expr",
    "coerce_predicate_expr",
    # Errors
    "NKDSLError",
    "SymbolicError",
    "SymbolicOperatorError",
    "SymbolicOperatorExecutionError",
    "SymbolicCompilationError",
    "SymbolicCompilerError",
    # Configs
    "cfg",
]
