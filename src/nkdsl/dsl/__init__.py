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


"""User-facing declarative symbolic operator DSL."""

from .context import ExpressionContext
from .clauses import register
from .iterators import AbstractIteratorClause
from .iterators import available_iterator_clause_names
from .iterators import register_iterator_clause
from .operator import SymbolicDiscreteJaxOperator
from .predicates import AbstractPredicateClause
from .predicates import available_predicate_clause_names
from .predicates import register_predicate_clause

from .rewrite import Update
from .rewrite import affine
from .rewrite import identity
from .rewrite import permute
from .rewrite import scatter
from .rewrite import shift
from .rewrite import shift_mod
from .rewrite import swap
from .rewrite import write

from .selectors import SiteSelector
from .selectors import emitted
from .selectors import global_index
from .selectors import site
from .selectors import symbol

__all__ = [
    "SymbolicDiscreteJaxOperator",
    "Update",
    "affine",
    "identity",
    "permute",
    "scatter",
    "shift",
    "shift_mod",
    "swap",
    "write",
    "AbstractIteratorClause",
    "AbstractPredicateClause",
    "register",
    "register_iterator_clause",
    "register_predicate_clause",
    "available_iterator_clause_names",
    "available_predicate_clause_names",
    "ExpressionContext",
    "site",
    "emitted",
    "symbol",
    "global_index",
    "SiteSelector",
]
