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


"""Custom exception types for nkdsl."""

from nkdsl.errors.exceptions import NKDSLError
from nkdsl.errors.exceptions import SymbolicCompilationError
from nkdsl.errors.exceptions import SymbolicCompilerError
from nkdsl.errors.exceptions import SymbolicError
from nkdsl.errors.exceptions import SymbolicOperatorError
from nkdsl.errors.exceptions import SymbolicOperatorExecutionError

__all__ = [
    "NKDSLError",
    "SymbolicError",
    "SymbolicOperatorError",
    "SymbolicOperatorExecutionError",
    "SymbolicCompilationError",
    "SymbolicCompilerError",
]
