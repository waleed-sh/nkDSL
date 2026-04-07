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
