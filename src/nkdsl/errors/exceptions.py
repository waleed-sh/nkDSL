"""Custom exception hierarchy for nkdsl."""

from __future__ import annotations


class NKDSLError(Exception):
    """Base exception for all nkdsl-specific errors."""

    default_message: str = "An nkdsl error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        hint: str | None = None,
        details: str | None = None,
    ) -> None:
        primary = message or self.default_message
        parts = [primary]
        if hint:
            parts.append(f"Hint: {hint}")
        if details:
            parts.append(f"Details: {details}")
        super().__init__("\n".join(parts))


class SymbolicError(NKDSLError):
    """Base exception for symbolic-DSL and symbolic-compiler errors."""


class SymbolicOperatorError(SymbolicError):
    """Base exception for symbolic operator construction and execution errors."""


class SymbolicOperatorExecutionError(SymbolicOperatorError):
    """Raised when a symbolic operator is executed before compilation."""

    default_message = "This symbolic operator cannot be executed before it is compiled."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message,
            hint="Call `.compile()` first, then run `get_conn_padded(...)` on the compiled operator.",
        )


class SymbolicCompilationError(SymbolicError):
    """Base exception for symbolic compilation failures."""


class SymbolicCompilerError(SymbolicCompilationError):
    """Raised when symbolic compilation fails at any pipeline stage."""

    default_message = "Symbolic compilation failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message,
            hint=(
                "Check the symbolic IR produced by `.to_ir()` and verify term predicates, "
                "update programs, and matrix elements."
            ),
        )


__all__ = [
    "NKDSLError",
    "SymbolicError",
    "SymbolicOperatorError",
    "SymbolicOperatorExecutionError",
    "SymbolicCompilationError",
    "SymbolicCompilerError",
]
