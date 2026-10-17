"""Render the package's existing RST markup in MkDocs API docstrings."""

from __future__ import annotations

import ast
import re
import textwrap

from griffe import Extension


def _literal_blocks(value: str) -> str:
    """Fence indented RST literals while preserving their section indentation."""
    lines = value.splitlines()
    result = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.rstrip().endswith("::"):
            indent = len(line) - len(line.lstrip())
            start = index + 1
            while start < len(lines) and not lines[start].strip():
                start += 1
            end = start
            while end < len(lines):
                candidate = lines[end]
                if candidate.strip() and len(candidate) - len(candidate.lstrip()) <= indent:
                    break
                end += 1
            if start < end:
                body = textwrap.dedent("\n".join(lines[start:end])).rstrip()
                section = line.strip().rstrip(":").lower() in ("example", "examples")
                padding = " " * (indent + 4 if section else indent)
                language = "python"
                if body.startswith(">>>"):
                    language = "pycon"
                else:
                    try:
                        ast.parse(body)
                    except SyntaxError:
                        language = "text"
                result.extend(
                    [line.rstrip()[:-1]]
                    + ([] if section else [""])
                    + [padding + "```" + language]
                    + [padding + part for part in body.splitlines()]
                    + [padding + "```", ""]
                )
                index = end
                continue
        result.append(line)
        index += 1
    return "\n".join(result)


def prose_without_semicolons(value: str) -> str:
    """Replace prose punctuation without touching code, math or HTML entities."""
    result = []
    fence = None
    math = False
    doctest_indent = None
    for line in value.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if doctest_indent is not None and (not stripped or indent < doctest_indent):
            doctest_indent = None
        if stripped.startswith(">>>"):
            doctest_indent = indent
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else marker
        elif stripped == "$$":
            math = not math
        elif fence is None and not math and doctest_indent is None:
            # Inline code and TeX spacing commands are syntax, not punctuation.
            parts = re.split(r"(`+[^`]*`+|\$[^$]*\$|&(?:#\w+|\w+);|\\;)", line)
            line = "".join(
                part if index % 2 else part.replace(";", ",") for index, part in enumerate(parts)
            )
        result.append(line)
    return "\n".join(result)


def markdown_docstring(value: str) -> str:
    """Convert inline roles and RST blocks without changing Google-style sections."""
    value = re.sub(r":(?:py:)?(?:class|func|meth|mod|attr|obj):`([^`]+)`", r"`\1`", value)
    value = re.sub(r":math:`([^`]+)`", r"$\1$", value)
    value = re.sub(r"``([^`]+)``", r"`\1`", value)

    def block(match: re.Match) -> str:
        indent, kind, language, body = match.groups()
        body = textwrap.dedent(body).strip("\n")
        if kind in ("code-block", "code"):
            return (
                indent
                + "```"
                + language.strip()
                + "\n"
                + textwrap.indent(body, indent)
                + "\n"
                + indent
                + "```\n\n"
            )
        if kind == "math":
            return indent + "$$\n" + textwrap.indent(body, indent) + "\n" + indent + "$$\n\n"
        return indent + "!!! " + kind + "\n\n" + textwrap.indent(body, indent + "    ") + "\n\n"

    value = re.sub(
        r"^([ \t]*)\.\. (code-block|code|math|note|warning|tip|important)::([^\n]*)\n((?:[ \t]+[^\n]*\n|\n)+)",
        block,
        value + "\n",
        flags=re.M,
    )
    value = _literal_blocks(value)
    # Google-style sections recognise the plural heading. A singular heading
    # becomes an admonition, where >>> is interpreted as nested blockquotes.
    value = re.sub(r"(?m)^([ \t]*)Example:[ \t]*$", r"\1Examples:", value)
    value = re.sub(
        r"(?m)^([ \t]*)Examples:[ \t]*\n(?:[ \t]*\n)+(?=[ \t]+\S)",
        r"\1Examples:\n",
        value,
    )
    return prose_without_semicolons(value).rstrip()


class MarkdownDocstrings(Extension):
    """Translate presentation markup after Griffe reads each source object."""

    def on_instance(self, *, obj, **kwargs) -> None:
        if obj.docstring:
            obj.docstring.value = markdown_docstring(obj.docstring.value)
