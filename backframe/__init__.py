# SPDX-License-Identifier: MIT
# (C) 2024-present Bartosz Sławecki (bswck)
"""
`backframe`.

Inspect the caller.
"""

from __future__ import annotations

import ast
import inspect
from contextlib import suppress
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType
    from typing import Any, TypeVar

    T = TypeVar("T")
    ExprT = TypeVar("ExprT", bound=ast.expr)

__all__ = (
    "resolve_expression",
    "map_args_to_identifiers",
)


def _get_frame_namespace(frame: FrameType) -> dict[str, Any]:
    """
    Get the namespace of the frame.

    Parameters
    ----------
    frame
        Frame to get the namespace from.

    Returns
    -------
    Namespace of the frame.

    """
    return {**frame.f_builtins, **frame.f_globals, **frame.f_locals}


def resolve_expression(
    lines: list[str],
    resolver: Callable[[ast.stmt], list[ExprT]],
) -> ExprT | None:
    """
    Resolve an expression of interest from `lines`.

    Parameters
    ----------
    lines
        Lines to get the statement from.
    resolver
        Typically a node visitor that extracts expressions of interest from statements.

    Returns
    -------
    First matching statement or `None` if no statement was found.

    """
    stmts: list[ast.stmt] = []

    for n in range(len(lines)):
        chunk = lines[: n + 1]
        with suppress(SyntaxError):
            stmts.extend(ast.parse("\n".join(chunk), mode="exec").body)
            break

    matching_exprs = [*chain.from_iterable(filter(None, map(resolver, stmts)))]

    if not matching_exprs:
        return None

    if len(matching_exprs) > 1:
        msg = (
            "Multiple matching statements found: "
            f"{', '.join(map(ast.dump, matching_exprs))}"
        )
        raise ValueError(msg)

    return matching_exprs[0]


class CallResolver(ast.NodeVisitor):
    """Resolve a simple call to named callable."""

    # ruff: noqa: N802

    def __init__(self, filter_name: str) -> None:
        self.filter_name = filter_name
        self.call_exprs: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == self.filter_name:
            self.call_exprs.append(node)


def _resolve_calls(stmt: ast.stmt, function_name: str) -> list[ast.Call]:
    """
    Find all calls to a function named as specified by `function_name`.

    Parameters
    ----------
    stmt
        Statement to walk through.
    function_name
        Function to match.

    Returns
    -------
    List of `ast.Call` objects.

    """
    resolver = CallResolver(function_name)
    resolver.visit(stmt)
    return resolver.call_exprs


def map_args_to_identifiers(
    *objects: Any,
    function: Callable[..., Any] | None = None,
    stack_offset: int = 2,
) -> dict[str, Any]:
    """
    Map objects (passed to the caller function) to their original identifiers.

    >>> def test(*args):
    ...     print(map_args_to_identifiers(*args))
    ...
    >>> foo = 1; bar = 2; biz = 3
    >>> test(foo)
    {'foo': 1}
    >>> test(
    ...     bar)
    {'bar': 2}
    >>> baz = 4; test(bar,
    ... biz,
    ...          baz,
    ... )
    {'bar': 2, 'biz': 3, 'baz': 4}

    Parameters
    ----------
    objects
        Objects to map to identifiers.
    function
        Function to get the caller expression from.
    stack_offset
        Stack level to get the caller expression from.

    Returns
    -------
    Dictionary with identifiers as keys and objects as values.

    """
    current_frame = inspect.currentframe()
    if current_frame is None:
        return {}

    caller_frame = current_frame.f_back
    if caller_frame is None:
        return {}

    caller_function_name = caller_frame.f_code.co_name
    if not caller_function_name.isidentifier():
        msg = "Cannot call `map_to_identifiers` outside functions."
        raise RuntimeError(msg)

    if function is None:
        function = _get_frame_namespace(caller_frame)[caller_function_name]

    frame = inspect.stack()[stack_offset].frame
    source_lines, bof = inspect.getsourcelines(frame)
    cutoff_lines = source_lines[frame.f_lineno - 1 - bof :]
    resolver = partial(_resolve_calls, function_name=function.__name__)

    call = resolve_expression(cutoff_lines, resolver)
    if call is None:
        return {}

    mapping: dict[str, Any] = {}

    for arg, obj in zip(call.args, objects):
        if not isinstance(arg, ast.Name):
            msg = f"Expected `ast.Name` but got `{arg}`."
            raise TypeError(msg)
        mapping[arg.id] = obj

    return mapping
