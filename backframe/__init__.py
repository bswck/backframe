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
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType
    from typing import Any, TypeVar

    T = TypeVar("T")

__all__ = (
    "get_first_expression",
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


def get_first_expression(
    lines: list[str],
    predicate: Callable[[ast.AST], bool] | None = None,
) -> ast.Expr | None:
    """
    Get the first expression from the lines that matches the predicate.

    Parameters
    ----------
    lines
        Lines to get the statement from.
    predicate
        Predicate to match the statement.

    Returns
    -------
    First matching statement or `None` if no statement was found.

    """
    exprs: list[ast.AST] = []

    for n in range(len(lines)):
        chunk = lines[: n + 1]
        with suppress(SyntaxError):
            exprs.extend(ast.parse("\n".join(chunk), mode="exec").body)
            break

    matching_exprs = [expr for expr in exprs if predicate is None or predicate(expr)]

    if not matching_exprs:
        return None

    if len(matching_exprs) > 1:
        msg = (
            "Multiple matching statements found: "
            f"{', '.join(map(ast.dump, matching_exprs))}"
        )
        raise ValueError(msg)

    return cast(ast.Expr, matching_exprs[0])


def _check_only_calls_function(expr: ast.AST, function_name: str) -> bool:
    """
    Get the predicate to match the statement that calls the function.

    Parameters
    ----------
    expr
        Expression to match.
    function_name
        Function to match.

    Returns
    -------
    Predicate to match the statement that calls the function.

    """
    if not isinstance(expr, (ast.Expr, ast.Assign)):
        return False

    if not isinstance(expr.value, ast.Call):
        return False

    if not isinstance(expr.value.func, ast.Name):
        return False

    if expr.value.func.id != function_name:
        return False

    return True


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
    >>> o = 1
    >>> test(o)
    {'o': 1}
    >>> test(
    ...     o)
    {'o': 1}
    >>> o = 2; test(
    ...     o,
    ... )
    {'o': 2}

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
    predicate = partial(_check_only_calls_function, function_name=function.__name__)

    expr = get_first_expression(cutoff_lines, predicate)
    if expr is None:
        return {}

    call: ast.Call = cast(ast.Call, expr.value)
    mapping: dict[str, Any] = {}

    for arg, obj in zip(call.args, objects):
        if not isinstance(arg, ast.Name):
            msg = f"Expected `ast.Name` but got `{arg}`."
            raise TypeError(msg)
        mapping[arg.id] = obj

    return mapping
