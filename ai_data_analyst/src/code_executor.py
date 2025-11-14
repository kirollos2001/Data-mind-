"""Safe execution utilities for running LLM-generated analysis code."""

from __future__ import annotations

import builtins
import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "range",
        "round",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
        "str",
        "repr",
        "print",
        "ValueError",
        "KeyError",
        "TypeError",
        "AttributeError",
        "IndexError",
    )
}


@dataclass
class ExecutionResult:
    """Artifacts produced when executing LLM-generated code."""

    figures: List[go.Figure] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    stdout: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def _collect_figures(values: Iterable[Any]) -> List[go.Figure]:
    """Extract Plotly figures from arbitrarily nested iterables."""
    figures: List[go.Figure] = []
    for value in values:
        if isinstance(value, go.Figure):
            figures.append(value)
        elif isinstance(value, (list, tuple, set)):
            figures.extend(_collect_figures(value))
        elif isinstance(value, dict):
            figures.extend(_collect_figures(value.values()))
    return figures


def _collect_tables(values: Iterable[Any], original_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Extract pandas DataFrames from arbitrarily nested iterables, excluding the original dataframe."""
    tables: List[pd.DataFrame] = []
    for value in values:
        if isinstance(value, pd.DataFrame):
            # Skip if it's the original dataframe (same shape and columns)
            is_original = (
                value.shape == original_df.shape and
                list(value.columns) == list(original_df.columns)
            )
            if not is_original:
                tables.append(value)
        elif isinstance(value, (list, tuple, set)):
            tables.extend(_collect_tables(value, original_df))
        elif isinstance(value, dict):
            tables.extend(_collect_tables(value.values(), original_df))
    return tables


def execute_code(code: str, dataframe: pd.DataFrame) -> ExecutionResult:
    """Execute LLM-provided Python code inside a restricted namespace."""
    if not code.strip():
        return ExecutionResult(error="No code to execute.")

    # Put df in safe_globals so it's accessible in list comprehensions and nested scopes
    safe_globals: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "df": dataframe.copy(),  # Make df globally accessible
        # Allow common exceptions
        "ValueError": ValueError,
        "KeyError": KeyError,
        "TypeError": TypeError,
    }
    local_scope: Dict[str, Any] = {}

    stdout_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer):
            # Split code into lines to capture last expression
            code_lines = code.strip().split('\n')
            
            # Check if last line is an expression (not assignment, not control flow)
            last_line = code_lines[-1].strip() if code_lines else ""
            is_expression = (
                last_line and 
                not last_line.startswith(('if ', 'for ', 'while ', 'def ', 'class ', 'import ', 'from ')) and
                '=' not in last_line.split('#')[0]  # Check before any comment
            )
            
            if is_expression and len(code_lines) > 1:
                # Execute all but last line
                exec('\n'.join(code_lines[:-1]), safe_globals, local_scope)
                # Evaluate and print last expression
                result = eval(last_line, safe_globals, local_scope)
                print(result)
            else:
                # Execute normally
                exec(code, safe_globals, local_scope)
                
    except Exception:
        error_message = traceback.format_exc(limit=4)
        return ExecutionResult(stdout=stdout_buffer.getvalue(), error=error_message)

    # Collect from both global and local scopes
    all_values = list(safe_globals.values()) + list(local_scope.values())
    figures = _collect_figures(all_values)
    # Pass original dataframe to filter it out from results
    tables = _collect_tables(all_values, dataframe)

    return ExecutionResult(
        figures=figures,
        tables=tables,
        stdout=stdout_buffer.getvalue().strip(),
    )
