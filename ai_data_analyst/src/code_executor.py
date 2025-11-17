"""Safe execution utilities for running LLM-generated analysis code."""

from __future__ import annotations

import builtins
import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

# --- Allowed libraries inside sandbox ---
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import re


# --- Allowed builtins (security sandbox) ---
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
        "ImportError",
        "NameError",
        "Exception",
    )
}

# Block __import__ by making it raise an error
def _blocked_import(*args, **kwargs):
    raise ImportError("Import statements are not allowed in this execution environment.")

SAFE_BUILTINS["__import__"] = _blocked_import


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
    """Extract pandas DataFrames from arbitrarily nested iterables, excluding original dataframe."""
    tables: List[pd.DataFrame] = []
    for value in values:
        if isinstance(value, pd.DataFrame):
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

    # --- Security Guard: block any import attempts ---
    dangerous_import_pattern = re.compile(
        r"(^|\n)\s*(import |from )|__import__",
        re.IGNORECASE
    )

    if dangerous_import_pattern.search(code):
        return ExecutionResult(
            error=(
                "Import statements are not allowed in this execution environment.\n"
                "All required libraries such as pandas (pd), numpy (np), plotly (px, go), "
                "and make_subplots are already available."
            )
        )

    # --- Allowed sandbox globals ---
    safe_globals: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,

        # Manually exposed libraries (no imports in user code)
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "pio": pio,

        # Original DataFrame
        "df": dataframe.copy(),

        # Exceptions
        "ValueError": ValueError,
        "KeyError": KeyError,
        "TypeError": TypeError,
        "ImportError": ImportError,
        "NameError": NameError,
        "Exception": Exception,
    }

    local_scope: Dict[str, Any] = {}
    stdout_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer):
            code_lines = code.strip().split("\n")

            last_line = code_lines[-1].strip() if code_lines else ""

            is_expression = (
                last_line 
                and not last_line.startswith(
                    ('if ', 'for ', 'while ', 'def ', 'class ', 'import ', 'from ')
                ) 
                and '=' not in last_line.split('#')[0]
            )

            if is_expression and len(code_lines) > 1:
                exec("\n".join(code_lines[:-1]), safe_globals, local_scope)
                result = eval(last_line, safe_globals, local_scope)
                print(result)
            else:
                exec(code, safe_globals, local_scope)

    except Exception:
        return ExecutionResult(
            stdout=stdout_buffer.getvalue(),
            error=traceback.format_exc(limit=4)
        )

    all_values = list(safe_globals.values()) + list(local_scope.values())
    figures = _collect_figures(all_values)
    tables = _collect_tables(all_values, dataframe)

    return ExecutionResult(
        figures=figures,
        tables=tables,
        stdout=stdout_buffer.getvalue().strip(),
    )
