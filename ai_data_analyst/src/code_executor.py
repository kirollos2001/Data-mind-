"""Safe execution utilities for running LLM-generated analysis code."""

from __future__ import annotations

import builtins
import io
import json
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

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
class ChartData:
    """Structured data extracted from a chart for LLM consumption."""
    chart_type: str  # e.g., "bar", "line", "pie", "scatter", etc.
    data: List[Dict[str, Any]]  # Data in records format
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional chart info


@dataclass
class TableData:
    """Structured data from a table for LLM consumption."""
    columns: List[str]
    data: List[Dict[str, Any]]  # Data in records format
    row_count: int
    summary_stats: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Artifacts produced when executing LLM-generated code."""
    figures: List[go.Figure] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    stdout: str = ""
    error: str | None = None
    chart_data: List[ChartData] = field(default_factory=list)  # Structured chart data
    table_data: List[TableData] = field(default_factory=list)  # Structured table data

    @property
    def success(self) -> bool:
        return self.error is None
    
    def get_chart_data_json(self) -> str:
        """Get all chart data as a JSON string for LLM consumption."""
        if not self.chart_data:
            return ""
        return json.dumps(
            [
                {
                    "chart_type": cd.chart_type,
                    "data": cd.data,
                    "metadata": cd.metadata,
                }
                for cd in self.chart_data
            ],
            indent=2,
            default=str
        )
    
    def get_table_data_json(self) -> str:
        """Get all table data as a JSON string for LLM consumption."""
        if not self.table_data:
            return ""
        return json.dumps(
            [
                {
                    "columns": td.columns,
                    "row_count": td.row_count,
                    "data": td.data,
                    "summary_stats": td.summary_stats,
                }
                for td in self.table_data
            ],
            indent=2,
            default=str
        )
    
    def get_all_data_summary(self) -> str:
        """Get a comprehensive summary of all chart and table data for LLM."""
        parts = []
        
        if self.chart_data:
            parts.append("## Chart Data\n")
            for idx, cd in enumerate(self.chart_data, 1):
                parts.append(f"### Chart {idx}: {cd.chart_type}\n")
                if cd.metadata:
                    parts.append(f"Metadata: {json.dumps(cd.metadata, indent=2, default=str)}\n")
                parts.append(f"Data records: {len(cd.data)}\n")
                if cd.data:
                    # Show first few records as sample
                    sample = cd.data[:5]
                    parts.append(f"Sample data (first {len(sample)} records):\n")
                    parts.append(json.dumps(sample, indent=2, default=str))
                    if len(cd.data) > 5:
                        parts.append(f"\n... and {len(cd.data) - 5} more records")
                parts.append("\n")
        
        if self.table_data:
            parts.append("## Table Data\n")
            for idx, td in enumerate(self.table_data, 1):
                parts.append(f"### Table {idx}\n")
                parts.append(f"Columns: {', '.join(td.columns)}\n")
                parts.append(f"Row count: {td.row_count}\n")
                if td.summary_stats:
                    parts.append(f"Summary statistics: {json.dumps(td.summary_stats, indent=2, default=str)}\n")
                if td.data:
                    # Show first few records as sample
                    sample = td.data[:5]
                    parts.append(f"Sample data (first {len(sample)} records):\n")
                    parts.append(json.dumps(sample, indent=2, default=str))
                    if len(td.data) > 5:
                        parts.append(f"\n... and {len(td.data) - 5} more records")
                parts.append("\n")
        
        return "\n".join(parts)


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


def _extract_data_from_figure(figure: go.Figure) -> Optional[ChartData]:
    """Extract underlying data from a Plotly figure.
    
    Attempts to reconstruct the DataFrame-like data from the figure's traces.
    Returns None if extraction fails.
    """
    if not figure.data:
        return None
    
    try:
        # Determine chart type from the first trace
        first_trace = figure.data[0]
        chart_type = first_trace.type if hasattr(first_trace, 'type') else "unknown"
        
        # Collect data from all traces
        all_data = []
        metadata = {
            "trace_count": len(figure.data),
            "layout_title": figure.layout.title.text if figure.layout.title else None,
        }
        
        # For each trace, extract the data
        for trace_idx, trace in enumerate(figure.data):
            trace_data = {}
            
            # Extract x, y, z values
            if hasattr(trace, 'x') and trace.x is not None:
                if isinstance(trace.x, (list, tuple, np.ndarray)):
                    trace_data['x'] = [float(x) if isinstance(x, (int, float, np.number)) else str(x) 
                                      for x in trace.x]
                else:
                    trace_data['x'] = [trace.x]
            
            if hasattr(trace, 'y') and trace.y is not None:
                if isinstance(trace.y, (list, tuple, np.ndarray)):
                    trace_data['y'] = [float(y) if isinstance(y, (int, float, np.number)) else str(y) 
                                      for y in trace.y]
                else:
                    trace_data['y'] = [trace.y]
            
            if hasattr(trace, 'z') and trace.z is not None:
                if isinstance(trace.z, (list, tuple, np.ndarray)):
                    trace_data['z'] = [float(z) if isinstance(z, (int, float, np.number)) else str(z) 
                                      for z in trace.z]
                else:
                    trace_data['z'] = [trace.z]
            
            # Extract labels (for pie charts, bar charts with categories, etc.)
            if hasattr(trace, 'labels') and trace.labels is not None:
                if isinstance(trace.labels, (list, tuple, np.ndarray)):
                    trace_data['labels'] = [str(l) for l in trace.labels]
                else:
                    trace_data['labels'] = [str(trace.labels)]
            
            # Extract values (for pie charts, bar charts, etc.)
            if hasattr(trace, 'values') and trace.values is not None:
                if isinstance(trace.values, (list, tuple, np.ndarray)):
                    trace_data['values'] = [float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                           for v in trace.values]
                else:
                    trace_data['values'] = [trace.values]
            
            # Extract text (for annotations, hover text, etc.)
            if hasattr(trace, 'text') and trace.text is not None:
                if isinstance(trace.text, (list, tuple, np.ndarray)):
                    trace_data['text'] = [str(t) for t in trace.text]
                else:
                    trace_data['text'] = [str(trace.text)]
            
            # Extract customdata if present
            if hasattr(trace, 'customdata') and trace.customdata is not None:
                trace_data['customdata'] = trace.customdata.tolist() if hasattr(trace.customdata, 'tolist') else list(trace.customdata)
            
            # Extract name for the trace
            if hasattr(trace, 'name') and trace.name:
                trace_data['trace_name'] = str(trace.name)
            
            # If we have x and y, create records (one per data point)
            if 'x' in trace_data and 'y' in trace_data:
                x_vals = trace_data['x']
                y_vals = trace_data['y']
                max_len = max(len(x_vals), len(y_vals))
                
                # Pad shorter list with None
                x_vals = list(x_vals) + [None] * (max_len - len(x_vals))
                y_vals = list(y_vals) + [None] * (max_len - len(y_vals))
                
                for i in range(max_len):
                    record = {
                        'x': x_vals[i],
                        'y': y_vals[i],
                    }
                    if 'text' in trace_data and i < len(trace_data['text']):
                        record['text'] = trace_data['text'][i]
                    if 'trace_name' in trace_data:
                        record['series'] = trace_data['trace_name']
                    all_data.append(record)
            elif 'labels' in trace_data and 'values' in trace_data:
                # Pie chart or similar
                labels = trace_data['labels']
                values = trace_data['values']
                max_len = max(len(labels), len(values))
                
                labels = list(labels) + [None] * (max_len - len(labels))
                values = list(values) + [None] * (max_len - len(values))
                
                for i in range(max_len):
                    record = {
                        'label': labels[i],
                        'value': values[i],
                    }
                    if 'trace_name' in trace_data:
                        record['series'] = trace_data['trace_name']
                    all_data.append(record)
            else:
                # Fallback: just store the trace data as-is
                if trace_data:
                    all_data.append(trace_data)
        
        # If we couldn't extract structured data, return None
        if not all_data:
            return None
        
        return ChartData(
            chart_type=chart_type,
            data=all_data,
            metadata=metadata
        )
    except Exception:
        # If extraction fails, return None
        return None


def _dataframe_to_table_data(df: pd.DataFrame) -> TableData:
    """Convert a pandas DataFrame to structured TableData."""
    # Convert to records format
    try:
        data_records = df.to_dict(orient="records")
        # Convert numpy types to native Python types for JSON serialization
        for record in data_records:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                elif pd.isna(value):
                    record[key] = None
    except Exception:
        # Fallback to string representation
        data_records = []
    
    # Calculate summary statistics for numeric columns
    summary_stats = None
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].describe().to_dict()
            # Convert numpy types
            for col in summary_stats:
                for stat in summary_stats[col]:
                    val = summary_stats[col][stat]
                    if isinstance(val, (np.integer, np.floating)):
                        summary_stats[col][stat] = float(val) if isinstance(val, np.floating) else int(val)
                    elif pd.isna(val):
                        summary_stats[col][stat] = None
    except Exception:
        pass
    
    return TableData(
        columns=list(df.columns),
        data=data_records,
        row_count=len(df),
        summary_stats=summary_stats
    )


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

    # Extract structured data from figures
    chart_data = []
    for figure in figures:
        extracted = _extract_data_from_figure(figure)
        if extracted:
            chart_data.append(extracted)
    
    # Convert tables to structured format
    table_data = [_dataframe_to_table_data(table) for table in tables]

    return ExecutionResult(
        figures=figures,
        tables=tables,
        stdout=stdout_buffer.getvalue().strip(),
        chart_data=chart_data,
        table_data=table_data,
    )
