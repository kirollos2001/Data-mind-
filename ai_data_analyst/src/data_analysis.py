"""Utilities for loading CSV data and producing concise dataset summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class DatasetSummary:
    """Container holding the dataframe alongside textual and structured summaries."""

    dataframe: pd.DataFrame
    text: str
    details: Mapping[str, Any]
    encoding: str = "utf-8"  # Track which encoding was used


def _format_float(value: Optional[float]) -> str:
    """Return a compact string representation for numeric values."""
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def _collect_column_summary(series: pd.Series, max_top_values: int = 3) -> Dict[str, Any]:
    """Build a structured summary for a single dataframe column."""
    dtype_name = str(series.dtype)
    missing_count = int(series.isna().sum())
    total = len(series)
    missing_pct = round((missing_count / total) * 100, 2) if total else 0.0

    summary: Dict[str, Any] = {
        "name": series.name,
        "dtype": dtype_name,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }

    if pd.api.types.is_numeric_dtype(series):
        summary["statistics"] = {
            "mean": _format_float(series.mean()),
            "median": _format_float(series.median()),
            "std": _format_float(series.std()),
            "min": _format_float(series.min()),
            "max": _format_float(series.max()),
        }
    elif pd.api.types.is_datetime64_any_dtype(series):
        summary["statistics"] = {
            "min": _format_float(series.min()),
            "max": _format_float(series.max()),
        }
    else:
        # For categorical columns
        unique_count = series.nunique(dropna=True)
        summary["unique_count"] = unique_count
        
        # If unique values < 50, include all unique values
        if unique_count < 50:
            all_unique = series.dropna().astype(str).unique().tolist()
            summary["all_unique_values"] = sorted(all_unique)
        else:
            # Otherwise, just show top N values
            value_counts = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(max_top_values)
            )
            summary["top_values"] = [
                {"value": index, "count": int(count)}
                for index, count in value_counts.items()
            ]

    return summary


def _build_text_summary(details: Mapping[str, Any]) -> str:
    """Compose a compact textual summary that can be sent to the LLM."""
    lines: List[str] = []

    shape = details.get("shape", {})
    lines.append(
        f"Dataset with {shape.get('rows', 'NA')} rows and {shape.get('columns', 'NA')} columns."
    )

    missing = details.get("missing_values", {})
    lines.append(
        f"Total missing cells: {missing.get('total_missing', 'NA')} "
        f"({missing.get('missing_pct', 'NA')}%)."
    )

    column_summaries: Iterable[Mapping[str, Any]] = details.get("columns", [])
    for column in column_summaries:
        column_line = [
            f"{column.get('name')} ({column.get('dtype')})",
            f"missing={column.get('missing_count')}",
            f"{column.get('missing_pct')}% missing",
        ]

        stats = column.get("statistics")
        if isinstance(stats, Mapping):
            stat_parts = [
                f"{key}={value}"
                for key, value in stats.items()
                if value is not None
            ]
            if stat_parts:
                column_line.append("stats[" + ", ".join(stat_parts) + "]")

        # Show all unique values if available (for categorical with < 50 unique values)
        all_unique = column.get("all_unique_values")
        if isinstance(all_unique, list):
            unique_count = column.get("unique_count", len(all_unique))
            formatted = ", ".join(str(v) for v in all_unique)
            column_line.append(f"unique_count={unique_count}")
            column_line.append("all_values[" + formatted + "]")
        else:
            # Show top values for high-cardinality categorical columns
            unique_count = column.get("unique_count")
            if unique_count is not None:
                column_line.append(f"unique_count={unique_count}")
            
            top_values = column.get("top_values")
            if isinstance(top_values, Iterable):
                formatted = ", ".join(
                    f"{item.get('value')} ({item.get('count')})"
                    for item in top_values
                )
                if formatted:
                    column_line.append("top_values[" + formatted + "]")

        lines.append(" - " + "; ".join(column_line))

    return "\n".join(lines)


def get_summary(file_like: Any, *, max_top_values: int = 3) -> DatasetSummary:
    """Load a CSV file-like object and return a structured summary for downstream use."""
    if hasattr(file_like, "seek"):
        file_like.seek(0)

    # Try multiple encodings to handle different file formats
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    dataframe = None
    last_error = None
    used_encoding = 'utf-8'
    
    for encoding in encodings:
        try:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            dataframe = pd.read_csv(file_like, encoding=encoding)
            used_encoding = encoding
            break  # Success! Stop trying other encodings
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
    
    if dataframe is None:
        raise ValueError(
            f"Unable to read CSV file with any of the supported encodings: {', '.join(encodings)}. "
            f"Last error: {last_error}"
        )
    
    dataframe.columns = [str(col) for col in dataframe.columns]

    column_details = [
        _collect_column_summary(dataframe[column], max_top_values=max_top_values)
        for column in dataframe.columns
    ]

    total_cells = dataframe.shape[0] * dataframe.shape[1]
    missing_cells = int(dataframe.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0

    details: Dict[str, Any] = {
        "shape": {"rows": dataframe.shape[0], "columns": dataframe.shape[1]},
        "columns": column_details,
        "missing_values": {
            "total_missing": missing_cells,
            "missing_pct": missing_pct,
        },
        "preview_rows": dataframe.head(5).to_dict(orient="records"),
    }

    text_summary = _build_text_summary(details)

    return DatasetSummary(
        dataframe=dataframe,
        text=text_summary,
        details=details,
        encoding=used_encoding,
    )
