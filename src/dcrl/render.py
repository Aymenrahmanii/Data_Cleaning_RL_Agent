"""Turning typed tables into the text an export would contain, and back."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from dcrl.schema import ColumnSpec, TargetSchema


def is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def fmt_plain(value: Any, spec: ColumnSpec) -> str:
    """The clean text form of one cell: what an uncorrupted export would contain."""
    if is_missing(value):
        return ""
    if spec.dtype == "int":
        return str(int(value))
    if spec.dtype == "float":
        return f"{float(value):.{spec.decimals}f}"
    if spec.dtype == "date":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value)


def render_plain(df: pd.DataFrame, schema: TargetSchema) -> pd.DataFrame:
    """All cells as clean text, columns in schema order."""
    return pd.DataFrame(
        {c.name: [fmt_plain(v, c) for v in df[c.name]] for c in schema.columns}, dtype=object
    )


def to_csv(df: pd.DataFrame) -> str:
    return str(df.to_csv(index=False, lineterminator="\n"))


def read_messy(csv_text: str) -> pd.DataFrame:
    """How the environment loads a messy export: every cell as text, nothing parsed."""
    return pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
