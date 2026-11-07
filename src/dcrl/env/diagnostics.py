"""What the agent can see about its table: profile() and check_schema().

Both read the table and say what is odd. Neither ever states the right answer.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dcrl.compare import dtype_ok
from dcrl.schema import ColumnSpec, TargetSchema

PLACEHOLDERS = frozenset({"k.A.", "n/a", "N/A", "-", "0", "NULL", "?", "#NV", "keine Angabe"})

PATTERNS: dict[str, re.Pattern[str]] = {
    "comma_decimal": re.compile(r"-?\d+,\d+"),
    "dot_thousands": re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"),
    "date_iso": re.compile(r"\d{4}-\d{2}-\d{2}"),
    "date_de": re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"),
    "date_de_short": re.compile(r"\d{1,2}\.\d{1,2}\.\d{2}"),
    "date_us": re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
    "date_text": re.compile(r"\d{1,2}\. [^\d\s]+\.? \d{4}"),
    "excel_serial": re.compile(r"\d{5}"),
    "timezone_suffix": re.compile(r"T\d{2}:\d{2}|UTC|MEZ|[+-]\d{2}:?\d{2}$"),
    "unit_in_cell": re.compile(r"\d\s*(?:kg|g|mm|cm|ml|bar|h|m|l)\b|°C"),
    "currency_or_percent": re.compile(r"€|EUR|%"),
    "whitespace": re.compile(r"^\s|\s$|\s{2,}| "),
    "mojibake": re.compile("Ã|Â|â€"),
}
SAMPLE_VALUES = 8


def _text(value: Any) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _column_name_counts(df: pd.DataFrame) -> list[Any]:
    return list(df.columns)


def profile(df: pd.DataFrame, column: str | None = None) -> str:
    if column is not None and column not in df.columns:
        return f"No column named {column!r}. Columns: {_column_name_counts(df)}"
    names = [column] if column is not None else _column_name_counts(df)
    blocks = []
    for name in names:
        if list(df.columns).count(name) != 1:
            blocks.append(f"## {name}: duplicated column name, cannot profile")
            continue
        values = [_text(v) for v in df[name]]
        present = [v for v in values if v is not None and v.strip() != ""]
        empty = len(values) - len(present)
        distinct = len(set(present))
        lines = [
            f"## {name}",
            f"type {df[name].dtype}; rows {len(values)}; empty {empty}; distinct {distinct}",
        ]
        if column is not None:
            sample = list(dict.fromkeys(present))[:SAMPLE_VALUES]
            lines.append("sample values: " + ", ".join(repr(v) for v in sample))
        counts: dict[str, int] = {}
        for label, pattern in PATTERNS.items():
            hits = sum(1 for v in present if pattern.search(v))
            if hits:
                counts[label] = hits
        placeholders = sum(1 for v in present if v.strip() in PLACEHOLDERS)
        if placeholders:
            counts["placeholder"] = placeholders
        lines.append(
            "odd patterns: "
            + (", ".join(f"{k} {v}" for k, v in counts.items()) if counts else "none")
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _is_empty(value: Any) -> bool:
    text = _text(value)
    return text is None or text.strip() == ""


def _breaks(text: str, spec: ColumnSpec) -> bool:
    if spec.dtype == "int":
        return re.fullmatch(r"-?\d+", text) is None
    if spec.dtype == "float":
        return re.fullmatch(r"-?\d+(?:\.\d+)?", text) is None
    if spec.dtype == "date":
        return re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) is None
    if spec.dtype == "category":
        return spec.allowed is not None and text not in spec.allowed
    return False


def check_schema(df: pd.DataFrame, schema: TargetSchema) -> str:
    problems: list[str] = []
    columns = _column_name_counts(df)
    missing = [n for n in schema.names if n not in columns]
    extra = [str(c) for c in columns if c not in schema]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"extra columns, not in the schema: {extra}")

    id_name = schema.columns[0].name
    if id_name in columns and columns.count(id_name) == 1:
        ids = [_text(v) for v in df[id_name]]
        repeated = len(ids) - len(set(i for i in ids if i is not None))
        if repeated:
            problems.append(f"{id_name}: {repeated} rows repeat the id of another row")

    for spec in schema.columns:
        if spec.name in missing or columns.count(spec.name) != 1:
            continue
        series = df[spec.name]
        if not dtype_ok(series, spec):
            problems.append(f"{spec.name}: dtype {series.dtype}, the schema says {spec.dtype}")
        empty_rows = 0
        bad_rows = 0
        examples: list[str] = []
        for value in series:
            if _is_empty(value):
                empty_rows += 1
                continue
            text = _text(value)
            assert text is not None
            if _breaks(text, spec):
                bad_rows += 1
                if len(examples) < 3 and text not in examples:
                    examples.append(text)
        if empty_rows and not spec.nullable:
            problems.append(
                f"{spec.name}: {empty_rows} rows are empty but the column is not nullable"
            )
        if bad_rows:
            shown = ", ".join(repr(e) for e in examples)
            problems.append(f"{spec.name}: {bad_rows} rows break the schema, e.g. {shown}")

    if not problems:
        return "No schema violations: types, allowed values and empty cells match the schema."
    return "\n".join(f"- {p}" for p in problems)
