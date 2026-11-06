"""Cell-by-cell comparison of a cleaned table with the clean original.

Numbers match within ``tol``, text matches exactly, dates match to the day, and a column
whose dtype does not fit the schema matches nowhere. Row alignment (for the row F1 of the
reward) is the environment's job; these functions assume rows are already aligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcrl.schema import ColumnSpec, TargetSchema

TOLERANCE = 1e-3


def dtype_ok(series: pd.Series, spec: ColumnSpec) -> bool:
    kind = series.dtype
    if spec.numeric:
        if pd.api.types.is_bool_dtype(kind) or not pd.api.types.is_numeric_dtype(kind):
            return False
        if spec.dtype == "int":
            values = series.dropna().astype("float64").to_numpy()
            return bool(np.all(values == np.round(values)))
        return True
    if spec.dtype == "date":
        return bool(pd.api.types.is_datetime64_dtype(kind))
    return pd.api.types.is_object_dtype(kind) or isinstance(kind, pd.StringDtype)


def _equal(result: pd.Series, clean: pd.Series, spec: ColumnSpec, tol: float) -> np.ndarray:
    r_na = result.isna().to_numpy()
    c_na = clean.isna().to_numpy()
    if spec.numeric:
        a = result.astype("float64").to_numpy()
        b = clean.astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):
            same = np.abs(a - b) <= tol + 1e-9
    elif spec.dtype == "date":
        same = result.to_numpy("datetime64[ns]") == clean.to_numpy("datetime64[ns]")
    else:
        a = result.to_numpy(object)
        b = clean.to_numpy(object)
        same = np.array([x == y for x, y in zip(a, b, strict=True)], dtype=bool)
    return np.where(r_na | c_na, r_na & c_na, same)


def cell_matches(
    result: pd.DataFrame, clean: pd.DataFrame, schema: TargetSchema, tol: float = TOLERANCE
) -> np.ndarray:
    """Boolean matrix (rows x schema columns) of matching cells; rows must be aligned."""
    if len(result) != len(clean):
        raise ValueError(f"row counts differ: {len(result)} vs {len(clean)}")
    result = result.reset_index(drop=True)
    clean = clean.reset_index(drop=True)
    out = np.zeros((len(clean), len(schema.columns)), dtype=bool)
    for j, spec in enumerate(schema.columns):
        if spec.name not in result.columns or not dtype_ok(result[spec.name], spec):
            continue
        out[:, j] = _equal(result[spec.name], clean[spec.name], spec, tol)
    return out


def mismatches(
    result: pd.DataFrame, clean: pd.DataFrame, schema: TargetSchema, tol: float = TOLERANCE
) -> list[str]:
    """Human-readable reasons the tables differ; empty when they match exactly."""
    problems: list[str] = []
    missing = [n for n in schema.names if n not in result.columns]
    extra = [str(n) for n in result.columns if n not in schema]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"extra columns: {extra}")
    if len(result) != len(clean):
        problems.append(f"row count {len(result)}, expected {len(clean)}")
        return problems
    for spec in schema.columns:
        if spec.name not in result.columns:
            continue
        if not dtype_ok(result[spec.name], spec):
            problems.append(f"{spec.name}: dtype {result[spec.name].dtype} is not {spec.dtype}")
            continue
        bad = ~_equal(result[spec.name].reset_index(drop=True), clean[spec.name], spec, tol)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            got = result[spec.name].iloc[i]
            want = clean[spec.name].iloc[i]
            problems.append(
                f"{spec.name}: {int(bad.sum())} cells differ, row {i}: {got!r} != {want!r}"
            )
    return problems


def frames_match(
    result: pd.DataFrame, clean: pd.DataFrame, schema: TargetSchema, tol: float = TOLERANCE
) -> bool:
    return not mismatches(result, clean, schema, tol)
