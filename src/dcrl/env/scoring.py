"""The reward, computed once at the end on the hidden copy.

R = F1_rows x A_cells - 0.02 x max(0, calls - 4), floored at 0.

- A is the share of cells equal to the clean table (numbers within 0.001, text exact, dates to
  the day, column types right). Rows are matched by the id column; a clean row with no
  matching result row counts as wrong in every cell.
- F1 is over rows: precision is matched rows over result rows, recall is matched rows over
  clean rows. Dropping real rows and keeping duplicates both cost.
- calls counts every tool call, including submit. Only calls beyond four are charged.
- R is 0 when nothing was submitted, when the episode ran no code, or when the code fails on
  the hidden copy. The floor at 0 is our choice: it keeps the reward non-negative for GRPO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcrl.compare import TOLERANCE, dtype_ok, equal_cells
from dcrl.schema import TargetSchema

CALL_PENALTY = 0.02
FREE_CALLS = 4


@dataclass(frozen=True)
class Score:
    cell_accuracy: float
    row_f1: float
    precision: float
    recall: float
    matched_rows: int
    calls: int
    reward: float


def _id_value(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        number = float(str(value)) if isinstance(value, str) else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or not number.is_integer():
        return None
    return int(number)


def score_table(
    result: pd.DataFrame | None,
    clean: pd.DataFrame,
    schema: TargetSchema,
    calls: int,
    *,
    ran_code: bool = True,
) -> Score:
    penalty = CALL_PENALTY * max(0, calls - FREE_CALLS)
    if result is None or not ran_code:
        return Score(0.0, 0.0, 0.0, 0.0, 0, calls, 0.0)

    n = len(clean)
    id_name = schema.columns[0].name
    first_at: dict[int, int] = {}
    if id_name in result.columns and list(result.columns).count(id_name) == 1:
        for pos, value in enumerate(result[id_name].tolist()):
            key = _id_value(value)
            if key is not None:
                first_at.setdefault(key, pos)

    positions = np.full(n, -1, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    for j, value in enumerate(clean[id_name].tolist()):
        found = first_at.get(int(value))
        if found is not None:
            positions[j] = found
            matched[j] = True
    matched_rows = int(matched.sum())

    cells = np.zeros((n, len(schema.columns)), dtype=bool)
    for j, spec in enumerate(schema.columns):
        if spec.name not in result.columns or list(result.columns).count(spec.name) != 1:
            continue
        column = result[spec.name].reset_index(drop=True)
        if not dtype_ok(column, spec):
            continue
        aligned = column.reindex(positions).reset_index(drop=True)
        same = equal_cells(aligned, clean[spec.name].reset_index(drop=True), spec, TOLERANCE)
        cells[:, j] = same & matched

    accuracy = float(cells.mean()) if cells.size else 0.0
    precision = matched_rows / len(result) if len(result) else 0.0
    recall = matched_rows / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reward = max(0.0, f1 * accuracy - penalty)
    return Score(accuracy, f1, precision, recall, matched_rows, calls, reward)
