"""Structure: split and merged columns, duplicate rows, totals rows, repeated headers.

Applied in ``order`` after all cell corruptions and undone first, in reverse order. The id
column (always first) is never split or merged, so the row fixes can rely on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dcrl.corruptions.base import StructuralCorruption, fmt_plain, pick
from dcrl.pools import MERGE_SEPARATORS
from dcrl.schema import TargetSchema


def _insert_column(df: pd.DataFrame, at: int, name: str, values: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.insert(at, name, values)
    return df


class SplitColumn(StructuralCorruption):
    """'Max Müller' exported as two columns name_teil1='Max', name_teil2='Müller'."""

    name = "split_column"
    order = 1

    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        options = [c.name for c in schema.columns if c.dtype == "str" and c.name not in taken]
        if not options:
            return None
        column = pick(rng, options)
        return {"column": column, "parts": [f"{column}_teil1", f"{column}_teil2"]}

    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        column = params["column"]
        first_part, second_part = params["parts"]
        pieces = [str(v).partition(" ") for v in df[column]]
        at = list(df.columns).index(column)
        df = df.drop(columns=[column])
        df = _insert_column(df, at, second_part, [p[2] for p in pieces])
        return _insert_column(df, at, first_part, [p[0] for p in pieces])

    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str:
        c = repr(params["column"])
        a, b = (repr(p) for p in params["parts"])
        return (
            f"df[{c}] = df[{a}].where(df[{b}] == '', df[{a}] + ' ' + df[{b}])\n"
            f"df = df.drop(columns=[{a}, {b}])\n"
        )


class MergeColumns(StructuralCorruption):
    """Two text columns exported as one: 'München / Vertrieb'."""

    name = "merge_columns"
    order = 2

    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        options = [c.name for c in schema.columns if c.textual and c.name not in taken]
        if len(options) < 2:
            return None
        idx = sorted(rng.choice(len(options), size=2, replace=False).tolist())
        first, second = options[idx[0]], options[idx[1]]
        return {
            "columns": [first, second],
            "merged": f"{first}_{second}",
            "sep": pick(rng, MERGE_SEPARATORS),
        }

    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        first, second = params["columns"]
        merged = [f"{a}{params['sep']}{b}" for a, b in zip(df[first], df[second], strict=True)]
        at = list(df.columns).index(first)
        df = df.drop(columns=[first, second])
        return _insert_column(df, at, params["merged"], merged)

    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str:
        first, second = (repr(c) for c in params["columns"])
        m = repr(params["merged"])
        return (
            f"_parts = df[{m}].str.split({params['sep']!r}, n=1, expand=True, regex=False)\n"
            f"df[{first}] = _parts[0]\n"
            f"df[{second}] = _parts[1]\n"
            f"df = df.drop(columns=[{m}])\n"
        )


class DuplicateRows(StructuralCorruption):
    """Exact copies of some rows, each placed somewhere after its original."""

    name = "duplicate_rows"
    order = 3

    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        return {"share": round(float(rng.uniform(0.02, 0.08)), 3)}

    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = len(df)
        k = max(1, round(params["share"] * n))
        source = rng.choice(n, size=k, replace=False)
        keys = np.concatenate([np.arange(n, dtype=float), source + rng.uniform(0.01, n - source)])
        rows = pd.concat([df, df.iloc[source]], ignore_index=True)
        return rows.iloc[np.argsort(keys, kind="stable")].reset_index(drop=True)

    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str:
        return "df = df.drop_duplicates().reset_index(drop=True)\n"


class TotalsRow(StructuralCorruption):
    """A 'Summe' row at the bottom with the column totals."""

    name = "totals_row"
    order = 4
    labels = ("Summe", "Gesamt", "Summe gesamt", "Total")

    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        if not any(c.numeric and not c.is_id for c in schema.columns):
            return None
        return {"label": pick(rng, self.labels)}

    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        row = dict.fromkeys(df.columns, "")
        row[schema.columns[0].name] = params["label"]
        for spec in schema.columns[1:]:
            if spec.numeric and spec.name in df.columns:
                row[spec.name] = fmt_plain(clean[spec.name].sum(), spec)
        return pd.concat([df, pd.DataFrame([row], dtype=object)], ignore_index=True)

    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str:
        first = repr(schema.columns[0].name)
        return f"df = df[df[{first}].str.strip() != {params['label']!r}].reset_index(drop=True)\n"


class RepeatedHeader(StructuralCorruption):
    """The header line repeated inside the data, as when exports are concatenated."""

    name = "repeated_header"
    order = 5

    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        return {"count": int(rng.integers(1, 4))}

    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = len(df)
        count = min(params["count"], n - 1)
        positions = sorted(rng.choice(np.arange(1, n), size=count, replace=False).tolist())
        header = pd.DataFrame([list(df.columns)], columns=df.columns, dtype=object)
        for p in reversed(positions):
            df = pd.concat([df.iloc[:p], header, df.iloc[p:]], ignore_index=True)
        return df

    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str:
        first = repr(schema.columns[0].name)
        return f"df = df[df[{first}] != {first}].reset_index(drop=True)\n"


CORRUPTIONS: list[StructuralCorruption] = [
    SplitColumn(),
    MergeColumns(),
    DuplicateRows(),
    TotalsRow(),
    RepeatedHeader(),
]
