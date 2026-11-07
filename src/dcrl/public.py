"""Public open-data tables: real, licence-checked clean tables for the held-out split.

Each table ships as a CSV inside the package, is cleaned to a target schema, and carries its
licence and source, which go into every manifest line that uses it. Rows are sampled
without replacement, so a table with fewer rows than a level asks for is capped at its size.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from dcrl.schema import ColumnSpec, TargetSchema

CC_BY_4 = "http://dcat-ap.de/def/licenses/cc-by/4.0"


@dataclass(frozen=True, eq=False)
class PublicTable:
    name: str
    licence: str
    licence_url: str
    source_url: str
    attribution: str
    clean: pd.DataFrame
    schema: TargetSchema

    @property
    def max_rows(self) -> int:
        return len(self.clean)

    def generate(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
        n = min(n_rows, len(self.clean))
        idx = np.sort(rng.permutation(len(self.clean))[:n])
        return self.clean.iloc[idx].reset_index(drop=True)

    def source(self) -> dict[str, Any]:
        return {
            "kind": "public",
            "name": self.name,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "source_url": self.source_url,
            "attribution": self.attribution,
        }


@cache
def _stuttgart_sv_employment() -> PublicTable:
    path = resources.files("dcrl").joinpath("data", "stuttgart_sv_employment.csv")
    with path.open("rb") as fh:
        raw = pd.read_csv(fh, sep=";", encoding="cp1252", dtype=str)
    if raw.shape[1] != 5:
        raise ValueError(f"unexpected layout in the source CSV: {raw.shape[1]} columns")

    district = raw.iloc[:, 1].str.strip()
    sector = raw.iloc[:, 2].str.strip()
    companies = raw.iloc[:, 3].astype("int64")
    employees = raw.iloc[:, 4].astype("int64")
    clean = pd.DataFrame(
        {
            "zeile_id": pd.Series(range(1, len(raw) + 1), dtype="Int64"),
            "stadtbezirk": district.astype(object),
            "wirtschaftsabschnitt": sector.astype(object),
            "anzahl_unternehmen": companies.astype("Int64"),
            "anzahl_beschaeftigte": employees.astype("Int64"),
        }
    )
    schema = TargetSchema(
        (
            ColumnSpec("zeile_id", "int", is_id=True, min_value=1.0),
            ColumnSpec(
                "stadtbezirk",
                "category",
                allowed=tuple(sorted(district.unique())),
            ),
            ColumnSpec(
                "wirtschaftsabschnitt",
                "category",
                allowed=tuple(sorted(sector.unique())),
            ),
            ColumnSpec(
                "anzahl_unternehmen",
                "int",
                min_value=0.0,
                max_value=float(companies.max()),
            ),
            ColumnSpec(
                "anzahl_beschaeftigte",
                "int",
                min_value=0.0,
                max_value=float(employees.max()),
            ),
        )
    )
    return PublicTable(
        name="stuttgart_sv_employment",
        licence="CC BY 4.0",
        licence_url=CC_BY_4,
        source_url="https://opendata.stuttgart.de/dataset/37edbf88-cec4-44fe-b5cf-e1ecc7ee87e6",
        attribution=(
            "Unternehmen und sozialversicherungspflichtig Beschäftigte, Stuttgart 2020, "
            "opendata.stuttgart.de (listed by Open Data Baden-Württemberg), CC BY 4.0"
        ),
        clean=clean,
        schema=schema,
    )


PUBLIC_TABLES: dict[str, PublicTable] = {t.name: t for t in (_stuttgart_sv_employment(),)}
