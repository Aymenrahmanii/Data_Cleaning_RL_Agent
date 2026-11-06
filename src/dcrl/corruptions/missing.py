"""Missing values written as placeholders: k.A., n/a, -, 0 and friends."""

from __future__ import annotations

from typing import Any

import numpy as np

from dcrl.corruptions.base import CellCorruption, fmt_plain, pick, pick_some
from dcrl.schema import ColumnSpec

TOKENS = ("k.A.", "n/a", "-", "NULL", "?", "#NV", "keine Angabe")


class MissingPlaceholders(CellCorruption):
    name = "missing_placeholders"
    group = "missing"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.nullable

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        options = list(TOKENS)
        if spec.numeric and spec.min_value is not None and spec.min_value > 0:
            options.append("0")  # only where a real zero cannot occur
        if spec.dtype == "date":
            options.append("00.00.0000")
        return {"tokens": pick_some(rng, options, 1, 3)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_plain(value, spec)

    def missing_cell(self, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return str(pick(rng, params["tokens"]))

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return f"df[{c}] = df[{c}].where(~df[{c}].str.strip().isin({params['tokens']!r}), '')\n"


CORRUPTIONS: list[CellCorruption] = [MissingPlaceholders()]
