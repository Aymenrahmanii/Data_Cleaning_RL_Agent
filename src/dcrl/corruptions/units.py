"""Units: unit text in the cell, mixed units in one column (g and kg, mm and m)."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from dcrl.corruptions.base import CellCorruption, pick, pick_some
from dcrl.corruptions.numbers import decimals_of, fmt_number
from dcrl.schema import ColumnSpec

# Factor from the schema's unit to each alternative: 1 kg = 1000 g.
FACTORS: dict[str, dict[str, int]] = {
    "weight": {"kg": 1, "g": 1000},
    "length": {"m": 1, "cm": 100, "mm": 1000},
    "volume": {"l": 1, "ml": 1000},
}


class UnitInCell(CellCorruption):
    name = "unit_in_cell"
    group = "units"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return (
            spec.numeric
            and spec.unit is not None
            and spec.quantity not in ("money", "percent")
            and not spec.is_id
        )

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        style = pick(rng, ("de", "en")) if spec.dtype == "float" else "en"
        return {"sep": pick(rng, (" ", "")), "style": style}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        number = fmt_number(value, decimals_of(spec), params["style"])
        return f"{number}{params['sep']}{spec.unit}"

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        pattern = r"\s*" + re.escape(str(spec.unit)) + r"\s*$"
        code = f"df[{c}] = df[{c}].str.replace({pattern!r}, '', regex=True)\n"
        if params["style"] == "de":
            code += f"df[{c}] = df[{c}].str.replace(',', '.', regex=False)\n"
        return code


class MixedUnits(CellCorruption):
    name = "mixed_units"
    group = "units"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.quantity in FACTORS and spec.unit == next(iter(FACTORS[spec.quantity]))

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        assert spec.quantity is not None
        units = list(FACTORS[spec.quantity])
        others = pick_some(rng, units[1:], 1, len(units) - 1)
        return {
            "units": [units[0], *others],
            "sep": pick(rng, (" ", "")),
            "style": pick(rng, ("de", "en")),
        }

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        assert spec.quantity is not None
        unit = pick(rng, params["units"])
        factor = FACTORS[spec.quantity][unit]
        decimals = max(0, spec.decimals - round(math.log10(factor)))
        number = fmt_number(float(value) * factor, decimals, params["style"])
        return f"{number}{params['sep']}{unit}"

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        assert spec.quantity is not None
        c = repr(spec.name)
        factors = {u: FACTORS[spec.quantity][u] for u in params["units"]}
        return (
            f"_FACTORS = {factors!r}\n"
            "def _to_base_unit(s):\n"
            "    s = str(s).strip()\n"
            "    if s == '':\n"
            "        return np.nan\n"
            "    m = re.fullmatch(r'(-?[0-9.,]+)\\s*([^\\s0-9.,-]+)', s)\n"
            "    return float(m.group(1).replace(',', '.')) / _FACTORS[m.group(2)]\n"
            f"df[{c}] = df[{c}].map(_to_base_unit)\n"
        )


CORRUPTIONS: list[CellCorruption] = [UnitInCell(), MixedUnits()]
