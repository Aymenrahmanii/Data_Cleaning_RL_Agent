"""Numbers: decimal comma, dot thousands, mixed formats, currency and percent signs."""

from __future__ import annotations

from typing import Any

import numpy as np

from dcrl.corruptions.base import CellCorruption, pick
from dcrl.schema import ColumnSpec

_DE = str.maketrans({",": ".", ".": ","})


def fmt_number(value: float, decimals: int, style: str = "en", thousands: bool = False) -> str:
    """12345.6 -> '12345.60', '12,345.60' (en) or '12345,60', '12.345,60' (de)."""
    out = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    return out.translate(_DE) if style == "de" else out


def decimals_of(spec: ColumnSpec) -> int:
    return spec.decimals if spec.dtype == "float" else 0


def german_to_dot(column: str) -> str:
    """Code that turns '1.234,5' into '1234.5' in place."""
    c = repr(column)
    return (
        f"df[{c}] = df[{c}].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)\n"
    )


class DecimalComma(CellCorruption):
    name = "decimal_comma"
    group = "numbers"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.dtype == "float" and not spec.is_id

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_number(float(value), spec.decimals, "de")

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return f"df[{c}] = df[{c}].str.replace(',', '.', regex=False)\n"


class ThousandsDot(CellCorruption):
    """German full format: 1.234,56 for floats, 1.234 for integers."""

    name = "thousands_dot"
    group = "numbers"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.numeric and not spec.is_id and (spec.max_value or 0) >= 2000

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_number(value, decimals_of(spec), "de", thousands=True)

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        return german_to_dot(spec.name)


class MixedNumberFormat(CellCorruption):
    """German and English formats with thousands separators in the same column."""

    name = "mixed_number_format"
    group = "numbers"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.dtype == "float" and not spec.is_id

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"share_de": round(float(rng.uniform(0.3, 0.7)), 2)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        style = "de" if rng.random() < params["share_de"] else "en"
        return fmt_number(float(value), spec.decimals, style, thousands=True)

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return (
            "def _parse_mixed_number(s):\n"
            "    s = str(s).strip()\n"
            "    if s == '':\n"
            "        return np.nan\n"
            "    cut = max(s.rfind(','), s.rfind('.'))  # the last separator is the decimal one\n"
            "    if cut == -1:\n"
            "        return float(s)\n"
            "    whole = s[:cut].replace(',', '').replace('.', '')\n"
            "    return float(whole + '.' + s[cut + 1:])\n"
            f"df[{c}] = df[{c}].map(_parse_mixed_number)\n"
        )


class CurrencySymbol(CellCorruption):
    name = "currency_symbol"
    group = "numbers"
    templates = ("{} €", "€ {}", "{} EUR", "EUR {}", "{}€")

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.quantity == "money"

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"template": pick(rng, self.templates), "style": pick(rng, ("de", "en"))}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        number = fmt_number(float(value), spec.decimals, params["style"], thousands=True)
        return str(params["template"]).format(number)

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        code = f"df[{c}] = df[{c}].str.replace(r'[^0-9,.\\-]', '', regex=True)\n"
        if params["style"] == "de":
            return code + german_to_dot(spec.name)
        return code + f"df[{c}] = df[{c}].str.replace(',', '', regex=False)\n"


class PercentSign(CellCorruption):
    name = "percent_sign"
    group = "numbers"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.quantity == "percent"

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"sep": pick(rng, ("", " ")), "style": pick(rng, ("de", "en"))}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return f"{fmt_number(float(value), spec.decimals, params['style'])}{params['sep']}%"

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        code = f"df[{c}] = df[{c}].str.replace('%', '', regex=False).str.strip()\n"
        if params["style"] == "de":
            code += f"df[{c}] = df[{c}].str.replace(',', '.', regex=False)\n"
        return code


CORRUPTIONS: list[CellCorruption] = [
    DecimalComma(),
    ThousandsDot(),
    MixedNumberFormat(),
    CurrencySymbol(),
    PercentSign(),
]
