"""Dates: German format, mixed formats, Excel serial numbers, two-digit years, time zones."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dcrl.corruptions.base import CellCorruption, pick, pick_some
from dcrl.schema import ColumnSpec

GERMAN_MONTHS = (
    "Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez.",
)  # fmt: skip
STRPTIME = {"iso": "%Y-%m-%d", "de": "%d.%m.%Y", "de_short": "%d.%m.%y", "us": "%m/%d/%Y"}
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def fmt_date(value: Any, style: str) -> str:
    ts = pd.Timestamp(value)
    if style == "text":
        return f"{ts.day}. {GERMAN_MONTHS[ts.month - 1]} {ts.year}"
    return ts.strftime(STRPTIME[style])


def _single_format_fix(column: str, fmt: str) -> str:
    c = repr(column)
    return (
        f"_d = pd.to_datetime(df[{c}].where(df[{c}] != ''), format={fmt!r})\n"
        f"df[{c}] = _d.dt.strftime('%Y-%m-%d').fillna('')\n"
    )


class _DateCorruption(CellCorruption):
    group = "dates"

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.dtype == "date"


class GermanDate(_DateCorruption):
    name = "german_date"

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_date(value, "de")

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        return _single_format_fix(spec.name, STRPTIME["de"])


class TwoDigitYear(_DateCorruption):
    """05.11.26: unambiguous because every generated date lies in 2000-2068."""

    name = "two_digit_year"

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_date(value, "de_short")

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        return _single_format_fix(spec.name, STRPTIME["de_short"])


class MixedDates(_DateCorruption):
    """2026-11-05, 05.11.2026, 05.11.26, 11/05/2026 and 5. Nov. 2026 in one column."""

    name = "mixed_dates"
    styles = ("iso", "de", "de_short", "us", "text")

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"styles": pick_some(rng, self.styles, 2, 4)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_date(value, pick(rng, params["styles"]))

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        formats = [STRPTIME[s] for s in params["styles"] if s != "text"]
        months = {m[:3].lower(): i + 1 for i, m in enumerate(GERMAN_MONTHS)}
        return (
            f"_DATE_FORMATS = {formats!r}\n"
            f"_MONTHS = {months!r}\n"
            "def _parse_mixed_date(s):\n"
            "    s = str(s).strip()\n"
            "    if s == '':\n"
            "        return ''\n"
            "    m = re.fullmatch(r'(\\d{1,2})\\.\\s*([^\\d\\s.]+)\\.?\\s*(\\d{4})', s)\n"
            "    if m:\n"
            "        month = _MONTHS[m.group(2)[:3].lower()]\n"
            "        return f'{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}'\n"
            "    for fmt in _DATE_FORMATS:\n"
            "        try:\n"
            "            return datetime.datetime.strptime(s, fmt).strftime('%Y-%m-%d')\n"
            "        except ValueError:\n"
            "            pass\n"
            "    return s\n"
            f"df[{c}] = df[{c}].map(_parse_mixed_date)\n"
        )


class ExcelSerial(_DateCorruption):
    """Some or all dates as Excel serial day numbers (46331 = 2026-11-05)."""

    name = "excel_serial"

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"share": round(float(rng.uniform(0.3, 1.0)), 2)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        if rng.random() < params["share"]:
            return str((pd.Timestamp(value) - EXCEL_EPOCH).days)
        return fmt_date(value, "iso")

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return (
            f"_n = pd.to_numeric(df[{c}], errors='coerce')\n"
            "_serial = _n.notna()\n"
            f"df.loc[_serial, {c}] = (pd.Timestamp('1899-12-30')"
            " + pd.to_timedelta(_n[_serial].astype('int64'), unit='D')).dt.strftime('%Y-%m-%d')\n"
        )


class TimezoneSuffix(_DateCorruption):
    """A date exported as a timestamp with a time zone: 2026-11-05T00:00:00+01:00."""

    name = "timezone_suffix"
    suffixes = (
        "T00:00:00+01:00",
        "T00:00:00Z",
        " 00:00:00 MEZ",
        " 00:00 UTC",
        "T00:00:00.000+0100",
    )

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"suffix": pick(rng, self.suffixes)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return fmt_date(value, "iso") + params["suffix"]

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return f"df[{c}] = df[{c}].str.slice(0, 10)\n"


CORRUPTIONS: list[CellCorruption] = [
    GermanDate(),
    TwoDigitYear(),
    MixedDates(),
    ExcelSerial(),
    TimezoneSuffix(),
]
