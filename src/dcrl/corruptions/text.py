"""Text: spelling variants of categories, stray whitespace, broken umlauts."""

from __future__ import annotations

from typing import Any

import numpy as np

from dcrl.corruptions.base import CellCorruption, pick, pick_some
from dcrl.pools import mojibake
from dcrl.schema import ColumnSpec

TRANSLIT = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
)


def normal_form(text: str) -> str:
    """The key spelling variants share: München, MUENCHEN and ' münchen' all give muenchen."""
    out = text.casefold()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        out = out.replace(a, b)
    return "".join(ch for ch in out if ch.isascii() and ch.isalnum())


class SpellingVariants(CellCorruption):
    """München, Muenchen, MÜNCHEN, ' münchen' for the same category."""

    name = "spelling_variants"
    group = "text"
    kinds = ("upper", "lower", "translit", "pad")

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.dtype == "category" and spec.allowed is not None

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "kinds": pick_some(rng, self.kinds, 1, len(self.kinds)),
            "share": round(float(rng.uniform(0.3, 0.8)), 2),
        }

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        original = out = str(value)
        if rng.random() >= params["share"]:
            return out
        for kind in pick_some(rng, params["kinds"], 1, 2):
            if kind == "upper":
                out = out.upper()
            elif kind == "lower":
                out = out.lower()
            elif kind == "translit":
                out = out.translate(TRANSLIT)
            else:
                out = pick(rng, (f" {out}", f"{out} ", f"{out}  "))
        if out == original:  # e.g. 'lower' on 'defekt': make the variant visible
            out = original.upper() if original.upper() != original else original.lower()
        if out == original:
            out = original + " "
        return out

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return (
            "def _normal_form(s):\n"
            "    s = str(s).casefold()\n"
            "    for a, b in (('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')):\n"
            "        s = s.replace(a, b)\n"
            "    return ''.join(ch for ch in s if ch.isascii() and ch.isalnum())\n"
            f"_CANONICAL = {{_normal_form(v): v for v in {list(spec.allowed or ())!r}}}\n"
            f"df[{c}] = df[{c}].map(lambda s: _CANONICAL.get(_normal_form(s), s))\n"
        )


class Whitespace(CellCorruption):
    """Leading, trailing, doubled and non-breaking spaces in free text."""

    name = "whitespace"
    group = "text"
    kinds = ("lead", "trail", "double", "nbsp")

    def applies_to(self, spec: ColumnSpec) -> bool:
        return spec.dtype == "str"

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "kinds": pick_some(rng, self.kinds, 1, len(self.kinds)),
            "share": round(float(rng.uniform(0.2, 0.6)), 2),
        }

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        out = str(value)
        if rng.random() >= params["share"]:
            return out
        kind = pick(rng, params["kinds"])
        if kind in ("double", "nbsp") and " " in out:
            return out.replace(" ", "  " if kind == "double" else " ", 1)
        if kind == "lead":
            return pick(rng, (" ", "  ", "\t")) + out
        return out + pick(rng, (" ", "  "))

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return f"df[{c}] = df[{c}].str.strip().str.replace(r'\\s+', ' ', regex=True)\n"


class Mojibake(CellCorruption):
    """UTF-8 read as Windows-1252: München becomes MÃ¼nchen."""

    name = "mojibake"
    group = "text"

    def applies_to(self, spec: ColumnSpec) -> bool:
        if spec.dtype == "category":  # needs umlauts to break
            return any(not v.isascii() for v in spec.allowed or ())
        return spec.dtype == "str"

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {"share": round(float(rng.uniform(0.3, 1.0)), 2)}

    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        out = str(value)
        if not out.isascii() and rng.random() < params["share"]:
            return mojibake(out)
        return out

    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        c = repr(spec.name)
        return (
            "def _undo_mojibake(s):\n"
            "    try:\n"
            "        return s.encode('cp1252').decode('utf-8')\n"
            "    except UnicodeError:\n"
            "        return s\n"
            f"df[{c}] = df[{c}].map(_undo_mojibake)\n"
        )


CORRUPTIONS: list[CellCorruption] = [SpellingVariants(), Whitespace(), Mojibake()]
