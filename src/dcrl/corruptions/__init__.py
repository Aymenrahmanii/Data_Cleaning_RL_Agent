"""All corruptions, by name."""

from __future__ import annotations

from dcrl.corruptions import dates, missing, numbers, structure, text, units
from dcrl.corruptions.base import Applied, CellCorruption, Corruption, StructuralCorruption

CELL: dict[str, CellCorruption] = {
    c.name: c
    for c in [
        *numbers.CORRUPTIONS,
        *dates.CORRUPTIONS,
        *missing.CORRUPTIONS,
        *units.CORRUPTIONS,
        *text.CORRUPTIONS,
    ]
}
STRUCTURAL: dict[str, StructuralCorruption] = {c.name: c for c in structure.CORRUPTIONS}
REGISTRY: dict[str, Corruption] = {**CELL, **STRUCTURAL}

GROUPS: dict[str, list[str]] = {}
for _c in REGISTRY.values():
    GROUPS.setdefault(_c.group, []).append(_c.name)

__all__ = [
    "CELL",
    "GROUPS",
    "REGISTRY",
    "STRUCTURAL",
    "Applied",
    "CellCorruption",
    "Corruption",
    "StructuralCorruption",
]
