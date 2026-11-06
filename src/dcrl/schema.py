"""The target schema handed to the agent with every task.

The schema removes ambiguity: without it, "clean" has no single right answer. Some fields
(quantity, decimals, value range, is_id) only steer the generator and are never shown to
the agent; ``public()`` is what the agent sees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DType = Literal["int", "float", "str", "category", "date"]
Quantity = Literal["money", "weight", "length", "volume", "percent"]

DTYPES: tuple[str, ...] = ("int", "float", "str", "category", "date")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: DType
    unit: str | None = None
    allowed: tuple[str, ...] | None = None
    nullable: bool = False
    # Generator-only fields.
    quantity: Quantity | None = None
    decimals: int = 0
    min_value: float | None = None
    max_value: float | None = None
    is_id: bool = False

    @property
    def numeric(self) -> bool:
        return self.dtype in ("int", "float")

    @property
    def textual(self) -> bool:
        return self.dtype in ("str", "category")

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.dtype}
        if self.dtype == "date":
            out["format"] = "ISO 8601 (YYYY-MM-DD)"
        if self.unit:
            out["unit"] = self.unit
        if self.allowed is not None:
            out["allowed"] = list(self.allowed)
        out["nullable"] = self.nullable
        if self.is_id:
            out["unique"] = True
        return out

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["allowed"] = list(self.allowed) if self.allowed is not None else None
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnSpec:
        data = dict(data)
        if data.get("allowed") is not None:
            data["allowed"] = tuple(data["allowed"])
        return cls(**data)


@dataclass(frozen=True)
class TargetSchema:
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate column names in schema: {names}")
        for c in self.columns:
            if c.dtype not in DTYPES:
                raise ValueError(f"unknown dtype {c.dtype!r} for column {c.name!r}")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def __getitem__(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(c.name == name for c in self.columns)

    def public(self) -> list[dict[str, Any]]:
        return [c.public() for c in self.columns]

    def to_dict(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.columns]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> TargetSchema:
        return cls(tuple(ColumnSpec.from_dict(d) for d in data))

    def describe(self) -> str:
        """Plain-text schema for a prompt."""
        lines = []
        for c in self.columns:
            parts: list[str] = [c.dtype]
            if c.dtype == "date":
                parts.append("ISO 8601 YYYY-MM-DD")
            if c.unit:
                parts.append(f"unit {c.unit}")
            parts.append("nullable" if c.nullable else "not null")
            if c.is_id:
                parts.append("unique")
            line = f"- {c.name}: {', '.join(parts)}"
            if c.allowed is not None:
                line += f"; allowed values: {', '.join(c.allowed)}"
            lines.append(line)
        return "\n".join(lines)
