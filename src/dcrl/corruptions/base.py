"""Corruption interfaces.

A corruption has task-level parameters, sampled once per task and shared by both copies
(which column, which placeholder, which date formats), and row-level randomness drawn from
each copy's own generator (which cells). Every corruption also writes the pandas code that
undoes it. That code fixes the pattern, never individual cells, so it works on any copy of
the task; it is the reference solution and the basis of the scripted agent.

Cell corruptions turn one typed column into text. Structural corruptions act on the whole
text table afterwards, in a fixed order, and are undone first, in reverse order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from dcrl.render import fmt_plain, is_missing
from dcrl.schema import ColumnSpec, TargetSchema


def jsonable(value: Any) -> Any:
    """Plain Python types only, so parameters survive a JSON round trip unchanged."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def pick(rng: np.random.Generator, options: Sequence[Any]) -> Any:
    return options[int(rng.integers(len(options)))]


def pick_some(rng: np.random.Generator, options: Sequence[Any], lo: int, hi: int) -> list[Any]:
    """Between lo and hi distinct options, kept in their original order."""
    k = int(rng.integers(lo, min(hi, len(options)) + 1))
    idx = sorted(rng.choice(len(options), size=k, replace=False).tolist())
    return [options[i] for i in idx]


@dataclass(frozen=True)
class Applied:
    """One recorded corruption: its name and its task-level parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Applied:
        return cls(data["name"], dict(data.get("params", {})))

    @property
    def columns(self) -> list[str]:
        if "column" in self.params:
            return [self.params["column"]]
        return list(self.params.get("columns", []))


class CellCorruption(ABC):
    name: ClassVar[str]
    group: ClassVar[str]
    structural: ClassVar[bool] = False

    @abstractmethod
    def applies_to(self, spec: ColumnSpec) -> bool: ...

    def sample_params(self, spec: ColumnSpec, rng: np.random.Generator) -> dict[str, Any]:
        return {}

    @abstractmethod
    def cell(self, value: Any, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        """Text for one non-missing cell."""

    def missing_cell(self, spec: ColumnSpec, params: dict[str, Any], rng: Any) -> str:
        return ""

    def render(
        self,
        values: pd.Series,
        spec: ColumnSpec,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> list[str]:
        return [
            self.missing_cell(spec, params, rng)
            if is_missing(v)
            else self.cell(v, spec, params, rng)
            for v in values
        ]

    @abstractmethod
    def fix_code(self, spec: ColumnSpec, params: dict[str, Any]) -> str:
        """Pandas code that undoes this corruption on ``df[spec.name]``."""


class StructuralCorruption(ABC):
    name: ClassVar[str]
    group: ClassVar[str] = "structure"
    structural: ClassVar[bool] = True
    order: ClassVar[int]

    @abstractmethod
    def sample_params(
        self, schema: TargetSchema, rng: np.random.Generator, taken: set[str]
    ) -> dict[str, Any] | None:
        """Parameters, or None when the schema offers nothing to corrupt."""

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        clean: pd.DataFrame,
        schema: TargetSchema,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def fix_code(self, schema: TargetSchema, params: dict[str, Any]) -> str: ...


Corruption = CellCorruption | StructuralCorruption

__all__ = [
    "Applied",
    "CellCorruption",
    "Corruption",
    "StructuralCorruption",
    "fmt_plain",
    "jsonable",
    "pick",
    "pick_some",
]
