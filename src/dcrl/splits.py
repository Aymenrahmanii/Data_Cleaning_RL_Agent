"""Splits: which table types and corruptions each set may use.

- train / dev: 14 table types, every corruption except the two held back.
- heldout: the other 6 table types, the same corruptions as training.
- unseen: training table types, every task includes one of the two held-back corruptions,
  so the only new thing is the corruption.

Public open-data tables and the hand-cleaned real exports join heldout later.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from dcrl.corruptions import REGISTRY
from dcrl.public import PUBLIC_TABLES
from dcrl.tables import TABLE_TYPES
from dcrl.tasks import Task, build_task

HELDOUT_TABLES: tuple[str, ...] = (
    "payroll",
    "returns",
    "energy_readings",
    "fleet",
    "quality_inspections",
    "warehouse_moves",
)
HELDOUT_PUBLIC: tuple[str, ...] = tuple(PUBLIC_TABLES)
TRAIN_TABLES: tuple[str, ...] = tuple(t for t in TABLE_TYPES if t not in HELDOUT_TABLES)
UNSEEN_CORRUPTIONS: frozenset[str] = frozenset({"excel_serial", "merge_columns"})
SEEN_CORRUPTIONS: frozenset[str] = frozenset(REGISTRY) - UNSEEN_CORRUPTIONS


@dataclass(frozen=True)
class Split:
    name: str
    code: int  # mixed into every task seed, so splits never share a seed
    tables: tuple[str, ...]
    corruptions: frozenset[str]
    required: tuple[str, ...] = ()  # each task includes one of these


SPLITS: dict[str, Split] = {
    s.name: s
    for s in (
        Split("train", 1, TRAIN_TABLES, SEEN_CORRUPTIONS),
        Split("dev", 2, TRAIN_TABLES, SEEN_CORRUPTIONS),
        Split("heldout", 3, HELDOUT_TABLES + HELDOUT_PUBLIC, SEEN_CORRUPTIONS),
        Split("unseen", 4, TRAIN_TABLES, frozenset(REGISTRY), tuple(sorted(UNSEEN_CORRUPTIONS))),
    )
}


def task_seed(split: str, base_seed: int, index: int) -> int:
    state = np.random.SeedSequence([SPLITS[split].code, base_seed, index]).generate_state(1)
    return int(state[0])


def make_split_task(split: str, base_seed: int, index: int, levels: tuple[int, ...]) -> Task:
    s = SPLITS[split]
    seed = task_seed(split, base_seed, index)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(s.tables)))
    level = levels[int(rng.integers(len(levels)))]
    required = s.required[int(rng.integers(len(s.required)))] if s.required else None
    # A required corruption may not fit every table (merge_columns needs two text columns):
    # move on to the next table type until one fits.
    for offset in range(len(s.tables)):
        table = s.tables[(start + offset) % len(s.tables)]
        try:
            return build_task(
                table,
                level,
                seed,
                task_id=f"{split}-{index:05d}",
                split=split,
                allowed=s.corruptions,
                required=required,
            )
        except ValueError:
            if required is None:
                raise
    raise ValueError(f"no table type in split {split!r} accepts {required!r}")


def generate_split(
    split: str, n: int, base_seed: int = 0, levels: tuple[int, ...] = (1, 2, 3)
) -> Iterator[Task]:
    for i in range(n):
        yield make_split_task(split, base_seed, i, levels)
