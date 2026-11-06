"""Tasks: a clean table broken on purpose by recorded steps, in two copies.

Both copies share the table type, the target schema and every corruption parameter; they
differ in their rows. The agent works on copy A, and the reward reruns its code on copy B,
so code that patches visible cells does not carry over while code that fixes the pattern
does.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dcrl.corruptions import CELL, REGISTRY, STRUCTURAL, Applied
from dcrl.corruptions.base import jsonable
from dcrl.render import fmt_plain, read_messy, render_plain, to_csv
from dcrl.schema import TargetSchema
from dcrl.tables import TABLE_TYPES, TableType


@dataclass(frozen=True)
class LevelSpec:
    n_rows: int
    total: tuple[int, int]  # corruptions per task, inclusive range
    structural: tuple[int, int]  # how many of them are structural


LEVELS: dict[int, LevelSpec] = {
    1: LevelSpec(50, (1, 1), (0, 0)),
    2: LevelSpec(200, (2, 3), (0, 0)),
    3: LevelSpec(1000, (4, 6), (1, 3)),
}

PRELUDE = "import datetime\nimport re\n\nimport numpy as np\nimport pandas as pd\n"


def cast_code(schema: TargetSchema) -> str:
    """Code that casts every column to its schema type and puts columns in schema order."""
    types = {c.name: c.dtype for c in schema.columns}
    return (
        f"_TARGET = {types!r}\n"
        "for _col, _type in _TARGET.items():\n"
        "    _s = df[_col]\n"
        "    if _s.dtype == object:\n"
        "        _s = _s.where(_s.astype(str).str.strip() != '')\n"
        "    if _type == 'int':\n"
        "        df[_col] = pd.to_numeric(_s).astype('Int64')\n"
        "    elif _type == 'float':\n"
        "        df[_col] = pd.to_numeric(_s).astype('float64')\n"
        "    elif _type == 'date':\n"
        "        df[_col] = pd.to_datetime(_s, format='%Y-%m-%d')\n"
        "    else:\n"
        "        df[_col] = _s.astype(object)\n"
        "df = df[list(_TARGET)].reset_index(drop=True)\n"
    )


def fix_program(schema: TargetSchema, corruptions: tuple[Applied, ...] | list[Applied]) -> str:
    """The reference solution: undo structure (in reverse), then cells, then cast."""
    parts = [PRELUDE]
    structural = sorted(
        (a for a in corruptions if a.name in STRUCTURAL),
        key=lambda a: STRUCTURAL[a.name].order,
        reverse=True,
    )
    for a in structural:
        parts.append(f"# undo {a.name}\n" + STRUCTURAL[a.name].fix_code(schema, a.params))
    for a in corruptions:
        if a.name in CELL:
            spec = schema[a.params["column"]]
            parts.append(
                f"# undo {a.name} on {spec.name}\n" + CELL[a.name].fix_code(spec, a.params)
            )
    parts.append("# cast to the target schema\n" + cast_code(schema))
    return "\n".join(parts)


def run_trusted(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run generator-written code in-process. Never use this for model-written code:
    that goes through the sandbox."""
    namespace: dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "re": re,
        "datetime": datetime,
    }
    exec(compile(code, "<fix>", "exec"), namespace)
    result = namespace["df"]
    assert isinstance(result, pd.DataFrame)
    return result


def sample_corruptions(
    schema: TargetSchema,
    level: int,
    rng: np.random.Generator,
    allowed: frozenset[str] | None = None,
    required: str | None = None,
) -> list[Applied]:
    """Pick the corruptions and their task-level parameters.

    A column gets at most one cell corruption, and each corruption type appears at most
    once per task. Structural corruptions come only at level 3, except a ``required`` one.
    """
    spec = LEVELS[level]
    allowed = frozenset(REGISTRY) if allowed is None else allowed
    if required is not None and required not in allowed:
        raise ValueError(f"required corruption {required!r} is not allowed")

    total = int(rng.integers(spec.total[0], spec.total[1] + 1))
    n_struct = int(rng.integers(spec.structural[0], spec.structural[1] + 1))
    if required in STRUCTURAL:
        n_struct = max(n_struct, 1)
    n_cell = max(0, total - n_struct)

    def ordered(names: list[str]) -> list[str]:
        names = sorted(n for n in names if n in allowed)
        names = [names[i] for i in rng.permutation(len(names))]
        if required in names:
            names.remove(required)
            names.insert(0, required)
        return names

    chosen: list[Applied] = []
    used: set[str] = set()
    for name in ordered(list(CELL)):
        if sum(a.name in CELL for a in chosen) >= n_cell:
            break
        corr = CELL[name]
        columns = [c for c in schema.columns if c.name not in used and corr.applies_to(c)]
        if not columns:
            continue
        col = columns[int(rng.integers(len(columns)))]
        params = {"column": col.name, **corr.sample_params(col, rng)}
        chosen.append(Applied(name, jsonable(params)))
        used.add(col.name)

    taken: set[str] = set()
    structural: list[Applied] = []
    for name in ordered(list(STRUCTURAL)):
        if len(structural) >= n_struct:
            break
        sparams = STRUCTURAL[name].sample_params(schema, rng, taken)
        if sparams is None:
            continue
        applied = Applied(name, jsonable(sparams))
        structural.append(applied)
        taken.update(applied.columns)

    structural.sort(key=lambda a: STRUCTURAL[a.name].order)
    chosen.extend(structural)
    if required is not None and all(a.name != required for a in chosen):
        raise ValueError(f"could not apply required corruption {required!r} to this table")
    if not chosen:
        raise ValueError("no corruption applies to this table")
    return chosen


def corrupt(
    clean: pd.DataFrame,
    schema: TargetSchema,
    corruptions: tuple[Applied, ...] | list[Applied],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Render the clean table as text, breaking it as recorded."""
    by_column = {a.params["column"]: a for a in corruptions if a.name in CELL}
    cells: dict[str, list[str]] = {}
    for spec in schema.columns:
        applied = by_column.get(spec.name)
        if applied is None:
            cells[spec.name] = [fmt_plain(v, spec) for v in clean[spec.name]]
        else:
            cells[spec.name] = CELL[applied.name].render(
                clean[spec.name], spec, applied.params, rng
            )
    df = pd.DataFrame(cells, dtype=object)
    for a in sorted(
        (a for a in corruptions if a.name in STRUCTURAL), key=lambda a: STRUCTURAL[a.name].order
    ):
        df = STRUCTURAL[a.name].apply(df, clean, schema, a.params, rng)
    return df


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class TaskCopy:
    clean: pd.DataFrame
    messy_csv: str
    schema: TargetSchema = field(repr=False)

    def messy(self) -> pd.DataFrame:
        """The table as the agent receives it: every cell a string."""
        return read_messy(self.messy_csv)

    def clean_csv(self) -> str:
        return to_csv(render_plain(self.clean, self.schema))


@dataclass(frozen=True, eq=False)
class Task:
    task_id: str
    split: str
    table_type: str
    level: int
    seed: int
    schema: TargetSchema
    corruptions: tuple[Applied, ...]
    a: TaskCopy  # the agent's copy
    b: TaskCopy  # the hidden copy

    @property
    def n_rows(self) -> int:
        return len(self.a.clean)

    def fix_program(self) -> str:
        return fix_program(self.schema, self.corruptions)

    def record(self) -> dict[str, Any]:
        """One manifest line: enough to rebuild the task, plus hashes to verify it."""
        return {
            "task_id": self.task_id,
            "split": self.split,
            "table_type": self.table_type,
            "level": self.level,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "corruptions": [a.to_dict() for a in self.corruptions],
            "schema": self.schema.public(),
            "sha256": {
                "messy_a": sha256(self.a.messy_csv),
                "messy_b": sha256(self.b.messy_csv),
                "clean_a": sha256(self.a.clean_csv()),
                "clean_b": sha256(self.b.clean_csv()),
            },
        }


def _make_copy(
    table: TableType,
    corruptions: tuple[Applied, ...],
    n_rows: int,
    seed: int,
    copy: int,
) -> TaskCopy:
    clean = table.generate(n_rows, np.random.default_rng([seed, copy, 0]))
    messy = corrupt(clean, table.schema, corruptions, np.random.default_rng([seed, copy, 1]))
    return TaskCopy(clean=clean, messy_csv=to_csv(messy), schema=table.schema)


def build_task(
    table_type: str,
    level: int,
    seed: int,
    *,
    task_id: str | None = None,
    split: str = "custom",
    allowed: frozenset[str] | None = None,
    required: str | None = None,
    corruptions: list[Applied] | None = None,
    n_rows: int | None = None,
) -> Task:
    """Build one task. Same arguments, same task, byte for byte.

    ``corruptions`` overrides sampling, which is how tests and hand-made tasks pin exactly
    what is broken.
    """
    table = TABLE_TYPES[table_type]
    if corruptions is None:
        corruptions = sample_corruptions(
            table.schema, level, np.random.default_rng([seed, 0]), allowed, required
        )
    applied = tuple(corruptions)
    rows = n_rows if n_rows is not None else LEVELS[level].n_rows
    return Task(
        task_id=task_id or f"{split}-{table_type}-L{level}-{seed}",
        split=split,
        table_type=table_type,
        level=level,
        seed=seed,
        schema=table.schema,
        corruptions=applied,
        a=_make_copy(table, applied, rows, seed, 1),
        b=_make_copy(table, applied, rows, seed, 2),
    )


def build_from_record(record: dict[str, Any]) -> Task:
    """Rebuild a task from its manifest line."""
    return build_task(
        record["table_type"],
        record["level"],
        record["seed"],
        task_id=record["task_id"],
        split=record["split"],
        corruptions=[Applied.from_dict(c) for c in record["corruptions"]],
        n_rows=record["n_rows"],
    )
