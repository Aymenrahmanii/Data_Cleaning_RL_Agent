"""Undoing the recorded corruptions gives back the clean table, on both copies."""

from __future__ import annotations

import numpy as np
import pytest

from dcrl.compare import mismatches
from dcrl.corruptions import CELL, REGISTRY, STRUCTURAL, Applied
from dcrl.splits import SPLITS, make_split_task
from dcrl.tables import TABLE_TYPES
from dcrl.tasks import Task, build_task, cast_code, run_trusted


def assert_fix_restores(task: Task) -> None:
    program = task.fix_program()
    for name, copy in (("A", task.a), ("B", task.b)):
        result = run_trusted(program, copy.messy())
        problems = mismatches(result, copy.clean, task.schema)
        assert not problems, f"{task.task_id} copy {name}: {problems}\n{task.corruptions}"


def single_corruption_cases() -> list[tuple[str, str, str]]:
    """Every cell corruption on every column it applies to, in every table type."""
    cases = []
    for table in TABLE_TYPES.values():
        for spec in table.schema.columns:
            for name, corr in CELL.items():
                if corr.applies_to(spec):
                    cases.append((table.name, name, spec.name))
    return cases


@pytest.mark.parametrize(("table", "corruption", "column"), single_corruption_cases())
def test_each_cell_corruption_roundtrips(table: str, corruption: str, column: str) -> None:
    schema = TABLE_TYPES[table].schema
    for seed in range(3):
        rng = np.random.default_rng(seed)
        params = {"column": column, **CELL[corruption].sample_params(schema[column], rng)}
        task = build_task(table, 2, seed, corruptions=[Applied(corruption, params)])
        assert_fix_restores(task)


@pytest.mark.parametrize("table", sorted(TABLE_TYPES))
@pytest.mark.parametrize("corruption", sorted(STRUCTURAL))
def test_each_structural_corruption_roundtrips(table: str, corruption: str) -> None:
    schema = TABLE_TYPES[table].schema
    params = STRUCTURAL[corruption].sample_params(schema, np.random.default_rng(1), set())
    if params is None:
        pytest.skip(f"{corruption} does not apply to {table}")
    task = build_task(table, 2, 7, corruptions=[Applied(corruption, params)])
    assert_fix_restores(task)


@pytest.mark.parametrize("table", sorted(TABLE_TYPES))
@pytest.mark.parametrize("level", [1, 2, 3])
def test_sampled_tasks_roundtrip(table: str, level: int) -> None:
    for seed in range(4):
        assert_fix_restores(build_task(table, level, 1000 * level + seed))


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_split_tasks_roundtrip(split: str) -> None:
    for i in range(25):
        assert_fix_restores(make_split_task(split, 0, i, (1, 2, 3)))


def test_heavy_level3_task_with_every_structural_fault() -> None:
    table = "customers"  # three text columns: split and merge can both apply
    schema = TABLE_TYPES[table].schema
    rng = np.random.default_rng(3)
    taken: set[str] = set()
    corruptions = []
    for name in sorted(STRUCTURAL, key=lambda n: STRUCTURAL[n].order):
        params = STRUCTURAL[name].sample_params(schema, rng, taken)
        assert params is not None
        applied = Applied(name, params)
        taken.update(applied.columns)
        corruptions.append(applied)
    cells = [
        Applied("whitespace", {"column": "firma", "kinds": ["lead", "nbsp"], "share": 0.5}),
        Applied("spelling_variants", {"column": "stadt", "kinds": ["upper", "translit", "pad"],
                                      "share": 0.8}),
        Applied("missing_placeholders", {"column": "ansprechpartner",
                                         "tokens": ["keine Angabe", "n/a"]}),
        Applied("currency_symbol", {"column": "umsatz_vorjahr", "template": "{} €",
                                    "style": "de"}),
        Applied("mixed_dates", {"column": "kunde_seit", "styles": ["de", "text", "us"]}),
    ]  # fmt: skip
    task = build_task(table, 3, 99, corruptions=cells + corruptions)
    assert_fix_restores(task)


def _cast_only_fails(task: Task) -> bool:
    program = "import numpy as np\nimport pandas as pd\n" + cast_code(task.schema)
    try:
        result = run_trusted(program, task.a.messy())
    except Exception:
        return True
    return bool(mismatches(result, task.a.clean, task.schema))


@pytest.mark.parametrize("corruption", sorted(REGISTRY))
def test_every_corruption_is_visible(corruption: str) -> None:
    """Alone on a 200-row table, each corruption breaks a cast-only "fix" almost always."""
    outcomes = []
    for table in TABLE_TYPES.values():
        schema = table.schema
        rng = np.random.default_rng(0)
        if corruption in CELL:
            specs = [c for c in schema.columns if CELL[corruption].applies_to(c)]
            params_list = [
                {"column": s.name, **CELL[corruption].sample_params(s, rng)} for s in specs
            ]
        else:
            p = STRUCTURAL[corruption].sample_params(schema, rng, set())
            params_list = [p] if p is not None else []
        for params in params_list:
            task = build_task(table.name, 2, 1, corruptions=[Applied(corruption, params)])
            outcomes.append(_cast_only_fails(task))
    assert outcomes, f"{corruption} applies to no table"
    assert np.mean(outcomes) >= 0.9, f"{corruption}: invisible in {outcomes.count(False)} cases"
